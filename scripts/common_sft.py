"""Common response-only QLoRA SFT; executed grammar corpus, never evaluation routes."""
import argparse
import fcntl
import gc
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import random
import resource
import subprocess
import threading
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer
from qlora_pilot import load_base
from stage3_common import CODE, read, write, append, digest, stamp


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    write(tmp, obj)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def capture(model, optimizer, progress):
    return dict(parameters={n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
                optimizer=optimizer.state_dict(), progress=progress,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())


def restore(model, optimizer, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state['parameters']: p.copy_(state['parameters'][n].to(p.device))
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])


def save_state(path, model, optimizer, progress):
    tmp = path.with_suffix('.tmp')
    torch.save(capture(model, optimizer, progress), tmp); tmp.replace(path)


def encode(rows, tokenizer, limit):
    encoded = []
    for r in rows:
        prompt = tokenizer.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True)
        before = tokenizer(prompt, add_special_tokens=False)['input_ids']
        after = tokenizer(r['response'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
        assert len(before) + len(after) <= limit, (r['example_id'], len(before), len(after))
        encoded.append(dict(example_id=r['example_id'], parent_sha256=r['parent_sha256'],
                            input_ids=before + after, labels=[-100] * len(before) + after,
                            prompt_tokens=len(before), response_tokens=len(after)))
    return encoded


def batch(row):
    ids = torch.tensor([row['input_ids']], device='cuda')
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                labels=torch.tensor([row['labels']], device='cuda'))


def score(model, rows):
    model.eval(); nll_sum = 0.; count = 0; detail = []
    with torch.inference_mode():
        for row in rows:
            loss = float(model(**batch(row)).loss)
            assert np.isfinite(loss)
            n = row['response_tokens']; nll_sum += loss * n; count += n
            detail.append(dict(example_id=row['example_id'], nll=loss, response_tokens=n))
    torch.cuda.synchronize()
    return dict(mean_response_nll=nll_sum / count, response_tokens=count, examples=len(rows), detail=detail)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--resume', action='store_true'); a = ap.parse_args(); root = a.root
    cfgpath = CODE / 'configs/sft_v1.json'; cfg = read(cfgpath)
    out = CODE / 'results/training/sft-v1'; ckdir = root / 'checkpoints/sft-v1'
    out.mkdir(parents=True, exist_ok=True); ckdir.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'completed.json').exists(), 'SFT already complete; do not retrain'
    assert a.resume or not (out / 'protocol_frozen.json').exists(), 'Existing run requires explicit --resume'
    corpus = CODE / cfg['corpus']; assert read(corpus / 'verification.json')['status'] == 'passed'
    assert read(CODE / 'results/training/qlora-pilot-v1/summary.json')['status'] == 'passed'
    manifest = read(root / 'artifacts/qwen_download.json'); assert manifest['config']['revision'] == cfg['base_revision']
    provenance = dict(protocol=cfg, protocol_sha256=digest(cfgpath),
                      corpus_sha256={s: digest(corpus / (s + '.jsonl')) for s in ('train', 'validation', 'heldout')},
                      source_sha256={str(p.relative_to(CODE)): digest(p) for p in
                                     (Path(__file__), CODE / 'scripts/qlora_pilot.py', CODE / 'scripts/stage3_common.py')})
    if (out / 'protocol_frozen.json').exists():
        old = read(out / 'protocol_frozen.json')
        for k, v in provenance.items(): assert old[k] == v, k
    else:
        write(out / 'protocol_frozen.json', dict(at_utc=stamp(), **provenance))
        source = out / 'executed_source'; source.mkdir()
        for p in (Path(__file__), cfgpath): (source / p.name).write_bytes(p.read_bytes())
    start, cpu = time.perf_counter(), time.process_time(); torch.set_num_threads(1); seed_all(cfg['seed'])
    attempt = stamp(); stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                                                 '--format=csv,noheader,nounits'], text=True).strip()
                append(out / 'gpu_samples.jsonl', dict(attempt=attempt, seconds=time.perf_counter()-start, gpu=value))
            except Exception: pass
            stop.wait(3)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    try:
        tokenizer = AutoTokenizer.from_pretrained(manifest['snapshot_path'], local_files_only=True)
        # Heldout response contents are not loaded until the selected adapter has been frozen.
        data = {s: encode([json.loads(x) for x in (corpus / (s + '.jsonl')).read_text().splitlines()],
                          tokenizer, cfg['max_sequence_tokens']) for s in ('train', 'validation')}
        stats = {s: dict(examples=len(v), prompt_tokens=sum(r['prompt_tokens'] for r in v),
                         response_tokens=sum(r['response_tokens'] for r in v),
                         min_total_tokens=min(len(r['input_ids']) for r in v),
                         max_total_tokens=max(len(r['input_ids']) for r in v)) for s, v in data.items()}
        write(out / 'token_lengths.json', stats)
        for s, rows in data.items():
            (out / (s + '_tokenized.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        torch.cuda.reset_peak_memory_stats(); load_start = time.perf_counter()
        lc = cfg['lora']
        model = get_peft_model(load_base(manifest['snapshot_path'], True), LoraConfig(r=lc['rank'],
            lora_alpha=lc['alpha'], lora_dropout=lc['dropout'], target_modules=lc['target_modules'],
            bias='none', task_type='CAUSAL_LM'))
        parameters = [p for p in model.parameters() if p.requires_grad]
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert names and all('lora_' in n for n in names)
        opt = torch.optim.AdamW(parameters, lr=cfg['optimizer']['learning_rate'], weight_decay=cfg['optimizer']['weight_decay'])
        torch.cuda.synchronize(); load_seconds = time.perf_counter() - load_start
        progress = dict(epoch=1, next_group=0, logical_updates=0, best_epoch=0, best_validation_nll=None)
        if a.resume and (ckdir / 'last.pt').exists():
            state = torch.load(ckdir / 'last.pt', map_location='cpu'); restore(model, opt, state); progress = state['progress']; del state
        else:
            val = score(model, data['validation']); val.update(epoch=0, at_utc=stamp())
            write(out / 'validation_epoch0.json', val); model.save_pretrained(ckdir / 'epoch0', safe_serialization=True)
            progress['best_validation_nll'] = val['mean_response_nll']; save_state(ckdir / 'last.pt', model, opt, progress)
        def step(indices, kind, logical_step):
            step_start, step_cpu = time.perf_counter(), time.process_time(); model.train(); opt.zero_grad(set_to_none=True)
            losses = []
            for index in indices:
                loss = model(**batch(data['train'][index])).loss; assert torch.isfinite(loss)
                (loss / len(indices)).backward(); losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['optimizer']['gradient_clip']); assert torch.isfinite(norm)
            opt.step(); torch.cuda.synchronize()
            assert all(p.grad is None for n, p in model.named_parameters() if 'lora_' not in n)
            event = dict(at_utc=stamp(), attempt=attempt, kind=kind, logical_step=logical_step,
                         example_ids=[data['train'][i]['example_id'] for i in indices],
                         mean_example_nll=float(np.mean(losses)), grad_norm=float(norm), microbatches=len(indices),
                         prompt_tokens=sum(data['train'][i]['prompt_tokens'] for i in indices),
                         response_tokens=sum(data['train'][i]['response_tokens'] for i in indices),
                         wall_seconds=time.perf_counter()-step_start, cpu_seconds=time.process_time()-step_cpu)
            append(out / 'update_events.jsonl', event)
            return event
        while progress['epoch'] <= cfg['epochs']:
            epoch = progress['epoch']; order = list(range(len(data['train'])))
            random.Random(cfg['seed'] + epoch).shuffle(order)
            size = cfg['gradient_accumulation']; groups = [order[i:i+size] for i in range(0, len(order), size)]
            while progress['next_group'] < len(groups):
                i = progress['next_group']; step(groups[i], 'training', progress['logical_updates'] + 1)
                progress['logical_updates'] += 1; progress['next_group'] += 1
                save_state(ckdir / 'last.pt', model, opt, progress)
                if progress['logical_updates'] == 1 and not (out / 'resume_verification.json').exists():
                    # Both diagnostic updates are rolled back: logical training resumes from update one.
                    expected_event = step(groups[1], 'restore_reference_rolled_back', 2)
                    expected = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
                    state = torch.load(ckdir / 'last.pt', map_location='cpu'); restore(model, opt, state)
                    actual_event = step(groups[1], 'restore_replay_rolled_back', 2)
                    error = max(float((p.detach().cpu()-expected[n]).abs().max()) for n, p in model.named_parameters() if p.requires_grad)
                    loss_error = abs(expected_event['mean_example_nll'] - actual_event['mean_example_nll'])
                    assert error <= 1e-6 and loss_error <= 1e-5, (error, loss_error)
                    restore(model, opt, state)
                    write(out / 'resume_verification.json', dict(status='passed', at_utc=stamp(), max_parameter_error=error,
                        mean_loss_error=loss_error, populated_optimizer_states=len(state['optimizer']['state']),
                        actual_corpus=True, extra_physical_updates=2, rolled_back_to_logical_update=1,
                        extra_wall_seconds=expected_event['wall_seconds']+actual_event['wall_seconds']))
                    del state, expected
                atomic_json(out / 'progress.json', dict(at_utc=stamp(), pid=os.getpid(), **progress))
                if progress['logical_updates'] % 10 == 0: print(json.dumps(progress), flush=True)
            val = score(model, data['validation']); val.update(epoch=epoch, at_utc=stamp())
            write(out / f'validation_epoch{epoch}.json', val); model.save_pretrained(ckdir / f'epoch{epoch}', safe_serialization=True)
            if val['mean_response_nll'] < progress['best_validation_nll']:
                progress['best_validation_nll'] = val['mean_response_nll']; progress['best_epoch'] = epoch
            progress['epoch'] += 1; progress['next_group'] = 0; save_state(ckdir / 'last.pt', model, opt, progress)
        selected = ckdir / f"epoch{progress['best_epoch']}"
        frozen = dict(at_utc=stamp(), selected_epoch=progress['best_epoch'], selected_adapter=str(selected),
                      adapter_sha256=digest(selected / 'adapter_model.safetensors'),
                      validation_nll_by_epoch=[read(out / f'validation_epoch{e}.json')['mean_response_nll'] for e in range(cfg['epochs']+1)],
                      criterion=cfg['checkpoint_selection'], family_heldout_opened=False, final_opened=False)
        if not (out / 'checkpoint_frozen.json').exists(): write(out / 'checkpoint_frozen.json', frozen)
        else: assert read(out / 'checkpoint_frozen.json')['adapter_sha256'] == frozen['adapter_sha256']
        del model, opt, parameters; gc.collect(); torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(load_base(manifest['snapshot_path'], False), selected, is_trainable=False).eval()
        rows = [json.loads(x) for x in (corpus / 'heldout.jsonl').read_text().splitlines()]
        heldout = encode(rows, tokenizer, cfg['max_sequence_tokens']); result = score(model, heldout)
        result.update(at_utc=stamp(), frozen_checkpoint_sha256=digest(out / 'checkpoint_frozen.json'),
                      interpretation='Program-family response NLL on shared discovery TSP instances, not unseen-instance route improvement')
        write(out / 'heldout_nll.json', result)
        diag = cfg['heldout_generation']; parents = set(); chosen = []
        for row in sorted(rows, key=lambda r: r['example_id']):
            if row['parent_sha256'] not in parents:
                parents.add(row['parent_sha256']); chosen.append(row)
            if len(chosen) == diag['unique_parent_prompts']: break
        model.config.use_cache = True
        for i, row in enumerate(chosen):
            dest = out / f'heldout_generation_{i:02d}.json'
            if dest.exists(): continue
            prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda'); seed_all(diag['seed'] + i)
            begin = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**ids, max_new_tokens=diag['max_new_tokens'], do_sample=True,
                    temperature=diag['temperature'], top_p=diag['top_p'], top_k=diag['top_k'], pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize(); output = generated[0, ids['input_ids'].shape[1]:]
            write(dest, dict(example_id=row['example_id'], messages=row['messages'], prompt=prompt,
                response=tokenizer.decode(output, skip_special_tokens=True), seed=diag['seed']+i,
                input_tokens=int(ids['input_ids'].numel()), output_tokens=int(output.numel()),
                wall_seconds=time.perf_counter()-begin, scope='heldout diagnostic only; never checkpoint or prompt selection'))
        events = [json.loads(x) for x in (out / 'update_events.jsonl').read_text().splitlines()]
        training = [e for e in events if e['kind'] == 'training']
        summary = dict(status='completed', at_utc=stamp(), selected_epoch=progress['best_epoch'],
            selected_adapter_sha256=frozen['adapter_sha256'], logical_updates=progress['logical_updates'],
            physical_optimizer_updates_including_restore_and_replay=len(events),
            physical_microbatches=sum(e['microbatches'] for e in events), trainable_parameters=2523136,
            training_event_prompt_tokens=sum(e['prompt_tokens'] for e in training),
            training_event_response_tokens=sum(e['response_tokens'] for e in training),
            all_update_prompt_tokens=sum(e['prompt_tokens'] for e in events),
            all_update_response_tokens=sum(e['response_tokens'] for e in events),
            last_attempt_wall_seconds=time.perf_counter()-start, last_attempt_cpu_seconds=time.process_time()-cpu,
            model_load_seconds=load_seconds, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            research_discovery_candidates=0, heldout_diagnostic_generations=len(chosen), final_opened=False,
            limitation='One validation family; grammar local-improvement targets; NLL does not establish route improvement')
        write(out / 'completed.json', summary); print(json.dumps(summary, indent=2))
    except Exception:
        append(out / 'failures.jsonl', dict(at_utc=stamp(), attempt=attempt, error=traceback.format_exc(),
            wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), final_opened=False))
        raise
    finally:
        stop.set(); thread.join(timeout=5)
        append(out / 'attempt_costs.jsonl', dict(attempt=attempt, ended_at_utc=stamp(),
            wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu))


if __name__ == '__main__': main()
