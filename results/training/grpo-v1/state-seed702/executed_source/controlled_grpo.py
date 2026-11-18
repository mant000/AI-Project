"""Small fixed-budget outcome GRPO vs bounded state auxiliary, on executed TSP rules."""
import argparse
from collections import defaultdict
import fcntl
import gc
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer
from stage3_common import CODE, read, write, append, digest, stamp
from common_sft import seed_all, capture, restore, save_state, atomic_json
from qlora_pilot import load_base
from training_study import TrainingStudy, group_rewards, monitor_gpu
from tsp_value.programs import canonical_program, INITIAL


def fixed_parents():
    rows = [json.loads(x) for x in (CODE/'results/corpus-v1/train.jsonl').read_text().splitlines()]
    families = defaultdict(dict)
    for row in rows: families[row['family']][row['target_sha256']] = canonical_program(row['response'])
    initial = canonical_program(INITIAL); parents = [initial]; seen = {initial['program_sha256']}
    indexed = {f: [v[k] for k in sorted(v)] for f, v in families.items()}
    depth = 0
    while len(parents) < 12:
        for family in sorted(indexed):
            if depth < len(indexed[family]):
                p = indexed[family][depth]
                if p['program_sha256'] not in seen: parents.append(p); seen.add(p['program_sha256'])
                if len(parents) == 12: break
        depth += 1
    return parents


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--arm', choices=['terminal', 'state'], required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--resume', action='store_true'); a = ap.parse_args(); root = a.root
    cfgpath = CODE/'configs/grpo_v1.json'; cfg = read(cfgpath); assert a.seed in cfg['optimization_seeds']
    sft = CODE/'results/training/sft-v1'; assert read(sft/'completed.json')['status'] == 'completed'
    common = read(sft/'checkpoint_frozen.json'); adapter = Path(common['selected_adapter'])
    assert digest(adapter/'adapter_model.safetensors') == common['adapter_sha256']
    out = CODE/f'results/training/grpo-v1/{a.arm}-seed{a.seed}'
    ckdir = root/f'checkpoints/grpo-v1/{a.arm}-seed{a.seed}'
    out.mkdir(parents=True, exist_ok=True); ckdir.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out/'completed.json').exists(), 'Completed run must never be retrained'
    assert a.resume or not (out/'protocol_frozen.json').exists(), 'Existing run requires --resume'
    study = TrainingStudy(root, out, CODE/'results/training/grpo-prepare-v1')
    assert study.cfg == cfg
    provenance = dict(protocol=cfg, protocol_sha256=digest(cfgpath), arm=a.arm, seed=a.seed,
        common_sft_adapter_sha256=common['adapter_sha256'], bank_sha256=study.bankhash,
        source_sha256={str(p.relative_to(CODE)): digest(p) for p in
            (Path(__file__), CODE/'scripts/training_study.py', CODE/'scripts/common_sft.py', CODE/'scripts/qlora_pilot.py',
             CODE/'scripts/stage3_worker.py', CODE/'scripts/core_discover.py', CODE/'scripts/stage3_discover.py',
             CODE/'src/tsp_value/programs.py', CODE/'src/tsp_value/solver.py')})
    if (out/'protocol_frozen.json').exists():
        old = read(out/'protocol_frozen.json')
        for k, v in provenance.items(): assert old[k] == v, k
    else:
        write(out/'protocol_frozen.json', dict(at_utc=stamp(), **provenance))
        source = out/'executed_source'; source.mkdir()
        for p in (Path(__file__), CODE/'scripts/training_study.py', cfgpath): (source/p.name).write_bytes(p.read_bytes())
    parents = fixed_parents(); assert len(parents) == cfg['groups_per_run']
    parentpath = out/'fixed_parents.json'
    if parentpath.exists(): assert read(parentpath)['parents'] == parents
    else: write(parentpath, dict(at_utc=stamp(), parents=parents, outcomes_used_for_parent_selection=False))
    start, cpu = time.perf_counter(), time.process_time(); torch.set_num_threads(1); seed_all(a.seed)
    attempt = stamp(); manifest = read(root/'artifacts/qwen_download.json'); torch.cuda.reset_peak_memory_stats()
    stop, thread = monitor_gpu(out, attempt, start)
    try:
        tokenizer = AutoTokenizer.from_pretrained(manifest['snapshot_path'], local_files_only=True)
        model = PeftModel.from_pretrained(load_base(manifest['snapshot_path'], True), adapter,
                                         adapter_name='policy', is_trainable=True)
        model.load_adapter(adapter, adapter_name='reference', is_trainable=False)
        def activate(name):
            model.set_adapter(name)
            for n, p in model.named_parameters(): p.requires_grad_('.policy.' in n and name == 'policy')
        activate('policy')
        parameters = [p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in parameters) == 2523136
        reference_start = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if '.reference.' in n}
        opt = torch.optim.AdamW(parameters, lr=cfg['optimizer']['learning_rate'], weight_decay=cfg['optimizer']['weight_decay'])
        progress = dict(next_group=0, logical_updates=0, zero_variance_groups=0)
        if a.resume and (ckdir/'last.pt').exists():
            state = torch.load(ckdir/'last.pt', map_location='cpu'); restore(model, opt, state); progress = state['progress']; del state
        def log_probs(raw, grad=False):
            ids = torch.tensor([raw['input_token_ids']+raw['output_token_ids']], device='cuda')
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            pred = result[:, len(raw['input_token_ids'])-1:-1, :].float()
            target = ids[:, len(raw['input_token_ids']):]
            return F.log_softmax(pred, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
        def update(group, kind, logical_step):
            activate('policy'); model.train(); opt.zero_grad(set_to_none=True); begin, begin_cpu = time.perf_counter(), time.process_time()
            losses = []; kls = []; ratios = []; clipfractions = []
            for raw, old, ref, advantage in zip(group['raw'], group['old_logps'], group['reference_logps'], group['advantages']):
                current = log_probs(raw, True)
                old = torch.tensor(old, device='cuda'); ref = torch.tensor(ref, device='cuda')
                logratio = current-old; ratio = torch.exp(logratio)
                surrogate = torch.minimum(ratio*advantage, ratio.clamp(1-cfg['ppo_clip'], 1+cfg['ppo_clip'])*advantage)
                delta = ref-current; kl = torch.exp(delta)-delta-1
                loss = (-surrogate+cfg['kl_beta']*kl).mean()/len(group['raw'])
                assert torch.isfinite(loss), 'Non-finite GRPO objective'
                loss.backward(); losses.append(float(loss.detach()))
                kls.append(float(kl.detach().mean())); ratios.append(float(ratio.detach().mean()))
                clipfractions.append(float(((ratio.detach()-1).abs()>cfg['ppo_clip']).float().mean()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['optimizer']['gradient_clip']); assert torch.isfinite(norm)
            opt.step(); torch.cuda.synchronize()
            assert all(p.grad is None for n, p in model.named_parameters() if '.policy.' not in n)
            event = dict(at_utc=stamp(), attempt=attempt, kind=kind, logical_step=logical_step, group_index=group['index'],
                loss=sum(losses), mean_kl=float(np.mean(kls)), mean_ratio=float(np.mean(ratios)),
                mean_clip_fraction=float(np.mean(clipfractions)), grad_norm=float(norm),
                input_tokens=sum(r['input_tokens'] for r in group['raw']), response_tokens=sum(r['output_tokens'] for r in group['raw']),
                wall_seconds=time.perf_counter()-begin, process_cpu_seconds=time.process_time()-begin_cpu)
            append(out/'update_events.jsonl', event); return event
        for index in range(progress['next_group'], cfg['groups_per_run']):
            groupstart = time.perf_counter(); parent = parents[index]; study.save_program(parent)
            ev, _ = study.endpoint(parent, 'discovery', cfg['endpoint_solver_seeds']); assert ev['ok']
            messages = study.fixed_parent_prompt(parent, ev); raws = []; candidates = []
            activate('policy'); model.eval(); model.config.use_cache = True
            for j in range(cfg['outputs_per_group']):
                cid = f'group{index:02d}-output{j}'; dest = out/'generations'/f'{cid}.json'
                if dest.exists():
                    raw = read(dest); assert raw['messages'] == messages
                else:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
                    n = int(ids['input_ids'].numel()); assert n+cfg['generation']['max_new_tokens'] <= cfg['max_context_tokens']
                    sampling_seed = a.seed*10000+index*10+j; seed_all(sampling_seed); begin = time.perf_counter()
                    with torch.inference_mode(): generated = model.generate(**ids, **cfg['generation'], pad_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize(); output = generated[0, n:].cpu().tolist()
                    raw = dict(candidate_id=cid, at_utc=stamp(), sampling_seed=sampling_seed, messages=messages,
                        rendered_prompt=prompt, input_token_ids=ids['input_ids'][0].cpu().tolist(), output_token_ids=output,
                        response=tokenizer.decode(output, skip_special_tokens=True), input_tokens=n, output_tokens=len(output),
                        generation_wall_seconds=time.perf_counter()-begin, hit_output_cap=len(output)==cfg['generation']['max_new_tokens'])
                    write(dest, raw)
                raws.append(raw); path = out/'candidates'/f'{cid}.json'
                if path.exists(): candidate = read(path)
                else:
                    candidate = dict(candidate_id=cid, group_index=index, output_index=j, at_utc=stamp(), valid=False,
                        parent_sha256=parent['program_sha256'], terminal_reward=cfg['reward']['invalid'])
                    try:
                        p = canonical_program(raw['response']); study.save_program(p); candidate.update(**p)
                        outcome, cached = study.endpoint(p, 'discovery', cfg['endpoint_solver_seeds'])
                        if not outcome['ok']: raise ValueError(outcome.get('error', 'Endpoint failed'))
                        candidate.update(valid=True, mean_discovery_gap_percent=outcome['mean_gap_percent'], evaluation_reused=cached,
                            measured_improvement_gap_pp=ev['mean_gap_percent']-outcome['mean_gap_percent'],
                            terminal_reward=float(np.clip(ev['mean_gap_percent']-outcome['mean_gap_percent'], -1., 1.)),
                            same_endpoint_routes_as_parent=all(x['best_tour']==y['best_tour'] for x, y in zip(ev['results'], outcome['results'])))
                        candidate['value_feedback'] = study.feedback(parent, p, 'B')
                    except Exception: candidate['error'] = traceback.format_exc()
                    write(path, candidate)
                candidates.append(candidate)
            reward = group_rewards(candidates, cfg, a.arm)
            groupfile = out/f'group{index:02d}.json'
            if groupfile.exists():
                group = read(groupfile); assert group['rewards'] == reward['rewards']
            else:
                model.eval(); model.config.use_cache = False
                activate('policy')
                with torch.no_grad(): old_logps = [log_probs(r).cpu().tolist() for r in raws]
                activate('reference')
                with torch.no_grad(): ref_logps = [log_probs(r).cpu().tolist() for r in raws]
                activate('policy')
                group = dict(index=index, parent_sha256=parent['program_sha256'], raw=raws,
                    old_logps=old_logps, reference_logps=ref_logps, **reward)
                write(groupfile, group)
            if reward['skipped_zero_variance']:
                progress['zero_variance_groups'] += 1
            else:
                event = update(group, 'training', progress['logical_updates']+1); progress['logical_updates'] += 1
                if not (out/'resume_verification.json').exists():
                    # Populate optimizer first. Two further same-group update probes are reverted.
                    save_state(ckdir/'restore_probe.pt', model, opt, progress)
                    expected_event = update(group, 'restore_reference_rolled_back', progress['logical_updates']+1)
                    expected = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
                    state = torch.load(ckdir/'restore_probe.pt', map_location='cpu'); restore(model, opt, state)
                    actual_event = update(group, 'restore_replay_rolled_back', progress['logical_updates']+1)
                    error = max(float((p.detach().cpu()-expected[n]).abs().max()) for n, p in model.named_parameters() if p.requires_grad)
                    loss_error = abs(expected_event['loss']-actual_event['loss']); assert error <= 1e-6 and loss_error <= 1e-5
                    restore(model, opt, state)
                    write(out/'resume_verification.json', dict(status='passed', at_utc=stamp(), max_parameter_error=error,
                        loss_error=loss_error, populated_optimizer_states=len(state['optimizer']['state']),
                        actual_group=index, extra_physical_updates=2, verification_updates_rolled_back=True))
                    del expected, state
            progress['next_group'] = index+1; save_state(ckdir/'last.pt', model, opt, progress)
            activate('policy'); model.save_pretrained(ckdir/f'group{index:02d}', selected_adapters=['policy'], safe_serialization=True)
            atomic_json(out/'progress.json', dict(at_utc=stamp(), pid=os.getpid(), **progress))
            append(out/'group_costs.jsonl', dict(at_utc=stamp(), attempt=attempt, group=index,
                wall_seconds=time.perf_counter()-groupstart, reward_std=reward['reward_std'],
                logical_updates=progress['logical_updates'], zero_variance=reward['skipped_zero_variance']))
            print(json.dumps(dict(group=index, **progress, rewards=reward['rewards'])), flush=True)
        activate('policy'); model.save_pretrained(ckdir/'final', selected_adapters=['policy'], safe_serialization=True)
        finaldir = ckdir/'final/policy'
        ref_error = max(float((p.detach().cpu()-reference_start[n]).abs().max()) for n, p in model.named_parameters() if n in reference_start)
        assert ref_error == 0.
        write(out/'adapter_frozen.json', dict(at_utc=stamp(), adapter_path=str(finaldir),
            adapter_sha256=digest(finaldir/'adapter_model.safetensors'), common_sft_sha256=common['adapter_sha256'],
            selection='last predefined group; no validation or favorable seed choice', **progress, final_opened=False))
        generations = [read(p) for p in sorted((out/'generations').glob('*.json'))]
        candidates = [read(p) for p in sorted((out/'candidates').glob('*.json'))]
        evaluations = [read(p) for p in (out/'evaluations').glob('*.json')]
        events = [json.loads(x) for x in (out/'update_events.jsonl').read_text().splitlines()] if (out/'update_events.jsonl').exists() else []
        summary = dict(status='completed', at_utc=stamp(), arm=a.arm, seed=a.seed,
            candidates=len(candidates), valid_candidates=sum(r['valid'] for r in candidates),
            unique_programs=len({r['program_sha256'] for r in candidates if r['valid']}),
            input_tokens=sum(r['input_tokens'] for r in generations), output_tokens=sum(r['output_tokens'] for r in generations),
            generation_wall_seconds=sum(r['generation_wall_seconds'] for r in generations),
            real_endpoint_routes=sum(len(r.get('results', [])) for r in evaluations),
            isolated_worker_cpu_seconds=sum(r['cpu_seconds'] for r in evaluations),
            physical_optimizer_updates_including_restore=len(events), **progress,
            frozen_reference_max_parameter_error=ref_error,
            last_attempt_wall_seconds=time.perf_counter()-start, last_attempt_cpu_seconds=time.process_time()-cpu,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, final_opened=False)
        write(out/'completed.json', summary); print(json.dumps(summary, indent=2))
    except Exception:
        append(out/'failures.jsonl', dict(at_utc=stamp(), attempt=attempt, error=traceback.format_exc(),
            wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), final_opened=False)); raise
    finally:
        stop.set(); thread.join(timeout=5)
        append(out/'attempt_costs.jsonl', dict(attempt=attempt, ended_at_utc=stamp(),
            wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu))


if __name__ == '__main__': main()
