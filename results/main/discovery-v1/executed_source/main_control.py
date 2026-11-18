"""Same-precision frozen/updated generators crossed with endpoint/state feedback."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from qlora_pilot import load_base
from stage3_common import CODE, read, write, append, digest, stamp
from training_study import TrainingStudy, monitor_gpu
from common_sft import atomic_json
from tsp_value.programs import canonical_program, INITIAL


def model_manifest(root, cfg):
    assert read(CODE/'results/training/grpo-v1/verification.json')['status'] == 'passed'
    common = read(CODE/'results/training/sft-v1/checkpoint_frozen.json')
    assert read(CODE/'results/training/sft-v1/verification.json')['status'] == 'passed'
    paths = {'original': None, 'sft': common['selected_adapter']}
    hashes = {'original': None, 'sft': common['adapter_sha256']}
    for arm in ('terminal', 'state'):
        for seed in cfg['optimization_seeds']:
            p = CODE/f'results/training/grpo-v1/{arm}-seed{seed}'
            assert read(p/'completed.json')['status'] == 'completed'
            meta = read(p/'adapter_frozen.json'); name = f'grpo-{arm}-{seed}'
            paths[name] = meta['adapter_path']; hashes[name] = meta['adapter_sha256']
    for name, path in paths.items():
        if path: assert digest(Path(path)/'adapter_model.safetensors') == hashes[name]
    assert set(paths) == set(cfg['generators']); return paths, hashes


def freeze_programs(study, contexts):
    cfg = study.cfg; frozen = []
    for (generator, seed, arm), ctx in contexts.items():
        ranked = sorted(ctx['population'], key=lambda p: study.endpoint(p, 'discovery', cfg['endpoint_solver_seeds'])[0]['mean_gap_percent'])
        elites = ranked[:cfg['elite_count']]; gaps = []
        for p in elites:
            result, _ = study.endpoint(p, 'program_select', cfg['program_select_solver_seeds'])
            gaps.append(result['mean_gap_percent'] if result['ok'] else 1e100)
        selected = elites[int(np.argmin(gaps))]
        frozen.append(dict(generator=generator, discovery_seed=seed, feedback_arm=arm,
            elites=[p['program_sha256'] for p in elites], program_select_mean_gap_percent=gaps,
            selected_program=selected, frozen_at_utc=stamp(), actual_outputs=len(ctx['history'])))
    result = dict(at_utc=stamp(), programs=frozen, all_programs_frozen_before_dev=True,
                  program_selection_uses_only_real_route_quality=True, final_opened=False)
    path = study.out/'frozen_programs.json'
    if path.exists():
        prior = read(path)
        def chosen(rows): return [(r['generator'], r['discovery_seed'], r['feedback_arm'], r['selected_program']['program_sha256']) for r in rows]
        assert chosen(prior['programs']) == chosen(frozen)
    else: write(path, result)
    return frozen


def evaluate_dev(study, frozen):
    cfg = study.cfg; baseline = canonical_program(INITIAL); runs = []
    for method in ('nn', 'two_opt', 'gls'):
        ev, reused = study.endpoint(baseline, 'program_dev_eval', cfg['program_dev_eval_solver_seeds'], method=method)
        assert ev['ok']; runs.append(dict(label=method, evaluation=ev, evaluation_reused=reused))
    for r in frozen:
        ev, reused = study.endpoint(r['selected_program'], 'program_dev_eval', cfg['program_dev_eval_solver_seeds'])
        assert ev['ok']; runs.append(dict(generator=r['generator'], feedback_arm=r['feedback_arm'], discovery_seed=r['discovery_seed'],
            program_sha256=r['selected_program']['program_sha256'], evaluation=ev, evaluation_reused=reused))
    write(study.out/'dev_evaluation.json', dict(at_utc=stamp(), runs=runs,
        frozen_programs_sha256=digest(study.out/'frozen_programs.json'), final_opened=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--resume', action='store_true'); a = ap.parse_args()
    cfgpath = CODE/'configs/main_control_v1.json'; cfg = read(cfgpath)
    out = CODE/'results/main/discovery-v1'; out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out/'completed.json').exists(), 'Study already complete'
    assert a.resume or not (out/'protocol_frozen.json').exists(), 'Use --resume for existing partial study'
    study = TrainingStudy(a.root, out, CODE/'results/main/prepare-v1'); assert study.cfg == cfg
    paths, hashes = model_manifest(a.root, cfg)
    sourcepaths = [Path(__file__), CODE/'scripts/training_study.py', CODE/'scripts/stage3_discover.py',
        CODE/'scripts/core_discover.py', CODE/'scripts/qlora_pilot.py', CODE/'scripts/common_sft.py',
        CODE/'scripts/stage3_worker.py', CODE/'src/tsp_value/programs.py', CODE/'src/tsp_value/solver.py']
    provenance = dict(protocol=cfg, protocol_sha256=digest(cfgpath), model_adapters=hashes, bank_sha256=study.bankhash,
        source_sha256={str(p.relative_to(CODE)): digest(p) for p in sourcepaths})
    if (out/'protocol_frozen.json').exists():
        old = read(out/'protocol_frozen.json')
        for k, v in provenance.items(): assert old[k] == v, k
    else:
        write(out/'protocol_frozen.json', dict(at_utc=stamp(), **provenance))
        src = out/'executed_source'; src.mkdir()
        for p in sourcepaths+[cfgpath]: (src/p.name).write_bytes(p.read_bytes())
    start, cpu = time.perf_counter(), time.process_time(); attempt = stamp(); torch.set_num_threads(1)
    manifest = read(a.root/'artifacts/qwen_download.json'); initial = canonical_program(INITIAL); study.save_program(initial)
    baseline, _ = study.endpoint(initial, 'discovery', cfg['endpoint_solver_seeds']); assert baseline['ok']
    study.tokenizer = AutoTokenizer.from_pretrained(manifest['snapshot_path'], local_files_only=True)
    contexts = {}; torch.cuda.reset_peak_memory_stats()
    stop, thread = monitor_gpu(out, attempt, start)
    try:
        for generator in cfg['generators']:
            needs_generation = any(not (out/'generations'/f'{generator}-{arm}-seed{seed}-candidate{i:03d}.json').exists()
                for seed in cfg['discovery_seeds'] for i in range(cfg['candidates_per_arm_per_seed']) for arm in cfg['feedback_arms'][generator])
            if needs_generation:
                load_start = time.perf_counter(); study.model = load_base(manifest['snapshot_path'], False)
                if paths[generator]: study.model = PeftModel.from_pretrained(study.model, paths[generator], is_trainable=False)
                study.model.eval().requires_grad_(False); study.model.config.use_cache = True
                append(out/'model_loads.jsonl', dict(at_utc=stamp(), attempt=attempt, generator=generator,
                    wall_seconds=time.perf_counter()-load_start, adapter_sha256=hashes[generator]))
            for seed in cfg['discovery_seeds']:
                for arm in cfg['feedback_arms'][generator]:
                    contexts[generator, seed, arm] = dict(parent=initial, parent_eval=baseline, last=None, population=[initial], history=[])
                for index in range(cfg['candidates_per_arm_per_seed']):
                    for arm in cfg['feedback_arms'][generator]:
                        ctx = contexts[generator, seed, arm]; parent = ctx['parent']
                        cid = f'{generator}-{arm}-seed{seed}-candidate{index:03d}'
                        candidatepath = out/'candidates'/f'{cid}.json'
                        if candidatepath.exists(): candidate = read(candidatepath)
                        else:
                            messages = study.prompt_for(ctx, arm); rawpath = out/'generations'/f'{cid}.json'
                            raw = read(rawpath) if rawpath.exists() else study.generate(messages, seed*1000+index, cid)
                            assert raw['messages'] == messages
                            candidate = dict(candidate_id=cid, generator=generator, arm=arm, discovery_seed=seed,
                                index=index, parent_sha256=parent['program_sha256'], valid=False, at_utc=stamp())
                            try:
                                p = canonical_program(raw['response']); duplicate = p['program_sha256'] in study.programs
                                study.save_program(p); candidate.update(**p, duplicate_program=duplicate)
                                result, reused = study.endpoint(p, 'discovery', cfg['endpoint_solver_seeds'])
                                if not result['ok']: raise ValueError(result.get('error', 'Worker evaluation failed'))
                                candidate.update(valid=True, mean_discovery_gap_percent=result['mean_gap_percent'],
                                    gap_by_size=result['by_nodes_mean_gap_percent'], evaluation_reused=reused,
                                    same_endpoint_routes_as_parent=all(x['best_tour']==y['best_tour'] for x,y in zip(result['results'],ctx['parent_eval']['results'])))
                                if arm == 'B': candidate['value_feedback'] = study.feedback(parent, p, 'B')
                            except Exception: candidate['error'] = traceback.format_exc()
                            write(candidatepath, candidate)
                        if candidate['valid']:
                            p = {k: candidate[k] for k in ('expression', 'source', 'program_sha256')}; study.save_program(p)
                            if p['program_sha256'] not in {r['program_sha256'] for r in ctx['population']}: ctx['population'].append(p)
                            if candidate['mean_discovery_gap_percent'] < ctx['parent_eval']['mean_gap_percent']-1e-12:
                                ctx['parent'] = p; ctx['parent_eval'] = study.endpoint(p, 'discovery', cfg['endpoint_solver_seeds'])[0]
                        ctx['last'] = candidate; ctx['history'].append(candidate)
                        atomic_json(out/'progress.json', dict(at_utc=stamp(), pid=__import__('os').getpid(),
                            candidate_id=cid, reconstructed_or_completed_outputs=sum(len(c['history']) for c in contexts.values()),
                            valid=candidate['valid'], incumbent_gap_percent=ctx['parent_eval']['mean_gap_percent']))
                        print(json.dumps(dict(candidate_id=cid, valid=candidate['valid'], gap=candidate.get('mean_discovery_gap_percent'))), flush=True)
            if needs_generation: del study.model; gc.collect(); torch.cuda.empty_cache()
        assert sum(len(c['history']) for c in contexts.values()) == cfg['total_outputs']
        frozen = freeze_programs(study, contexts); assert len(frozen) == 45
        evaluate_dev(study, frozen)
        calls = [read(p) for p in (out/'generations').glob('*.json')]
        evaluations = [read(p) for p in (out/'evaluations').glob('*.json')]
        write(out/'completed.json', dict(status='completed', at_utc=stamp(), outputs=len(calls), frozen_programs=len(frozen),
            input_tokens=sum(r['input_tokens'] for r in calls), output_tokens=sum(r['output_tokens'] for r in calls),
            generation_wall_seconds=sum(r['generation_wall_seconds'] for r in calls),
            real_endpoint_routes=sum(len(r.get('results', [])) for r in evaluations),
            isolated_worker_cpu_seconds=sum(r['cpu_seconds'] for r in evaluations),
            last_attempt_wall_seconds=time.perf_counter()-start, last_attempt_cpu_seconds=time.process_time()-cpu,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, final_opened=False))
    except Exception:
        append(out/'failures.jsonl', dict(at_utc=stamp(), attempt=attempt, error=traceback.format_exc(),
            wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu)); raise
    finally:
        stop.set(); thread.join(timeout=5)
        append(out/'attempt_costs.jsonl', dict(attempt=attempt, ended_at_utc=stamp(), wall_seconds=time.perf_counter()-start,
            process_cpu_seconds=time.process_time()-cpu))


if __name__ == '__main__': main()
