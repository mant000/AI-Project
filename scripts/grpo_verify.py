"""Audit all executed RL rewards/routes/updates; limited real branch mismatch reference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from stage3_common import CODE, read, write, digest, stamp, worker
from training_study import TrainingStudy, group_rewards
from controlled_grpo import fixed_parents
from core_mechanism import cycle_hash
from stage3_worker import restore_state
from tsp_value.programs import canonical_program
from tsp_value.solver import clone, penalize, run
from tsp_value.data import distances, length, validate_tour


def audit(root, directory, cfg):
    meta = read(directory/'protocol_frozen.json'); arm, seed = meta['arm'], meta['seed']
    assert meta['protocol'] == cfg and read(directory/'completed.json')['status'] == 'completed'
    for name, expected in meta['source_sha256'].items(): assert digest(CODE/name) == expected
    frozen = read(directory/'adapter_frozen.json'); assert digest(Path(frozen['adapter_path'])/'adapter_model.safetensors') == frozen['adapter_sha256']
    assert read(directory/'fixed_parents.json')['parents'] == fixed_parents()
    study = TrainingStudy(root, directory, CODE/'results/training/grpo-prepare-v1')
    rows = {r['instance_id']: r for r in study.pools['discovery']}
    ds = {k: distances(r['coords']) for k, r in rows.items()}
    checked = 0; max_error = 0.
    for path in sorted((directory/'evaluations').glob('*.json')):
        ev = read(path); assert ev['pool'] == 'discovery'
        for r in ev.get('results', []):
            assert r['project_use'] == 'discovery' and r['instance_id'] in rows
            validate_tour(r['best_tour'], r['nodes']); cost = length(r['best_tour'], ds[r['instance_id']])
            max_error = max(max_error, abs(cost-r['original_length'])); checked += 1
            assert abs(cost-r['original_length']) < 1e-10
            assert r['delta_evaluations'] == cfg['delta_budget_by_nodes'][str(r['nodes'])]
    logical = 0; feedback_rows = 0; nonzero = 0; nonzero_terminal = 0
    all_candidates = []
    for index, parent in enumerate(fixed_parents()):
        ev, _ = study.endpoint(parent, 'discovery', cfg['endpoint_solver_seeds']); candidates = []
        for j in range(cfg['outputs_per_group']):
            cid = f'group{index:02d}-output{j}'; r = read(directory/'candidates'/f'{cid}.json')
            raw = read(directory/'generations'/f'{cid}.json')
            assert raw['messages'] == study.fixed_parent_prompt(parent, ev)
            assert raw['sampling_seed'] == seed*10000+index*10+j
            assert r['parent_sha256'] == parent['program_sha256']
            if r['valid']:
                p = canonical_program(raw['response']); assert p['program_sha256'] == r['program_sha256']
                outcome, _ = study.endpoint(p, 'discovery', cfg['endpoint_solver_seeds'])
                actual = float(np.clip(ev['mean_gap_percent']-outcome['mean_gap_percent'], -1., 1.))
                assert abs(actual-r['terminal_reward']) < 1e-12
                fb = study.feedback(parent, p, 'B'); assert fb == r['value_feedback']
                feedback_rows += len(fb['states_detail'])
            else: assert r['terminal_reward'] == cfg['reward']['invalid']
            candidates.append(r); all_candidates.append(r)
        actual = group_rewards(candidates, cfg, arm); group = read(directory/f'group{index:02d}.json')
        for k, v in actual.items(): assert group[k] == v, (index, k)
        if actual['reward_std'] > 1e-8: logical += 1
        nonzero += sum(abs(x) > 1e-10 for x in actual['rewards'])
        nonzero_terminal += sum(abs(r['terminal_reward']) > 1e-10 for r in candidates if r['valid'])
        for raw, old, ref in zip(group['raw'], group['old_logps'], group['reference_logps']):
            assert len(old) == len(ref) == raw['output_tokens']
            assert np.isfinite(old).all() and np.isfinite(ref).all()
    events = [json.loads(x) for x in (directory/'update_events.jsonl').read_text().splitlines()] if (directory/'update_events.jsonl').exists() else []
    updates = [x for x in events if x['kind'] == 'training']; assert len(updates) == logical
    assert [r['logical_step'] for r in updates] == list(range(1, logical+1))
    if logical:
        check = read(directory/'resume_verification.json'); assert check['status'] == 'passed' and check['verification_updates_rolled_back']
        assert len(events) == logical+2
        assert all(abs(r['mean_ratio']-1) < .01 for r in updates), 'On-policy initial ratio mismatch'
    summary = read(directory/'completed.json'); assert summary['logical_updates'] == logical
    assert len(all_candidates) == 48 and summary['frozen_reference_max_parameter_error'] == 0.
    result = dict(status='passed', at_utc=stamp(), arm=arm, seed=seed, routes_recomputed=checked,
        max_route_cost_error=max_error, feedback_rows_recomputed=feedback_rows, logical_updates=logical,
        zero_variance_groups=12-logical, nonzero_total_rewards=nonzero, nonzero_valid_terminal_rewards=nonzero_terminal,
        exact_prompt_and_pairing_and_reward_checks=48, reference_adapter_unchanged=True, final_opened=False)
    write(directory/'verification.json', result)
    return result, all_candidates


def branch_diagnostic(root, cfg, candidates):
    out = CODE/'results/training/grpo-v1'; bankpath = CODE/'results/training/grpo-prepare-v1/bank_observables.jsonl'
    bank = [json.loads(x) for x in bankpath.read_text().splitlines()]
    # Prespecified deterministic interleave by size gives this tiny diagnostic both scales.
    ordered = [row for pair in zip([r for r in bank if r['nodes']==50], [r for r in bank if r['nodes']==100]) for row in pair]
    chosen = []; seen = set()
    for candidate in candidates:
        if not candidate['valid']: continue
        fb = {r['bank_id']: r for r in candidate['value_feedback']['states_detail']}
        for state in ordered:
            row = fb[state['bank_id']]; key = (row['bank_id'], tuple(row['parent_edge']), tuple(row['child_edge']))
            if row['same_action'] or key in seen: continue
            seen.add(key); chosen.append(dict(candidate_id=candidate['candidate_id'], candidate_program_sha256=candidate['program_sha256'],
                parent_program_sha256=candidate['parent_sha256'], **row))
            if len(chosen) == 12: break
        if len(chosen) == 12: break
    selection_path = out/'branch_diagnostic_selection.json'
    if selection_path.exists(): assert read(selection_path)['pairs'] == chosen
    else: write(selection_path, dict(at_utc=stamp(), pairs=chosen, bank_sha256=digest(bankpath),
        order='optimization seeds ascending, terminal/state, then group/output then size-interleaved bank; first twelve distinct changed pairs',
        adapters_frozen_before_outcomes=True, no_reward_or_threshold_adjustment=True))
    by_id = {r['bank_id']: r for r in bank}; measured = []; wall, cpu = time.perf_counter(), time.process_time()
    path = out/'branch_diagnostic'; path.mkdir(exist_ok=True)
    for i, pair in enumerate(chosen):
        target = path/f'pair{i:02d}.json'
        if target.exists(): measured.append(read(target)); continue
        s = restore_state(by_id[pair['bank_id']]['state']); branches = []
        for role in ('parent', 'child'):
            branch = clone(s); branch['rng_state'] = np.random.default_rng(71).bit_generator.state
            edge = pair[role+'_edge']; penalize(branch, edge); before = branch['delta_evaluations']
            h = cfg['horizon_by_nodes'][str(pair['nodes'])]; run(branch, h)
            assert branch['delta_evaluations']-before == h
            validate_tour(branch['best_tour'], pair['nodes']); cost = length(branch['best_tour'], branch['distance'])
            assert abs(cost-branch['best_length']) < 1e-10
            branches.append(dict(role=role, edge=edge, best_tour=branch['best_tour'].tolist(), original_length=cost,
                best_cycle_sha256=cycle_hash(branch['best_tour']), delta_evaluations=h))
        true = 100*(branches[0]['original_length']-branches[1]['original_length'])/s['best_length']
        result = dict(pair=pair, branches=branches, measured_child_minus_parent_pp=true,
            predicted_child_minus_parent_pp=pair['child_minus_parent_pp'],
            true_tie=abs(true)<=1e-7, indicated_preference=pair['indicated_preference'],
            false_preference_on_tie=abs(true)<=1e-7 and pair['indicated_preference']!='none',
            scope='separate measured finite-H conventional-continuation reference, not Transformer success', seed=71)
        write(target, result); measured.append(result)
    summary = dict(at_utc=stamp(), pairs=len(measured), real_branches=2*len(measured),
        true_ties=sum(r['true_tie'] for r in measured), false_preferences_on_ties=sum(r['false_preference_on_tie'] for r in measured),
        mean_absolute_error_pp=float(np.mean([abs(r['measured_child_minus_parent_pp']-r['predicted_child_minus_parent_pp']) for r in measured])) if measured else None,
        wall_seconds=time.perf_counter()-wall, process_cpu_seconds=time.process_time()-cpu,
        finite_horizon_single_seed_conditional_reference=True, used_to_tune_rewards=False, final_opened=False)
    write(out/'branch_diagnostic_summary.json', summary); return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); a = ap.parse_args()
    torch.set_num_threads(1); cfg = read(CODE/'configs/grpo_v1.json'); rootout = CODE/'results/training/grpo-v1'
    dirs = [rootout/f'{arm}-seed{seed}' for seed in cfg['optimization_seeds'] for arm in cfg['arms']]
    assert all((p/'adapter_frozen.json').exists() and (p/'completed.json').exists() for p in dirs)
    summaries = []; candidates = []
    for p in dirs:
        summary, cc = audit(a.root, p, cfg); summaries.append(summary); candidates.extend(cc)
    diagnostics = branch_diagnostic(a.root, cfg, candidates)
    write(rootout/'verification.json', dict(status='passed', at_utc=stamp(), runs=summaries,
        measured_branch_reference=diagnostics, all_optimization_seeds_retained=True, final_opened=False))
    print(json.dumps(dict(runs=summaries, diagnostics=diagnostics), indent=2))


if __name__ == '__main__': main()
