"""Observable-only discovery banks for RL and the later frozen main comparison."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from stage3_common import CODE, read, write, digest, stamp
from core_mechanism import select_slice
from stage2_labels import states_for
from tsp_value.solver import clone, select_edge
from tsp_value.value_model import EdgeValue, state_features, features, collate
from tsp_value.data import coordinate_hash


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--phase', choices=['grpo', 'main'], required=True); a = ap.parse_args()
    cfgpath = CODE / ('configs/grpo_v1.json' if a.phase == 'grpo' else 'configs/main_control_v1.json')
    cfg = read(cfgpath)
    out = CODE / ('results/training/grpo-prepare-v1' if a.phase == 'grpo' else 'results/main/prepare-v1')
    out.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time(); torch.set_num_threads(1)
    datapath = a.root / 'data/prepared/stage1/project_instances.jsonl'
    assert digest(datapath) == '43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    rows = [json.loads(x) for x in datapath.read_text().splitlines()]
    pools = ['discovery'] if a.phase == 'grpo' else ['discovery', 'program_select', 'program_dev_eval']
    selected = {p: select_slice(rows, p, cfg[p+'_offset_per_size'], cfg[p+'_instances_per_size']) for p in pools}
    sets = [{coordinate_hash(r['coords']) for r in selected[p]} for p in pools]
    for i, s in enumerate(sets):
        for j, t in enumerate(sets):
            if i != j: assert s.isdisjoint(t)
    write(out / 'protocol_frozen.json', dict(at_utc=stamp(), protocol=cfg, protocol_sha256=digest(cfgpath),
        source_sha256={str(p.relative_to(CODE)): digest(p) for p in
            (Path(__file__), CODE/'scripts/stage2_labels.py', CODE/'src/tsp_value/solver.py', CODE/'src/tsp_value/value_model.py')},
        data_sha256=digest(datapath), prediction_backend='CPU float32; fixed inference, no fitted transformation'))
    write(out / 'selection.json', {p: [{k: r[k] for k in ('instance_id', 'nodes', 'project_use')}
                                     for r in rr] for p, rr in selected.items()})
    models = []; checkpoints = []
    for seed in cfg['transformer_seeds']:
        path = a.root / f'checkpoints/stage2/training-v1/transformer-seed{seed}/best.pt'
        expected = read(CODE / f'results/stage2/training-v1/transformer-seed{seed}_summary.json')['checkpoint_sha256']
        assert digest(path) == expected
        ck = torch.load(path, map_location='cpu'); model = EdgeValue('transformer', **ck['config']['architecture']).eval()
        model.load_state_dict(ck['model']); models.append(model); checkpoints.append(dict(seed=seed, sha256=expected))
    items = []
    for n in (50, 100): items.extend([r for r in selected['discovery'] if r['nodes'] == n][:cfg['bank_instances_per_size']])
    bank = []; encoded_checks = 0
    for item in items:
        for event, s in states_for(item, cfg['bank_penalty_events']):
            ref = select_edge(clone(s)); edges = sorted({tuple(sorted((int(u), int(v))))
                for u, v in zip(s['tour'], np.roll(s['tour'], -1))} - {ref}) + [ref]
            f = state_features(s, edges)
            # The labelled training API is checked with artificial zero records, not future labels.
            old = features(dict(state=s, candidates=edges[:-1], reference_edge=ref,
                                records=[dict(relative_benefit=0.) for _ in range(2*(len(edges)-1))]))
            for k in f: assert np.array_equal(f[k], old[k]), k
            encoded_checks += 1
            with torch.inference_mode(): scores = np.stack([m(collate([f], 'cpu')).numpy()[0] for m in models])
            assert np.isfinite(scores).all() and np.array_equal(scores[:, -1], np.zeros(3))
            snapshot = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()
                        if k not in ('distance', 'pair_i', 'pair_j')}
            bank.append(dict(bank_id=item['instance_id']+':event'+str(event), instance_id=item['instance_id'],
                nodes=item['nodes'], project_use='discovery', state_index=event, state=snapshot,
                reference_edge=list(ref), edges=[list(e) for e in edges], model_scores_pp=scores.tolist()))
    bankpath = out / 'bank_observables.jsonl'; bankpath.write_text(''.join(json.dumps(r)+'\n' for r in bank))
    result = dict(status='completed', at_utc=stamp(), bank_sha256=digest(bankpath), states=len(bank),
        current_edges=sum(len(r['edges']) for r in bank), checkpoints=checkpoints,
        future_labels_read=False, training_online_feature_exact_checks=encoded_checks,
        all_current_edges_scored=True, inference_device='CPU', wall_seconds=time.perf_counter()-start,
        process_cpu_seconds=time.process_time()-cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        final_opened=False)
    write(out / 'completed.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
