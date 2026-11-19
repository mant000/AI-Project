"""Recompute value-learning, ABC and generator-adaptation results on the CPU.

The optional --replay reruns the frozen rules on their recorded development
inputs. All output goes to a new directory; released records remain untouched.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import time

import numpy as np
from analyze_results import CODE, digest, read, write
from tsp.data import distances, length, validate_tour
from tsp.programs import canonical_program
from tsp.solver import create_state, run
from tsp.programs import compile_rule


def labels(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        rows = list(map(json.loads, stream))
    values = {}
    repeated = 0
    for r in rows:
        ys = []
        for index in range(8):
            records = [x for x in r['records'] if x['candidate_index'] == index]
            assert {x['seed'] for x in records} == {71, 72}
            for x in records:
                actual = (x['reference_length']-x['candidate_length'])/x['normalization_length']
                assert abs(actual-x['relative_benefit']) < 1e-14
            assert records[0]['relative_benefit'] == records[1]['relative_benefit']
            ys.append(100*np.mean([x['relative_benefit'] for x in records]))
        branches = {(tuple(x['edge']), x['seed']):x for x in r['branch_measurements']}
        for edge in [r['reference_edge'], *r['candidates']]:
            a, b = (branches[(tuple(edge), seed)] for seed in (71, 72))
            assert a['trajectory_signature'] == b['trajectory_signature']
            repeated += 1
        values[(r['instance_id'], r['state_index'])] = np.array([*ys, 0.])
    stats = dict(states=len(rows), instances=len({r['instance_id'] for r in rows}),
                 duplicate_seed_branch_pairs=repeated, zero_candidate_percent={})
    for n in (50, 100):
        a = np.array([values[(r['instance_id'],r['state_index'])][:-1] for r in rows if r['nodes']==n])
        stats['zero_candidate_percent'][str(n)] = float(100*np.mean(np.abs(a)<1e-10))
    return values, stats


def value_results(folder):
    train, train_stats = labels(folder/'train/states.jsonl.gz')
    val, val_stats = labels(folder/'val/states.jsonl.gz')
    assert not {k[0] for k in train} & {k[0] for k in val}
    methods = {}
    for kind in ('mlp', 'transformer'):
        seeds = []
        for seed in (101, 202, 303):
            rows = [json.loads(x) for x in (folder/f'models/{kind}-seed{seed}_predictions.jsonl').read_text().splitlines()]
            assert {(r['instance_id'],r['state_index']) for r in rows} == set(val)
            benefits, regrets = [], []
            for r in rows:
                y, pred = np.array(r['labels_pp']), np.array(r['predictions_pp'])
                assert np.max(np.abs(y-val[(r['instance_id'],r['state_index'])])) < 1e-6
                choices = np.flatnonzero(pred>=pred.max()-1e-9)
                choice = 8 if 8 in choices else int(choices[0])
                assert choice == r['chosen_index']
                assert abs(y[choice]-r['selected_benefit_pp']) < 1e-12
                benefits.append(y[choice]); regrets.append(y.max()-y[choice])
            seeds.append(dict(seed=seed, selected_benefit_pp=float(np.mean(benefits)),
                              top1_regret_pp=float(np.mean(regrets))))
        methods[kind] = dict(runs=seeds, mean_selected_benefit_pp=float(np.mean([r['selected_benefit_pp'] for r in seeds])),
                             mean_top1_regret_pp=float(np.mean([r['top1_regret_pp'] for r in seeds])))
    return dict(train=train_stats, validation=val_stats, methods=methods)


def route_audit(rows, inputs):
    for r in rows:
        item = inputs[r['instance_id']]
        validate_tour(r['best_tour'], item['nodes'])
        actual = length(np.array(r['best_tour']), distances(item['coords']))
        assert abs(actual-r['original_length']) < 1e-10
        assert abs(100*(actual/item['reference_length']-1)-r['gap_percent']) < 1e-10


def instance_means(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['instance_id']].append(r['gap_percent'])
    return {k:float(np.mean(v)) for k,v in grouped.items()}


def abc_results(folder, inputs):
    frozen = read(folder/'frozen_programs.json')
    dev = read(folder/'dev_evaluation.json')
    assert dev['frozen_programs_sha256'] == digest(folder/'frozen_programs.json')
    assert frozen['all_programs_frozen_before_dev_outcomes']
    scores = {}
    jobs = {}
    programs = {p['selected_program']['program_sha256']:p['selected_program'] for p in frozen['programs']}
    for q in dev['runs']:
        if 'arm' not in q:
            continue
        rows = q['evaluation']['results']; route_audit(rows, inputs)
        assert len(rows)==96 and {r['seed'] for r in rows}=={0,1,2}
        scores[(q['arm'],q['discovery_seed'])] = float(np.mean(list(instance_means(rows).values())))
        ph = q['program_sha256']; program=programs[ph]
        assert canonical_program(program['source'])['program_sha256']==ph
        for r in rows:
            jobs[(ph,r['instance_id'],r['seed'])]=(inputs[r['instance_id']],program['expression'],r)
    directional = {}
    for arm in ('B', 'C'):
        flags=[]
        for file in sorted((folder/'generations').glob(f'{arm}-*.json')):
            g=read(file); prompt=g['messages'][-1]['content']
            feedback=json.loads(prompt.split('ADDITIONAL STATE VALUE FEEDBACK:\n',1)[1])
            flags.append(any(k!='none' and v for k,v in feedback['indicated_preference_counts'].items()))
        assert len(flags)==96
        directional[arm]=sum(flags)
    return dict(mean_gap_percent={a:float(np.mean([scores[(a,s)] for s in (501,502,503,504)])) for a in ('A','B','C')},
                added_feedback_minus_endpoint_pp={a:[scores[(a,s)]-scores[('A',s)] for s in (501,502,503,504)] for a in ('B','C')},
                prompts_with_directional_feedback=directional, prompt_count_per_arm=96), jobs


def adaptation_results(folder, inputs):
    protocol=read(folder/'evaluation/protocol_frozen.json')
    frozen=read(folder/'frozen_programs.json')
    assert protocol['all_programs_frozen_before_this_dev'] and frozen['all_programs_frozen_before_dev']
    programs={r['selected_program']['program_sha256']:r['selected_program'] for r in frozen['programs']}
    groups=defaultdict(list); allrows={}; jobs={}
    for ph,p in programs.items():
        assert canonical_program(p['source'])['program_sha256']==ph
        rows=read(folder/f'evaluation/routes/gls-{ph}.json')['results'];route_audit(rows,inputs)
        assert len(rows)==320 and {r['seed'] for r in rows}==set(range(5))
        allrows[ph]=instance_means(rows)
        for r in rows:
            jobs[(ph,r['instance_id'],r['seed'])]=(inputs[r['instance_id']],p['expression'],r)
    for a in protocol['aliases']:
        if not a['label'].startswith('main/'):
            continue
        _,generator,arm,seed=a['label'].split('/')
        if generator.startswith('grpo-terminal-'):generator='terminal_grpo'
        if generator.startswith('grpo-state-'):generator='state_grpo'
        groups[generator+'/'+arm].append(allrows[a['program_sha256']])
    ids=sorted(next(iter(allrows.values())))
    per_instance={k:np.array([[r[i] for i in ids] for r in rows]).mean(0) for k,rows in groups.items()}
    contrasts={}
    # As in the original analysis, resample originals conditional on the frozen
    # programs; these intervals do not add independent training/discovery runs.
    indices=np.random.default_rng(1821).integers(len(ids),size=(5000,len(ids)))
    for a,b in [('sft/A','original/A'),('state_grpo/B','terminal_grpo/A')]:
        d=per_instance[a]-per_instance[b]
        boot=d[indices].mean(1)
        contrasts[a+' minus '+b]=dict(mean_gap_pp=float(d.mean()), ci95=np.quantile(boot,[.025,.975]).tolist())
    return dict(mean_gap_percent={k:float(v.mean()) for k,v in per_instance.items()},
                frozen_aliases={k:len(v) for k,v in groups.items()}, contrasts=contrasts,
                uncertainty='Original-instance bootstrap conditional on all frozen programs; one shared SFT initialization and three GRPO optimization seeds.'),jobs


def replay(job):
    item,expr,old=job
    state=create_state(item['coords'],old['seed'],mode='gls')
    run(state,{50:235000,100:970000}[item['nodes']],rule=compile_rule(expr))
    assert state['best_tour'].tolist()==old['best_tour']
    assert state['delta_evaluations']==old['delta_evaluations']
    assert abs(length(state['best_tour'],state['distance'])-old['original_length'])<1e-12
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--replay',action='store_true')
    ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args(); start=time.perf_counter()
    assert not (a.output/'progress_summary.json').exists(), 'Use a new output directory'
    records=read(CODE/'results/research_records.json')
    for r in records['files']:
        assert digest(CODE/r['path'])==r['sha256'], r['path']
    assert digest(CODE/'results/research_inputs.jsonl')==read(CODE/'results/research_inputs_source.json')['sha256']
    inputs={r['instance_id']:r for r in map(json.loads,(CODE/'results/research_inputs.jsonl').read_text().splitlines())}
    value=value_results(CODE/'results/value_learning')
    abc,abcjobs=abc_results(CODE/'results/abc',inputs)
    adaptation,adaptjobs=adaptation_results(CODE/'results/adaptation',inputs)
    jobs={**abcjobs,**adaptjobs}; replays=0
    if a.replay:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            replays=sum(pool.map(replay,jobs.values(),chunksize=4))
    result=dict(status='passed',source_records_verified=len(records['files']),value_learning=value,
                abc=abc,generator_adaptation=adaptation,unique_frozen_routes_repriced=len(jobs),
                exact_solver_replays=replays,wall_seconds=time.perf_counter()-start)
    write(a.output/'progress_summary.json',result)
    print(json.dumps(dict(status='passed',unique_routes=len(jobs),exact_replays=replays,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':
    main()
