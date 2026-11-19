"""CPU-only solver and state-bank reproduction from the released 256 inputs.

Run in a new output directory. The public records are never modified. This
replays all selected dev routes, deduplicating identical programs across seeds,
and reconstructs every state from its input trajectory without branch labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from analyze_results import CODE, read, write, digest
from state_feedback import collect_states,intervene
from tsp.solver import create_state, run
from tsp.programs import compile_rule
from tsp.data import length


def replay_job(job):
    item,seed,expr,method,cap,old=job
    start,cpu=time.perf_counter(),time.process_time()
    state=create_state(item['coords'],seed,mode='two_opt' if method=='two_opt' else 'gls')
    if method!='nn':run(state,cap,rule=compile_rule(expr) if method=='gls' else None)
    actual=length(state['best_tour'],state['distance'])
    assert state['best_tour'].tolist()==old['best_tour']
    assert abs(actual-old['original_length'])<1e-12
    assert state['delta_evaluations']==old['delta_evaluations']
    return dict(instance_id=item['instance_id'],seed=seed,method=method,expression=expr,
                route_exact=True,delta_evaluations=state['delta_evaluations'],original_length=actual,
                wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',type=Path,default=CODE/'results/mechanism/bank')
    ap.add_argument('--study',type=Path,default=CODE/'results/discovery/run')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=10)
    ap.add_argument('--skip-bank',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert not (a.output/'replay_verification.json').exists(),'use a new output directory'
    start,cpu=time.perf_counter(),time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    cfg=read(a.study/'protocol_frozen.json')['protocol'];inputs=[json.loads(x) for x in (a.prepare/'benchmark_inputs.jsonl').read_text().splitlines()]
    bank_checks=0
    if not a.skip_bank:
        with gzip.open(a.prepare/'states.jsonl.gz','rt') as f:oldbank=list(map(json.loads,f))
        newbank=[]
        for item in [x for x in inputs if x['project_use']=='bank']:newbank.extend(collect_states(item,cfg))
        assert newbank==oldbank
        bank_checks=len(newbank)
        with gzip.open(a.output/'states_rebuilt.jsonl.gz','wt') as f:
            for r in newbank:f.write(json.dumps(r)+'\n')
    items={x['instance_id']:x for x in inputs if x['project_use']=='program_dev_eval'}
    frozen=read(a.study/'frozen_programs.json')['programs'];programs={p['selected_program']['program_sha256']:p['selected_program'] for p in frozen}
    from tsp.programs import canonical_program, INITIAL
    std=canonical_program(INITIAL);programs[std['program_sha256']]=std
    jobs=[]
    for file in (a.study/'evaluations').glob('*.json'):
        ev=read(file)
        if ev['pool']!='program_dev_eval' or ev['program_sha256'] not in programs:continue
        p=programs[ev['program_sha256']]
        for r in ev['results']:
            item=items[r['instance_id']]
            jobs.append((item,r['seed'],p['expression'],ev['method'],cfg['delta_budget_by_nodes'][str(item['nodes'])],r))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(replay_job,jobs,chunksize=4))
    write(a.output/'solver_replay_records.json',results)
    with gzip.open(a.prepare/'states.jsonl.gz','rt') as f:fullbank=list(map(json.loads,f))
    wanted=set(read(a.prepare/'feedback_selection.json')['bank_ids']);feedback_bank=[r for r in fullbank if r['bank_id'] in wanted]
    pair_replays=[];seen=set()
    # Post-outcome audit sampling exercises nonzero paths; these are verification
    # cases, never discovery hints or new independent effectiveness observations.
    for arm in ('P','F'):
        count=0
        for file in sorted((a.study/'feedback').glob(f'{arm}-*.json')):
            fb=read(file)
            if fb['status']!='available':continue
            informative=[i for i,r in enumerate(fb['states_detail']) if abs(r['child_minus_parent_benefit_pp'])>1e-7]
            if not informative:continue
            pa=canonical_program(fb['display']['compared_parent_code']);ch=canonical_program(fb['display']['compared_child_code']);mode='single' if arm=='P' else 'fragment'
            key=(mode,pa['program_sha256'],ch['program_sha256'])
            if key in seen:continue
            seen.add(key);index=informative[0];row=feedback_bank[index]
            for p in (pa,ch):
                old=read(a.study/'rollouts'/f'{mode}-{p["program_sha256"]}.json')['results'][index]
                result=intervene(row,p['expression'],mode,cfg)
                for field in ('best_tour','best_length','total_work','controlled_action_sha256'):assert result[field]==old[field]
                pair_replays.append(dict(bank_id=row['bank_id'],mode=mode,program_sha256=p['program_sha256'],exact=True))
            count+=1
            if count==2:break
    write(a.output/'feedback_pair_replays.json',pair_replays)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    proof=dict(status='passed',scope='new CPU environment on the same rented physical server; no Torch/model/GPU',
               bank_states_reconstructed_exact=bank_checks,unique_selected_programs_including_standard=len(programs),
               complete_dev_routes_replayed_exact=len(results),all_64_dev_instances=len(items)==64,
               solver_seeds=cfg['program_dev_eval_solver_seeds'],wall_seconds=time.perf_counter()-start,
               actual_feedback_branches_independently_replayed=len(pair_replays),
               parent_cpu_seconds=time.process_time()-cpu,waited_child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
               inputs_sha256=digest(a.prepare/'benchmark_inputs.jsonl'),study_freeze_sha256=digest(a.study/'frozen_programs.json'),
               python_version=sys.version,source_sha256=digest(Path(__file__)))
    write(a.output/'replay_verification.json',proof);print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
