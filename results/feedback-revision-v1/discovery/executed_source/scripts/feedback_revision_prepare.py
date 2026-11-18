"""Budget-aligned measured feedback preparation; CPU only, never opens final.

Two separately recorded diagnostics: matched remeasurement of old R1 states,
and a preregistered fresh discovery bank. Program-select/dev receive no labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'src'))
from tsp_value.data import distances, length, validate_tour, coordinate_hash
from tsp_value.programs import canonical_program, compile_rule, INITIAL
from tsp_value.solver import create_state, run, clone, select_edge, penalize, ARRAY_TYPES


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
def snapshot(s):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in s.items()
            if k not in ('distance','pair_i','pair_j')}
def restore(row):
    s = create_state(row['state']['coords'],17); s.update(copy.deepcopy(row['state']))
    for k,d in ARRAY_TYPES.items(): s[k] = np.asarray(s[k],dtype=d)
    return s


def states_for(item, events, cfg):
    s = create_state(item['coords'], cfg['bank_trajectory_seed'])
    cap = cfg['delta_budget_by_nodes'][str(item['nodes'])]
    for target in events:
        while True:
            # State collection itself is capped, including descent to a boundary.
            remain = cap-s['delta_evaluations']
            if remain <= 0: return
            run(s, remain, stop_at_decision=True)
            if s['delta_evaluations'] >= cap: return
            if s['penalty_events'] >= target:
                yield target, clone(s); break
            penalize(s, select_edge(s))


def measured(row, cfg):
    t,c = time.perf_counter(),time.process_time(); s=restore(row)
    remain=cfg['delta_budget_by_nodes'][str(row['nodes'])]-s['delta_evaluations']
    assert remain > 0 and s['needs_penalty']
    records=[]
    for edge in row['edges']:
        b=clone(s); b['rng_state']=np.random.default_rng(cfg['continuation_seed']).bit_generator.state
        penalize(b,edge); run(b,remain)
        assert b['delta_evaluations']==cfg['delta_budget_by_nodes'][str(row['nodes'])]
        validate_tour(b['best_tour'],row['nodes'])
        actual=length(b['best_tour'],b['distance']); assert abs(actual-b['best_length'])<1e-10
        records.append(dict(edge=edge,best_length=actual,best_tour=b['best_tour'].tolist(),
                            delta_evaluations=remain,total_work=b['delta_evaluations']))
    reference=records[-1]['best_length']
    return dict(bank_id=row['bank_id'],instance_id=row['instance_id'],nodes=row['nodes'],remaining_work=remain,
        true_scores_pp=[100*(reference-r['best_length'])/s['best_length'] for r in records], records=records,
        branches=len(records), wall_seconds=time.perf_counter()-t,worker_cpu_seconds=time.process_time()-c)


def full_rules(item, programs, cfg):
    t,c=time.perf_counter(),time.process_time(); rows=[]
    for program in programs:
        s=create_state(item['coords'],17);rule=compile_rule(program['expression'])
        run(s,cfg['delta_budget_by_nodes'][str(item['nodes'])],rule)
        actual=length(s['best_tour'],s['distance']);assert abs(actual-s['best_length'])<1e-10
        rows.append(dict(program_sha256=program['program_sha256'],instance_id=item['instance_id'],nodes=item['nodes'],seed=17,
            gap_percent=100*(actual/item['reference_length']-1),best_length=actual,best_tour=s['best_tour'].tolist(),
            delta_evaluations=s['delta_evaluations']))
    return dict(instance_id=item['instance_id'],results=rows,wall_seconds=time.perf_counter()-t,worker_cpu_seconds=time.process_time()-c)


def execute_bank(bank, cfg, out):
    out.mkdir(parents=True,exist_ok=True); finished=[]
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs={}
        for i,row in enumerate(bank):
            path=out/f'state{i:03d}.json'
            if path.exists():
                r=read(path);assert r['bank_id']==row['bank_id'];finished.append(r)
            else:jobs[pool.submit(measured,row,cfg)]=path
        for f in as_completed(jobs):
            r=f.result();write(jobs[f],r);finished.append(r)
            print('measured',len(finished),'/',len(bank),flush=True)
    return {r['bank_id']:r for r in finished}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--phase',choices=['old','fresh'],required=True);a=ap.parse_args()
    cfg=read(CODE/'configs/feedback_revision_v1.json'); start=time.perf_counter()
    assert not cfg['parameter_training'] and not cfg['final_evaluation']
    base=CODE/'results/feedback-revision-v1';out=base/(a.phase+'-prepare');out.mkdir(parents=True,exist_ok=True)
    assert not (out/'completed.json').exists(),'Completed preparation must not be repeated'
    source=[Path(__file__),CODE/'src/tsp_value/solver.py',CODE/'src/tsp_value/programs.py',CODE/'src/tsp_value/data.py']
    frozen=dict(at_utc=stamp(),protocol=cfg,protocol_sha256=digest(CODE/'configs/feedback_revision_v1.json'),
        phase=a.phase,source_sha256={str(p.relative_to(CODE)):digest(p) for p in source},final_opened=False)
    if (out/'protocol_frozen.json').exists():
        prev=read(out/'protocol_frozen.json');assert prev['protocol']==cfg and prev['source_sha256']==frozen['source_sha256']
    else:
        write(out/'protocol_frozen.json',frozen)
        for p in source:
            dst=out/'executed_source'/p.relative_to(CODE);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    data=a.root/'data/prepared/stage1/project_instances.jsonl'
    assert digest(data)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    bank=[];excluded=[]
    if a.phase=='old':
        old=CODE/'results/core/prepare-v1'
        raw=[json.loads(x) for x in (old/'bank_observables.jsonl').read_text().splitlines()]
        for row in raw:
            if row['state']['delta_evaluations']>=cfg['delta_budget_by_nodes'][str(row['nodes'])]:
                excluded.append(dict(bank_id=row['bank_id'],reason='state at/after full work cap'));continue
            bank.append(row)
        # Only explicit non-final discovery rows required for matched full-rule runs.
        wanted={r['instance_id'] for r in raw};items=[]
        with data.open() as f:
            for line in f:
                if any('"'+iid+'"' in line for iid in wanted):
                    r=json.loads(line);assert r['project_use']=='discovery';items.append(r)
        assert len(items)==8
    else:
        eligible=[]
        with data.open() as f:
            for line in f:
                if '"project_use": "value_train"' in line:
                    eligible.append(json.loads(line))
        pools={}
        for role,(offset,count) in cfg['pools'].items():
            pools[role]=[]
            for n in (50,100):
                rr=[r for r in eligible if r['nodes']==n][offset:offset+count];assert len(rr)==count
                for r in rr:
                    r=copy.deepcopy(r);r['project_use']=role;pools[role].append(r)
        selected=[r for rr in pools.values() for r in rr]
        ids=[r['instance_id'] for r in selected];coords=[coordinate_hash(r['coords']) for r in selected]
        assert len(ids)==len(set(ids))==len(set(coords))==128
        # Historical value_train use was source1..264 at most. Fixed source513..576
        # is additionally checked against all small selection/isolation manifests.
        manifests=[];oldids=set()
        for p in (CODE/'results').rglob('*.json'):
            if 'feedback-revision-v1' in str(p) or 'final' in str(p).lower():continue
            if 'selection' in p.name or p.name=='pool_isolation.json':
                import re
                text=p.read_text();oldids.update(re.findall(r'joshi:tsp(?:50|100)_train_concorde\.txt:\d+',text));manifests.append(p)
        assert not set(ids)&oldids
        write(out/'selection.json',{k:[{q:r[q] for q in ('instance_id','nodes','project_use')} for r in rr] for k,rr in pools.items()})
        write(out/'pool_isolation.json',dict(selected_instances=128,original_ids_unique=True,coordinate_hashes_unique=True,
            historical_manifest_id_overlap=False,historical_manifests={str(p.relative_to(CODE)):digest(p) for p in manifests},
            selected_coordinate_hashes=coords,data_sha256=digest(data),final_opened=False))
        # Readable non-final selected records, so discovery need not parse global data.
        for role,rr in pools.items():write(out/(role+'_instances.json'),rr)
        items=pools['discovery']
        for item in items:
            seen=[]
            for event,s in states_for(item,cfg['bank_penalty_events'],cfg):
                seen.append(event);ref=select_edge(clone(s))
                edges=sorted({tuple(sorted(map(int,e))) for e in zip(s['tour'],np.roll(s['tour'],-1))}-{ref})+[ref]
                bank.append(dict(bank_id=item['instance_id']+':event'+str(event),instance_id=item['instance_id'],nodes=item['nodes'],
                    state_index=event,state=snapshot(s),edges=[list(e) for e in edges],reference_edge=list(ref)))
            for event in cfg['bank_penalty_events']:
                if event not in seen:excluded.append(dict(instance_id=item['instance_id'],event=event,reason='no boundary before full work cap'))
    (out/'bank_observables.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bank))
    write(out/'excluded_states.json',excluded)
    exact=execute_bank(bank,cfg,out/'exact_states')
    if a.phase=='old':
        programs=[canonical_program(p.read_text()) for p in sorted((CODE/'results/core/discovery-v1/programs').glob('*.py'))]
        endpoints=out/'full_seed17';endpoints.mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            jobs={pool.submit(full_rules,r,programs,cfg):endpoints/(hashlib.sha256(r['instance_id'].encode()).hexdigest()+'.json')
                  for r in items if not (endpoints/(hashlib.sha256(r['instance_id'].encode()).hexdigest()+'.json')).exists()}
            for f in as_completed(jobs):write(jobs[f],f.result())
        # Controlled replay: same original H for first edge of first valid state per size.
        replay=[]
        for n in (50,100):
            row=next(r for r in bank if r['nodes']==n);s=restore(row);b=clone(s)
            b['rng_state']=np.random.default_rng(cfg['continuation_seed']).bit_generator.state;penalize(b,row['edges'][0])
            run(b,cfg['old_horizon_by_nodes'][str(n)])
            oldobj=next(read(p) for p in (CODE/'results/core/prepare-v1/exact_states').glob('*.json') if read(p)['bank_id']==row['bank_id'])
            assert np.array_equal(b['best_tour'],oldobj['records'][0]['best_tour']) and abs(b['best_length']-oldobj['records'][0]['best_length'])<1e-10
            replay.append(dict(bank_id=row['bank_id'],same_old_H_tour_and_cost=True))
        write(out/'old_H_replay.json',replay)
    informative=[r for r in exact.values() if np.ptp(r['true_scores_pp'])>1e-7]
    summary=dict(at_utc=stamp(),phase=a.phase,states=len(bank),excluded_states=len(excluded),original_instances=len(items),
        informative_states=len(informative),informative_original_instances=len({r['instance_id'] for r in informative}),
        positive_states=sum(max(r['true_scores_pp'])>1e-7 for r in exact.values()),branches=sum(r['branches'] for r in exact.values()),
        worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in exact.values()),wall_seconds=time.perf_counter()-start,
        bank_sha256=digest(out/'bank_observables.jsonl'),protocol_sha256=frozen['protocol_sha256'],
        signal_gate_passed=len(informative)>=cfg['signal_gate']['minimum_informative_states'] and
                           len({r['instance_id'] for r in informative})>=cfg['signal_gate']['minimum_original_instances'],
        parameters_trained=False,gpu_used=False,final_opened=False,python=sys.version,numpy=np.__version__,platform=platform.platform())
    write(out/'completed.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
