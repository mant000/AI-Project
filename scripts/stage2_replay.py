"""Independently replay the actions selected by trained checkpoints on validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))
from tsp_value.solver import create_state,ARRAY_TYPES,paired_action_set


def worker(r,choices):
    s=create_state(r['state']['coords'],17)
    for k,v in r['state'].items(): s[k]=np.asarray(v,dtype=ARRAY_TYPES[k]) if k in ARRAY_TYPES else v
    edges=[tuple(e) for e in r['candidates']]+[tuple(r['reference_edge'])]
    selected=sorted({edges[c] for c in choices.values()}-{edges[-1]})
    if not selected: selected=[edges[-1]]
    result=paired_action_set(s,edges[-1],selected,[71,72],r['horizon'],trace=True)
    gains={e:np.mean([v['relative_benefit'] for v in result['records'] if tuple(v['candidate_edge'])==e])*100 for e in selected}
    gains[edges[-1]]=0.
    ys=np.r_[np.asarray([x['relative_benefit'] for x in r['records']]).reshape(2,-1).mean(0)*100,0]
    output=[]
    for name,c in choices.items():
        observed=float(gains[edges[c]]); expected=float(ys[c]); assert abs(observed-expected)<1e-10
        output.append(dict(method=name,instance_id=r['instance_id'],nodes=r['nodes'],state_index=r['state_index'],
            edge=edges[c],replayed_benefit_pp=observed,stored_benefit_pp=expected,abs_error_pp=abs(observed-expected),
            regret_pp=float(ys.max()-observed)))
    return dict(records=output,branches_executed=result['branches_executed'],branch_measurements=result['branch_measurements'])


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--training',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    with gzip.open(a.dataset,'rt') as f: rows=[json.loads(l) for l in f]
    decisions={}
    for p in sorted(a.training.glob('*_predictions.jsonl')):
        name=p.name.removesuffix('_predictions.jsonl')
        for line in p.read_text().splitlines():
            x=json.loads(line); decisions.setdefault((x['instance_id'],x['state_index']),{})[name]=x['chosen_index']
    start=time.perf_counter(); count=0; output=[]
    with (a.output/'raw.jsonl').open('x') as raw,(a.output/'branches.jsonl').open('x') as branches,ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(worker,r,decisions[(r['instance_id'],r['state_index'])]) for r in rows]):
            result=f.result(); count+=result['branches_executed']; output.extend(result['records'])
            for x in result['records']: raw.write(json.dumps(x)+'\n')
            for x in result['branch_measurements']: branches.write(json.dumps(x)+'\n')
    summary=dict(states=len(rows),instances=len({r['instance_id'] for r in rows}),branches_executed=count,
        wall_seconds=time.perf_counter()-start,max_abs_error_pp=max(x['abs_error_pp'] for x in output),
        dataset_sha256=hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
        interpretation='Replay of paired finite-horizon action outcomes on value_val, not a complete neural-guided solver or independent final evaluation.',methods={})
    for name in sorted({x['method'] for x in output}):
        g=[x for x in output if x['method']==name]
        summary['methods'][name]=dict(states=len(g),mean_benefit_pp=float(np.mean([x['replayed_benefit_pp'] for x in g])),
            mean_regret_pp=float(np.mean([x['regret_pp'] for x in g])))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
