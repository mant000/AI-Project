"""Measure single-penalty, fragment and persistent outcomes from a rebuilt state bank."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import csv
import gzip
import json
from pathlib import Path
import re
import resource
import time
import numpy as np
from experiment_io import CODE,read,write,digest,stamp
from state_feedback import collect_states,intervene
from tsp.programs import canonical_program
from tsp.data import coordinate_hash

BANK=None;CFG=None;PROGRAMS=None


def initialize(bank,cfg,programs):
    global BANK,CFG,PROGRAMS
    BANK,CFG,PROGRAMS=bank,cfg,programs


def measure(index):
    row=BANK[index];begin,cpu=time.perf_counter(),time.process_time()
    baseline=intervene(row,None,'persistent',CFG);results=[]
    for program in PROGRAMS:
        variants=[]
        for mode in CFG['modes']:
            result=intervene(row,program['expression'],mode,CFG)
            result['benefit_pp']=100*(baseline['best_length']-result['best_length'])/row['state']['best_length'];variants.append(result)
        results.append(dict(program_sha256=program['program_sha256'],expression=program['expression'],variants=variants))
    return dict(bank_id=row['bank_id'],instance_id=row['instance_id'],nodes=row['nodes'],policy_index=row['policy_index'],
        phase_index=row['phase_index'],baseline=baseline,programs=results,branches_executed=1+len(PROGRAMS)*len(CFG['modes']),
        wall_seconds=time.perf_counter()-begin,worker_cpu_seconds=time.process_time()-cpu)


def prepare(root,out,cfg):
    assert (out/'bank_frozen.json').exists(), 'Run build_state_bank.py first'
    saved=read(out/'bank_frozen.json')
    assert saved['config_sha256']==digest(CODE/'configs/state_feedback.json')
    assert saved['bank_sha256']==digest(out/'states.jsonl.gz')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=['pilot','full'],required=True)
    ap.add_argument('--output',type=Path,default=CODE/'results/mechanism/bank');a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    cfg=read(CODE/'configs/state_feedback.json');begin,cpu=time.perf_counter(),time.process_time()
    sources=[Path(__file__),CODE/'scripts/state_feedback.py',*[CODE/f'src/tsp/{name}.py' for name in ('solver','programs','data')]]
    provenance=dict(config_sha256=digest(CODE/'configs/state_feedback.json'),source_sha256={p.relative_to(CODE).as_posix():digest(p) for p in sources},protocol=cfg)
    if (out/'protocol_frozen.json').exists():
        old=read(out/'protocol_frozen.json')
        for k,v in provenance.items():assert old[k]==v,k
    else:
        write(out/'protocol_frozen.json',dict(at_utc=stamp(),**provenance))
        for p in sources:
            dst=out/'executed_source'/p.relative_to(CODE);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    assert not (out/f'{a.phase}_completed.json').exists(),'Completed phase is immutable'
    prepare(a.root,out,cfg)
    with gzip.open(out/'states.jsonl.gz','rt') as f:bank=[json.loads(x) for x in f]
    pool=read(out/'bank_instances.json');pilot_ids={r['instance_id'] for n in (50,100) for r in [x for x in pool if x['nodes']==n][:cfg['pilot_bank_instances_per_size']]}
    wanted=[i for i,r in enumerate(bank) if a.phase=='full' or r['instance_id'] in pilot_ids]
    programs=[canonical_program(e) for e in cfg['diagnostic_rules']];assert len(programs)==12
    write(out/'diagnostic_programs.json',programs);measures=out/'measurements';measures.mkdir(exist_ok=True)
    done=0;reused=0;before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with ProcessPoolExecutor(max_workers=cfg['workers'],initializer=initialize,initargs=(bank,cfg,programs)) as workers:
        pending={}
        for i in wanted:
            path=measures/f'state{i:04d}.json'
            if path.exists():assert read(path)['bank_id']==bank[i]['bank_id'];reused+=1;done+=1
            else:pending[workers.submit(measure,i)]=path
        for future in as_completed(pending):
            write(pending[future],future.result());done+=1
            if done%32==0:print(a.phase,'states',done,'/',len(wanted),flush=True)
    after=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.perf_counter()-begin
    result=dict(status='completed',at_utc=stamp(),phase=a.phase,states=done,states_reused_from_nested_pilot=reused,
        original_instances=len({bank[i]['instance_id'] for i in wanted}),unique_branches_in_phase=done*37,new_branches=(done-reused)*37,
        wall_seconds=elapsed,parent_cpu_seconds=time.process_time()-cpu,
        waited_child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
        projected_full_measurement_wall_seconds=elapsed*len(bank)/len(wanted) if a.phase=='pilot' else None,
        bank_sha256=digest(out/'states.jsonl.gz'),final_opened=False,parameter_training=False)
    write(out/f'{a.phase}_completed.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
