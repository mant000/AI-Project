"""New original-instance allocation and nested pilot/full policy-fragment measurement on server CPUs."""
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
from stage3_common import CODE,read,write,digest,stamp
from fragment_core import collect_states,intervene
from tsp_value.programs import canonical_program
from tsp_value.data import coordinate_hash

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
    if (out/'bank_frozen.json').exists():
        saved=read(out/'bank_frozen.json');assert saved['config_sha256']==digest(CODE/'configs/milestone_fragment_v1.json')
        assert saved['bank_sha256']==digest(out/'states.jsonl.gz');return
    data=root/'data/prepared/stage1/project_instances.jsonl';assert digest(data)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    rows=[json.loads(x) for x in data.read_text().splitlines()];selected=[];pools={}
    for role,(lo,hi) in cfg['roles'].items():
        rr=[copy.deepcopy(r) for r in rows if r['project_use']=='value_train' and lo<=r['source_line']<=hi]
        assert len(rr)==2*(hi-lo+1)
        for r in rr:r['original_project_use']='value_train';r['project_use']=role
        pools[role]=rr;selected.extend(rr)
    ids={r['instance_id'] for r in selected};assert len(ids)==256
    assert len({coordinate_hash(r['coords']) for r in selected})==256
    oldids=set();checked=[]
    for file in (CODE/'results').rglob('*.json'):
        if 'milestone-fragment-v1' in str(file):continue
        if 'selection' not in file.name and file.name not in ('pool_isolation.json','instance_selection.json'):continue
        content=file.read_text();oldids.update(re.findall(r'joshi:tsp(?:50|100)_train_concorde\.txt:\d+',content));checked.append(file)
    assert not ids&oldids,sorted(ids&oldids)
    source_manifest={r['instance_id']:r for r in csv.DictReader((CODE/'splits/manifest.csv').open())}
    for r in selected:
        old=source_manifest[r['instance_id']];assert old['coordinate_set_sha256']==coordinate_hash(r['coords'])
        r.update(source_row_sha256=old['source_row_sha256'],source_file=old['source_file'],source_file_sha256=old['source_file_sha256'])
    write(out/'selection.json',{role:[dict(instance_id=r['instance_id'],nodes=r['nodes'],project_use=role) for r in rr] for role,rr in pools.items()})
    write(out/'pool_isolation.json',dict(status='passed',original_instances=256,ids_and_coordinates_unique=True,historical_manifest_overlap=False,
        reassigned_only_previously_unexecuted_nominal_value_train_rows=True,historical_manifests={p.relative_to(CODE).as_posix():digest(p) for p in checked},
        input_sha256=digest(data),final_opened=False))
    for role,rr in pools.items():write(out/f'{role}_instances.json',rr)
    with (out/'benchmark_inputs.jsonl').open('w') as f:
        for r in selected:f.write(json.dumps(r)+'\n')
    bank=[]
    for item in pools['bank']:bank.extend(collect_states(item,cfg))
    assert len({r['instance_id'] for r in bank})==128
    with gzip.open(out/'states.jsonl.gz','wt') as f:
        for row in bank:f.write(json.dumps(row)+'\n')
    feedback=[]
    for index,item in enumerate(pools['bank']):
        rr=[r for r in bank if r['instance_id']==item['instance_id']]
        wanted=[r for r in rr if r['policy_index']==index%3 and r['phase_index']==(index//3)%4]
        feedback.append((wanted or rr)[0]['bank_id'])
    write(out/'feedback_selection.json',dict(bank_ids=feedback,original_instances=128,selection_uses_outcomes=False,
        fallback='first available state only when requested boundary absent before cap'))
    write(out/'bank_frozen.json',dict(at_utc=stamp(),states=len(bank),original_instances=128,config_sha256=digest(CODE/'configs/milestone_fragment_v1.json'),
        bank_sha256=digest(out/'states.jsonl.gz'),inputs_sha256=digest(out/'benchmark_inputs.jsonl'),feedback_original_instances=128,final_opened=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=['pilot','full'],required=True)
    ap.add_argument('--output',type=Path,default=CODE/'results/milestone-fragment-v1/prepare');a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    cfg=read(CODE/'configs/milestone_fragment_v1.json');begin,cpu=time.perf_counter(),time.process_time()
    sources=[Path(__file__),CODE/'scripts/fragment_core.py',*[CODE/f'src/tsp_value/{name}.py' for name in ('solver','programs','data')]]
    provenance=dict(config_sha256=digest(CODE/'configs/milestone_fragment_v1.json'),source_sha256={p.relative_to(CODE).as_posix():digest(p) for p in sources},protocol=cfg)
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
