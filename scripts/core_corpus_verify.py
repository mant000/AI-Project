"""Independent route, strict-improvement and family/target-isolation audit of SFT corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from stage3_common import read,write,digest,stamp
from core_mechanism import cycle_hash,select_slice
from tsp_value.data import distances,length,validate_tour
from tsp_value.programs import canonical_program


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--corpus',type=Path,required=True);a=ap.parse_args()
    start=time.perf_counter();out=a.corpus;cfg=read(out/'protocol_frozen.json')['protocol'];assert (out/'completed.json').exists()
    data=a.root/'data/prepared/stage1/project_instances.jsonl';assert digest(data)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    rows=[json.loads(x) for x in data.read_text().splitlines()];items=select_slice(rows,'discovery',cfg['discovery_offset_per_size'],cfg['discovery_instances_per_size']);byid={r['instance_id']:r for r in items}
    programs={r['program_sha256']:r for r in read(out/'grammar_candidates.json')};outcomes={};checked=0;error=0.;matrices={}
    for path in (out/'evaluations').glob('*.json'):
        result=read(path);key=result['program_sha256'];assert key in programs
        assert canonical_program(programs[key]['source'])['program_sha256']==key
        for row in result.get('results',[]):
            assert row['instance_id'] in byid and row['project_use']=='discovery';item=byid[row['instance_id']]
            if row['instance_id'] not in matrices:matrices[row['instance_id']]=distances(item['coords'])
            validate_tour(row['best_tour'],item['nodes']);actual=length(np.asarray(row['best_tour']),matrices[row['instance_id']])
            error=max(error,abs(actual-row['original_length']));assert abs(actual-row['original_length'])<1e-10
            assert abs(row['gap_percent']-100*(actual/item['reference_length']-1))<1e-10
            assert row['delta_evaluations']==cfg['delta_budget_by_nodes'][str(item['nodes'])];checked+=1
        if result['ok']:
            result['gap']=float(np.mean([r['gap_percent'] for r in result['results']]))
            vector=[(r['instance_id'],r['seed'],cycle_hash(r['best_tour'])) for r in result['results']]
            result['behavior']=hashlib.sha256(json.dumps(vector).encode()).hexdigest()
        outcomes[key]=result
    prior_targets=set();prior_behaviors=set();prior_families=set();coverage={};total=0
    for split in ('train','validation','heldout'):
        examples=[json.loads(x) for x in (out/f'{split}.jsonl').read_text().splitlines()];targets=set();behaviors=set();families=set()
        for example in examples:
            parent=programs[example['parent_sha256']];child=programs[example['target_sha256']]
            assert example['split']==parent['split']==child['split']==split and parent['family']==child['family']==example['family']
            pr,ch=outcomes[parent['program_sha256']],outcomes[child['program_sha256']];assert pr['ok'] and ch['ok']
            assert ch['gap']<pr['gap']-1e-9 and pr['behavior']!=ch['behavior']
            assert abs(example['measured_improvement_gap_pp']-(pr['gap']-ch['gap']))<1e-10
            assert example['response']==child['source'] and child['source'] not in json.dumps(example['messages'])
            assert example['target_behavior_sha256']==ch['behavior']
            assert example['source_kind']=='executed_predeclared_grammar'
            targets.add(child['program_sha256']);behaviors.add(ch['behavior']);families.add(child['family']);total+=1
        assert not targets&prior_targets and not behaviors&prior_behaviors and not families&prior_families
        prior_targets.update(targets);prior_behaviors.update(behaviors);prior_families.update(families)
        coverage[split]=dict(examples=len(examples),targets=len(targets),target_behaviors=len(behaviors),families=len(families))
    summary=dict(status='passed',at_utc=stamp(),routes_recomputed=checked,max_route_cost_error=error,strict_improvement_pairs_checked=total,
        coverage=coverage,family_and_target_hash_and_target_behavior_splits_disjoint=True,target_not_in_prompt=True,
        only_discovery_instances_used=True,normalized_data_sha256=digest(data),final_opened=False,
        limitations='Behavior isolation is for target route vectors; only one validation family survives deduplication. Same discovery instances underlie all family splits.',wall_seconds=time.perf_counter()-start)
    write(out/'verification.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
