"""Build explicitly sourced, executed and family-isolated local-improvement examples."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time
import numpy as np
from stage3_common import CODE,read,write,digest,stamp,evaluate
from core_mechanism import select_slice,cycle_hash
from stage3_discover import SYSTEM,TASK
from tsp_value.programs import canonical_program,INITIAL


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'completed.json').exists();start,cpu=time.perf_counter(),time.process_time();cfg=read(CODE/'configs/core_corpus_v1.json')
    frozen=dict(at_utc=stamp(),protocol=cfg,protocol_sha256=digest(CODE/'configs/core_corpus_v1.json'),source_sha256={str(p.relative_to(CODE)):digest(p) for p in [Path(__file__),CODE/'src/tsp_value/programs.py',CODE/'scripts/stage3_worker.py']})
    if (out/'protocol_frozen.json').exists():assert read(out/'protocol_frozen.json')['protocol']==cfg
    else:write(out/'protocol_frozen.json',frozen)
    for name in ('programs','evaluations'):(out/name).mkdir(exist_ok=True)
    rows=[json.loads(x) for x in (a.root/'data/prepared/stage1/project_instances.jsonl').read_text().splitlines()]
    items=select_slice(rows,'discovery',cfg['discovery_offset_per_size'],cfg['discovery_instances_per_size'])
    write(out/'instance_selection.json',[{k:r[k] for k in ('instance_id','nodes','project_use')} for r in items])
    programs=[];seen=set()
    for family in cfg['families']:
        names=list(family['parameters'])
        for values in itertools.product(*(family['parameters'][k] for k in names)):
            parameters=dict(zip(names,values));p=canonical_program(family['template'].format(**parameters))
            assert p['program_sha256'] not in seen,'Cross-family source duplicate must be resolved explicitly';seen.add(p['program_sha256'])
            p.update(family=family['id'],split=family['split'],parameters=parameters,source_kind='predeclared_grammar')
            programs.append(p);(out/'programs'/f"{p['program_sha256']}.py").write_text(p['source'])
    baseline=canonical_program(INITIAL);baseline.update(family='reference',split='reference',parameters={},source_kind='reference')
    programs.append(baseline);write(out/'grammar_candidates.json',programs)
    outcomes={}
    def execute(program):
        p=out/'evaluations'/f"{program['program_sha256']}.json"
        if p.exists():result=read(p)
        else:
            result=evaluate(program['expression'],items,cfg['solver_seeds'],cfg)
            result.update(at_utc=stamp(),program_sha256=program['program_sha256'],family=program['family'],split=program['split'],source_kind=program['source_kind'])
            # The wrapper's per-call RUSAGE_CHILDREN delta overlaps under threads.
            # Use measured worker process CPU; report aggregate parent children once.
            result['child_cpu_seconds']=None;result['child_cpu_note']='Not attributed per task under concurrent subprocesses; cpu_seconds is the isolated worker measurement'
            write(p,result)
        if result['ok']:
            result['mean_gap_percent']=float(np.mean([r['gap_percent'] for r in result['results']]))
            result['gap_by_size']={str(n):float(np.mean([r['gap_percent'] for r in result['results'] if r['nodes']==n])) for n in (50,100)}
            key=[(r['instance_id'],r['seed'],cycle_hash(r['best_tour'])) for r in result['results']]
            result['behavior_sha256']=hashlib.sha256(json.dumps(key).encode()).hexdigest()
        return result
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs={pool.submit(execute,p):p for p in programs}
        for f in as_completed(jobs):
            result=f.result();outcomes[result['program_sha256']]=result
            print('evaluated '+str(len(outcomes))+'/'+str(len(programs)),flush=True)
    byhash={p['program_sha256']:p for p in programs};rng=np.random.default_rng(cfg['pair_order_seed']);seen_target_hashes=set();seen_target_behaviors=set();dataset={};coverage={}
    for split in ('train','validation','heldout'):
        examples=[];candidate_count=overlap_removed=0
        for family in [f for f in cfg['families'] if f['split']==split]:
            group=[p for p in programs if p['family']==family['id'] and outcomes[p['program_sha256']]['ok']]
            for child in group:
                child_result=outcomes[child['program_sha256']]
                parents=[p for p in group if outcomes[p['program_sha256']]['mean_gap_percent']>child_result['mean_gap_percent']+1e-9 and outcomes[p['program_sha256']]['behavior_sha256']!=child_result['behavior_sha256']]
                if not parents:continue
                candidate_count+=1
                if child['program_sha256'] in seen_target_hashes or (split!='train' and child_result['behavior_sha256'] in seen_target_behaviors):
                    overlap_removed+=1;continue
                order=rng.permutation(len(parents))[:cfg['maximum_pairs_per_target']]
                for j in order:
                    parent=parents[j];parent_result=outcomes[parent['program_sha256']]
                    user=TASK+'\nREAL PARENT ENDPOINT FEEDBACK:\n'+json.dumps(dict(parent_code=parent['source'],parent_mean_discovery_gap_percent=parent_result['mean_gap_percent'],
                        parent_gap_by_size=parent_result['gap_by_size'],original_instances=len(items),solver_seeds=cfg['solver_seeds'],work_budget_by_nodes=cfg['delta_budget_by_nodes']),indent=2)
                    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':user}];assert child['source'] not in user
                    examples.append(dict(example_id=f"{family['id']}:{parent['program_sha256'][:12]}:{child['program_sha256'][:12]}",split=split,family=family['id'],
                        source_kind='executed_predeclared_grammar',parent_sha256=parent['program_sha256'],target_sha256=child['program_sha256'],
                        target_behavior_sha256=child_result['behavior_sha256'],parent_gap_percent=parent_result['mean_gap_percent'],child_gap_percent=child_result['mean_gap_percent'],
                        measured_improvement_gap_pp=parent_result['mean_gap_percent']-child_result['mean_gap_percent'],
                        messages=messages,response=child['source']))
        rng.shuffle(examples);examples=examples[:cfg['maximum_pairs'][split]];dataset[split]=examples
        seen_target_hashes.update(r['target_sha256'] for r in examples);seen_target_behaviors.update(r['target_behavior_sha256'] for r in examples)
        coverage[split]=dict(examples=len(examples),families=len({r['family'] for r in examples}),distinct_targets=len({r['target_sha256'] for r in examples}),
            distinct_target_behaviors=len({r['target_behavior_sha256'] for r in examples}),eligible_targets_before_cross_split_filter=candidate_count,targets_removed_for_prior_split_overlap=overlap_removed)
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
    families={split:{r['family'] for r in rr} for split,rr in dataset.items()}
    for left in dataset:
        for right in dataset:
            if left!=right:assert not families[left]&families[right]
    summary=dict(at_utc=stamp(),protocol_sha256=frozen['protocol_sha256'],grammar_candidates_excluding_reference=len(programs)-1,
        valid_programs=sum(r['ok'] for r in outcomes.values()),invalid_programs=sum(not r['ok'] for r in outcomes.values()),
        actual_endpoint_routes=sum(len(r.get('results',[])) for r in outcomes.values()),worker_cpu_seconds=sum(r.get('cpu_seconds',0) for r in outcomes.values()),
        wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        coverage=coverage,family_splits_disjoint=True,source_target_hash_splits_disjoint=True,cross_split_target_behavior_overlap_removed=True,
        actual_llm_generations=0,final_opened=False,teacher_heldout_or_program_dev_read=False,
        limitation='Positive local improvements within hand-specified arithmetic families, not necessarily better than standard GLS. Explicitly grammar-generated supervision, not automatically discovered LLM successes.')
    write(out/'completed.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
