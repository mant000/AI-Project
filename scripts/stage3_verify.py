"""Audit raw A/B boundaries, program hashes and real tours; independently replay frozen rules."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from stage3_common import CODE,read,write,digest,stamp,choose,evaluate,worker
from tsp_value.programs import canonical_program,INITIAL
from tsp_value.data import distances,length,validate_tour


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--prepare',type=Path,required=True)
    ap.add_argument('--study',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start,cpu=time.perf_counter(),time.process_time(); a.output.mkdir(parents=True,exist_ok=False)
    cfg=read(a.prepare/'protocol_frozen.json')['protocol']; completed=read(a.study/'completed.json')
    assert completed['logical_outputs']==len(cfg['discovery_seeds'])*2*cfg['candidates_per_arm_per_seed']
    assert not completed['final_opened'] and not completed['sft_grpo_started']
    assert digest(a.prepare/'bank.jsonl')==read(a.prepare/'bank_frozen.json')['sha256']
    data=a.root/'data/prepared/stage1/project_instances.jsonl'
    rows=[json.loads(x) for x in data.read_text().splitlines()]; by_id={r['instance_id']:r for r in rows}
    protected={r['instance_id'] for r in rows if r['project_use'] in ('value_val','program_select','program_dev_eval')}
    bank=[json.loads(x) for x in (a.prepare/'bank.jsonl').read_text().splitlines()]
    for state in bank:
        assert by_id[state['instance_id']]['project_use']=='discovery'
        assert 'records' not in state and 'y' not in state and 'future' not in state['state']
        assert len(state['edges'])==state['nodes']==len({tuple(e) for e in state['edges']})
    generations=[read(p) for p in (a.study/'generations').glob('*.json')]
    candidates=[read(p) for p in (a.study/'candidates').glob('*.json')]
    assert {r['candidate_id'] for r in generations}=={r['candidate_id'] for r in candidates}
    counts=Counter((r['discovery_seed'],r['arm']) for r in candidates)
    for seed in cfg['discovery_seeds']:
        for arm in ('A','B'): assert counts[seed,arm]==16
    for generation in generations:
        candidate=next(r for r in candidates if r['candidate_id']==generation['candidate_id'])
        assert generation['sampling_seed']==candidate['discovery_seed']*1000+candidate['index']
        assert len(generation['input_token_ids'])==generation['input_tokens']
        assert len(generation['output_token_ids'])==generation['output_tokens']
        prompt=generation['rendered_prompt']
        assert not any(iid in prompt for iid in protected)
        assert ('ADDITIONAL STATE VALUE FEEDBACK:' in prompt)==(candidate['arm']=='B')
        if candidate['valid']:
            normalized=canonical_program(candidate['source'])
            assert normalized['program_sha256']==candidate['program_sha256']
            assert (a.study/'programs'/f"{candidate['program_sha256']}.py").read_text()==candidate['source']
            assert candidate['value_feedback']['bank_sha256']==digest(a.prepare/'bank.jsonl')
    frozen=read(a.study/'frozen_programs.json'); dev=read(a.study/'dev_evaluation.json')
    assert dev['frozen_programs_sha256']==digest(a.study/'frozen_programs.json')
    for chosen in frozen['programs']:
        idx=int(np.argmin(chosen['select_mean_gaps_percent']))
        assert chosen['selected_program']['program_sha256']==chosen['elites'][idx]
    checked=0; max_error=0.; matrices={}
    for p in (a.study/'evaluations').glob('*.json'):
        evaluation=read(p)
        for result in evaluation.get('results',[]):
            item=by_id[result['instance_id']]; assert item['project_use']==evaluation['pool']
            key=item['instance_id']; matrices.setdefault(key,distances(item['coords']))
            validate_tour(result['best_tour'],item['nodes'])
            actual=length(np.asarray(result['best_tour']),matrices[key]); err=abs(actual-result['original_length'])
            max_error=max(max_error,err); assert err<1e-10
            assert abs(result['gap_percent']-100*(actual/item['reference_length']-1))<1e-10
            if evaluation['method']=='gls': assert result['delta_evaluations']==cfg['delta_budget_by_nodes'][str(item['nodes'])]
            checked+=1
    # Replay the first declared dev instance of each size, all three fixed solver seeds.
    # This is additional verification cost, never additional independent dev instances.
    replay_items=choose(rows,'program_dev_eval',1); replay=[]; initial=canonical_program(INITIAL)
    programs={initial['program_sha256']:initial}
    programs.update({r['selected_program']['program_sha256']:r['selected_program'] for r in frozen['programs']})
    for program in programs.values():
        result=evaluate(program['expression'],replay_items,cfg['program_dev_eval_solver_seeds'],cfg)
        write(a.output/f"replay-{program['program_sha256']}.json",result); assert result['ok']
        original=next(r['evaluation'] for r in dev['runs'] if (r.get('program_sha256')==program['program_sha256'] or
            (r['label']=='gls' and program['program_sha256']==initial['program_sha256'])))
        original_by_key={(r['instance_id'],r['seed']):r for r in original['results']}
        for r in result['results']:
            old=original_by_key[r['instance_id'],r['seed']]
            for k in ('best_tour','original_length','gap_percent','delta_evaluations','penalty_events'): assert old[k]==r[k],k
        replay.append(dict(program_sha256=program['program_sha256'],replayed_routes=len(result['results']),
            cpu_seconds=result['cpu_seconds'],subprocess_wall_seconds=result['subprocess_wall_seconds']))
    summary=dict(status='passed',at_utc=stamp(),logical_outputs=len(candidates),raw_generations=len(generations),
        valid_outputs=sum(r['valid'] for r in candidates),independently_recomputed_tours=checked,
        max_route_length_abs_error=max_error,paired_generation_seeds_match=True,prompt_pool_leakage_found=False,
        A_has_no_state_value_prompt=True,B_has_only_discovery_bank_state_values=True,
        bank_all_edge_and_program_hashes_valid=True,frozen_selection_precedes_dev=True,
        independent_frozen_program_replays=replay,final_opened=False,
        wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_used=False)
    write(a.output/'summary.json',summary); print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
