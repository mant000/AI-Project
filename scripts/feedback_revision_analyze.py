"""Analyze aligned labels and frozen discovery separately; no model/solver runs."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))
from feedback_revision_prepare import read,write,digest,stamp,restore
from tsp_value.programs import canonical_program,compile_rule,INITIAL
from tsp_value.solver import clone,select_edge
from tsp_value.data import length,distances,validate_tour
from feedback_information_audit import rank_correlation


def table(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])


def load_bank(path):
    rows=[json.loads(x) for x in (path/'bank_observables.jsonl').read_text().splitlines()]
    exact={r['bank_id']:r for r in (read(p) for p in (path/'exact_states').glob('*.json'))}
    return rows,exact


def analyze_prepare(base,out):
    old,aligned=load_bank(base/'old-prepare');fresh,newvalues=load_bank(base/'fresh-prepare')
    original={r['bank_id']:r for r in (read(p) for p in (CODE/'results/core/prepare-v1/exact_states').glob('*.json'))}
    changes=[]
    for s in old:
        a=np.asarray(original[s['bank_id']]['true_scores_pp']);b=np.asarray(aligned[s['bank_id']]['true_scores_pp'])
        changes.append(dict(bank_id=s['bank_id'],instance_id=s['instance_id'],nodes=s['nodes'],state_index=s['state_index'],
            work_used=s['state']['delta_evaluations'],aligned_horizon=aligned[s['bank_id']]['remaining_work'],
            old_positive_actions=int(np.sum(a>1e-7)),aligned_positive_actions=int(np.sum(b>1e-7)),
            old_nonzero_actions=int(np.sum(np.abs(a)>1e-7)),aligned_nonzero_actions=int(np.sum(np.abs(b)>1e-7)),
            changed_action_values=int(np.sum(np.abs(a-b)>1e-7)),max_abs_change_pp=float(np.abs(a-b).max())))
    table(out/'old_horizon_changes.csv',changes)
    full={}
    for path in (base/'old-prepare/full_seed17').glob('*.json'):
        for r in read(path)['results']:full[r['program_sha256'],r['instance_id']]=r
    programs=[canonical_program(p.read_text()) for p in sorted((CODE/'results/core/discovery-v1/programs').glob('*.py'))]
    baseline=canonical_program(INITIAL)['program_sha256'];paired=[];score_rows=[]
    states={s['bank_id']:restore(s) for s in old}
    old_byid=defaultdict(list)
    for s in old:old_byid[s['instance_id']].append(s)
    for p in programs:
        key=p['program_sha256'];rule=compile_rule(p['expression']);vals_old=[];vals_new=[];gains=[]
        for iid,ss in old_byid.items():
            oo=[];nn=[]
            for s in ss:
                edge=select_edge(clone(states[s['bank_id']]),rule);j=list(map(tuple,s['edges'])).index(edge)
                oo.append(original[s['bank_id']]['true_scores_pp'][j]);nn.append(aligned[s['bank_id']]['true_scores_pp'][j])
            gain=full[baseline,iid]['gap_percent']-full[key,iid]['gap_percent']
            paired.append(dict(program_sha256=key,instance_id=iid,states=len(ss),old_mean_pp=float(np.mean(oo)),aligned_mean_pp=float(np.mean(nn)),
                old_any_nonzero=any(abs(v)>1e-7 for v in oo),aligned_any_nonzero=any(abs(v)>1e-7 for v in nn),
                full_seed17_gain_gap_pp=gain))
            vals_old.extend(oo);vals_new.extend(nn);gains.append(gain)
        score_rows.append(dict(program_sha256=key,expression=p['expression'],old_mean_pp=float(np.mean(vals_old)),
            aligned_mean_pp=float(np.mean(vals_new)),old_nonzero_states=sum(abs(v)>1e-7 for v in vals_old),
            aligned_nonzero_states=sum(abs(v)>1e-7 for v in vals_new),full_seed17_gain_gap_pp=float(np.mean(gains))))
    table(out/'old_rule_local_full_pairs.csv',paired);table(out/'old_rule_program_scores.csv',score_rows)
    # Fixed pre-existing grammar is evaluated on fresh states without fitting.
    grammar=read(CODE/'results/corpus-v1/grammar_candidates.json');coverage=[]
    for s in fresh:
        st=restore(s);edges=list(map(tuple,s['edges']));y=np.asarray(newvalues[s['bank_id']]['true_scores_pp']);vv=[]
        for p in grammar:
            e=select_edge(clone(st),compile_rule(p['expression']));vv.append(float(y[edges.index(e)]))
        coverage.append(dict(bank_id=s['bank_id'],instance_id=s['instance_id'],nodes=s['nodes'],state_index=s['state_index'],
            positive_actions=int(np.sum(y>1e-7)),nonzero_actions=int(np.sum(np.abs(y)>1e-7)),all_edge_best_pp=float(y.max()),
            grammar_positive_choices=sum(v>1e-7 for v in vv),grammar_nonzero_choices=sum(abs(v)>1e-7 for v in vv),
            grammar_best_pp=max(vv)))
    table(out/'fresh_state_coverage.csv',coverage)
    routes=0;maxerr=0.
    for bank,exact in [(old,aligned),(fresh,newvalues)]:
        for s in bank:
            d=distances(s['state']['coords']);r=exact[s['bank_id']]
            for record,value in zip(r['records'],r['true_scores_pp']):
                validate_tour(record['best_tour'],s['nodes']);cost=length(np.asarray(record['best_tour']),d)
                maxerr=max(maxerr,abs(cost-record['best_length']));assert abs(cost-record['best_length'])<1e-10
                expected=100*(r['records'][-1]['best_length']-cost)/s['state']['best_length']
                assert abs(expected-value)<1e-10;routes+=1
    summary=dict(at_utc=stamp(),old_control=dict(states=len(old),changed_states=sum(r['changed_action_values']>0 for r in changes),
        changed_action_values=sum(r['changed_action_values'] for r in changes),
        old_nonzero_actions=sum(r['old_nonzero_actions'] for r in changes),aligned_nonzero_actions=sum(r['aligned_nonzero_actions'] for r in changes),
        old_positive_actions=sum(r['old_positive_actions'] for r in changes),aligned_positive_actions=sum(r['aligned_positive_actions'] for r in changes),
        programs=len(programs),same_seed17_original_instances=len(old_byid),
        old_all_zero_programs=sum(r['old_nonzero_states']==0 for r in score_rows),aligned_all_zero_programs=sum(r['aligned_nonzero_states']==0 for r in score_rows),
        old_zero_local_but_nonzero_full_pairs=sum(not r['old_any_nonzero'] and abs(r['full_seed17_gain_gap_pp'])>1e-7 for r in paired),
        aligned_zero_local_but_nonzero_full_pairs=sum(not r['aligned_any_nonzero'] and abs(r['full_seed17_gain_gap_pp'])>1e-7 for r in paired),
        old_descriptive_spearman=rank_correlation([r['old_mean_pp'] for r in score_rows],[r['full_seed17_gain_gap_pp'] for r in score_rows]),
        aligned_descriptive_spearman=rank_correlation([r['aligned_mean_pp'] for r in score_rows],[r['full_seed17_gain_gap_pp'] for r in score_rows])),
        fresh=dict(states=len(fresh),original_instances=len({s['instance_id'] for s in fresh}),
            informative_states=sum(r['nonzero_actions']>0 for r in coverage),positive_states=sum(r['positive_actions']>0 for r in coverage),
            positive_states_reachable_by_fixed_grammar=sum(r['grammar_positive_choices']>0 for r in coverage),
            positive_actions=sum(r['positive_actions'] for r in coverage),nonzero_actions=sum(r['nonzero_actions'] for r in coverage),
            measured_actions=sum(len(s['edges']) for s in fresh)),
        integrity=dict(routes_revalued=routes,max_route_cost_error=maxerr),
        resources=dict(old=read(base/'old-prepare/completed.json'),fresh=read(base/'fresh-prepare/completed.json')),
        limitations=['Old controlled comparison reuses8 originals; no new-generalization claim.',
            'New bank differs from old bank: between-bank differences cannot be attributed only to horizon correction.',
            'Matching seed17 removes initial-trajectory seed discrepancy, not the single-action versus repeated-rule difference.',
            'Fixed-bank rank correlations with near-constant scores are descriptive, not causal or calibrated significance tests.'],
        parameter_training_started=False,final_opened=False)
    write(out/'prepare_summary.json',summary);print(json.dumps(summary,indent=2))


def analyze_discovery(base,out):
    study=base/'discovery';assert (study/'completed.json').exists()
    cfg=read(study/'protocol_frozen.json')['protocol'];dev=read(study/'dev_evaluation.json');frozen=read(study/'frozen_programs.json')
    assert dev['frozen_programs_sha256']==digest(study/'frozen_programs.json')
    candidates=[read(p) for p in (study/'candidates').glob('*.json')]
    calls=[read(p) for p in (study/'generations').glob('*.json')]
    assert len(candidates)==len(calls)==cfg['total_outputs']
    groups=[]
    for arm in cfg['arms']:
        rr=[r for r in candidates if r['arm']==arm];gg=[r for r in calls if r['candidate_id'].startswith(arm+'-')]
        ff=[read(p) for p in (study/'feedback').glob(arm+'-*.json')]
        groups.append(dict(arm=arm,outputs=len(rr),valid=sum(r['valid'] for r in rr),
            unique_programs=len({r['program_sha256'] for r in rr if r['valid']}),
            same_parent_endpoint_outputs=sum(r.get('same_endpoint_routes_as_parent',False) for r in rr),
            inputs=sum(r['input_tokens'] for r in gg),output_tokens=sum(r['output_tokens'] for r in gg),
            generation_seconds=sum(r['generation_wall_seconds'] for r in gg),
            prompts_with_direction=sum(any(a['nonzero_shown_spread'] for a in r['audits']) for r in ff)))
    table(out/'discovery_groups.csv',groups)
    grids={};devrows=[];instance_ids=[]
    for r in dev['runs']:
        byinst=defaultdict(list)
        for q in r['evaluation']['results']:
            byinst[q['instance_id']].append(q['gap_percent'])
            devrows.append(dict(label=r['label'],arm=r.get('arm','gls'),discovery_seed=r.get('discovery_seed'),**q))
        ids=sorted(byinst)
        if not instance_ids:instance_ids=ids
        assert instance_ids==ids
        grids[r['label']]=np.array([np.mean(byinst[i]) for i in ids])
    # Do not write raw best_tour arrays into summary tables.
    table(out/'dev_rows.csv',[{k:v for k,v in r.items() if k!='best_tour'} for r in devrows])
    comparisons=[];perseed=[];rng=np.random.default_rng(11919)
    for reference in ['A','P','S']:
        matrix=np.stack([grids[f'R-seed{s}']-grids[f'{reference}-seed{s}'] for s in cfg['discovery_seeds']])
        boot=[]
        for _ in range(10000):
            si=rng.integers(len(matrix),size=len(matrix));ii=rng.integers(matrix.shape[1],size=matrix.shape[1])
            boot.append(float(matrix[si][:,ii].mean()))
        low,high=np.quantile(boot,[.025,.975])
        comparisons.append(dict(comparison='R-'+reference,mean_gap_difference_pp=float(matrix.mean()),
            hierarchical_bootstrap95_low_pp=float(low),hierarchical_bootstrap95_high_pp=float(high),
            discovery_pairs=len(matrix),original_instances=matrix.shape[1],
            improving_discovery_seeds=int(np.sum(matrix.mean(1)<-1e-9)),
            practical_005pp_gain_excluded=bool(low>-.05)))
        for seed,row in zip(cfg['discovery_seeds'],matrix):
            perseed.append(dict(comparison='R-'+reference,discovery_seed=seed,mean_gap_difference_pp=float(row.mean())))
    table(out/'comparisons.csv',comparisons);table(out/'per_seed_comparisons.csv',perseed)
    primary=comparisons[0];sham=comparisons[-1]
    supported=primary['mean_gap_difference_pp']<=-.05 and primary['hierarchical_bootstrap95_high_pp']<0 and sham['mean_gap_difference_pp']<0
    summary=dict(at_utc=stamp(),outputs=len(candidates),groups=groups,comparisons=comparisons,
        success_gate_passed=bool(supported),frozen_lineages=len(frozen['programs']),
        distinct_selected_programs=len({r['selected_program']['program_sha256'] for r in frozen['programs']}),
        mean_gap_by_arm={a:float(np.mean([grids[f'{a}-seed{s}'].mean() for s in cfg['discovery_seeds']])) for a in cfg['arms']},
        resources=read(study/'process_resources.json'),limitations=[
            'Fixed existing common-SFT checkpoint and discovery budget; no conclusion about other generators/training scales.',
            'States used for feedback are selected for nonzero spread; natural bank denominator is reported separately.',
            'Sequential trajectories may diverge across arms; same scheduling/construction does not mean all later prompts are identical.',
            'Eight discovery repetitions; bootstrap is an estimate, not a universal equivalence proof.',
            'Oracle alternatives/labels cost CPU work; no learned-teacher or wall-time superiority claim.'],
        parameter_training_started=False,final_opened=False)
    write(out/'discovery_summary.json',summary);print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','discovery'],required=True);a=ap.parse_args()
    base=CODE/'results/feedback-revision-v1';out=base/('analysis-'+a.phase);out.mkdir(parents=True,exist_ok=True)
    (analyze_prepare if a.phase=='prepare' else analyze_discovery)(base,out)
    write(out/'analysis_source.json',dict(at_utc=stamp(),script_sha256=digest(Path(__file__)),protocol_sha256=digest(CODE/'configs/feedback_revision_v1.json')))


if __name__=='__main__':main()
