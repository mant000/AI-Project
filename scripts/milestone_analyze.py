"""CPU-only audit and analysis of the fixed policy-fragment milestone study.

Inputs are immutable raw records, not hand-entered summary numbers. The original
TSP instance is the sampling unit; solver seeds are averaged before inference.
Discovery and instance indices are resampled jointly (crossed paired bootstrap).
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import spearmanr

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'src'))
from tsp_value.data import distances, length, validate_tour
from tsp_value.programs import canonical_program, INITIAL


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def matrix(evaluation, ids):
    rows = evaluation['results']
    return np.array([np.mean([r['gap_percent'] for r in rows if r['instance_id']==key]) for key in ids])


def bootstrap(delta, ids, draws=10000):
    """Crossed resampling retains pairing across arms and balances both sizes."""
    rng = np.random.default_rng(20261119)
    strata = [[j for j,k in enumerate(ids) if f'tsp{n}_' in k] for n in (50,100)]
    samples = []
    for _ in range(draws):
        ii = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        dd = rng.integers(delta.shape[0], size=delta.shape[0])
        samples.append(float(delta[np.ix_(dd,ii)].mean()))
    low, high = np.quantile(samples, [.025,.975])
    return dict(mean_gap_pp=float(delta.mean()), ci95_low=float(low), ci95_high=float(high),
                per_discovery_seed_gap_pp=delta.mean(1).tolist(), original_instances=len(ids),
                discovery_repetitions=delta.shape[0], bootstrap_draws=draws,
                interpretation='crossed paired discovery-seed and size-stratified original-instance bootstrap; four discovery repetitions limit precision')


def audit_study(study, prepare):
    protocol=read(study/'protocol_frozen.json'); cfg=protocol['protocol']
    for relative,expected in protocol['source_sha256'].items():
        assert digest(study/'executed_source'/relative)==expected,(relative,'executed source hash')
    complete = read(study/'completed.json')
    assert complete['status']=='completed' and complete['outputs']==cfg['total_outputs']
    pools = {key:read(prepare/f'{key}_instances.json') for key in ('bank','discovery','program_select','program_dev_eval')}
    flat = [r for rows in pools.values() for r in rows]
    assert len({r['instance_id'] for r in flat})==256
    items = {r['instance_id']:r for r in flat}
    route_checks=0; max_error=0.; fingerprints={}
    evaluations = {}
    for file in sorted((study/'evaluations').glob('*.json')):
        ev=read(file)
        expected={(r['instance_id'],seed) for r in pools[ev['pool']] for seed in ev['seeds']}
        observed={(r['instance_id'],r['seed']) for r in ev['results']}
        assert observed<=expected and len(observed)==len(ev['results'])
        if ev['ok']:assert observed==expected
        for r in ev['results']:
            item=items[r['instance_id']]; validate_tour(r['best_tour'],item['nodes'])
            actual=length(np.array(r['best_tour']), distances(item['coords']))
            err=abs(actual-r['original_length']); max_error=max(max_error,err); assert err<1e-10
            assert abs(r['gap_percent']-100*(actual/item['reference_length']-1))<1e-10
            if ev['method']=='gls': assert r['delta_evaluations']==cfg['delta_budget_by_nodes'][str(item['nodes'])]
            route_checks+=1
        if ev['ok']:assert abs(np.mean([r['gap_percent'] for r in ev['results']])-ev['mean_gap_percent'])<1e-12
        evaluations[(ev['pool'],ev['method'],ev['program_sha256'])]=ev
        if ev['pool']=='discovery' and ev['ok']:
            fingerprints[ev['program_sha256']]=hashlib.sha256(json.dumps([r['best_tour'] for r in ev['results']]).encode()).hexdigest()
    initial=canonical_program(INITIAL); frozen=read(study/'frozen_programs.json')
    dev=read(study/'dev_evaluation.json'); assert dev['frozen_programs_sha256']==digest(study/'frozen_programs.json')
    assert dev['at_utc']>=frozen['at_utc'] and frozen['all_frozen_before_dev']
    assert all(ev['at_utc']>=frozen['at_utc'] for ev in evaluations.values() if ev['pool']=='program_dev_eval')
    stats={}; contexts={}; feedback_checked=0
    with gzip.open(prepare/'states.jsonl.gz','rt') as f: bank=list(map(json.loads,f))
    wanted=set(read(prepare/'feedback_selection.json')['bank_ids']); bank=[r for r in bank if r['bank_id'] in wanted]
    rollout_route_checks=0
    for path in (study/'rollouts').glob('*.json'):
        obj=read(path);assert len(obj['results'])==128
        for row,r in zip(bank,obj['results']):
            assert row['bank_id']==r['bank_id'];validate_tour(r['best_tour'],row['nodes'])
            assert abs(length(np.array(r['best_tour']),distances(row['state']['coords']))-r['best_length'])<1e-10
            rollout_route_checks+=1
    for seed in cfg['discovery_seeds']:
        for arm in cfg['arms']:
            contexts[seed,arm]=dict(parent=initial,score=evaluations['discovery','gls',initial['program_sha256']]['mean_gap_percent'],population=[initial],last=None)
            stats[f'{arm}/{seed}']=dict(arm=arm,seed=seed,outputs=0,valid=0,unique=set(),behaviors=set(),same_parent_routes=0,parent_updates=0,
                                        directional_feedback_prompts=0,feedback_states_nonzero=0,feedback_states_total=0,input_tokens=0,output_tokens=0,generation_seconds=0.)
    for seed in cfg['discovery_seeds']:
        for index in range(cfg['candidates_per_arm_per_seed']):
            for arm in cfg['arms']:
                cid=f'{arm}-seed{seed}-candidate{index:03d}'; c=read(study/'candidates'/f'{cid}.json'); raw=read(study/'generations'/f'{cid}.json')
                ctx=contexts[seed,arm]; st=stats[f'{arm}/{seed}']; assert c['parent_sha256']==ctx['parent']['program_sha256']
                assert raw['candidate_id']==cid and raw['sampling_seed']==seed*1000+index
                assert 'REAL ENDPOINT FEEDBACK:' in raw['messages'][1]['content']
                st['outputs']+=1
                for field in ('input_tokens','output_tokens'):st[field]+=raw[field]
                st['generation_seconds']+=raw['generation_wall_seconds']
                try:p=canonical_program(raw['response'])
                except Exception:p=None
                previous=ctx['last']; feedback_default=ctx['parent']
                if c['valid']:
                    assert p and p['program_sha256']==c['program_sha256']
                    score=evaluations['discovery','gls',p['program_sha256']]['mean_gap_percent']; assert score==c['mean_discovery_gap_percent']
                    st['valid']+=1; st['unique'].add(p['program_sha256']); st['behaviors'].add(fingerprints[p['program_sha256']]); st['same_parent_routes']+=c['same_endpoint_routes_as_parent']
                    if p['program_sha256'] not in {q['program_sha256'] for q in ctx['population']}:ctx['population'].append(p)
                    if score<ctx['score']-1e-12:ctx.update(parent=p,score=score); st['parent_updates']+=1
                if arm!='A':
                    fb=read(study/'feedback'/f'{cid}.json')
                    assert json.dumps(fb['display'],separators=(',',':')) in raw['messages'][1]['content']
                    if fb['status']=='available':
                        d=fb['display']; rows=fb['states_detail']; assert len(rows)==128
                        assert [r['bank_id'] for r in rows]==[r['bank_id'] for r in bank]
                        parent=canonical_program(d['compared_parent_code']); child=canonical_program(d['compared_child_code']); mode='single' if arm=='P' else 'fragment'
                        assert parent['program_sha256']==(previous['parent_sha256'] if previous and previous['valid'] else feedback_default['program_sha256'])
                        assert child['program_sha256']==(previous['program_sha256'] if previous and previous['valid'] else feedback_default['program_sha256'])
                        pa=read(study/'rollouts'/f'{mode}-{parent["program_sha256"]}.json')['results']
                        ch=read(study/'rollouts'/f'{mode}-{child["program_sha256"]}.json')['results']
                        values=np.array([100*(p['best_length']-c['best_length'])/r['state']['best_length'] for r,p,c in zip(bank,pa,ch)])
                        assert np.max(np.abs(values-np.array([r['child_minus_parent_benefit_pp'] for r in rows])))<1e-12
                        assert abs(values.mean()-d['mean_measured_child_minus_parent_benefit_pp'])<1e-12
                        nz=int((np.abs(values)>1e-7).sum()); assert nz==d['child_better']+d['parent_better']
                        assert d['tied']==128-nz
                        selected=[(seed+13*index)%128,(seed+13*index+64)%128]
                        assert all(abs(ex['measured_child_minus_parent_benefit_pp']-values[j])<1e-12 for ex,j in zip(d['fixed_schedule_examples'],selected))
                        st['directional_feedback_prompts']+=nz>0; st['feedback_states_nonzero']+=nz; st['feedback_states_total']+=128; feedback_checked+=1
                ctx['last']=c
    for choice in frozen['programs']:
        ctx=contexts[choice['discovery_seed'],choice['arm']]
        elite=sorted(ctx['population'],key=lambda p:evaluations['discovery','gls',p['program_sha256']]['mean_gap_percent'])[:cfg['elite_count']]
        assert [p['program_sha256'] for p in elite]==choice['elite_hashes']
        scores=[evaluations['program_select','gls',p['program_sha256']]['mean_gap_percent'] for p in elite]
        assert scores==choice['program_select_gaps']
        assert elite[int(np.argmin(scores))]['program_sha256']==choice['selected_program']['program_sha256']
    for st in stats.values():
        st['unique_programs']=len(st.pop('unique'));st['unique_endpoint_route_vectors']=len(st.pop('behaviors'))
    ids=sorted(r['instance_id'] for r in pools['program_dev_eval']); arrays={}
    for arm in cfg['arms']:
        arrays[arm]=np.stack([matrix(next(r['evaluation'] for r in dev['runs'] if r.get('arm')==arm and r['discovery_seed']==seed),ids) for seed in cfg['discovery_seeds']])
    contrasts={f'{lhs}-{rhs}':bootstrap(arrays[lhs]-arrays[rhs],ids) for lhs,rhs in [('F','A'),('F','P'),('P','A')]}
    contrasts['F-A']['predeclared_success']=bool(contrasts['F-A']['mean_gap_pp']<=-.05 and contrasts['F-A']['ci95_high']<0)
    costs=[json.loads(x) for x in (study/'attempt_costs.jsonl').read_text().splitlines()]
    return dict(status='passed',study=study.name,protocol_sha256=digest(study/'protocol_frozen.json'),
                endpoint_routes_repriced=route_checks,feedback_rollout_routes_repriced=rollout_route_checks,max_route_cost_error=max_error,feedback_prompts_recomputed=feedback_checked,
                candidates_audited=complete['outputs'],selection_recomputed=True,freeze_before_dev_verified=True,
                instances_per_role={k:len(v) for k,v in pools.items()},lineages=list(stats.values()),contrasts=contrasts,
                arm_mean_gap_percent={k:float(v.mean()) for k,v in arrays.items()},
                baseline_mean_gap_percent={r['label']:r['evaluation']['mean_gap_percent'] for r in dev['runs'] if 'label' in r},
                costs=dict(attempts=costs,generation_seconds=sum(v['generation_seconds'] for v in stats.values()),
                           input_tokens=complete['input_tokens'],output_tokens=complete['output_tokens']),final_opened=False),arrays,ids


def analyze_mechanism(prepare):
    """Recompute mechanism metrics from every raw branch, with no solver calls."""
    records=[]; price_checks=0
    with gzip.open(prepare/'states.jsonl.gz','rt') as f:bank=list(map(json.loads,f))
    for i,row in enumerate(bank):
        m=read(prepare/'measurements'/f'state{i:04d}.json'); assert m['bank_id']==row['bank_id']
        d=distances(row['state']['coords']); ref=m['baseline']
        for obj in [ref,*[v for p in m['programs'] for v in p['variants']]]:
            validate_tour(obj['best_tour'],row['nodes']);assert abs(length(np.array(obj['best_tour']),d)-obj['best_length'])<1e-10;price_checks+=1
        for j,p in enumerate(m['programs']):
            values={v['mode']:100*(ref['best_length']-v['best_length'])/row['state']['best_length'] for v in p['variants']}
            if j in (0,9):assert all(abs(x)<1e-12 for x in values.values());continue
            records.append(dict(instance_id=row['instance_id'],**values))
    target=np.array([r['persistent'] for r in records]);result={}
    for mode in ('single','fragment','persistent'):
        pred=np.array([r[mode] for r in records]); nz=np.abs(pred)>1e-7;informative=np.abs(target)>1e-7
        result[mode]=dict(comparisons=len(pred),nonzero=int(nz.sum()),nonzero_fraction=float(nz.mean()),
                          spearman=float(spearmanr(pred,target).statistic),informative_target=int(informative.sum()),
                          tie_aware_agreement=float(np.where(nz,np.sign(pred)==np.sign(target),.5)[informative].mean()))
    old=read(prepare.parent/'mechanism/summary.json')
    for mode in result:
        assert result[mode]['nonzero']==old['overall'][mode]['nonzero']
        assert abs(result[mode]['spearman']-old['overall'][mode]['spearman_to_persistent_conditional_target'])<1e-12
    per_instance=[]
    for key in sorted({r['instance_id'] for r in records}):
        rr=[r for r in records if r['instance_id']==key];target=np.array([r['persistent'] for r in rr]);valid=np.abs(target)>1e-7; metrics={}
        for mode in ('single','fragment'):
            pred=np.array([r[mode] for r in rr]);covered=np.abs(pred)>1e-7
            metrics[mode]=dict(nonzero_fraction=float(covered.mean()),
                tie_aware_agreement_on_informative_target=float(np.where(covered,np.sign(pred)==np.sign(target),.5)[valid].mean()) if valid.any() else None,
                non_abstaining_target_coverage=float(covered[valid].mean()) if valid.any() else None)
        per_instance.append(metrics)
    rng=np.random.default_rng(20261119);paired={}
    for metric in ('nonzero_fraction','tie_aware_agreement_on_informative_target','non_abstaining_target_coverage'):
        delta=np.array([m['fragment'][metric]-m['single'][metric] for m in per_instance if m['single'][metric] is not None])
        draws=rng.integers(len(delta),size=(5000,len(delta)));ci=np.quantile(delta[draws].mean(1),[.025,.975])
        paired[metric]=dict(mean_instance_paired_difference=float(delta.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),original_instances=len(delta))
        for field in paired[metric]:assert abs(paired[metric][field]-old['paired_instance_bootstrap'][metric][field])<1e-12
    return dict(status='passed',routes_repriced=price_checks,states=len(bank),overall=result,
                paired_instance_bootstrap=paired,bootstrap_source='all metrics and resamples independently recomputed from raw branches, no solver calls')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',type=Path,default=CODE/'results/milestone-fragment-v1/prepare')
    ap.add_argument('--study',type=Path,default=CODE/'results/milestone-fragment-expression-v1/discovery')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    summary,arrays,ids=audit_study(a.study,a.prepare)
    write(a.output/'study_summary.json',summary)
    write(a.output/'mechanism_summary.json',analyze_mechanism(a.prepare))
    with (a.output/'dev_instance_means.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','discovery_seed','instance_id','mean_gap_percent'])
        seeds=read(a.study/'protocol_frozen.json')['protocol']['discovery_seeds']
        for arm,array in arrays.items():
            for s,values in zip(seeds,array):
                for key,value in zip(ids,values):w.writerow([arm,s,key,value])
    write(a.output/'analysis_verification.json',dict(status='passed',wall_seconds=time.perf_counter()-start,source_sha256=digest(Path(__file__)),
                                                   scope='raw record repricing, parent/selection/freeze/feedback checks and bootstrap; no model inference or new solver executions'))
    print(json.dumps(dict(arms=summary['arm_mean_gap_percent'],contrasts=summary['contrasts'],route_checks=summary['endpoint_routes_repriced']),indent=2))


if __name__=='__main__':main()
