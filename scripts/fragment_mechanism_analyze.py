"""Audit all branch costs and measure information about persistent conditional rule effects."""
import argparse
from collections import defaultdict,Counter
import csv
import gzip
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
from stage3_common import CODE,read,write,digest,stamp
from fragment_core import restore,intervene
from tsp_value.data import length,validate_tour


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',type=Path,default=CODE/'results/milestone-fragment-v1/prepare')
    ap.add_argument('--output',type=Path,default=CODE/'results/milestone-fragment-v1/mechanism');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    start,cpu=time.perf_counter(),time.process_time();cfg=read(a.prepare/'protocol_frozen.json')['protocol'];complete=read(a.prepare/'full_completed.json');assert complete['status']=='completed'
    with gzip.open(a.prepare/'states.jsonl.gz','rt') as f:bank=[json.loads(x) for x in f]
    records=[];prices=0;maxerror=0.;equivalent=0;examples=[]
    for index,row in enumerate(bank):
        measured=read(a.prepare/'measurements'/f'state{index:04d}.json');assert measured['bank_id']==row['bank_id'];state=restore(row)
        ref=measured['baseline'];rr=[ref,*[r for p in measured['programs'] for r in p['variants']]]
        for result in rr:
            t=np.array(result['best_tour']);validate_tour(t,row['nodes']);actual=length(t,state['distance'])
            error=abs(actual-result['best_length']);maxerror=max(maxerror,error);assert error<1e-10
            assert result['total_work']==cfg['delta_budget_by_nodes'][str(row['nodes'])];prices+=1
        for rule_index,p in enumerate(measured['programs']):
            modes={r['mode']:r for r in p['variants']};assert len({tuple(r['first_edge']) for r in p['variants']})==1
            values={name:r['benefit_pp'] for name,r in modes.items()}
            for r in p['variants']:assert abs(r['benefit_pp']-100*(ref['best_length']-r['best_length'])/state['best_length'])<1e-12
            if rule_index in (0,9):
                assert all(r['best_tour']==ref['best_tour'] and r['best_length']==ref['best_length'] for r in p['variants']);equivalent+=1
            record=dict(bank_id=row['bank_id'],instance_id=row['instance_id'],nodes=row['nodes'],policy_index=row['policy_index'],phase_index=row['phase_index'],
                rule_index=rule_index,program_sha256=p['program_sha256'],behavior_control=rule_index in (0,9),
                same_first_action=modes['persistent']['first_edge']==ref['first_edge'],**values)
            records.append(record)
            if record['same_first_action'] and abs(values['persistent'])>1e-7 and abs(values['single'])<=1e-7 and abs(values['fragment'])>1e-7 and len(examples)<8:examples.append(record)
    rr=[r for r in records if not r['behavior_control']];counts=Counter(r['instance_id'] for r in rr);ids=sorted(counts)
    def summary(group):
        true=np.array([r['persistent'] for r in group]);output={}
        for name in ('single','fragment','persistent'):
            pred=np.array([r[name] for r in group]);valid=np.abs(true)>1e-7;covered=np.abs(pred)>1e-7
            agreement=np.where(covered,.0+(np.sign(pred)==np.sign(true)),.5)
            rho=float(spearmanr(pred,true).statistic) if np.ptp(pred)>1e-7 and np.ptp(true)>1e-7 else None
            output[name]=dict(comparisons=len(group),nonzero=int(covered.sum()),nonzero_fraction=float(covered.mean()),
                persistent_target_informative=int(valid.sum()),tie_aware_agreement_on_informative_target=float(agreement[valid].mean()) if valid.any() else None,
                non_abstaining_target_coverage=float(covered[valid].mean()) if valid.any() else None,
                sign_accuracy_when_both_nonzero=float((np.sign(pred[covered&valid])==np.sign(true[covered&valid])).mean()) if (covered&valid).any() else None,
                spearman_to_persistent_conditional_target=rho)
        return output
    overall=summary(rr);per_instance={key:summary([r for r in rr if r['instance_id']==key]) for key in ids}
    rng=np.random.default_rng(20261119);paired={}
    for metric in ('nonzero_fraction','tie_aware_agreement_on_informative_target','non_abstaining_target_coverage'):
        delta=np.array([m['fragment'][metric]-m['single'][metric] for m in per_instance.values() if m['single'][metric] is not None and m['fragment'][metric] is not None])
        draws=rng.integers(len(delta),size=(5000,len(delta)));ci=np.quantile(delta[draws].mean(1),[.025,.975])
        paired[metric]=dict(mean_instance_paired_difference=float(delta.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),original_instances=len(delta))
    strata={}
    for field in ('nodes','policy_index','phase_index'):
        for value in sorted({r[field] for r in rr}):strata[f'{field}/{value}']=summary([r for r in rr if r[field]==value])
    replays=[]
    for n in (50,100):
        for policy in range(3):
            row=next(r for r in bank if r['nodes']==n and r['policy_index']==policy);idx=next(i for i,r in enumerate(bank) if r['bank_id']==row['bank_id'])
            p=read(a.prepare/'measurements'/f'state{idx:04d}.json')['programs'][1]
            for mode in ('single','fragment'):
                actual=intervene(row,p['expression'],mode,cfg);old=next(r for r in p['variants'] if r['mode']==mode)
                assert actual['best_tour']==old['best_tour'] and actual['total_work']==old['total_work'] and actual['controlled_action_sha256']==old['controlled_action_sha256']
                replays.append(dict(bank_id=row['bank_id'],mode=mode,exact=True))
    with (a.output/'paired_records.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    same=[r for r in rr if r['same_first_action'] and abs(r['persistent'])>1e-7]
    result=dict(status='passed',at_utc=stamp(),original_instances=len(ids),states=len(bank),rules=12,noncontrol_rules=10,unique_branches_repriced=prices,max_cost_error=maxerror,
        reference_and_positive_rescaling_controls_exact=equivalent,overall=overall,paired_instance_bootstrap=paired,strata=strata,
        same_first_action_informative_persistent_comparisons=len(same),same_first_action_subgroup=summary(same) if same else None,
        independent_replays=replays,illustrative_cases=examples,signal_gate_passed=overall['fragment']['nonzero']>0,
        uncertainty='5000 paired original-instance resamples of per-instance metrics; pooled Spearman descriptive only, conditional persistent outcomes are not whole-run generalization',
        future_direction='A policy-conditioned critic is a different learning target from the old single-edge value; no new predictor trained',
        final_opened=False,wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,
        protocol_sha256=digest(a.prepare/'protocol_frozen.json'),source_sha256=digest(Path(__file__)))
    write(a.output/'summary.json',result);print(json.dumps(dict(overall=overall,paired=paired,states=len(bank),original_instances=len(ids),routes=prices),indent=2),flush=True)


if __name__=='__main__':main()
