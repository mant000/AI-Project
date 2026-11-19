"""Reprice reference/timing tours and assemble the measured experiment cost ledger.

Run after analyze_results.py and replay_solvers.py have succeeded.
The output records the actual code revision and Python environment.
"""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import platform
import sys
import time
import numpy as np
from analyze_results import CODE,read,write,digest
from tsp.data import distances,length,validate_tour


def git(*args):return subprocess.check_output(['git','-C',str(CODE),*args],text=True).strip()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis-proof',type=Path,required=True);ap.add_argument('--replay-proof',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    analysis=read(a.analysis_proof);replay=read(a.replay_proof);assert analysis['status']==replay['status']=='passed'
    base=CODE/'results/discovery';prepare=CODE/'results/mechanism/bank'
    items={r['instance_id']:r for r in read(prepare/'program_dev_eval_instances.json')}
    refs={};reference_checks=0
    for file in (base/'references').glob('*.json'):
        obj=read(file)
        if 'results' not in obj:continue
        for r in obj['results']:
            item=items[r['instance_id']];validate_tour(r['best_tour'],item['nodes']);actual=length(np.array(r['best_tour']),distances(item['coords']))
            assert abs(actual-r['original_length'])<1e-10
            assert abs(100*(actual/item['reference_length']-1)-r['gap_percent'])<1e-10;reference_checks+=1
        assert len(obj['results'])==320
        assert abs(obj['mean_gap_percent']-np.mean([r['gap_percent'] for r in obj['results']]))<1e-12
        refs[obj['label']]=dict(mean_gap_percent=obj['mean_gap_percent'],wall_seconds=obj['wall_seconds'],routes=len(obj['results']))
    timing=read(base/'timing/records.json')
    for r in timing:
        item=items[r['instance_id']];validate_tour(r['best_tour'],item['nodes'])
        assert abs(length(np.array(r['best_tour']),distances(item['coords']))-r['original_length'])<1e-10
    ledger={};physical_branches=0
    for phase in ('pilot','full'):
        obj=read(prepare/f'{phase}_completed.json');physical_branches+=obj['new_branches']
        ledger['mechanism_'+phase]=dict(wall_seconds=obj['wall_seconds'],parent_cpu_seconds=obj['parent_cpu_seconds'],waited_child_cpu_seconds=obj['waited_child_cpu_seconds'],new_physical_branches=obj['new_branches'])
    assert physical_branches==56832
    for label,study in [('strict',CODE/'results/interface_check/run'),('expression',base/'run')]:
        calls=[read(p) for p in (study/'generations').glob('*.json')]
        attempts=[json.loads(x) for x in (study/'attempt_costs.jsonl').read_text().splitlines()]
        rollouts=[read(p) for p in (study/'rollouts').glob('*.json')]
        endpoint=[read(p) for p in (study/'evaluations').glob('*.json')]
        samples=[json.loads(x) for x in (study/'gpu_samples.jsonl').read_text().splitlines()]
        power=[]
        for sample in samples:
            values=sample['gpu'].split(',')
            try:power.append([sample['seconds'],float(values[-3]),float(values[-2]),float(values[-1])])
            except ValueError:pass
        power=np.array(power)
        ledger[label]=dict(wall_seconds=sum(x['wall_seconds'] for x in attempts),parent_cpu_seconds=sum(x['parent_cpu_seconds'] for x in attempts),
            waited_child_cpu_seconds=sum(x['waited_child_cpu_seconds'] for x in attempts),generation_synchronized_wall_seconds=sum(x['generation_wall_seconds'] for x in calls),
            input_tokens=sum(x['input_tokens'] for x in calls),output_tokens=sum(x['output_tokens'] for x in calls),outputs=len(calls),
            unique_endpoint_solves=sum(len(x['results']) for x in endpoint),physical_feedback_branches=sum(x['physical_branches'] for x in rollouts),
            cpu_feedback_worker_seconds=sum(x['worker_cpu_seconds'] for x in rollouts),
            cuda_peak_allocated_bytes=max(x['cuda_peak_allocated_bytes'] for x in attempts),cuda_peak_reserved_bytes=max(x['cuda_peak_reserved_bytes'] for x in attempts),
            sampled_device_peak_memory_mib=float(power[:,2].max()) if len(power) else None,
            sampled_device_utilization_mean_percent=float(power[:,1].mean()) if len(power) else None,
            sampled_gpu_energy_wh=float(np.trapz(power[:,3],power[:,0])/3600) if len(power)>1 else None,
            energy_scope='3-second device-level sampling, approximate and includes idle/overhead; not kernel-only energy')
    ledger['references']=read(base/'references/completed.json');ledger['references']['methods']=refs
    ledger['isolated_timing']=read(base/'timing/completed.json')
    ledger['cpu_replay']=replay;ledger['record_audit']=analysis
    write(a.output/'cost_ledger.json',dict(operations=ledger,accounting='Pilot/full new branches summed once. Whole process waited-child CPU is authoritative. Inner concurrent subprocess child deltas overlap and are never added to it. CPU feedback subtotals are included in process totals.',
          hardware='RTX3090 24GB; Xeon6330 14-vCPU quota; 60GiB RAM',billing='existing monthly rental; no fabricated per-run monetary price; device/host seconds and approximate sampled GPU energy reported',
          historical_costs='Earlier stages are retained at tag milestone-2026-11-19; they are separate protocols, excluded from this study subtotal.'))
    environment=dict(python=sys.version,prefix=sys.prefix,packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','matplotlib']},
                     torch_installed=importlib.util.find_spec('torch') is not None)
    write(a.output/'environment.json',environment)
    proof=dict(status='passed',source_commit=git('rev-parse','HEAD'),origin=git('remote','get-url','origin'),
          verification_environment=dict(platform=platform.platform(),python_prefix=sys.prefix),analysis=analysis,replay=replay,
          reference_routes_repriced=reference_checks,timing_routes_repriced=len(timing),
          wall_seconds=time.perf_counter()-start,source_sha256=digest(Path(__file__)),
          not_claimed=['new physical machine','full stochastic GPU discovery regeneration','retraining all historical models'])
    write(a.output/'release_verification.json',proof);print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
