"""Freeze all methods and evaluation inputs before parsing final reference costs."""
import argparse
from pathlib import Path
from stage3_common import CODE,read,write,digest,stamp
from solver_dev_benchmark import collect_programs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=CODE/'results/final/freeze-v1');a=ap.parse_args()
    required=['results/main/verification-v1/summary.json','results/eoh-control-v1/verification.json',
        'results/solver-controls/dev-v1/verification.json','results/baselines/fixture-v2/verification.json',
        'results/solver-controls/fixture-v1/verification.json','results/final/fixture-v1/verification.json']
    for p in required:assert read(CODE/p)['status']=='passed',p
    cfgpath=CODE/'configs/final_evaluation_v1.json';cfg=read(cfgpath)
    assert cfg['final_evaluation'] and cfg['instances']==1194
    programs,aliases=collect_programs(read(CODE/'configs/solver_controls_v1.json'),read(CODE/'configs/dev_benchmark_v1.json'))
    dev=read(CODE/'results/solver-controls/dev-v1/protocol_frozen.json')
    assert sorted(programs)==dev['program_hashes'] and aliases==dev['aliases']
    # Include the entire local solver/runtime dependency surface; old executed
    # files retain their bytes, new adapters have their own explicit provenance.
    sources=[p for folder in ('src/tsp_value','scripts') for p in (CODE/folder).glob('*.py')]
    configs=['configs/final_evaluation_v1.json','configs/final_identity_policy.json','configs/tsplib_catalog.json',
        'configs/external_baselines_v1.json','configs/dev_benchmark_v1.json','configs/solver_controls_v1.json','configs/joshi_received_files.json']
    metadata=['splits/manifest.csv','results/main/discovery-v1/frozen_programs.json',
        'results/eoh-control-v1/frozen_programs.json','results/baselines/prepare-v1/completed.json',
        'results/baselines/prepare-v1/lehd_completed.json',*required]
    metadata.extend(f'results/stage2/training-v1/transformer-seed{s}_summary.json' for s in (101,202,303))
    record=dict(status='frozen',at_utc=stamp(),protocol=cfg,programs=programs,aliases=aliases,
        files_sha256={p.relative_to(CODE).as_posix():digest(p) for p in [*sources,*[CODE/x for x in configs+metadata]]},
        tsplib_reference_file_sha256=digest(a.root/'data/raw/tsplib_best_known.json'),
        final_reference_costs_or_solver_results_read_before_freeze=False,
        all_optimization_and_discovery_seeds_retained=True,method_changes_after_final_prohibited=True)
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/'protocol_frozen.json'
    if path.exists():
        old=read(path)
        for key in record:
            if key!='at_utc':assert old[key]==record[key],key
    else:write(path,record)
    print('frozen',len(programs),'unique programs,',len(aliases),'logical aliases',digest(path),flush=True)


if __name__=='__main__':main()
