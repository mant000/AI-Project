"""Exercise the final worker/rounded distance adapter on invented geometry only."""
import argparse
import copy
from pathlib import Path
import numpy as np
from stage3_common import CODE,read,write,digest,stamp
from final_evaluate import isolated
from final_prepare import parse_invit_reference
from tsp_value.programs import INITIAL,canonical_program
from tsp_value.data import distances,length,validate_tour


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=CODE/'results/final/fixture-v1');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);cfg=copy.deepcopy(read(CODE/'configs/final_evaluation_v1.json'))
    cfg.update(solver_seeds=[0,1,2],deadline_seconds=[.02,.05],worker_cpu_limit_seconds=30,worker_wall_timeout_seconds=40)
    xy=np.array([[0.,0.],[.9,.1],[.8,.8],[.3,.9],[.15,.35],[.6,.4]])
    items=[]
    for index,mode in enumerate(('continuous_euclidean','tsplib_euc_2d')):
        points=xy if index==0 else xy*100+np.array([1000.,2500.])
        items.append(dict(instance_id=f'invented-{index}',nodes=6,group='invented',coords=points.tolist(),
            distance_convention=mode,reference_length=length(np.arange(6),distances(points,mode))))
    records={};std=canonical_program(INITIAL)['expression']
    for kind in ('exact','timing'):
        for method in ('gls','native_gls','two_opt','nn'):
            result=isolated(dict(kind=kind,method=method,expression=std,items=items,config=cfg));assert result['ok']
            for r in result['results']:
                item=next(i for i in items if i['instance_id']==r['instance_id']);assert not r['solver_failed']
                if kind=='timing':assert all(q['available_by_deadline'] for q in r['checkpoints'])
                for q in r['checkpoints'] if kind=='timing' else [r]:
                    t=np.asarray(q['best_tour']);validate_tour(t,6)
                    assert abs(length(t,distances(item['coords'],item['distance_convention']))-q['original_length'])<1e-10
            records[kind+'/'+method]=result
    assert all(x['best_tour']==y['best_tour'] and x['delta_evaluations']==y['delta_evaluations']
        for x,y in zip(records['exact/gls']['results'],records['exact/native_gls']['results']))
    for raw in (b'0,1,2,3,4,5 6.0 1.0',b'1,2,3,4,5,6,1 6.0 1.0'):
        tour,_,_=parse_invit_reference(raw,6);assert tour.tolist()==list(range(6))
    write(a.output/'results.json',records)
    write(a.output/'verification.json',dict(status='passed',at_utc=stamp(),routes_and_time_snapshots_repriced=True,
        continuous_and_rounded_search_tested=True,compiled_and_native_exact_equal=True,reference_parser_formats_tested=True,
        final_opened=False,invented_geometry_only=True,source_sha256={p.relative_to(CODE).as_posix():digest(p) for p in
            [Path(__file__),CODE/'scripts/final_state.py',CODE/'scripts/final_worker.py',CODE/'scripts/final_evaluate.py',CODE/'scripts/final_prepare.py']}))
    print('final worker fixture passed',flush=True)


if __name__=='__main__':main()
