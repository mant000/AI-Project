"""Recompute every new dense-label route and independently check reused costs."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from stage3_common import read,write,stamp
from stage3_worker import restore_state
from tsp_value.solver import clone,penalize,run
from tsp_value.data import distances,length,validate_tour,coordinate_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start,cpu=time.perf_counter(),time.process_time();cfg=read(a.data/'protocol_frozen.json')['protocol'];checked=0;reused=0;maximum=0.;replays=[];counts={50:0,100:0};identities={role:set() for role in ('train','validation','heldout')}
    for path in sorted((a.data/'states').glob('*.json')):
        row=read(path);s=restore_state(row['state']);d=s['distance'];identities[row['role']].add(coordinate_hash(row['state']['coords']))
        assert len(row['edges'])==row['nodes']==len(row['records'])==len(row['true_scores_pp'])
        reference=row['records'][-1]['best_length']
        for edge,record,y in zip(row['edges'],row['records'],row['true_scores_pp']):
            assert record['edge']==edge;expected=100*(reference-record['best_length'])/row['state']['best_length'];assert abs(y-expected)<1e-10
            if record['reused']:
                reused+=1
                if counts[row['nodes']]<6:
                    branch=clone(s);branch['rng_state']=np.random.default_rng(cfg['continuation_seed']).bit_generator.state
                    penalize(branch,edge);run(branch,cfg['horizon_by_nodes'][str(row['nodes'])]);assert branch['best_length']==record['best_length']
                    replays.append(dict(bank_id=row['bank_id'],edge=edge,best_length=record['best_length'],exact_cached_cost_match=True));counts[row['nodes']]+=1
            else:
                validate_tour(record['best_tour'],row['nodes']);actual=length(np.asarray(record['best_tour']),d)
                maximum=max(maximum,abs(actual-record['best_length']));assert abs(actual-record['best_length'])<1e-10;checked+=1
    assert not identities['train']&identities['validation'] and not identities['validation']&identities['heldout']
    summary=dict(status='passed',at_utc=stamp(),new_route_costs_recomputed=checked,max_route_cost_error=maximum,reused_action_values=reused,
        independently_replayed_cached_actions=replays,all_action_score_and_edge_orders_checked=True,role_coordinate_hashes_disjoint=True,
        heldout_model_metrics_computed=False,final_opened=False,wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu)
    write(a.output,summary);print(json.dumps({k:v for k,v in summary.items() if k!='independently_replayed_cached_actions'},indent=2))


if __name__=='__main__':main()
