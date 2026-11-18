"""Verify all bank routes/values and feedback boundaries before any new 7B output."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from stage3_common import CODE,read,write,stamp,digest
from core_discover import CoreStudy
from tsp_value.programs import canonical_program,INITIAL
from tsp_value.data import distances,length,validate_tour


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--prepare',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    for name in ('decisions','programs','evaluations','candidates','generations'):(a.output/name).mkdir()
    study=CoreStudy(a.root,a.output,a.prepare);checked=0;error=0.
    for state in study.bank:
        d=distances(state['state']['coords']);exact=study.true[state['bank_id']]
        assert len(state['edges'])==state['nodes']==len(exact['records'])
        for edge,record,value in zip(state['edges'],exact['records'],exact['true_scores_pp']):
            assert edge==record['edge'];validate_tour(record['best_tour'],state['nodes'])
            actual=length(np.asarray(record['best_tour']),d);error=max(error,abs(actual-record['best_length']))
            assert abs(actual-record['best_length'])<1e-10
            expected=100*(exact['records'][-1]['best_length']-actual)/state['state']['best_length']
            assert abs(value-expected)<1e-10
            assert record['delta_evaluations']==study.cfg['horizon_by_nodes'][str(state['nodes'])]
            checked+=1
        assert exact['true_scores_pp'][-1]==0
        assert np.all(np.asarray(state['model_scores_pp'])[:,-1]==0)
    initial=canonical_program(INITIAL)
    fake_eval=dict(mean_gap_percent=0,by_nodes_mean_gap_percent={'50':0,'100':0})
    for arm in study.cfg['arms']:
        ctx=dict(parent=initial,parent_eval=fake_eval,last=None,history=[])
        messages=study.prompt_for(ctx,arm);text=messages[1]['content']
        assert ('ADDITIONAL STATE VALUE FEEDBACK:' in text)==(arm!='A')
        if arm=='B':assert 'measured single-action branch rollouts' not in text
        if arm=='C':assert 'Transformer ensemble predictions' not in text
        if arm!='A':
            feedback=study.feedback(initial,initial,arm)
            assert all(r['same_action'] and r['child_minus_parent_pp']==0 for r in feedback['states_detail'])
    result=dict(status='passed',at_utc=stamp(),measured_routes_recomputed=checked,max_route_cost_error=error,
        all_edges_and_reference_order_valid=True,same_action_delta_zero=True,feedback_source_boundaries_checked=True,
        protocol_sha256=digest(CODE/'configs/core_abc_v1.json'),bank_sha256=study.bankhash,final_opened=False,
        wall_seconds=time.perf_counter()-start)
    write(a.output/'summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
