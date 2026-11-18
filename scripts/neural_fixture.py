"""CPU functional check of the actual fixed teacher and exact conventional fallback."""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
import torch
from stage3_common import CODE,write,digest,stamp
from tsp_value.solver import create_state,clone,run
from tsp_value.neural_direct import NeuralDirect
from tsp_value.data import validate_tour,length


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1)
    out=CODE/'results/solver-controls/fixture-v1';out.mkdir(parents=True,exist_ok=False)
    xy=np.asarray([[0.,0.],[.9,.1],[.8,.8],[.3,.9],[.15,.35],[.6,.4]])
    initial=create_state(xy,17);normal=clone(initial);fallback=clone(initial);actual=clone(initial)
    teacher=NeuralDirect(a.root,'cpu',threshold=1e9)
    run(normal,3000);run(fallback,3000,rule=teacher)
    for key in ('tour','best_tour','penalties'):assert np.array_equal(normal[key],fallback[key])
    for key in ('best_length','current_length','delta_evaluations','moves','penalty_events'):assert normal[key]==fallback[key]
    assert teacher.profile['calls']>0 and teacher.profile['changed_actions']==0
    fallback_profile=dict(teacher.profile);teacher.threshold=.001;teacher.reset_profile();run(actual,3000,rule=teacher)
    validate_tour(actual['best_tour'],len(xy));assert abs(length(actual['best_tour'],actual['distance'])-actual['best_length'])<1e-10
    result=dict(status='passed',at_utc=stamp(),fixture='hand-written six cities, not experimental data',
        exact_conventional_fallback=True,loaded_actual_predeclared_checkpoints=teacher.checkpoints,
        fallback_profile=fallback_profile,ordinary_threshold_profile=teacher.profile,
        neural_route_valid=True,future_labels_used=False,gpu_used=False,final_opened=False,
        source_sha256=digest(CODE/'src/tsp_value/neural_direct.py'))
    write(out/'verification.json',result);print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
