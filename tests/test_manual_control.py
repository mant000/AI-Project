"""Preserve the preregistered geometric penalty gate under the literal-power interface."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tsp_value.programs import canonical_program,compile_rule
from tsp_value.solver import create_state


class ManualGate(unittest.TestCase):
    def test_legal_implementation_matches_frozen_intent(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/dev_benchmark_v1.json').read_text())
        p=canonical_program(cfg['manual_implementations']['stagnation_gate']);rule=compile_rule(p['expression'])
        state=create_state([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        d=np.array([.13,.4,.7,1.2]);penalties=np.array([0,1,12,1024])
        for counter in (0,1,3,20,150,1000):
            state['stagnation_sweeps']=counter
            expected=d/(1+penalties)**(1+np.clip(np.log1p(counter)/5,0,1))
            np.testing.assert_allclose(rule(d,penalties,state),expected,rtol=1e-12,atol=1e-14)


if __name__=='__main__':unittest.main()
