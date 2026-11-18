"""An improvement returned after a deadline cannot leak backwards into its quality snapshot."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from timed_engine import run_deadlines
from tsp_value.solver import create_state
from tsp_value.data import length


class DeadlineQuality(unittest.TestCase):
    def test_late_improvement_and_early_termination(self):
        xy=[[0.,0.],[.9,.1],[.8,.8],[.3,.9],[.15,.35],[.6,.4]]
        s=create_state(xy);s['best_tour']=np.arange(6);s['best_length']=length(s['best_tour'],s['distance'])
        original=s['best_tour'].copy();optimal=min((np.array((0,)+p) for p in itertools.permutations(range(1,6))),key=lambda t:length(t,s['distance']))
        now=[0.];calls=[0]
        def operation():
            calls[0]+=1;now[0]+=1.2;s['best_tour']=optimal;s['best_length']=length(optimal,s['distance'])
            if calls[0]==2:s['terminated']=True
        result=run_deadlines(s,[1.,5.],0.,clock=lambda:now[0],operation=operation)
        self.assertEqual(result['checkpoints'][0]['best_tour'],original.tolist())
        self.assertEqual(result['checkpoints'][1]['best_tour'],optimal.tolist())
        self.assertTrue(result['terminated_early']);self.assertAlmostEqual(result['checkpoints'][0]['atomic_overshoot_seconds'],.2)
        self.assertTrue(all(r['available_by_deadline'] for r in result['checkpoints']))


if __name__=='__main__':unittest.main()
