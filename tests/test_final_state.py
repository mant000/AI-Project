"""Rounding changes the actual search, while continuous evaluation is bit-identical."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from final_state import create_evaluation_state,delta_budget
from tsp_value.solver import create_state,run,nearest_neighbor
from tsp_value.data import distances,length,validate_tour


class EvaluationDistance(unittest.TestCase):
    def test_distance_and_partial_resume(self):
        xy=np.array([[0.,0.],[1.49,0.],[1.1,.2],[3.1,.7],[.2,3.4],[2.7,3.5]])
        a=create_state(xy,7);b=create_evaluation_state(xy,7)
        run(a,1234);run(b,1234)
        self.assertEqual(a['rng_state'],b['rng_state'])
        for k in ('tour','best_tour','penalties'):np.testing.assert_array_equal(a[k],b[k])
        c=create_evaluation_state(xy,7,convention='tsplib_euc_2d')
        d=np.floor(np.sqrt(((xy[:,None]-xy[None,:])**2).sum(2))+.5)
        np.testing.assert_array_equal(c['distance'],d)
        np.testing.assert_array_equal(c['tour'],nearest_neighbor(d,int(create_state(xy,7)['tour'][0])))
        run(c,1234);validate_tour(c['best_tour'],6)
        self.assertEqual(c['best_length'],length(c['best_tour'],d))
        self.assertEqual(c['delta_evaluations'],1234)
        self.assertEqual([delta_budget(n) for n in (50,100,200,500)],[235000,970000,3940000,24850000])


if __name__=='__main__':unittest.main()
