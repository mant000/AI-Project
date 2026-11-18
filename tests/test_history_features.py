"""History removal must be invariant to removed memory and leave solver state untouched."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from tsp_value.solver import create_state, clone
from tsp_value.history_features import history_features


class HistoryEncoding(unittest.TestCase):
    def test_removed_memory_and_observable_current_configuration(self):
        xy = [[0.,0.],[.9,.1],[.8,.8],[.3,.9],[.15,.35],[.6,.4]]
        a = create_state(xy); b = clone(a); edges = np.column_stack((a['tour'],np.roll(a['tour'],-1)))
        b['penalties'][:] = 9; b['penalty_events'] = 30; b['delta_evaluations'] = 500
        b['stagnation_sweeps'] = 13; b['best_length'] *= .9; b['best_tour'] = np.roll(b['tour'][::-1], 2)
        fa, fb = history_features(a, edges, 'no_history'), history_features(b, edges, 'no_history')
        for key in fa: np.testing.assert_array_equal(fa[key], fb[key])
        self.assertEqual(b['penalty_events'], 30); self.assertTrue((b['penalties']==9).all())
        fulla, fullb = history_features(a, edges, 'full'), history_features(b, edges, 'full')
        self.assertFalse(np.array_equal(fulla['node'], fullb['node']))
        self.assertFalse(np.array_equal(fulla['glob'], fullb['glob']))
        self.assertTrue((fb['pair'][:,:,2]==0).all())
        np.testing.assert_array_equal(fb['pair'][:,:,1], fb['pair'][:,:,3])


if __name__ == '__main__': unittest.main()
