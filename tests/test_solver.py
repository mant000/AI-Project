"""Checks that protect route quality, budget fairness, and label validity."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from tsp.data import distances, length, parse_author, validate_tour
from tsp.solver import (create_state, clone, run, save_state, load_state,
                              penalize, select_edge, paired_continuation, paired_action_set)

# Hand-written asymmetric six-city fixture, not an experimental instance.
COORDS = np.array([[0.,0.], [.9,.1], [.8,.8], [.3,.9], [.15,.35], [.6,.4]])


def as_json(state):
    return json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in state.items()}, sort_keys=True)


class SolverCorrectness(unittest.TestCase):
    def test_reference_parse_exact_enumeration_and_rounding(self):
        d = distances(COORDS)
        tours = [np.array((0,)+p) for p in itertools.permutations(range(1, len(d)))]
        optimal = min(tours, key=lambda t: length(t, d))
        raw = (' '.join(map(str, COORDS.ravel()))+' output '+' '.join(map(str, list(optimal+1)+[1]))+'\n').encode()
        x, reference = parse_author(raw, len(d))
        self.assertAlmostEqual(length(reference, distances(x)), min(length(t, d) for t in tours))
        self.assertEqual(distances([[0,0],[1.5,0]], 'tsplib_euc_2d')[0,1], 2.)
        with self.assertRaises(ValueError):
            parse_author(raw.replace(b'output', b'output 1'), len(d))

    def test_all_raw_and_augmented_deltas(self):
        s = create_state(COORDS)
        p = np.arange(36).reshape(6,6)
        p = p+p.T
        lam = .27
        augmented = s['distance']+lam*p
        for d in (s['distance'], augmented):
            for i,j in zip(s['pair_i'], s['pair_j']):
                t = s['tour']
                a,b,c,e = t[i],t[i+1],t[j],t[(j+1)%len(t)]
                delta = d[a,c]+d[b,e]-d[a,b]-d[c,e]
                changed = t.copy()
                changed[i+1:j+1] = changed[i+1:j+1][::-1]
                self.assertAlmostEqual(delta, length(changed,d)-length(t,d), places=12)

    def test_every_step_tours_and_costs(self):
        s = create_state(COORDS, seed=17)
        for _ in range(1000):
            run(s, 1)
            validate_tour(s['tour'], len(COORDS))
            validate_tour(s['best_tour'], len(COORDS))
            self.assertAlmostEqual(s['current_length'], length(s['tour'],s['distance']))
            self.assertAlmostEqual(s['best_length'], length(s['best_tour'],s['distance']))
        self.assertEqual(s['delta_evaluations'], 1000)
        self.assertTrue(np.array_equal(s['penalties'], s['penalties'].T))

    def test_checkpoint_arbitrary_partial_sweeps(self):
        full = create_state(COORDS, 11)
        run(full, 1201)
        for cut in (1, 8, 9, 10, 53, 799, 1200):
            partial = create_state(COORDS, 11)
            run(partial, cut)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'state.json'
                save_state(partial,path)
                restored = load_state(path)
            run(restored,1201-cut)
            self.assertEqual(as_json(full), as_json(restored))

    def test_same_action_zero_pair_and_copy_isolation(self):
        s = create_state(COORDS, 3)
        run(s, 10000, stop_at_decision=True)
        self.assertTrue(s['needs_penalty'])
        before = as_json(s)
        chooser = clone(s)
        edge = select_edge(chooser)
        rows = paired_continuation(s, edge, edge, [4,5,6], 1001)
        self.assertTrue(all(r['relative_benefit']==0 for r in rows))
        self.assertEqual(before, as_json(s))
        changed = clone(s)
        penalize(changed, edge)
        changed['distance'][0,1] = -9
        self.assertEqual(before, as_json(s))

    def test_two_actions_equal_work_and_replay(self):
        s = create_state(COORDS, 3)
        run(s,10000,stop_at_decision=True)
        t = s['tour']
        e0,e1 = tuple(t[:2]),tuple(t[1:3])
        first = paired_continuation(s,e0,e1,[9,10],1013)
        self.assertEqual(first, paired_continuation(s,e0,e1,[9,10],1013))
        self.assertTrue(all(r['horizon_delta_evaluations']==1013 for r in first))

    def test_rng_checkpoint_with_utility_ties(self):
        s = create_state([[0,0],[1,0],[1,1],[0,1]],4)
        run(s,2,stop_at_decision=True)
        self.assertTrue(s['needs_penalty'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'tie.json'
            save_state(s,path)
            restored = load_state(path)
        run(s,131)
        run(restored,131)
        self.assertEqual(as_json(s),as_json(restored))
        self.assertGreater(s['penalty_events'], 1)

    def test_two_opt_local_optimum_and_gls_equivalence_before_penalty(self):
        plain, guided = create_state(COORDS,2,'two_opt'), create_state(COORDS,2)
        run(plain,10000)
        run(guided,10000,stop_at_decision=True)
        self.assertTrue(plain['terminated'])
        self.assertEqual(plain['delta_evaluations'], guided['delta_evaluations'])
        self.assertTrue(np.array_equal(plain['tour'], guided['tour']))
        run(guided,500)
        self.assertLessEqual(guided['best_length'],plain['best_length']+1e-12)

    def test_rule_and_action_errors(self):
        s = create_state(COORDS)
        run(s,10000,stop_at_decision=True)
        with self.assertRaises(ValueError):
            select_edge(s,lambda d,p,s: np.full(len(d),np.nan))
        with self.assertRaises(ValueError):
            penalize(s,(0,0))
        with self.assertRaises(ValueError):
            run(s,-1)

    def test_multi_candidate_reference_is_executed_once_per_seed(self):
        s=create_state(COORDS)
        run(s,10000,stop_at_decision=True)
        t=s['tour']
        with patch('tsp.solver.run',wraps=run) as measured:
            result=paired_action_set(s,tuple(t[:2]),[tuple(t[1:3]),tuple(t[2:4])],[41,42],103)
        self.assertEqual(result['branches_executed'],6)
        self.assertEqual(result['reference_branches_executed'],2)
        self.assertEqual(measured.call_count,6)
        self.assertEqual(len(result['records']),4)


if __name__ == '__main__':
    unittest.main()
