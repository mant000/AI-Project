"""Check the scientific reward invariant, including tied and subnormal improvements."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from training_study import group_rewards


def row(terminal, prediction, valid=True):
    return dict(valid=valid, terminal_reward=terminal, value_feedback=dict(states_detail=[
        dict(child_minus_parent_pp=prediction, indicated_preference='child' if prediction > 0 else 'parent')]))


class RewardInvariant(unittest.TestCase):
    cfg = {'reward': {'state_max_absolute': .02}}

    def test_proxy_cannot_reverse_real_reward(self):
        rng = np.random.default_rng(3)
        for _ in range(1000):
            bases = rng.normal(0, .01, 4)
            candidates = [row(float(x), float(y)) for x, y in zip(bases, rng.normal(0, 1, 4))]
            actual = group_rewards(candidates, self.cfg, 'state')['rewards']
            self.assertEqual(list(np.argsort(bases)), list(np.argsort(actual)))
        candidates = [row(0., 100.), row(1e-12, -100.), row(-2., 0., False), row(1., -100.)]
        actual = group_rewards(candidates, self.cfg, 'state')['rewards']
        self.assertGreater(actual[1], actual[0]); self.assertGreater(actual[0], actual[2])

    def test_ties_and_terminal_control(self):
        candidates = [row(0., 1.), row(0., -1.), row(0., 1.), row(0., -1.)]
        terminal = group_rewards(candidates, self.cfg, 'terminal')
        self.assertTrue(terminal['skipped_zero_variance']); self.assertEqual(terminal['rewards'], [0.]*4)
        state = group_rewards(candidates, self.cfg, 'state')
        self.assertFalse(state['skipped_zero_variance'])
        self.assertTrue(all(abs(x) <= .02 for x in state['rewards']))


if __name__ == '__main__': unittest.main()
