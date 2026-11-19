"""Expose decisions without changing native run semantics; matched interventions and budgets."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from state_feedback import controlled_run,collect_states,intervene,restore
from tsp.solver import create_state,run,clone
from tsp.programs import compile_rule


class FragmentSemantics(unittest.TestCase):
    def test_trace_and_reference_identity(self):
        xy=np.random.default_rng(5491).uniform(size=(14,2));a=create_state(xy,17);b=clone(a);rule=compile_rule('d/(1+0.25*p)')
        run(a,12345,rule);controlled_run(b,12345,rule)
        for key in ('tour','best_tour','penalties'):np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(a['rng_state'],b['rng_state']);self.assertEqual(a['cursor'],b['cursor'])
        cfg=dict(delta_budget_by_nodes={'14':18480},trajectory_seed=17,trajectory_expressions=['d/(1+p)'],state_work_fractions=[.1,.3,.5,.7],continuation_seed=71,fragment_fraction_of_remaining=.25)
        rows=list(collect_states(dict(coords=xy.tolist(),instance_id='invented',nodes=14),cfg));self.assertTrue(rows)
        row=rows[0];original=copy.deepcopy(row)
        rr=[intervene(row,'d/(1+p)',mode,cfg) for mode in ('single','fragment','persistent')]
        self.assertEqual(rr[0]['best_tour'],rr[1]['best_tour']);self.assertEqual(rr[1]['best_tour'],rr[2]['best_tour'])
        self.assertTrue(all(r['total_work']==18480 for r in rr));self.assertEqual(row,original)


if __name__=='__main__':unittest.main()
