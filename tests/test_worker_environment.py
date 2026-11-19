"""Solver isolation must not depend on the caller's BLAS thread settings."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_io import worker


class WorkerEnvironment(unittest.TestCase):
    def test_large_parent_thread_pool_does_not_break_limited_worker(self):
        item = dict(instance_id='thread-fixture', project_use='fixture', nodes=6,
                    coords=[[0., 0.], [.9, .1], [.8, .8], [.3, .9], [.15, .35], [.6, .4]],
                    reference_length=6.)
        request = dict(mode='endpoint', expression='d/(1+p)', method='gls',
                       items=[item], seeds=[0], budget_by_nodes={'6': 300})
        cfg = dict(threads=1, worker_address_space_gib=2,
                   worker_cpu_limit_seconds=10, worker_wall_timeout_seconds=20)
        routes = []
        for threads in ('1', '96'):
            with patch.dict(os.environ, {key: threads for key in
                            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}):
                result = worker(request, cfg)
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['results'][0]['delta_evaluations'], 300)
            routes.append(result['results'][0]['best_tour'])
        self.assertEqual(routes[0], routes[1])


if __name__ == '__main__':
    unittest.main()
