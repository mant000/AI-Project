"""Shared isolated real endpoint and frozen action-value feedback, without future labels."""
import json
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from stage3_common import CODE, read, write, append, digest, worker, evaluate
from stage3_discover import Study, SYSTEM, TASK
from core_discover import CoreStudy
from core_mechanism import select_slice


def monitor_gpu(out, attempt, start):
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                    '--format=csv,noheader,nounits'], text=True).strip()
                append(out/'gpu_samples.jsonl', dict(attempt=attempt, seconds=time.perf_counter()-start, gpu=value))
            except Exception: pass
            stop.wait(3)
    thread = threading.Thread(target=sample, daemon=True); thread.start()
    return stop, thread


class TrainingStudy(Study):
    feedback = CoreStudy.feedback
    prompt_for = CoreStudy.prompt_for

    def __init__(self, root, out, prepare):
        self.root, self.out, self.prepare = root, out, prepare
        self.cfg = read(prepare / 'protocol_frozen.json')['protocol']
        self.bankpath = prepare / 'bank_observables.jsonl'; self.bankhash = digest(self.bankpath)
        assert self.bankhash == read(prepare / 'completed.json')['bank_sha256']
        self.bank = [json.loads(x) for x in self.bankpath.read_text().splitlines()]
        rows = [json.loads(x) for x in (root / 'data/prepared/stage1/project_instances.jsonl').read_text().splitlines()]
        names = read(prepare / 'selection.json').keys()
        self.pools = {p: select_slice(rows, p, self.cfg[p+'_offset_per_size'], self.cfg[p+'_instances_per_size']) for p in names}
        self.programs = {}; self.cache = {}; self.decisions = {}; self.candidates = []; self.calls = []
        for name in ('programs', 'evaluations', 'decisions', 'generations', 'candidates'):
            (out / name).mkdir(exist_ok=True)

    def decision(self, program):
        key = program['program_sha256']; path = self.out / 'decisions' / f'{key}.json'
        if key not in self.decisions:
            if path.exists():
                result = read(path); assert result['bank_sha256'] == self.bankhash
            else:
                result = worker(dict(mode='decisions', expression=program['expression'], states=self.bank), self.cfg)
                result.update(program_sha256=key, bank_sha256=self.bankhash); write(path, result)
            if not result['ok']: raise ValueError(result.get('error', 'Decision worker failed'))
            self.decisions[key] = result
        return self.decisions[key]['results']

    def fixed_parent_prompt(self, parent, ev):
        endpoint = dict(parent_code=parent['source'], parent_mean_discovery_gap_percent=ev['mean_gap_percent'],
            parent_gap_by_size=ev['by_nodes_mean_gap_percent'], original_instances=len(self.pools['discovery']),
            solver_seeds=self.cfg['endpoint_solver_seeds'], work_budget_by_nodes=self.cfg['delta_budget_by_nodes'])
        return [dict(role='system', content=SYSTEM), dict(role='user', content=TASK+'\nREAL PARENT ENDPOINT FEEDBACK:\n'+json.dumps(endpoint, indent=2))]


def group_rewards(candidates, cfg, arm):
    """Bound auxiliary feedback so it cannot invert any distinct terminal rewards."""
    reward_cfg = cfg['reward']; bases = [r['terminal_reward'] for r in candidates if r['valid']]
    unique = sorted(set(bases)); separation = min(np.diff(unique)) if len(unique) > 1 else None
    bound = reward_cfg['state_max_absolute'] if separation is None else min(reward_cfg['state_max_absolute'], float(separation)/4)
    rewards = []
    for row in candidates:
        aux = 0.
        if row['valid'] and arm == 'state':
            fb = row['value_feedback']['states_detail']
            qualified = [r['child_minus_parent_pp'] if r['indicated_preference'] != 'none' else 0. for r in fb]
            aux = float(bound * np.tanh(float(np.mean(qualified))/.01))
        rewards.append(float(row['terminal_reward'] + aux))
    for i, x in enumerate(candidates):
        for j, y in enumerate(candidates):
            if x['terminal_reward'] > y['terminal_reward']: assert rewards[i] > rewards[j]
    arr = np.asarray(rewards); std = float(arr.std())
    advantage = ((arr-arr.mean())/(std+1e-8)).tolist() if std > 1e-8 else [0.]*len(arr)
    return dict(rewards=rewards, advantages=advantage, reward_std=std, auxiliary_bound=bound,
                minimum_distinct_valid_terminal_separation=separation, skipped_zero_variance=std <= 1e-8,
                distinct_terminal_order_preserved=True)
