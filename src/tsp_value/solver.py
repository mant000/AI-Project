"""Budget-exact, resumable best-improvement 2-opt and edge-penalty GLS.

A work unit is one valid unordered nonadjacent 2-opt pair delta evaluation.
Incomplete sweeps keep their cursor and best move. A move is committed only
after a full sweep; truncating a budget therefore never selects a move from
an unevaluated suffix. All returned quality uses the original distance matrix.
"""
import copy
import json
from pathlib import Path
import numpy as np
from .data import distances, length, validate_tour

EPS = 1e-12
RULE_ID = 'distance_over_one_plus_penalty_single_max_random_tie_v1'


def nearest_neighbor(d, start):
    """Deterministic city-index tie breaking, with a seeded start city."""
    n = len(d)
    route = [int(start)]
    seen = np.zeros(n, dtype=bool)
    seen[start] = True
    while len(route) < n:
        scores = np.where(seen, np.inf, d[route[-1]])
        nxt = int(np.argmin(scores))
        route.append(nxt)
        seen[nxt] = True
    return np.asarray(route, dtype=np.int64)


def create_state(coords, seed=0, mode='gls', alpha=0.1):
    """Create NN state; GLS lambda is frozen at the first 2-opt local optimum."""
    if mode not in ('gls', 'two_opt') or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Invalid mode/alpha')
    d = distances(coords)
    n = len(d)
    if n < 4:
        raise ValueError('Search requires at least four cities')
    rng = np.random.default_rng(seed)
    tour = nearest_neighbor(d, rng.integers(n))
    pairs = [(i, j) for i in range(n) for j in range(i+2, n) if not (i == 0 and j == n-1)]
    pi, pj = np.asarray(pairs, dtype=np.int64).T
    return dict(schema_version=1, mode=mode, alpha=float(alpha), coords=np.asarray(coords, dtype=np.float64).copy(),
                distance=d, pair_i=pi, pair_j=pj, tour=tour, best_tour=tour.copy(),
                current_length=length(tour, d), best_length=length(tour, d),
                penalties=np.zeros((n, n), dtype=np.int64), lambda_=0.0,
                phase='initial_descent', needs_penalty=False, terminated=False,
                cursor=0, scan_best_delta=0.0, scan_best_pair=-1,
                delta_evaluations=0, moves=0, sweeps=0, penalty_events=0,
                stagnation_sweeps=0, rng_state=copy.deepcopy(rng.bit_generator.state),
                rule_id=RULE_ID, last_penalized_edge=None)


def clone(state):
    """Independent arrays, caches, partial scan, and RNG state."""
    return copy.deepcopy(state)


def select_edge(state, rule=None):
    """Rule returns one finite priority per current route edge; ties use RNG.

The conventional rule penalizes ONE uniformly chosen maximum-utility edge,
an explicit single-edge variant of GLS (rather than penalizing all ties).
"""
    t = state['tour']
    nxt = np.roll(t, -1)
    dist = state['distance'][t, nxt]
    p = state['penalties'][t, nxt]
    scores = dist/(1+p) if rule is None else np.asarray(rule(dist.copy(), p.copy(), state), dtype=float)
    if scores.shape != (len(t),) or not np.isfinite(scores).all():
        raise ValueError('Rule must return a finite priority for every current edge')
    choices = np.flatnonzero(scores == scores.max())
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state['rng_state'])
    idx = int(rng.choice(choices))
    state['rng_state'] = copy.deepcopy(rng.bit_generator.state)
    return tuple(sorted((int(t[idx]), int(nxt[idx]))))


def penalize(state, edge):
    """Apply one intervention at a local optimum, without charging delta work."""
    if not state['needs_penalty'] or state['cursor'] != 0 or state['mode'] != 'gls':
        raise ValueError('Interventions require a GLS penalty decision boundary')
    a, b = map(int, edge)
    edges = {tuple(sorted((int(u), int(v)))) for u, v in zip(state['tour'], np.roll(state['tour'], -1))}
    if tuple(sorted((a, b))) not in edges:
        raise ValueError('Action must be a current tour edge')
    state['penalties'][a, b] += 1
    state['penalties'][b, a] += 1
    state['penalty_events'] += 1
    state['last_penalized_edge'] = [min(a, b), max(a, b)]
    state['needs_penalty'] = False


def run(state, delta_budget, rule=None, stop_at_decision=False):
    """Consume at most delta_budget additional evaluations; mutate and return.

GLS consumes exactly the budget unless asked to stop at a decision boundary.
Pure 2-opt can terminate early on a completed nonimproving sweep.
"""
    if not isinstance(delta_budget, (int, np.integer)) or delta_budget < 0:
        raise ValueError('Budget must be a nonnegative integer')
    left = int(delta_budget)
    while left and not state['terminated']:
        if state['needs_penalty']:
            if stop_at_decision:
                break
            penalize(state, select_edge(state, rule))
        cursor = state['cursor']
        amount = min(left, len(state['pair_i'])-cursor)
        sl = slice(cursor, cursor+amount)
        i, j = state['pair_i'][sl], state['pair_j'][sl]
        t, d, p, lam = state['tour'], state['distance'], state['penalties'], state['lambda_']
        a, b, c, e = t[i], t[(i+1) % len(t)], t[j], t[(j+1) % len(t)]
        delta = d[a,c]+d[b,e]-d[a,b]-d[c,e]
        if lam:
            delta = delta+lam*(p[a,c]+p[b,e]-p[a,b]-p[c,e])
        idx = int(np.argmin(delta))
        if float(delta[idx]) < min(state['scan_best_delta'], -EPS):
            state['scan_best_delta'] = float(delta[idx])
            state['scan_best_pair'] = cursor+idx
        state['cursor'] += amount
        state['delta_evaluations'] += amount
        left -= amount
        if state['cursor'] != len(state['pair_i']):
            continue
        state['sweeps'] += 1
        best = state['scan_best_pair']
        if best >= 0:
            lo, hi = int(state['pair_i'][best])+1, int(state['pair_j'][best])+1
            t[lo:hi] = t[lo:hi][::-1]
            state['current_length'] = length(t, d)
            state['moves'] += 1
            if state['current_length'] < state['best_length']-EPS:
                state['best_tour'] = t.copy()
                state['best_length'] = state['current_length']
                state['stagnation_sweeps'] = 0
            else:
                state['stagnation_sweeps'] += 1
        else:
            state['stagnation_sweeps'] += 1
            if state['mode'] == 'two_opt':
                state['terminated'] = True
            else:
                if state['phase'] == 'initial_descent':
                    state['lambda_'] = state['alpha']*state['current_length']/len(t)
                    state['phase'] = 'guided'
                state['needs_penalty'] = True
        state['cursor'], state['scan_best_delta'], state['scan_best_pair'] = 0, 0.0, -1
        if state['needs_penalty'] and stop_at_decision:
            break
    return state


ARRAY_TYPES = {'coords': float, 'distance': float, 'pair_i': int, 'pair_j': int,
               'tour': int, 'best_tour': int, 'penalties': int}


def save_state(state, path):
    """Atomic JSON checkpoint, including full distance/scan caches and PCG64 RNG."""
    payload = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)


def load_state(path):
    """Restore arrays exactly; reject incompatible checkpoint schema."""
    state = json.loads(Path(path).read_text())
    if state['schema_version'] != 1:
        raise ValueError('Unsupported solver checkpoint')
    for k, dtype in ARRAY_TYPES.items():
        state[k] = np.asarray(state[k], dtype=dtype)
    validate_tour(state['tour'], len(state['coords']))
    validate_tour(state['best_tour'], len(state['coords']))
    return state


def paired_action_set(state, reference_edge, candidate_edges, seeds, horizon, rule=None):
    """Same-state/action intervention, common RNG and exact work per branch.

Returns empirical finite-horizon relative benefit, never an optimal Q value.
    The baseline snapshot is unchanged. Exactly one reference rollout is run
    per seed and reused for all candidates, costing (K+1)*len(seeds) branches.
"""
    if not state['needs_penalty'] or horizon <= 0 or not seeds or not candidate_edges:
        raise ValueError('Need a decision state, positive horizon and paired seeds')
    records = []
    for seed in seeds:
        base = clone(state)
        base['rng_state'] = np.random.default_rng(seed).bit_generator.state
        def rollout(edge):
            branch = clone(base)
            penalize(branch, edge)
            before = branch['delta_evaluations']
            run(branch, horizon, rule=rule)
            assert branch['delta_evaluations']-before == horizon
            return branch
        ref = rollout(reference_edge)
        for candidate_index, candidate_edge in enumerate(candidate_edges):
            cand = rollout(candidate_edge)
            records.append(dict(seed=int(seed), candidate_index=candidate_index,
                            reference_edge=list(reference_edge), candidate_edge=list(candidate_edge),
                            horizon_delta_evaluations=int(horizon), reference_length=ref['best_length'],
                            candidate_length=cand['best_length'], normalization_length=state['best_length'],
                            relative_benefit=(ref['best_length']-cand['best_length'])/state['best_length']))
    return dict(records=records, branches_executed=(len(candidate_edges)+1)*len(seeds),
                reference_branches_executed=len(seeds))


def paired_continuation(state, reference_edge, candidate_edge, seeds, horizon, rule=None):
    """Convenience wrapper for one candidate versus one reference action."""
    return paired_action_set(state, reference_edge, [candidate_edge], seeds, horizon, rule)['records']
