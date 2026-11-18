"""Distance-aware evaluation adapter; the previously executed GLS engine stays immutable."""
import time
import numpy as np
from tsp_value.data import distances, length
from tsp_value.solver import create_state, nearest_neighbor
from timed_engine import run_deadlines


def create_evaluation_state(coords, seed=0, method='gls', convention='continuous_euclidean'):
    state=create_state(coords,seed,'two_opt' if method=='two_opt' else 'gls')
    if convention!='continuous_euclidean':
        d=distances(coords,convention)
        # Reuse the exact same seeded start and post-start RNG state. Rounded
        # distances must determine NN, every delta, lambda and reported quality.
        tour=nearest_neighbor(d,int(state['tour'][0]))
        state.update(distance=d,tour=tour,best_tour=tour.copy(),
            current_length=length(tour,d),best_length=length(tour,d))
    if method=='nn':state['terminated']=True
    return state


def timed_evaluation(coords,seed,deadlines,method='gls',rule=None,convention='continuous_euclidean'):
    begin,cpu=time.perf_counter(),time.process_time()
    state=create_evaluation_state(coords,seed,method,convention)
    result=run_deadlines(state,deadlines,begin,rule)
    result.update(seed=seed,method=method,distance_convention=convention,
        process_cpu_seconds=time.process_time()-cpu)
    return result


def delta_budget(nodes,sweeps=200):
    return int(sweeps*nodes*(nodes-3)//2)
