"""Anytime deadline checkpoints; never attribute a late atomic improvement to an earlier budget."""
import time
import traceback
import numpy as np
from tsp_value.solver import create_state,run,select_edge,penalize
from tsp_value.data import length,validate_tour


def snapshot(state,seconds):
    return dict(best_tour=state['best_tour'].copy(),original_length=state['best_length'],
        delta_evaluations=state['delta_evaluations'],penalty_events=state['penalty_events'],available_seconds=seconds)


def run_deadlines(state,deadlines,begin,rule=None,clock=time.perf_counter,operation=None):
    """A single nested trajectory; final atomic overshoot is charged and disclosed."""
    deadlines=sorted(deadlines);index=0;records=[];safe=snapshot(state,clock()-begin);error=None
    def advance():
        if state['needs_penalty']:
            edge=select_edge(state,rule)
            if clock()-begin < deadlines[-1]:penalize(state,edge)
        else:run(state,4096,stop_at_decision=True)
    step=advance if operation is None else operation
    while index<len(deadlines) and not state['terminated']:
        previous=safe
        try:step()
        except Exception:error=traceback.format_exc();break
        after=clock()-begin
        while index<len(deadlines) and after>=deadlines[index]:
            deadline=deadlines[index]
            records.append(dict(**previous,deadline_seconds=deadline,deadline_observed_seconds=after,
                atomic_overshoot_seconds=max(0.,after-deadline),available_by_deadline=previous['available_seconds']<=deadline))
            index+=1
        safe=snapshot(state,after)
    after=clock()-begin
    while index<len(deadlines):
        deadline=deadlines[index];records.append(dict(**safe,deadline_seconds=deadline,deadline_observed_seconds=after,
            atomic_overshoot_seconds=max(0.,after-deadline),available_by_deadline=safe['available_seconds']<=deadline))
        index+=1
    for r in records:
        validate_tour(r['best_tour'],len(state['tour']));actual=length(r['best_tour'],state['distance'])
        assert abs(actual-r['original_length'])<1e-10;r['best_tour']=r['best_tour'].tolist()
    return dict(checkpoints=records,error=error,terminated_early=bool(state['terminated']),
        wall_seconds=after,actual_delta_evaluations=state['delta_evaluations'],
        convention='Conservative last-completed-operation quality at deadlines, not a hard process-stop SLA; atomic overrun reported')


def timed_solve(coords,seed,deadlines,method='gls',rule=None):
    begin,cpu=time.perf_counter(),time.process_time()
    state=create_state(coords,seed,mode='two_opt' if method=='two_opt' else 'gls')
    if method=='nn':state['terminated']=True
    result=run_deadlines(state,deadlines,begin,rule)
    result.update(seed=seed,method=method,process_cpu_seconds=time.process_time()-cpu)
    return result
