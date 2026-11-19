"""Matched-work interventions on a complete rule, with immutable legacy GLS semantics."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'src'))
from tsp.solver import create_state,run,clone,select_edge,penalize,ARRAY_TYPES
from tsp.programs import compile_rule
from tsp.data import length,validate_tour


def snapshot(state):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in state.items() if k not in ('distance','pair_i','pair_j')}


def restore(row):
    state=create_state(row['state']['coords'],17);state.update(copy.deepcopy(row['state']))
    for k,dtype in ARRAY_TYPES.items():state[k]=np.asarray(state[k],dtype=dtype)
    return state


def controlled_run(state,work,rule):
    """Exactly run(...,rule), exposing actual chosen edges rather than max-score guesses."""
    end=state['delta_evaluations']+int(work);trace=[]
    while state['delta_evaluations']<end and not state['terminated']:
        if state['needs_penalty']:
            edge=select_edge(state,rule);penalize(state,edge);trace.append(list(edge))
        run(state,end-state['delta_evaluations'],stop_at_decision=True)
    return trace


def intervene(row,expression,mode,cfg):
    begin,cpu=time.perf_counter(),time.process_time();state=restore(row)
    assert state['needs_penalty'] and state['cursor']==0
    state['rng_state']=np.random.default_rng(cfg['continuation_seed']).bit_generator.state
    cap=cfg['delta_budget_by_nodes'][str(row['nodes'])];remaining=cap-state['delta_evaluations'];assert remaining>0
    rule=None if expression is None else compile_rule(expression)
    if mode=='single':
        edge=select_edge(state,rule);penalize(state,edge);trace=[list(edge)];controlled=0;run(state,remaining)
    elif mode=='fragment':
        controlled=max(1,int(remaining*cfg['fragment_fraction_of_remaining']));trace=controlled_run(state,controlled,rule)
        run(state,remaining-controlled)
    elif mode=='persistent':controlled=remaining;trace=controlled_run(state,remaining,rule)
    else:raise ValueError(mode)
    assert state['delta_evaluations']==cap;validate_tour(state['best_tour'],row['nodes'])
    actual=length(state['best_tour'],state['distance']);assert abs(actual-state['best_length'])<1e-10
    return dict(mode=mode,best_length=actual,best_tour=state['best_tour'].tolist(),remaining_work=remaining,
        controlled_delta_work=controlled,controlled_decisions=len(trace),first_edge=trace[0] if trace else None,
        controlled_action_sha256=hashlib.sha256(json.dumps(trace).encode()).hexdigest(),controlled_first_edges=trace[:8],
        total_work=cap,wall_seconds=time.perf_counter()-begin,worker_cpu_seconds=time.process_time()-cpu)


def collect_states(item,cfg):
    """Sample at the next penalty boundary after fixed work fractions, without outcome filters."""
    cap=cfg['delta_budget_by_nodes'][str(item['nodes'])]
    for policy,expression in enumerate(cfg['trajectory_expressions']):
        state=create_state(item['coords'],cfg['trajectory_seed']);rule=compile_rule(expression);seen=set()
        for phase,fraction in enumerate(cfg['state_work_fractions']):
            target=int(cap*fraction)
            while state['delta_evaluations']<cap:
                if state['needs_penalty']:
                    if state['delta_evaluations']>=target:break
                    penalize(state,select_edge(state,rule))
                run(state,cap-state['delta_evaluations'],stop_at_decision=True)
            if not state['needs_penalty'] or state['delta_evaluations']>=cap:continue
            signature=(state['delta_evaluations'],state['penalty_events'])
            if signature in seen:continue
            seen.add(signature)
            yield dict(bank_id=f"{item['instance_id']}:policy{policy}:phase{phase}",instance_id=item['instance_id'],nodes=item['nodes'],
                policy_index=policy,phase_index=phase,target_work_fraction=fraction,actual_work_fraction=state['delta_evaluations']/cap,
                state=snapshot(state))
