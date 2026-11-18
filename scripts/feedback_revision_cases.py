"""Revised measured cases, neutral action order, and explicit sham-score control."""
import hashlib
import json
import numpy as np
from feedback_revision_prepare import restore
from tsp_value.solver import select_edge, clone
from tsp_value.programs import compile_rule

EPS=1e-7


def case_seed(*parts):
    return int(hashlib.sha256(':'.join(map(str,parts)).encode()).hexdigest()[:16],16)


def scheduled_states(bank, exact, seed, index, count=2):
    eligible=[r for r in bank if np.ptp(exact[r['bank_id']]['true_scores_pp'])>EPS]
    order=np.random.default_rng(seed).permutation(len(eligible))
    return [eligible[int(order[(count*index+j)%len(order)])] for j in range(count)]


def make_case(row, measured, parent, previous, arm, seed, index):
    """R/S have identical construction given the same parent; only values differ.

    After sequential search diverges, parents/actions can differ across arms.
    All case selection and shuffling are independent of selected/dev outcomes.
    """
    s=restore(row);edges=list(map(tuple,row['edges']));truth=np.asarray(measured['true_scores_pp'])
    rule=compile_rule(parent['expression']);chosen=select_edge(clone(s),rule)
    if arm=='P':
        other=select_edge(clone(s),compile_rule(previous['expression']))
        options=list(dict.fromkeys([chosen,other]))
    else:
        options=list(dict.fromkeys([chosen,edges[int(np.argmax(truth))],edges[int(np.argmin(truth))],edges[-1]]))
    rng=np.random.default_rng(case_seed(seed,index,row['bank_id'],'option-order'))
    options=[options[int(j)] for j in rng.permutation(len(options))]
    true_values=[float(truth[edges.index(e)]) for e in options]
    shown=list(true_values)
    if arm=='S' and len(shown)>1:
        # A nonidentity cyclic shift avoids accidentally keeping all scores in
        # place when values differ; preserve exactly the same score multiset.
        shift=1+case_seed(seed,index,row['bank_id'],'score-shift')%(len(shown)-1)
        shown=np.roll(np.asarray(shown),int(shift)).tolist()
        if len(set(true_values))>1 and shown==true_values:
            shown=np.roll(np.asarray(shown),1).tolist()
        assert sorted(shown)==sorted(true_values)
    t=s['tour'];d=s['distance'][t,np.roll(t,-1)];pen=s['penalties'][t,np.roll(t,-1)]
    rows=[]
    for j,(edge,value) in enumerate(zip(options,shown)):
        rows.append(dict(option=f'o{j}',current_rule_choice=edge==chosen,
            distance_over_mean=float(s['distance'][edge]/d.mean()),penalty=int(s['penalties'][edge]),
            supplied_value_pp=value))
    output=dict(nodes=row['nodes'],progress=float(np.log1p(s['penalty_events'])),
        stagnation=float(np.log1p(s['stagnation_sweeps'])),best_ratio=s['best_length']/s['current_length'],
        mean_d=float(d.mean()),max_d=float(d.max()),lam=s['lambda_'],remaining_work=measured['remaining_work'],
        options=rows,preferred_options=[r['option'] for r in rows if r['supplied_value_pp']>=max(shown)-EPS])
    audit=dict(bank_id=row['bank_id'],true_values_pp=true_values,shown_values_pp=shown,
        actual_edges=[list(e) for e in options],nonzero_shown_spread=bool(np.ptp(shown)>EPS),
        accurate=bool(np.allclose(shown,true_values,rtol=0,atol=1e-12)),
        actual_current_value_pp=float(truth[edges.index(chosen)]),actual_best_value_pp=float(truth.max()))
    return output,audit


def case_feedback(bank,exact,parent,previous,arm,seed,index):
    states=scheduled_states(bank,exact,seed,index)
    cases=[];audits=[]
    for row in states:
        case,audit=make_case(row,exact[row['bank_id']],parent,previous,arm,seed,index)
        cases.append(case);audits.append(audit)
    display=dict(scope='conditional single-action continuation to remaining work cap; conventional later GLS; values are percentage points, not whole-program rewards',
        instruction='Use features and patterns to suggest a general rule. Do not fit constants to one example. Local cases can be misleading about repeated use; judge progress by REAL ENDPOINTS.',cases=cases)
    return display,audits
