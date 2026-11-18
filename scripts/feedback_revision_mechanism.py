"""Separate blind-spot repair from local/global alignment; no LLM or solver runs."""
from pathlib import Path
import json
import sys
import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'scripts'))
from feedback_revision_prepare import read,write,digest,stamp,restore
from feedback_revision_cases import make_case
from feedback_revision_analyze import load_bank,table
from tsp_value.programs import canonical_program,compile_rule,INITIAL
from tsp_value.solver import clone,select_edge


def main():
    base=CODE/'results/feedback-revision-v1';out=base/'analysis-mechanism';out.mkdir(parents=True,exist_ok=True)
    bank,exact=load_bank(base/'fresh-prepare');baseline=canonical_program(INITIAL)
    rows=[];example=None
    for row in bank:
        measured=exact[row['bank_id']]
        p,pa=make_case(row,measured,baseline,baseline,'P',1101,0)
        r,ra=make_case(row,measured,baseline,baseline,'R',1101,0)
        assert ra['accurate'] and pa['accurate']
        values=[v['supplied_value_pp'] for v in r['options']]
        assert max(values)==max(measured['true_scores_pp'])
        assert min(values)==min(measured['true_scores_pp'])
        assert 'actual_edges' not in r and all('penalty' in v and 'distance_over_mean' in v for v in r['options'])
        rows.append(dict(bank_id=row['bank_id'],instance_id=row['instance_id'],nodes=row['nodes'],
            all_action_spread_pp=float(np.ptp(measured['true_scores_pp'])),
            positive_alternative_exists=max(measured['true_scores_pp'])>1e-7,
            parent_self_comparison_has_direction=pa['nonzero_shown_spread'],
            revised_case_has_direction=ra['nonzero_shown_spread'],
            revised_case_displays_positive_alternative=any(v>1e-7 for v in values),
            remaining_work=measured['remaining_work']))
        if example is None and max(values)>1e-7:
            example=dict(bank_id=row['bank_id'],old_parent_self_case=p,new_alternative_case=r,
                         scope='same aligned labels; baseline fixed; not an LLM response or whole-rule improvement')
    table(out/'blind_spot_cases.csv',rows);write(out/'example_old_new_case.json',example)
    # Mathematical scope: changing reference to the same statewise best value
    # only adds a rule-independent constant; it cannot repair scalar ranking.
    old,aligned=load_bank(base/'old-prepare');programs=[canonical_program(p.read_text()) for p in
        sorted((CODE/'results/core/discovery-v1/programs').glob('*.py'))]
    local={};regret={};bestmean=float(np.mean([max(aligned[r['bank_id']]['true_scores_pp']) for r in old]))
    for program in programs:
        values=[];gaps=[];rule=compile_rule(program['expression'])
        for row in old:
            edge=select_edge(clone(restore(row)),rule);j=list(map(tuple,row['edges'])).index(edge)
            y=aligned[row['bank_id']]['true_scores_pp'];values.append(y[j]);gaps.append(max(y)-y[j])
        local[program['program_sha256']]=float(np.mean(values));regret[program['program_sha256']]=float(np.mean(gaps))
    error=max(abs(regret[k]-(bestmean-v)) for k,v in local.items());assert error<1e-12
    pair_error=max(abs((regret[a]-regret[b])-(local[b]-local[a])) for a in local for b in local)
    assert pair_error<1e-12
    source=read(base/'analysis-prepare/prepare_summary.json')
    summary=dict(at_utc=stamp(),scope='mechanism checks only; no frozen-model utility conclusions',
        hypotheses=dict(old='Fixed-bank parent/child single-action values improve whole-rule discovery/adaptation',
            revised='Measured alternatives expose jointly missed opportunities; remaining-budget labels improve target correspondence'),
        blind_spot_test=dict(status='passed_for_measured_case_interface',natural_states=len(bank),
            original_instances=len({r['instance_id'] for r in bank}),
            informative_states=sum(r['all_action_spread_pp']>1e-7 for r in rows),
            positive_alternative_states=sum(r['positive_alternative_exists'] for r in rows),
            old_parent_self_direction_states=sum(r['parent_self_comparison_has_direction'] for r in rows),
            revised_direction_states=sum(r['revised_case_has_direction'] for r in rows),
            revised_positive_alternative_visible_states=sum(r['revised_case_displays_positive_alternative'] for r in rows),
            caveat='Pass means information is represented accurately and visibly, not that the LLM uses it or whole-rule quality improves.'),
        local_global_test=dict(status='not_passed_by_current_revision',
            horizon_alignment='implemented and route-verified; necessary budget correction, not full target equivalence',
            same_initial_seed17_control=source['old_control'],
            scalar_reference_identity='regret(h,s)=max_a q(s,a)-q(s,a_h); regret(h1,s)-regret(h2,s)=q(s,a_h2)-q(s,a_h1)',
            mean_identity_max_abs_error=error,pairwise_identity_max_abs_error=pair_error,
            interpretation='Under fixed common states/labels/weights, changing the scalar reference to the best action cannot create distinctions between rules with equal local values. Rich cases add behavioral information, but do not establish whole-rule credit.'),
        next_decision='Milestone may report partial mechanism repair and unresolved target correspondence; no justification yet for scaling training solely from nonzero case values.',
        parameter_training_started=False,final_opened=False,additional_solver_runs=0,additional_llm_calls=0,
        script_sha256=digest(Path(__file__)))
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
