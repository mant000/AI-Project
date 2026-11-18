"""Check that the sham treatment changes values without leaking preferred IDs."""
import json
from pathlib import Path
import sys
import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'scripts'))
from feedback_revision_cases import make_case,scheduled_states
from tsp_value.programs import canonical_program,INITIAL


def old_bank():
    bank=[json.loads(x) for x in (CODE/'results/core/prepare-v1/bank_observables.jsonl').read_text().splitlines()]
    true={r['bank_id']:r for r in (json.loads(p.read_text()) for p in (CODE/'results/core/prepare-v1/exact_states').glob('*.json'))}
    # Unit-test display metadata only. Old fixed-H values remain old fixtures,
    # never substitute for the newly measured aligned experimental labels.
    bank=[r for r in bank if r['state']['delta_evaluations'] < {50:235000,100:970000}[r['nodes']]]
    for row in bank:
        true[row['bank_id']]['remaining_work']={50:235000,100:970000}[row['nodes']]-row['state']['delta_evaluations']
    return bank,true


def test_sham_preserves_cases_features_and_score_multiset():
    bank,true=old_bank();p=canonical_program(INITIAL)
    for row in bank:
        if np.ptp(true[row['bank_id']]['true_scores_pp'])<=1e-7:continue
        r,ra=make_case(row,true[row['bank_id']],p,p,'R',1101,0)
        s,sa=make_case(row,true[row['bank_id']],p,p,'S',1101,0)
        assert ra['actual_edges']==sa['actual_edges']
        assert ra['shown_values_pp']!=sa['shown_values_pp']
        assert sorted(ra['shown_values_pp'])==sorted(sa['shown_values_pp'])
        for a,b in zip(r['options'],s['options']):
            assert {k:v for k,v in a.items() if k!='supplied_value_pp'}=={k:v for k,v in b.items() if k!='supplied_value_pp'}
        text=json.dumps(r)
        assert 'edge' not in text and 'reference' not in text and 'best_action' not in text
        assert ra['accurate'] and not sa['accurate']


def test_parent_self_comparison_has_no_fabricated_direction():
    bank,true=old_bank();p=canonical_program(INITIAL)
    for row in bank:
        case,audit=make_case(row,true[row['bank_id']],p,p,'P',1101,0)
        assert len(case['options'])==1
        assert not audit['nonzero_shown_spread']
        assert case['options'][0]['current_rule_choice']


def test_case_schedule_does_not_use_parent_or_arm():
    bank,true=old_bank()
    aa=scheduled_states(bank,true,1101,3)
    bb=scheduled_states(bank,true,1101,3)
    assert [r['bank_id'] for r in aa]==[r['bank_id'] for r in bb]
    assert len({r['bank_id'] for r in aa})==2
    assert all(np.ptp(true[r['bank_id']]['true_scores_pp'])>1e-7 for r in aa)
