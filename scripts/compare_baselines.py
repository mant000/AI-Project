"""Compare deterministic solver outputs while keeping timing measurements separate."""
import argparse
import json
from pathlib import Path

FIELDS=('instance_id','nodes','project_use','seed','method','original_length',
        'reference_length','gap_percent','delta_evaluations','sweeps','moves',
        'penalty_events','lambda_','scan_cursor','terminated','best_tour','error_type')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference',type=Path)
    ap.add_argument('replay',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    a=[json.loads(x) for x in args.reference.read_text().splitlines()]
    b=[json.loads(x) for x in args.replay.read_text().splitlines()]
    assert len(a)==len(b)
    differences=[]
    for i,(first,second) in enumerate(zip(a,b)):
        for field in FIELDS:
            if first.get(field)!=second.get(field):
                differences.append(dict(row=i,field=field))
    result=dict(status='passed' if not differences else 'failed',run_count=len(a),
                compared_fields=FIELDS,differences=differences,timing_compared=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert not differences


if __name__=='__main__':
    main()
