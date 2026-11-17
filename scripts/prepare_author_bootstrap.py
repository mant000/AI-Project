"""Extract complete TSP50 training rows from an incomplete author tar.gz prefix.

This prepares a narrowly scoped solver-development bootstrap, not the complete
training/validation split. Full archive SHA256 and gzip CRC remain unverified.
Replaying from the complete author archive must produce the same row hashes.
"""
import argparse
import csv
import hashlib
import json
import math
import struct
import tarfile
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def parse_row(raw, line_number):
    tokens = raw.decode('utf-8').split()
    split = tokens.index('output')
    assert split == 100, f'Expected TSP50 coordinates at line {line_number}'
    values = [float(x) for x in tokens[:split]]
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in values)
    coords = list(zip(values[::2],values[1::2]))
    tour = [int(x)-1 for x in tokens[split+1:]]
    assert len(tour) == 51 and tour[0] == tour[-1]
    assert sorted(tour[:-1]) == list(range(50))
    # Decimal arithmetic plus a declared output precision avoids last-bit
    # math.dist differences between Python/platform builds in normalized hashes.
    with localcontext() as context:
        context.prec = 40
        precise = [(Decimal(tokens[i]),Decimal(tokens[i+1])) for i in range(0,split,2)]
        precise_length = Decimal(0)
        for i in range(50):
            x,y = precise[tour[i]]
            xx,yy = precise[tour[i+1]]
            precise_length += ((x-xx)**2+(y-yy)**2).sqrt()
        length = float(precise_length.quantize(Decimal('0.000000000001')))
    assert abs(length-sum(math.dist(coords[tour[i]],coords[tour[i+1]]) for i in range(50))) < 1e-11
    canonical_coords = b''.join(struct.pack('<dd',x,y) for x,y in sorted(coords))
    return {'instance_id':f'joshi-tsp50-train-line-{line_number:07d}',
            'source_member':'data/tsp/tsp50_train_concorde.txt', 'source_line':line_number,
            'original_split':'train','nodes':50,'coords':coords,'reference_tour':tour[:-1],
            'reference_length':length,'reference_kind':'author_Concorde_tour_recomputed_continuous_Euclidean',
            'reference_length_precision':'decimal geometry rounded to 12 decimal places',
            'source_row_sha256':hashlib.sha256(raw).hexdigest(),
            'coordinate_set_sha256':hashlib.sha256(canonical_coords).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=512)
    a = p.parse_args()
    assert 16 <= a.count <= 2048
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/sources.json').read_text(encoding='utf-8'))['joshi']
    rows = []
    with tarfile.open(a.archive_prefix,'r|gz') as archive:
        for member in archive:
            if member.name == 'data/tsp/tsp50_train_concorde.txt':
                f = archive.extractfile(member)
                for i in range(1,a.count+1):
                    raw = f.readline()
                    assert raw.endswith(b'\n'), f'Incomplete training row {i}'
                    rows.append(parse_row(raw,i))
                break
    assert len(rows)==a.count
    assert len({r['coordinate_set_sha256'] for r in rows})==a.count, 'Duplicate instances'
    a.output.mkdir(parents=True,exist_ok=True)
    data_path = a.output/'tsp50_author_train.jsonl'
    with data_path.open('w',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    fields = ['instance_id','source_member','source_line','original_split','nodes',
              'source_row_sha256','coordinate_set_sha256','reference_kind']
    with (a.output/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    record = {'status':'bootstrap_rows_validated','prepared_at_utc':datetime.now(timezone.utc).isoformat(),
              'source':config,'source_archive_path':str(a.archive_prefix.resolve()),
              'source_prefix_bytes':a.archive_prefix.stat().st_size,'source_prefix_sha256':file_hash(a.archive_prefix),
              'full_archive_sha256':None,'full_archive_gzip_crc_verified':False,
              'instance_count':len(rows),'nodes':50,'original_split':'train',
              'original_line_range':[1,a.count], 'data_file':str(data_path.resolve()),
              'data_sha256':file_hash(data_path), 'manifest_sha256':file_hash(a.output/'manifest.csv'),
              'purpose':'Stage-1 solver implementation and engineering checks; reserve these author train rows for training/development only',
              'scope_limits':['TSP50 only','No author validation set yet','No complete project split yet',
                              'No full-archive integrity claim','No final test data used'],
              'validations':['complete newline-terminated source rows','finite unit-square coordinates',
                             'Hamiltonian closed reference tours','continuous Euclidean reference recomputed',
                             'coordinate-set duplicates absent','per-row and normalized-file SHA256'],
              'reproduction':'Run this same command on the complete author archive; selected row hashes must match.'}
    (a.output/'bootstrap_audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
