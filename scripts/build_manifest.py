"""Freeze project pools and final identifiers; audit coordinates without final solving.

Only project train/validation pools get normalized coordinates and reference
costs. Final sources are read for identity/format checks only, not evaluated.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE/'src'))
from tsp_value.data import coordinate_hash, parse_author, distances, length


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--tsplib-archive', type=Path, required=True)
    args = ap.parse_args()
    wall, cpu = time.perf_counter(), time.process_time()
    policy = json.loads((CODE/'configs/data_split_policy.json').read_text())
    sources = json.loads((CODE/'configs/sources.json').read_text())
    expected = json.loads((CODE/'configs/joshi_received_files.json').read_text())
    final_policy = json.loads((CODE/'configs/final_identity_policy.json').read_text())
    out = args.root/'data/prepared/stage1'
    out.mkdir(parents=True, exist_ok=True)
    results = CODE/'results/stage1'
    results.mkdir(parents=True, exist_ok=True)
    records, identities, overlaps, hash32_collisions = [], {}, [], []
    counts = Counter()
    checked = Counter()
    project32 = {}
    def add(coords, source, file, file_hash, line, n, split, pool, raw_hash, reference, convention='continuous_euclidean'):
        h, h32 = coordinate_hash(coords), coordinate_hash(coords, True)
        ident = f'{source}:{Path(file).name}:{line}'
        if h in identities:
            overlaps.append([identities[h], ident])
        identities[h] = ident
        if not pool.startswith('final'):
            project32[h32] = ident
        elif h32 in project32:
            hash32_collisions.append([project32[h32], ident])
        records.append(dict(instance_id=ident, source=source, download_url=sources[source]['url'],
                            source_revision=sources[source].get('revision', 'TSPLIB95'),
                            source_file=file, source_file_sha256=file_hash, source_line=line,
                            nodes=n, original_split=split, project_use=pool,
                            coordinate_set_sha256=h, coordinate_float32_sha256=h32,
                            source_row_sha256=raw_hash, reference_solution=reference,
                            distance_convention=convention, final_evaluated=False))
        counts[pool] += 1
        return ident, h

    with (out/'project_instances.jsonl').open('w') as normalized:
        for n in (50, 100):
            name = f'data/tsp/tsp{n}_train_concorde.txt'
            path = args.root/'data/author_received'/name
            digest = sha(path)
            assert digest == expected['files'][name]['sha256']
            with path.open('rb') as f:
                upper = max(b for a, b in policy['pools_per_size'].values())
                for row in range(1, upper+1):
                    raw = f.readline()
                    pool = next((k for k, (a,b) in policy['pools_per_size'].items() if a <= row <= b), None)
                    if pool is None:
                        continue
                    coords, ref = parse_author(raw, n)
                    raw_hash = hashlib.sha256(raw).hexdigest()
                    ident, coord_hash = add(coords, 'joshi', name, digest, row, n, 'train', pool, raw_hash, 'source output; closed 1-based author Concorde route')
                    normalized.write(json.dumps(dict(instance_id=ident, nodes=n, project_use=pool,
                        source_line=row, coordinate_set_sha256=coord_hash, coords=coords.tolist(),
                        reference_tour=ref.tolist(), reference_length=length(ref, distances(coords))))+'\n')
    reserved = list(csv.DictReader((CODE/'splits/bootstrap_reserved_rows.csv').open()))
    lookup = {(r['nodes'], r['source_line']): r for r in records}
    for r in reserved:
        actual = lookup[50, int(r['source_line'])]
        assert actual['project_use'] == 'value_train'
        assert actual['source_row_sha256'] == r['source_row_sha256']
        assert actual['coordinate_set_sha256'] == r['coordinate_set_sha256']
    project_hashes = {r['coordinate_set_sha256']: r['instance_id'] for r in records}
    for n in (50, 100, 200):
        name = f'data/tsp/tsp{n}_test_concorde.txt'
        path = args.root/'data/author_received'/name
        digest = sha(path)
        assert digest == expected['files'][name]['sha256']
        with path.open('rb') as f:
            for row, raw in enumerate(f, 1):
                coords, ref = parse_author(raw, n)
                h, h32 = coordinate_hash(coords), coordinate_hash(coords, True)
                checked[f'joshi_test_{n}'] += 1
                ident = f'joshi:{Path(name).name}:{row}'
                if h in project_hashes:
                    overlaps.append([project_hashes[h], ident])
                if h32 in project32:
                    hash32_collisions.append([project32[h32], ident])
                if row <= final_policy['joshi_test_rows_per_size']:
                    add(coords, 'joshi', name, digest, row, n, 'test', 'final_id_size',
                        hashlib.sha256(raw).hexdigest(), 'source output; final route cost deliberately not computed')
    invit = args.root/'data/raw/INViT_data.zip'
    assert sha(invit) == final_policy['invit_archive_sha256']
    with zipfile.ZipFile(invit) as z:
        for distribution in final_policy['invit_distributions']:
            name = f'data/data_farm/tsp/tsp100/tsp100_{distribution}.txt'
            raw_file = z.read(name)
            digest = hashlib.sha256(raw_file).hexdigest()
            for row, raw in enumerate(raw_file.splitlines(keepends=True), 1):
                coords = np.asarray([list(map(float, pair.split(','))) for pair in raw.decode().split()])
                assert coords.shape == (100, 2) and np.isfinite(coords).all()
                h, h32 = coordinate_hash(coords), coordinate_hash(coords, True)
                checked[f'invit_{distribution}'] += 1
                ident = f'invit:{Path(name).name}:{row}'
                if h in project_hashes:
                    overlaps.append([project_hashes[h], ident])
                if h32 in project32:
                    hash32_collisions.append([project32[h32], ident])
                if row <= final_policy['invit_rows_per_distribution']:
                    add(coords, 'invit', name, digest, row, 100, 'published_ood', 'final_ood',
                        hashlib.sha256(raw).hexdigest(), f'data/solution_farm/tsp100_{distribution}/Gurobi.txt; verify at final')
    assert sha(args.tsplib_archive) == final_policy['tsplib_archive_sha256']
    catalog = json.loads((CODE/'configs/tsplib_catalog.json').read_text())
    with tarfile.open(args.tsplib_archive, 'r:gz') as tar:
        for item in catalog['selected_instances']:
            name = item['instance_id']+'.tsp.gz'
            member = next(m for m in tar.getmembers() if Path(m.name).name == name)
            content = gzip.decompress(tar.extractfile(member).read())
            text = content.decode()
            assert 'EUC_2D' in text and 'FIXED_EDGES_SECTION' not in text
            rows = text.split('NODE_COORD_SECTION', 1)[1].split('EOF')[0].strip().splitlines()
            coords = np.asarray([[float(t) for t in line.split()[1:3]] for line in rows])
            assert len(coords) == item['dimension']
            checked['tsplib'] += 1
            add(coords, 'tsplib', member.name, hashlib.sha256(content).hexdigest(), 1,
                len(coords), 'benchmark', 'final_tsplib', hashlib.sha256(content).hexdigest(),
                'official best-known length; verify at final', 'tsplib_euc_2d')
    assert not overlaps and not hash32_collisions, f'Overlap: {overlaps[:3]} {hash32_collisions[:3]}'
    assert dict(counts) == dict(value_train=4096, discovery=128, value_val=512,
        program_select=128, program_dev_eval=128, final_id_size=768, final_ood=384, final_tsplib=42)
    manifest = CODE/'splits/manifest.csv'
    with manifest.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    report = dict(status='passed', counts=dict(counts), total_manifest_instances=len(records),
                  all_final_source_rows_checked=dict(checked), exact_overlap_pairs=overlaps,
                  float32_overlap_pairs=hash32_collisions, bootstrap_512_reserved=True,
                  final_solver_or_reference_cost_computed=False,
                  validation='project heldout from author train; no author val',
                  final_policy=final_policy, manifest_sha256=sha(manifest),
                  project_instances_sha256=sha(out/'project_instances.jsonl'),
                  wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
    (results/'manifest_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
