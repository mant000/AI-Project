# AI-Project: Learning TSP Heuristic Programs from Search-State Values

Research question: can feedback about decisions at a shared guided local search
state help a frozen code LLM discover better TSP penalty rules? This repository
currently implements **stage 0 infrastructure and a TSP50 author-data bootstrap**. Solver evaluation, paired
labels, graph Transformer training, and A/B program discovery are subsequent
stages. No research performance results are claimed here.

## Verified environment

Measured on 2026-11-16: NVIDIA RTX 3090 (24 GiB), Intel Xeon Gold 6330,
14 vCPU cgroup quota, 60 GiB memory limit, Ubuntu 22.04, Python 3.10.8,
PyTorch 2.1.2+cu121, NVIDIA driver 580.105.08. The driver advertises CUDA 13.0;
the **PyTorch runtime is CUDA 12.1**. Host CPU count and host RAM are not the
container allocation. `scripts/system_probe.py` records both with cgroup limits.

The project venv is on the data disk and inherits the image's existing PyTorch
wheel using `--system-site-packages`. Project dependencies install into that
venv; the base environment is preserved. On a fresh Python 3.10 machine, create
a normal venv and install both requirements files:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-torch.txt -r requirements.txt
pip check
```

The AutoDL-specific setup is:

```bash
# Repository copied to /root/autodl-tmp/tsp-state-value/code
cd /root/autodl-tmp/tsp-state-value/code
bash scripts/bootstrap_remote.sh
source scripts/env.sh
```

The bootstrap verifies a CUDA attention forward/backward update, optimizer and
checkpoint restore, and tiny randomly initialized Qwen forward/backward and
generation. It does **not** validate loading the full 7B model or GRPO.
Outputs: `$TSP_ROOT/artifacts/system_probe.json`, `environment.freeze.txt`, and
`environment.packages.json`. The CUDA smoke itself took about 0.82 s; package
installation time depends on the network. The full inherited environment is
recorded for auditing; `requirements.txt` lists the project overlay.

The full fixed 7B snapshot was subsequently verified against all four author
weight hashes and loaded in BF16 for one short greedy smoke generation:
4.68 s loading, 0.57 s generation, peak allocated 15,292,781,568 bytes. This
37-input-token / 2-output-token check is not a throughput benchmark or program
discovery result. Repeat it after author hash verification with
`python scripts/model_smoke.py`; output is `artifacts/model_smoke.json`.

## Storage and cache

`source scripts/env.sh` is required for non-interactive SSH too. It places
pip/Hugging Face/PyTorch/Matplotlib/temp caches on the 50 GiB local data disk.
Code lives in `$TSP_ROOT/code`, downloaded author releases in
`$TSP_ROOT/data/raw`, logs in `$TSP_ROOT/logs`, and measurements in
`$TSP_ROOT/artifacts`. The verified file-storage mount resolves from
`/root/autodl-fs` to `/autodl-fs/data` (FUSE, 200 GiB capacity).

Keep a single copy of Qwen weights in the Hub cache (about 15.23 GB decimal).
Reserve at least 8 GiB local free space for subsequent work. Keep author
archives compressed and extract only required members or selected rows;
inspect the manifest's uncompressed size before extraction. File storage is
for compact results/checkpoints/backups, with an independent local backup.
No weights or raw data are committed.

## Author releases and model

`configs/sources.json` fixes upstream repository commits and the model
revision. `configs/qwen_weight_hashes.json` contains the author's four LFS
SHA256 hashes, checked on completion of the model download.

- [Joshi et al., CP 2021](https://github.com/chaitjo/learning-tsp):
  `tsp-data.tar.gz`, approximately 5.84 GB compressed. Author train supplies
  disjoint project training/validation pools; author test is final-only.
- [INViT, ICML 2024](https://github.com/Kasumigaoka-Utaha/INViT):
  `INViT_data.zip`, approximately 163 MB. Reserved for final OOD evaluation.
- [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct):
  revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`.

```bash
bash scripts/start_job.sh tsp-data-joshi python scripts/download_sources.py joshi
bash scripts/start_job.sh tsp-data-invit python scripts/download_sources.py invit
bash scripts/start_job.sh tsp-model-qwen env HF_ENDPOINT=https://hf-mirror.com python scripts/download_sources.py qwen
```

The current server cannot reach Google Drive directly. A local Windows worker
downloads with resume and uploads via SFTP; its operational entry is outside
the public code directory in `milestone/continue_author_downloads.py`. Generic
downloads above work where the author host is reachable. Resume rejects
responses that fail to honor the requested byte range. A download is complete
only when its JSON status says `completed_verified`. Archive inventory does
not imply instance parsing, split correctness, or solver evaluation is done.

The legacy `/uc` endpoint may report a quota error while Google's confirmation
page download domain still serves the same public author file. The following
alternative uses that public download domain, small validated byte ranges,
per-chunk SHA256, resumable chunk files, and a final full gzip CRC/tar audit:

```bash
# Run on a machine that can reach drive.usercontent.google.com.
python scripts/download_drive_ranges.py joshi --root "$TSP_ROOT" --workers 6 --chunk-mib 8
```

Keep sufficient space for both chunks and the assembled archive. No third-party
dataset mirror or regenerated instances are substituted by this command.

Before starting another job, inspect existing tmux sessions, process state,
download JSON, partial sizes, and logs. The current job may still be running.

```bash
tmux ls
tail -n 30 "$TSP_ROOT/logs/tsp-model-qwen-mirror.log"
cat "$TSP_ROOT/artifacts/qwen_download.json"
bash scripts/stage1_preflight.sh
```

The preflight requires a verified Joshi archive and the passed environment
probe. It intentionally fails while data is incomplete. Stage 1 must add
instance-level manifests/splits, distance/reference validation, NN/2-opt/GLS,
state restore and paired-run correctness checks before any learning results.
Do not use final datasets to tune methods.

A separate bootstrap mode allows stage-1 solver development from 512 complete
author **train** rows already recovered from the tar.gz prefix. It validates
coordinates, tours, duplicate coordinate sets, source-row hashes, and the
normalized data hash. This mode makes no full-archive CRC/SHA256 claim and
provides no author validation set or TSP100 data:

```bash
python scripts/prepare_author_bootstrap.py \
  --archive-prefix /path/to/tsp-data.tar.gz.prefix \
  --output "$TSP_ROOT/data/bootstrap" --count 512
bash scripts/stage1_preflight.sh bootstrap
```

The same preparation command can read the full author archive later; selected
source-row hashes must match `splits/bootstrap_reserved_rows.csv`. Keep these
rows in training/development pools when building the full instance split.
This bootstrap supports engineering checks, not final performance conclusions.

## Received author data and validation policy

On 2026-11-17 a user-supplied seven-volume 7z archive was merged without
modifying its source volumes. It contains 11 extracted author text files
(12,755,124,869 bytes), rather than the original `tsp-data.tar.gz`. The merged
7z is 5,739,945,309 bytes with SHA256
`ea6ed955a48028627653c6674fb5038b4073db64e7a2805ff1c88ab6399a64d3`.
This identifies the repack and does not establish the original gzip's hash.
`configs/joshi_received_files.json` records exact per-file hashes and counts.
The independent original download prefix matches the complete TSP50 test file
by bytes and the first 8 MiB of TSP50 train; test data is only hashed for
integrity. All 512 bootstrap source-row hashes match.

No separate `*_val` file exists in this received payload. Before any solver
or learning experiment, `configs/data_split_policy.json` fixes disjoint
original train-file line ranges, separately for TSP50 and TSP100: value train
1–2048, discovery 2049–2112, value validation 100001–100256, program selection
100257–100320, and program development evaluation 100321–100384. These are
**project validation held out from author train**, never “author val”. The
512 bootstrap rows stay in value train. Final author test/INViT/TSPLIB remain
reserved. Stage 1 must still create the complete per-instance manifest and
check final-source overlap using hashes before experiments.

`scripts/audit_author_directory.py` validates file hashes, source-prefix
agreement, all 4,992 selected source rows, disjoint coordinate sets, and
bootstrap reservations. It creates two 384-row project-validation views.
With `--extract` on Linux it verifies the received 7z hash, checks member paths,
extracts via `7z` (including member CRC checks), and compares every extracted
file against the recorded manifest. Ubuntu needs `p7zip-full` for that option.
The original author's gzip can also be used after extracting it with tar;
the auditor consumes the full gzip stream for its CRC and checks the same
per-file hashes. That original-gzip route has not been replayed end to end
here because Google Drive quota prevents completing the original download.

Directory audit, as run locally (set paths to the files you actually have):

```bash
python scripts/audit_author_directory.py \
  --directory /path/to/received_author_payload --project-root "$TSP_ROOT" \
  --archive /path/to/tsp-data.received.7z \
  --archive-prefix /path/to/tsp-data.tar.gz.prefix
```

The server's full gate requires completed archive integrity and received-data
audit records. Until its transfer and extraction finish, only the previously
verified bootstrap gate is ready. No solver, baseline, labels or learning
results are claimed by these data checks.

## Logs and backups

`start_job.sh SESSION COMMAND ...` starts one tmux session and records
`logs/SESSION.log` plus the true command exit code in `logs/SESSION.exit`.
Duplicate live session names are rejected. Confirmed disconnect survival is
recorded in `logs/tsp-tmux-survival.log`.

```bash
bash scripts/backup_remote.sh
```

This refuses an unmounted file-storage path, archives code, artifacts, logs,
and checkpoints, runs an archive integrity check, and writes a SHA256 sidecar.
Copy both outputs to an independent machine and verify the sidecar there.
Small audited bootstrap data is included when present. Weights, caches, and
large raw archives are excluded; author release URLs, hashes, and download
scripts support retrieval.

## Publication

The public repository is [mant000/AI-Project](https://github.com/mant000/AI-Project).
Only the `implementation` directory is published; the original repository
history is preserved. Scoped repository SSH authorization is used for Git
pushes. A submission tag is pending later milestone results. Private course
attachments, instructor comments, SSH credentials, and tokens stay outside
this repository. Never add an experiment command to this README before it
exists and has actually run.
