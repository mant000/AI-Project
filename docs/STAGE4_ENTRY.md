# Stage 4 entry: report, reproducibility and public milestone freeze

Stage3 completed on 18 November 2026. Read `docs/STAGE3_RESULTS.md`,
`results/stage3/analysis-v1/summary.json`, `resource_accounting.json`,
`verification-v1/summary.json`, and the external milestone status first.
Deadline remains **19 November 2026 23:59 BJT**; recalculate remaining time.

Actual evidence: 64 logical outputs, 47 valid /17 invalid; 2 discovery seeds,
16 outputs per arm per seed. A401/B401/B402 selected standard GLS; A402 selected
`(d-mean_d)/(1+p)`. Dev gaps are 2.674253% for standard GLS and 2.321641% for
A402. Seed401 A/B ties; seed402 B is worse by .352612 gap pp. This is a
preliminary negative Transformer-feedback result; do not pick the favorable
A seed or reinterpret the real-branch diagnostic as a successful third arm.

The discovery-only probe executed 154 branches, with 67/69 true ties and
39 thresholded predicted preferences on those ties. Both continuation seeds
are redundant. Training validation selected checkpoints in stage2; stage3
predeclared the equal-weight ensemble 101/202/303. No feedback is calibrated
uncertainty. Four frozen lineage selections yield only two distinct rules.
No expansion, final evaluation, SFT or GRPO is authorized by this entry.

Finish the English PDF (A4, Times New Roman 12pt, 2.5cm margins, single spacing,
at most four pages including milestone references), covering problem,
intervention values, actual A/B, limitations, CPU/GPU costs and future gates.
The editable draft, original data/solver results and stage2 figures remain
available. `discovery_and_dev` and `feedback_probe` figures plus source CSVs
are in `results/stage3/analysis-v1/`.

Verify the reproduction chain in a fresh source checkout using existing
environment/inputs; state precisely which expensive portions are replayed
versus newly generated. Do not rerun stage2 or final by default. Publish the
implementation-only repository after checking credentials/private materials,
verify anonymous reads, then fix the milestone tag and full SHA. Public
release is still stage1 before this step; local commits/backups are not public
reproduction. Large weights/checkpoints need accurate retrieval/hash/availability
instructions. Canvas submission remains the user's responsibility unless
explicitly delegated later.

Update the external status/plan and give the PDF, public URL, exact SHA/tag and
paste-ready submission note. Check real remote processes first; do not start
a duplicate discovery or any additional candidate search to improve B.
