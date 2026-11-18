# Five-seed independent development controls

All 67 logical frozen-program aliases were retained, deduplicating only ten
exact canonical source hashes. Every method ran on the same 64 original
`program_dev_eval` instances with solver seeds 0–4 and 200 complete-sweep
equivalents (235,000 / 970,000 valid 2-opt deltas for TSP50/100). Pure 2-opt may
terminate early. The native conventional implementation is separately timed
to avoid charging the conventional baseline generated-program overhead.
Native and compiled conventional GLS returned exactly equal routes/work.

| Frozen condition / reference | Mean gap-to-reference (%) |
|---|---:|
| Original/A, all three discovery seeds | 2.333741 |
| SFT/A, all three discovery seeds | 2.202174 |
| SFT/B | 2.209870 |
| Terminal GRPO/A, all three optimization and discovery seeds | 2.202334 |
| State GRPO/B, all three optimization and discovery seeds | 2.202174 |
| EoH-inspired | 2.336856 |
| Native GLS | 2.336856 |
| Manual stagnation gate | 2.549958 |
| Manual phase centering | 2.134361 |
| Fixed stage2 ensemble directly guiding GLS | 2.255846 |
| Pure 2-opt | 3.757943 |
| Nearest neighbor | 23.392303 |
| LEHD author greedy | 0.544305 |
| LEHD author RRC10 | 0.092007 |
| LKH3.0.6, RUNS1 / MAX_TRIALS1000 | −0.028233 |

SFT/A minus original/A is −0.131567 pp, with a paired original-instance
bootstrap interval [−0.210681, −0.055730]. The primary state-GRPO/B minus
terminal-GRPO/A effect is −0.000160 pp, interval [−0.008923, +0.008655]. The
auxiliary-reward-only effects are zero. All repetitions remain in the results;
no best discovery/training seed or manual rule is selected from this table.
Bootstrap intervals condition on the programs; discovery variability is
separately recorded. These are development results, not final evidence.

Neural direct guidance averaged 1.023 s per exact-work solve on the 3090 with
one CPU thread, versus 0.0595 s for native GLS, including the online feature and
inference costs. This comparison does not establish a wall-budget advantage.
LEHD greedy/RRC10 averaged 0.212/1.329 s per solve on the GPU; LKH averaged
0.183 s on CPU. External methods use their separately disclosed budgets and
search procedures, not identical internal work or strict 1/5-second limits.
LKH continuous input was rounded at scale 1e6; returned tours were repriced in
the original continuous geometry and the quantization bound checked. Negative
gaps remain, because the author continuous reference is not a certified optimum.

The nested 1/5-second engineering timing pilot covered only four fixed original
instances × five seeds, all ten programs plus native GLS/NN/2-opt/neural direct
(280 trajectories). Last-completed-operation snapshots exclude improvements
returned after a deadline. Maximum measured atomic overrun was 0.00733 s;
this is not a hard process-stop guarantee. The pilot cannot substitute for the
predeclared complete final timing evaluation.

Independent audit repriced 6,000 routes/snapshots with zero error, checked
coverage, source hashes, profile fields, integer input costs, deadline
availability, and native/compiled equality, and passed six independent exact
replays. Seventeen unit tests and the invented-geometry final worker fixture
passed. A preflight failure occurred before any manual-rule result: the old
intended dynamic power was outside the AST grammar. Its exp/log1p equivalent
was fixed before evaluation and mathematically checked; constants and the
frozen original intent were retained. That failure's separate CPU/wall time
was not instrumented and is not reported as zero.

The completed driver used 2,307.70 wall seconds and 928.48 parent CPU seconds;
worker CPU and per-method costs are saved separately in `attempt_costs.jsonl`
and `analysis/costs.csv`. All cost scopes remain explicit. Results, tables and
figures: `results/solver-controls/dev-v1`. Final protocol draft:
`configs/final_evaluation_v1.json`; final remains unopened until its freeze gate.
