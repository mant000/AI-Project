# Stage 3 measured results — 18 November 2026

**The small frozen-7B A/B study is complete and does not support a benefit from Transformer feedback.** Two paired discovery seeds generated 16 outputs per arm per seed (64 total). All four selected programs were frozen before independent development evaluation. B selected conventional GLS in both seeds; A did so in seed401 and selected a new rule in seed402. No expansion, SFT/GRPO or final evaluation follows from these results.

Entry was 18 November 03:45 BJT, with about 44h14 until the unchanged milestone deadline, 19 November 23:59 BJT. Discovery and dev evaluation finished at 03:56:40 BJT. The environment, full existing Qwen weights and stage2 checkpoints were reused; nothing was installed, downloaded or retrained. The remote GPU was idle and no prior experiment/tmux process was running before launch.

## Protocol fixed before discovery

`configs/stage3_protocol.json` (SHA256 `82cf8684fbdf8842139308b4b5cd0802adb7be1986d38c3778d0d2bbd4583496`) was persisted at 03:50:20 BJT, before the bank probe or generated candidates. Generator: Qwen2.5-Coder-7B-Instruct, revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`, full BF16, frozen 7,615,616,512 parameters. Every local snapshot file was rehashed against the previously verified author manifest. Transformer feedback is the arithmetic mean of all predeclared seeds 101/202/303, using their existing best checkpoints. No best-seed selection occurred.

Common A/B settings: discovery seeds 401/402; sampling seed `1000*discovery_seed + candidate_index`; temperature .8, top-p .95, top-k 50, max-new-tokens 384, context cap 4096; same initial `d/(1+p)` rule, solver, instances and work budget. The first 8 discovery instances per size give 16 original instances, each with solver seeds 0/1. Endpoint caps are 235,000 /970,000 delta evaluations for TSP50/100. A receives real endpoint summaries and previous code/error feedback. B receives the same fields plus same-state parent/child predicted action differences. The intended prompt-content difference also increases B's input-token cost.

Rules are a single arithmetic expression in `priority(d,p,mean_d,max_d,lam,progress,stagnation,best_ratio)`. The AST permits only finite numerical constants, named observables, arithmetic and whitelisted numerical calls, with a node/power limit. Each rule is evaluated in a separate process with CPU, address-space, file-size, descriptor and wall-time limits. Generated code cannot access imports, attributes, subscripts, loops, files, networks or processes. Invalid and exact-hash duplicate outputs consume their original budget; there are no free retries. Exact deterministic evaluations are reused and referenced. Numeric validity is checked at actual decisions, including NaN/Inf failures. The configured worker capacity is eight; actual evaluation concurrency was **one** isolated worker, one BLAS thread, equally for A/B.

The initial nested closed loop comprised two outputs per arm for seed401. It took 6.83s, projecting 109.25s for 64 outputs, so the fixed small study continued. This is included in the 64, not extra work. The nonzero-pair sign gate did not trigger, but only one probe pair qualified; passing this gate is not evidence that feedback is reliable. The protocol explicitly excludes a 240-output expansion in this stage.

## Live encoding and diagnostic feedback

`state_features(state, edges)` reads observable geometry, current/incumbent routes, penalties and progress only. Label extraction remains in the training wrapper. All 1,024 saved **training** states match the original stage2 encoder bit for bit, including node/pair/global/edge/utility tensors. Sixteen fixed training states also match all three models' predictions exactly. No value_val example or error was used for engineering, prompting or method adjustment in stage3. The original encoder version and checkpoint hashes are retained.

The fixed bank contains the first four discovery instances per size, four conventional-GLS decision boundaries each (events 0/8/32/96): **8 instances, 32 states**. Every current undirected edge is scored relative to the same conventional reference edge. Predictions and the bank hash were saved before real probe labels. All bank states are within their size's endpoint work cap. Parent and child decisions use identical snapshots and RNG states; their paired per-model score differences are averaged. The model never scores two different trajectories as if they were the same state.

The separate probe uses the first discovery instance of each size at all four boundaries, eight uniformly sampled nonreference actions plus the model's top action when new. It executed **154 real branches**, 69 action comparisons on 8 states /2 original instances. This is a diagnostic, **not a direct-branch-feedback discovery arm**; no real probe outcome was placed in either prompt.

| Diagnostic | Actual result |
| --- | ---: |
| True zero effects | 67/69 (97.10%) |
| Identical labels from continuation seeds 71/72 | 100% |
| Mean absolute prediction error | 0.040365 pp |
| Nonzero true pairs exceeding the provisional prediction threshold | 1 |
| Opposite signs among those pairs | 0/1 |
| True ties exceeding the same threshold | 39/67 |

The threshold `abs(prediction)>max(.001 pp, across-model std)` is **not calibrated uncertainty**. The 39 false preferences on true ties are a substantial limitation, even though the tiny nonzero sign check did not trigger its stopping condition. Optimization-seed variation does not capture horizon/model error. Feedback values describe one intervention followed by conventional continuation; candidate programs change all later decisions. This scope mismatch and the static conventional bank limit what a score can predict about a complete candidate. Figures: `results/stage3/analysis-v1/feedback_probe.{png,pdf,svg}`.

## Generated candidates and true development results

| Arm / discovery seed | Outputs | Valid | Invalid | Distinct valid hashes within run | Exact duplicates against earlier study programs |
| --- | ---: | ---: | ---: | ---: | ---: |
| A /401 | 16 | 14 | 2 | 2 | 12 |
| B /401 | 16 | 14 | 2 | 3 | 12 |
| A /402 | 16 | 10 | 6 | 9 | 1 |
| B /402 | 16 | 9 | 7 | 6 | 3 |

All 64 full prompts/replies, token IDs, seeds, hashes, parent lineage and errors are retained. Of 17 invalid outputs, 12 used a forbidden call and five violated the one-expression/function structure. There were no output-cap hits or endpoint worker failures. Across the whole study there are 20 evaluated program hashes including the initial rule. Distinct syntax can still yield identical action rankings or routes; exact-hash uniqueness is not behavioral diversity.

Each run retains up to four unique elites using **real discovery gap**, with the initial program eligible. The first eight `program_select` instances per size, seeds 0/1/2, select the final program by true gap; ties retain the earlier elite. No select/dev outcomes enter prompts. All selected hashes were saved together at 03:56:26 BJT, before dev outcomes. Dev uses the first 16 `program_dev_eval` instances per size, seeds 0/1/2: **32 original instances**, 96 solver conditions per program. Identical selected programs reuse identical outcomes explicitly; they are not new measured routes or independent replications.

| Frozen method | TSP50 gap (%) | TSP100 gap (%) | Combined gap (%) |
| --- | ---: | ---: | ---: |
| NN | 23.5210 | 23.6905 | 23.6058 |
| 2-opt | 3.9353 | 4.4069 | 4.1711 |
| Conventional GLS | 1.9893 | 3.3592 | 2.6743 |
| A, seed401 | 1.9893 | 3.3592 | 2.6743 |
| B, seed401 | 1.9893 | 3.3592 | 2.6743 |
| A, seed402 | 1.5776 | 3.0656 | 2.3216 |
| B, seed402 | 1.9893 | 3.3592 | 2.6743 |

Entries are mean per-instance/solver-seed gap-to-reference, not certified optimality gaps. All GLS programs consume identical size-specific delta work; NN and 2-opt can use less work. Seed401 ties exactly. In seed402, B minus A is **+0.352612 gap pp** (B worse), with 10 B wins, 18 B losses and four ties after averaging the three solver seeds within each original instance. A's seed402 mean paired relative length gain over GLS is +0.335773 pp, a different normalization from the gap difference. Only two independent discovery seeds exist; do not select seed402 to represent A or count solver seeds/states as independent discovery replications.

Frozen code:

- A401, B401, B402: `d/(1+p)`, SHA256 `b77a2dde96bbe9cc797043440a875ca67ce76d542a14cd7faad2fbaca568ee21`.
- A402: `(d-mean_d)/(1+p)`, SHA256 `3e79c9ea6c8882a465ae75bb014bd354f60806554dbc4fd6041537733b09594b`.

The result is a **preliminary negative learned-feedback A/B**, not Transformer success. A's one favorable seed demonstrates that the generator found a useful rule on this dev subset, not a robust overall discovery advantage. No further candidates, extra dev instances, prompt tuning or seed selection were added after this result. `discovery_and_dev.{png,pdf,svg}`, source CSV tables, all instance-paired outcomes and complete frozen-program metadata are in `results/stage3/`.

## Measured costs and verification

| Scope | Wall s | Process CPU s | Child CPU s | CUDA peak allocated /reserved |
| --- | ---: | ---: | ---: | --- |
| Preparation, hashes, bank inference, real probe and security checks | 35.22 | 23.38 | 7.19 | 20.30 /34.00 MiB |
| Full 7B load +64 generations +selection +dev | 206.48 | 103.40 | 108.58 | 14.54 /15.15 GiB |
| Independent outcome/hash/prompt audit and 12 route replays | 2.65 | 1.31 | 1.34 | CPU only |

The generation subset used 95.98 synchronized wall seconds; loading used 4.29s. A used 28,473 input /1,605 output tokens, B 46,062 /1,469. Total 74,535 input /3,074 output tokens. Equal output budget therefore does not mean equal discovery cost. CUDA timing is synchronized wall time, not kernel-only GPU time. Raw utilization, power and device-memory samples are retained. Core 7B-process peak RSS was 5.17 GiB. Isolated endpoint worker CPU totaled 93.85s, already included in child CPU rather than extra to add again.

There were **1,600 unique real route evaluations**: 640 discovery, 576 program_select, 384 program_dev_eval. The latter includes NN/2-opt/GLS and the one additional distinct selected program. The 154 action-probe branches, runtime security probes and **12** independent frozen-program route replays are separate, not additional original instances. Stage2 label/training/replay costs remain in the total-method accounting; they are not erased by reusing checkpoints. Plotting/analysis, prior analysis pass and verification-command costs are recorded separately. Transfers, implementation, backup and infrastructure monitoring are outside experiment wall totals. One failed verification CLI launch is retained: PowerShell expanded a remote variable before argparse, so it ran no solver or generation; only its orchestration wall time is available, not CPU time.

Ten existing solver correctness checks pass again. Eight unsafe AST examples were rejected; isolated valid/scalar/zero-division/overflow checks passed. All 1,600 saved tours were independently recomputed against original coordinates with **zero cost error**; program hashes, paired generation seeds and prompt pool boundaries passed. Both distinct frozen rules replayed two dev instances with all three solver seeds (12 routes), matching exact tours, lengths, work and penalty events. See `verification-v1/summary.json` and `solver_regression.xml`.

## Reproduction and handoff

Use the existing environment, prepared project instances, saved stage2 train snapshots/checkpoints and fixed local Qwen snapshot. Commands with **new** output paths are in the README. `executed_source_v1/` preserves the executed core source, in addition to hashes and the frozen protocol. This stage did not repeat stage2, use value_val examples, access final data, train the generator, publish results or create the milestone tag.

Next: `docs/STAGE4_ENTRY.md`. Consolidate the negative A/B, sparse-label limitations and full costs into the English report; verify reproduction and publish/freeze the milestone source in stage4. The public repository still contains stage1 until that explicit publication step. Do not expand this experiment to search for a favorable B result.
