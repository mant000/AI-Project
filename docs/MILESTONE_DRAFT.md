# Learning TSP Heuristic Programs from Search-State Values

XuHongye, 72655146. Working report text, stage 1, 17 November 2026.
This editable draft will be typeset within four pages for the milestone.

## Problem and motivation

For a Euclidean traveling salesperson instance with cities V and distances
d(i,j), the task is to find a Hamiltonian cycle minimizing its original
length L. We study automated discovery of the edge-priority rule used by
guided local search (GLS), while keeping its initializer, neighborhood,
penalty mechanics and evaluation budget fixed. A discovered rule will run
without an LLM or a neural network at deployment.

Terminal solution quality provides a useful program-level objective but
aggregates many decisions along a search trajectory. Our hypothesis is that
feedback about alternative decisions at a shared state can help a code LLM
propose better penalty rules. An edge's usefulness for escaping a particular
local optimum need not coincide with its presence in a reference tour.
The research question concerns the usefulness of this intervention feedback;
combining a Transformer, an LLM and GLS alone is not a novelty claim.

## Shared solver and controlled interventions

Each run starts from nearest neighbor with a seeded start city and deterministic
city-index distance ties. Best-improvement 2-opt scans all n(n-3)/2 unordered
nonadjacent pairs. GLS first reaches an unpenalized local optimum, then freezes
lambda = 0.1 L_local/n and optimizes

L_aug(tour) = L(tour) + lambda sum_(i,j in tour) p(i,j).

The conventional comparator selects one current edge maximizing
d(i,j)/(1+p(i,j)); exact utility ties are sampled uniformly by a saved PCG64
generator. This is the explicitly specified single-edge variant of GLS.
Each intervention increases that undirected edge's penalty by one. Future
generated programs will supply only finite priorities for current tour edges.
Final scores always use the best original, unpenalized length seen.

One search work unit is one valid 2-opt delta evaluation. A checkpoint stores
coordinates, the distance and pair-index caches, current and best tours/costs,
the full symmetric penalty matrix, lambda and phase, the partial-sweep cursor
and best pending move, progress/stagnation counters, and RNG state. A move is
committed only after its entire sweep. Thus a budget that ends mid-sweep can
be resumed without silently restarting work or changing the evaluated suffix.

At a penalty decision state s, let e0 denote the conventional edge and e a
candidate edge. Copy s, apply each intervention, and continue under the same
fixed policy b, common RNG initialization z, and exactly H additional delta
evaluations. The proposed target is

A_H(s,e) = mean_z [(L_H(s,e0;b,z) - L_H(s,e;b,z))/L_best(s)].

The reference branch is executed once per state/seed and reused across
candidates. This is empirical finite-horizon relative benefit under b,
not an optimal Q value. Stage 1 establishes execution correctness; it does
not establish that these targets are informative or learnable.

## Data and separation

The received Joshi author data is a seven-volume 7z repack with verified member
hashes and CRCs, not a verified hash of the original gzip archive. No separate
author validation file is present. Both TSP50 and TSP100 use fixed original
author-train rows: 1–2048 for value training, 2049–2112 for discovery,
100001–100256 for value validation, 100257–100320 for program selection,
and 100321–100384 for program development evaluation. These are project
validation pools held out from author train. All 512 TSP50 bootstrap rows
remain in value training. States inherit the instance's pool.

The full 6,186-row manifest contains 4,992 project instances and 1,194 final
identifiers. Coordinate identities were checked against all 3,840 author
test rows, all 6,000 INViT rows in the three selected distributions, and the
42 audited TSPLIB instances. No project/final exact or float32-quantized
coordinate-set overlap was found. Such hashes detect duplicate coordinates,
not every possible geometric equivalence.

Final use remains locked: author test TSP50/100/200, 256 each; INViT TSP100
clustered1/explosion/implosion, 128 each; and 42 TSPLIB EUC_2D instances.
The clustered1 variant was fixed at the identity audit before any final
solver evaluation. No final solver result or reference-route cost was
computed. The milestone uses continuous Euclidean distances; TSPLIB will use
its rounded EUC_2D convention at final evaluation.

## Stage 1 preliminary results

The predeclared engineering baseline uses the first 16 project value-validation
instances per size and three common solver seeds: 32 instances and 288 runs.
The 2-opt/GLS caps are 235,000 deltas for TSP50 and 970,000 for TSP100,
equivalent to 200 sweeps including initial descent. Pure 2-opt stops on
convergence; NN uses no delta evaluations. These are shared budget caps,
not identical actual work for all three algorithms.

| Size | NN gap (%) | 2-opt gap (%) | Conventional GLS gap (%) |
| --- | ---: | ---: | ---: |
| TSP50 | 22.7997 | 3.2449 | 1.8333 |
| TSP100 | 27.4375 | 4.0042 | 2.8971 |

Entries average per-run gap-to-reference, first over three seeds and then
over equally weighted instances. No runs failed in the corrected evaluation.
Two TSP50 GLS runs improved the supplied reference by 0.0513%; their complete
tours and independent scalar distance sums are retained. The author evaluator
uses continuous Euclidean length, while its published generator calls
Concorde with `norm="GEO"`. That source-level discrepancy is a possible
explanation, not proof of how each received row was generated. We report
gap-to-reference throughout and make no optimality claim for those routes.

Ten solver checks pass, including exact enumeration on a hand-written small
instance, raw/augmented delta identities, tour validity, arbitrary partial-sweep
checkpoint replay, seeded utility ties, clone isolation, identical-action zero
benefit, equal-work interventions and reference-rollout reuse. On one author
training instance per size, serialized restored and uninterrupted states match
byte for byte. Same-action benefits are zero for both paired seeds. The two
different-action engineering probes also tie at H=50,000: these are not a label
pilot or evidence of action-value signal. H must be diagnosed on train only.

## Feasibility, limitations and planned experiments

The server provides an RTX 3090 24GB, a 14-vCPU quota on Xeon Gold 6330 and
60GiB RAM. Stage 1 used one CPU process and one BLAS thread, with zero GPU
compute. The identity/manifest audit took 20.99 wall seconds / 20.98 CPU seconds.
The successful check/evaluation command took 9.51 wall seconds / 9.51 child CPU
seconds and peaked at 139.1 MiB child RSS. Baselines alone took 7.39 seconds
including independent cost checks and logging. These are engineering
measurements on this machine, not a controlled solver speed competition.

Two failed verification attempts consumed 31.59 wall seconds / 31.56 child
CPU seconds in total, including the manifest audit: first an integer-coordinate
checkpoint representation mismatch, then an overly strict assertion against
negative reference gaps. A prior launch failed immediately because a system
timing executable was absent. Logs and failed outputs are retained. No model
training or candidate generation cost has yet been incurred by stage 1.

Next, a 64-instance training-only pilot will measure finite-horizon label
signal, ties, seed sensitivity and branch throughput before H and state
sampling are frozen. A graph Transformer and simple MLP will then be trained
and evaluated by ranking, action regret and realized continuation benefit.
The planned frozen Qwen2.5-Coder-7B-Instruct A/B compares endpoint-only
feedback with endpoint plus state-value feedback, sharing seeds and candidate
budgets. Programs will still be selected by real solution quality, using
separate discovery/selection/development pools. Transformer training,
program discovery, SFT and GRPO have not run. The current small baseline
establishes feasibility and does not test the main hypothesis.

## Reproducibility and source notes

Raw run records, audit hashes, checks and costs are in `results/stage1/`;
earlier failed attempts are in `results/stage1_attempts/`. Exact commands are
in the README. The repository is https://github.com/mant000/AI-Project.

- Joshi et al., Learning TSP Requires Rethinking Generalization, CP 2021:
  https://github.com/chaitjo/learning-tsp/tree/bb28f5795924d99f7e7d945695e884f8d1f7df45
- Cached author cost definition, lines 77–82, and generator, line 39, at the
  same pinned revision; source hashes are retained in the stage 1 audit.
- INViT author data repository:
  https://github.com/Kasumigaoka-Utaha/INViT/tree/f11a6024870036f5d7a7826e0955bd4e6dd350dd
- TSPLIB95: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

Full related-work positioning and the final formatted bibliography remain
for the report stage; the current evidence does not support a novelty or
statistical-significance claim.
