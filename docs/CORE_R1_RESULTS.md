# Core strengthening R1: sparse mechanisms and measured-feedback control

18 November 2026. R1 is complete. The broader project remains in progress;
report packaging is deferred under the user's research-first instruction.

The new study preserves stage3-v1 and freezes `configs/core_abc_v1.json`
(SHA256 `3fd2d97f3855284eeb3c0fbba61f48978093c0d8d778031f404e53fb8bd85531`)
before discovery. Four paired discovery seeds 501–504 generated 24 outputs
per arm: **288 outputs, 261 valid, 27 invalid**. A receives real endpoint
feedback; B additionally receives existing equal-weight Transformer
101/202/303 predictions; C instead receives measured same-state conditional
single-action continuation values. All three use the same frozen BF16 7B,
initial GLS rule, generation settings, work budgets and real-route selection.
No outcome-dependent budget/prompt/model change or free retry occurred.

## Train-only mechanism result

Fresh value_train rows 257–264 per size supply 16 original instances and
64 states. Eight uniform nonreference actions plus reference were continued
once through seven fixed horizons: **576 continued branches**, not 4,032
independent branches. Sixteen additional direct-budget replays matched the
resumed trajectories. Diagnosis took 6.20 wall /41.31 worker CPU seconds.

Of 512 candidate action sequences, 51 show a nonzero best-length effect at
some horizon; **37** of those have zero effect at the last horizon. Another
**376** sequences change the current cycle at some horizon while their best
quality remains tied at every sampled horizon. At one sweep all best-quality
effects tie, despite 244 changed current cycles. Thus both temporal washout
and best-so-far masking are real measured mechanisms. This does not prove
that optimizing a different proxy would improve the final objective.

There were **zero exact maximum-priority ties in 70,822 observed future
decisions**. The later all-edge discovery bank also had zero ties across
222,088 observed decisions. Under these measured conventional trajectories,
changing the continuation tie-breaking RNG supplies no independent repeat.

## All-edge discovery feedback and fresh evaluation

The first four new discovery instances per size, four boundaries each,
yield 32 fixed bank states. All 2,400 current-edge conditional rollouts were
measured once using common seed71 and the original H (58,750 /970,000).
Their extra cost is **150.38 worker CPU seconds**, about 22.62s from initial
preparation start until the final saved rollout. This is a separately costed
C control, not an optimal oracle or proof of Transformer success. Every
saved branch route recomputed with zero error before generation.

Measured nonreference zero rates are 93.75% for TSP50 and 98.99% for TSP100.
The model threshold flags 280 /895 true-zero actions respectively. The
ensemble's selected benefit over all current edges is 0, versus a measured
best-action mean benefit of **0.186192 pp**. This exposes action coverage and
teacher ranking issues; it does not make the measured best-action control a
deployable policy or an estimate of a complete program's gain.

Pools use previously unused slices: discovery offsets8–23 per size,
program_select offsets8–23, program_dev_eval offsets16–31. IDs and coordinate
hashes are disjoint across selected pools and from the stage3-v1 selected
instances. Old dev outcomes do not enter prompts. All 12 lineage selections
were frozen before any new dev result. Fresh dev has 32 original instances,
three common solver seeds.

| Frozen method | Seeds501 /502 /503 combined gap (%) | Seed504 combined gap (%) |
|---|---:|---:|
| Conventional GLS | 2.687714 | 2.687714 |
| A: endpoint | 2.687714 | 2.555683 |
| B: endpoint + predicted | 2.687714 | 2.687714 |
| C: endpoint + measured | 2.687714 | 2.687714 |

B−A and C−A are [0,0,0,+0.132031] gap pp across the four discovery pairs;
their mean is +0.033008 pp. C−B is zero in all four. The additional A504 rule
is `d/(1+p+max_d/100+(progress+stagnation)/20)`, hash
`9c6e581f82b5504ecd6c05a37553a367f1862c8688ef782d7a111c661f540db7`.
Every other selection is standard GLS. These are preliminary negative
results for both current feedback variants. C failing to help means teacher
prediction error is not an adequate sole explanation; it does not rule out
other state information, feedback designs or a trained generator.

## Diversity, cost and verification

There are only two distinct frozen programs. Nine of the twelve discovery
runs generate only one distinct endpoint route vector, despite multiple
program hashes. A504 generates seven route vectors. Thus grammar validity
alone does not establish useful SFT data diversity. Syntax differs much more
often than solver behavior.

The full generation/select/dev process took **1,107.03 wall seconds**,
545.85 process CPU seconds and 579.42 child CPU seconds. Generation used
513,132 input /15,122 output tokens and 531.80 synchronized wall seconds.
CUDA peak allocated/reserved was 15,752,160,256 /18,563,989,504 bytes.
There are **7,744 unique endpoint routes**: discovery4,288, select3,072,
dev384. Cache reuse is explicit; repeated selected standard rules are not
new route executions or independent observations. The 2,400 measured bank
branches, 576 diagnostic branches, 16 diagnostic direct replays and 12
frozen-program verification replays are separate costs. Prior stage2/3
costs remain part of overall method accounting.

`verification-r2/summary.json` passed: all 7,744 endpoint routes recomputed
with zero cost error; 5,536 saved same-state feedback rows agree exactly
with their declared source; source/protocol hashes, paired generation seeds,
fresh dev slice and prompt boundaries pass. Both distinct programs replayed
six route conditions each with exact tours, quality, work and penalty events.

Two engineering failures are preserved. First, bank summary JSON could not
serialize NumPy scalars after all rollouts had been saved; the scalar cast
was repaired and every rollout reused. Initial parent-process CPU and GPU
peak are unavailable, rather than zero; resumed-attempt figures do not
describe original inference. Second, a prompt-audit substring check mistook
discovery ID2057 for protected train ID2; exact instance-token matching fixed
the audit without rerunning discovery. Original logs/source and the passing
audit are retained. No evidence of actual protected-pool prompt leakage was
found.

## Next: R2 teacher control and executed corpus

Do not expand this frozen study until a preferred arm wins. Preserve the
negative result. Next use a bounded, separately declared dense-action
teacher pilot, with a GNN control and fresh teacher holdout, to assess rare
action coverage. Construct execution-verified, behaviorally varied program
changes with explicit source and lineage/family isolation before SFT. The
same-initialization terminal/state training comparison is still pending;
there has been no parameter training or final evaluation.

Tables, plots, raw prompts/token IDs, all candidates/failures, exact bank,
mechanism records and source snapshots are under `results/core/`. Figures:
`analysis-v1/core_discovery_dev` and `mechanism_horizons` (PNG/PDF/SVG).
