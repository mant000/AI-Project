# From milestone to final

The milestone establishes that a rule fragment carries more conditional
information than one penalty. Four discovery repetitions do not establish a
stable improvement in discovered programs. The next experiment asks whether
the measured feedback can choose better candidates when all methods see the
same candidates.

## Common candidate pool

By November 21, freeze a pool of 64 valid, behaviorally distinct arithmetic
rules, using the existing generator and development-only inputs. Include the
initial rule and all fixed manual controls. Deduplicate both program hashes and
endpoint route vectors. Candidate construction must not use evaluation outcomes.
Reserve separate originals for screening and for evaluating selected rules.

Compare endpoint-only screening with fragment-assisted screening at equal
measured CPU budgets. Include random selection and the best rule in the fixed
pool as references. Report selection regret, top-k quality, ties and total
solver work, rather than only agreement with a value label. Fix the candidate
pool, instances, budgets, random seeds and analysis before measuring outcomes.

Use the rented 14-vCPU host for branch/endpoint evaluation and its RTX3090 for
candidate generation. Start with 16 screening originals and 16 independent
development originals as a runtime pilot. Expand once to 64 per role if the
pilot indicates that the complete comparison fits within six CPU-host hours.
Budget at most two GPU hours and twelve CPU-host hours for this experiment,
including independent route repricing. Inspect progress after pool construction,
the pilot and the single complete run. Timing may change the affordable pool
size before the complete run; it must not be used to select favorable outcomes.

## Decision about a learned predictor

Make this decision by November 24. Proceed only if fragment-assisted screening
reduces held-out selection regret relative to endpoint-only screening at an
equal measured budget, with a paired interval excluding zero and consistent
direction across both problem sizes. Otherwise report the controlled screening
result and stop expanding the feedback pipeline.

If the gate passes, train one compact policy-conditioned fragment-value model
on separate screening-training originals. Its input must describe both state
and candidate rule; old edge-action checkpoints predict a different target.
Use one fixed architecture and three predetermined seeds, averaging predictions.
Budget at most eight GPU hours and twelve CPU-host hours for labels. Require
held-out calibration, nonzero coverage and useful ranking before inserting it
into screening. SFT/GRPO is not justified merely by a successful training run.

## Final evaluation and delivery

By November 28, freeze the method, all seed aggregation, candidate budget,
selection rule, primary comparison and analysis code. Then evaluate once on
previously untouched final originals. Report every predetermined repetition;
do not select a favorable seed or retune after final outcomes. If the learned
predictor gate fails, the frozen final method is the measured-feedback screening
comparison, with that failure retained.

Reserve November 29–December 2 for independent CPU reproduction, cost accounting,
figures, the final English report and an immutable public version. The course
final deadline is December 3, 2026, 23:59 Beijing time. Keep at least one full
day for publication and document checks. Final evaluation has not yet occurred
in the milestone release.
