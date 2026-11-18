# CS5491 milestone — Group G042

**Hongye XU (72655146), 19 November 2026.**

Public repository: **https://github.com/mant000/AI-Project**  
Grading tag: **[milestone-2026-11-19](https://github.com/mant000/AI-Project/tree/milestone-2026-11-19)**

- [English milestone PDF](reports/milestone-2026-11-19/milestone.pdf): four A4 body
  pages plus one reference page, Times New Roman 12pt, 2.5cm margins, single spacing.
- [Editable DOCX](reports/milestone-2026-11-19/milestone.docx),
  [Markdown](reports/milestone-2026-11-19/milestone.md),
  [canonical content JSON](reports/milestone-2026-11-19/report_content.json),
  [figures](reports/milestone-2026-11-19/figures), and
  [format verification](reports/milestone-2026-11-19/format_verification.json).
- [Actual reproduction proof](results/milestone-reproduction-v1/release_verification.json),
  [cost ledger](results/milestone-reproduction-v1/cost_ledger.json),
  [exact commands and result map](docs/MILESTONE_REPRODUCTION.md).
- [Results and interpretation in Chinese](docs/MILESTONE_RESULTS_CN.md),
  [concrete route to final](docs/FINAL_ROADMAP_CN.md),
  [data/model entry points](docs/DATA_AND_MODELS.md).

The anonymous-clone numerical verification used computational commit
**110fba1ed847509910f74ebaa4c2053ac76c4282**. The grading tag adds the report,
actual verification records and documentation; frozen experimental inputs,
programs and outcomes are unchanged. Resolve the exact grading commit with:

```bash
git fetch origin tag milestone-2026-11-19
git rev-parse 'milestone-2026-11-19^{commit}'
```

The new-environment CPU verification reconstructed 1,536 states and replayed
2,560 complete dev solves plus six actual feedback branches exactly; 2,240
reference routes, 576 timing routes and all 70 released model files also passed.
This was a separate venv on the same rented server, not a different physical
machine, a full stochastic GPU regeneration or a rerun of historical training.

Primary F−A dev difference: −0.069672 gap pp, joint paired 95% interval
[−0.173928,0]. It does **not** pass the declared success gate. All four discovery
repetitions, failures and duplicate budgets are retained. Strong manual and
author-solver references are reported. Final test data were not evaluated.

The PDF, source, code, input subset, raw logs, program lineage and own trained
inference exports are included in this tag. Full third-party base weights have
pinned download/hash entry points. Original optimizer backups are separate from
inference exports. The user must upload the PDF and paste the repository/tag into
Canvas; this release does not claim a Canvas submission.
