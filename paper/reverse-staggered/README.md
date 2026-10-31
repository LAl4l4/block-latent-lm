# Reverse-staggered decoding — arXiv v1 manuscript

- `main.tex`: the manuscript.
- `refs.bib`: references. All arXiv IDs, titles and authors were checked against the arXiv API on 2026-10-31.
- `scripts/make_figures.py`: reads `runs/schedule-mask/summary.md` and `runs/schedule-mask-long/summary.md`, and writes `data/per_slot_ce_*.csv` plus `figures/per_slot_ce.tex`, `figures/schedule_grid.tex` and `figures/tradeoff.tex` (pgfplots/TikZ, no plotting library needed). If `runs/` is missing, it reads the CSV files instead.
- `scripts/make_upload.py`: rebuilds `arxiv-upload/` and `arxiv-upload.zip` (`main.tex`, `refs.bib`, `figures/*.tex`; relative paths only). Rerun it after every edit to `main.tex`.
- `arxiv-metadata.md`: title, plain-text abstract, categories, comments, license.

## Building and submitting

This machine has no TeX, so the paper has only been syntax-checked, not compiled.

1. Upload `arxiv-upload.zip` to Overleaf as a new project. The compiler is pdfLaTeX; pgfplots and natbib are standard there. Compile until the cross-references and the shared figure legend resolve.
2. arXiv does not run BibTeX. Download `main.bbl` from Overleaf (Logs and output files → Other logs and files) and add it next to `main.tex`.
3. Submit `main.tex`, `main.bbl` and `figures/*.tex` to arXiv. `refs.bib` may stay in the upload but is not used.

Local build, if TeX is installed: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Sources of numbers

- Mask study: `docs/paper-v1-draft.md` and `docs/idea-reverse-staggered-decoding.md`.
- Per-slot values (figure and Appendix B): the `summary.md` files.
- Sample tables: `judge_ce` (whole continuation) and `judge_steady_first_ce` (seam) from the sample-eval JSONs under `runs/schedule-mask*/`.
- Meet-in-the-middle and seed-42 costs: `runs/schedule-mask/*-s42/final.json`.
- Plan experiment: `runs/schedule-plan/*/final.json`.
- Speed/quality sweep (Table and Figure in §4): `runs/schedule-mask-tradeoff/*/final.json` plus the seed-42 arms of `runs/schedule-mask`, copied to `data/tradeoff.csv` by `make_figures.py`.
- Wide 20k scaling: `runs/schedule-mask-wide-long/*/metrics.jsonl` (22M tokens, overfits) and `runs/schedule-mask-wide-1m/*/metrics.jsonl` (219M tokens, `data/prepared-1m`).
- Single-token refinement: `runs/schedule-mask-refine/*/final.json` (refinement CE) and `runs/schedule-mask-refine/sample-eval.json` (`judge_ce`, `judge_steady_by_slot`).
- Intra-block seam in samples: `judge_steady_by_slot` in the sample-eval JSONs.
- Parameter counts (7.2M base, 31.5M wide): the run `config.json` files. The docs' "about 10M" for the base model is wrong.

## Code

- Schedules (visibility): `seam/schedules.py`. Models: `seam/models.py`. Training: `seam/train_mask.py`, `seam/train_plan.py`. Sample evaluation: `seam/sample_eval.py`. Summaries: `seam/summarize.py`.
- Tests: `tests/test_seam_schedules.py` (distance bound, meet d+1, sampler rounds) and `tests/test_seam_plan.py`.
- Run scripts: `scripts/run_schedule_*.bat`.
