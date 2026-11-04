# Reverse-staggered decoding — arXiv manuscript

- `main.tex`: the manuscript. ICLR 2027 layout with a "Preprint" header and the author named (Qiran Li, independent researcher). For an anonymous venue submission, remove `\iclrfinalcopy` and the `\lhead{Preprint}` line after `\maketitle`. The earlier 11pt-article manuscript was frozen in commit 9270038 and replaced on 2026-11-04; it is only in the git history now.
- `iclr2027_conference.sty`, `iclr2027_conference.bst`: the official ICLR 2027 style (github.com/ICLR/Master-Template), downloaded 2026-11-04.
- `refs.bib`: references. All arXiv IDs, titles and authors were checked against the arXiv API on 2026-10-31.
- `scripts/make_figures.py`: reads `runs/schedule-mask/summary.md` and `runs/schedule-mask-long/summary.md`, and writes `data/per_slot_ce_*.csv` plus `figures/per_slot_ce.tex` and `figures/tradeoff.tex` (pgfplots/TikZ, no plotting library needed). It also writes `figures/teaser_schedule.tex` and `figures/teaser_bars.tex` from the 3-seed costs in `summary.md` and the seam judge CE in `runs/schedule-mask/sample-eval-s42-judge-s43.json`, cached in `data/teaser.csv`. If `runs/` is missing, it reads the CSV files instead.
- `scripts/make_upload.py`: rebuilds `arxiv-upload/` and `arxiv-upload.zip` from `main.tex`, `build/main.bbl`, the ICLR style file, `refs.bib` and `figures/*.tex`; relative paths only. It refuses a `.bbl` older than the source. Rerun it after every edit.
- `arxiv-metadata.md`: title, author, plain-text abstract, categories, comments, license.

## Building and submitting

The manuscript compiles locally with Tectonic (installed in `E:\APP\tectonic`, package cache in `E:\APP\tectonic\cache`). From this folder, in PowerShell:

```
$env:TECTONIC_CACHE_DIR = 'E:\APP\tectonic\cache'; & 'E:\APP\tectonic\tectonic.exe' -X compile --outdir build --keep-intermediates main.tex
```

(In Git Bash: `TECTONIC_CACHE_DIR=E:/APP/tectonic/cache /e/APP/tectonic/tectonic.exe -X compile --outdir build --keep-intermediates main.tex`.) The `Fontconfig error` line it prints is harmless. Output goes to `build/` (gitignored); Tectonic runs BibTeX itself, and `--keep-intermediates` keeps the `.bbl` that arXiv needs.

To submit:

1. Compile as above, then run `D:\Code\venv313\Scripts\python.exe scripts\make_upload.py`.
2. Upload `arxiv-upload.zip` to arXiv. arXiv does not run BibTeX; it reads `main.bbl`. `refs.bib` stays in the upload but is not used.
3. Fill in the form from `arxiv-metadata.md`, and check arXiv's PDF preview before confirming.

## Sources of numbers

- Mask study: `stage2/docs/idea-reverse-staggered.md` (repository root).
- Per-slot values (figure and Appendix B): the `summary.md` files.
- Sample tables: `judge_ce` (whole continuation) and `judge_steady_first_ce` (seam) from the sample-eval JSONs under `runs/schedule-mask*/`.
- Meet-in-the-middle and seed-42 costs: `runs/schedule-mask/*-s42/final.json`.
- Plan experiment: `runs/schedule-plan/*/final.json`.
- Speed/quality sweep (Table and Figure in §4): `runs/schedule-mask-tradeoff/*/final.json` plus the seed-42 arms of `runs/schedule-mask`, copied to `data/tradeoff.csv` by `make_figures.py`.
- Wide 20k scaling: `runs/schedule-mask-wide-long/*/metrics.jsonl` (22M tokens, overfits) and `runs/schedule-mask-wide-1m/*/metrics.jsonl` (219M tokens, `data/prepared-1m`).
- Single-token refinement: `runs/schedule-mask-refine/*/final.json` (refinement CE) and `runs/schedule-mask-refine/sample-eval.json` (`judge_ce`, `judge_steady_by_slot`).
- Intra-block seam in samples: `judge_steady_by_slot` in the sample-eval JSONs.
- Example samples and flagged-token shares (appendix "Example samples"): `runs/schedule-mask/sample-examples-s42-judge-s43.json`, written by `python -m stage2.mask.examples --runs causal-s42 forward8-s42 reverse8-s42 reverse8g8-s42 delayed-s42 --judge causal-s43 --out runs/schedule-mask/sample-examples-s42-judge-s43.json`. Full-parallel samples (`landing`, `reverse0`, `forward0`): `runs/schedule-mask/sample-examples-parallel-s42-judge-s43.json`, same command with `--runs landing-s42 reverse0-s42 forward0-s42`. `scripts/make_examples.py` turns both into `figures/examples.tex` and `figures/examples_parallel.tex`.
- Parameter counts (7.2M base, 31.5M wide): the run `config.json` files. The docs' "about 10M" for the base model is wrong.

## Code

- Schedules (visibility): `stage2/schedules.py`. Models: `stage2/models.py`. Training: `stage2/mask/train.py`, `stage2/plan/train.py`. Sample evaluation: `stage2/mask/sample_eval.py`. Summaries: `stage2/mask/summarize.py`.
- Tests: `stage2/tests/test_schedules.py` (distance bound, meet d+1, sampler rounds) and `stage2/tests/test_plan.py`.
- Run scripts: `stage2/scripts/run_schedule_*.bat` (repository root).
