# arXiv submission metadata (v1)

The upload is `arxiv-upload.zip`, built from `main.tex` (ICLR 2027 layout, "Preprint" header, named author) by `scripts/make_upload.py`.

## Title

Write the Seam Last: Which Dependencies Should a Block-Parallel Decoder Drop?

## Authors

Qiran Li

(The paper lists the affiliation as "Independent researcher". No email is printed.)

## Abstract (plain text, 1210 characters; arXiv limit 1920)

Paste this exactly. It contains no LaTeX (the paper's $20\times$ is written 20x).

```text
Block-parallel decoders speed up language-model generation by starting a block before the previous one is finished, so some dependencies between tokens must be dropped. Prior work asks how many dependencies a decoder can afford to drop; we ask which ones. Staggered pipelines write every block left to right and so drop the seam, the dependency between a block's first token and the token just before it. We show that reordering the tokens inside each block so that the seam is written last leaves both the parallelism and the number of dropped dependencies unchanged, yet moves every dropped dependency at least a block away. To measure what this buys, we train models that differ only in their attention mask, which separates the information a schedule loses from the cost of learning a non-left-to-right order. Moved away from the seam, the same dropped dependencies become more than 20x cheaper, and the seam failure largely disappears from samples. What remains is an ordering cost concentrated at a single token per block, which neither scale nor a block-level plan removes in our experiments; this points to learnability, rather than lost information, as the next bottleneck for block-parallel decoding.
```

## Categories

- Primary: cs.CL (Computation and Language)
- Cross-list: cs.LG (Machine Learning)

If the endorser can only endorse cs.LG, make cs.LG primary and cross-list cs.CL.

## Comments field

`23 pages, 3 figures, 15 tables`

## License

CC BY 4.0 (Creative Commons Attribution 4.0 International). This is the author's choice; confirm or change it on the arXiv form.

## Before submitting

- Recompile with `--keep-intermediates` and rerun `scripts/make_upload.py` after any edit, so that `main.bbl` matches the source; the script refuses a stale `.bbl`.
- arXiv publishes the source, comments included. `main.tex` has no TODO comments; keep it that way.
- The Reproducibility statement says the code is in a single code base but gives no link, because the repository is private. Add a URL if the code is released.
- Check the PDF that arXiv builds in its preview before confirming.
