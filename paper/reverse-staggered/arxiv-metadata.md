# arXiv submission metadata (v1)

## Title

Write the Seam Last: Reverse-Staggered Decoding for Block-Parallel Language Models

## Authors

TODO(author): names as they should appear on arXiv, matching `\author{}` in `main.tex`.

## Abstract (plain text, 1891 characters; arXiv limit 1920)

Paste this exactly. It contains no LaTeX, and the formula is written in ASCII.

```text
Block-parallel and pipelined generators start a block before its predecessor is finished, so some token pairs can no longer condition on each other. It is known that dependency between tokens decays with distance, so distant tokens are safer to generate in parallel. We turn this into a schedule-design rule for block pipelines and quantify it. Staggered schemes that keep left-to-right order inside each block drop the strongest dependency: the one between a block's first token and the end of the block before it. Our rule is to write the seam last: if each block's head is written after the previous block's tail, the seam token can condition on that tail, whatever the exact schedule. We compare three instances: token-reversed staggering, segment-reversed staggering and a block-level meet-in-the-middle scheme. We prove that reversed staggering drops only pairs at distance at least L-g+1+g*max(0, ceil((s-g+1)/g)). We then run a TinyStories study with small models in which schedules differ only in the attention mask (3 seeds), which separates information cost from ordering cost. Forward and token-reversed staggering drop the same 216 pairs per window, yet the dropped pairs cost 0.350 nats/token at distance 1-8 and 0.016 at distance 24 or more. Segment reversal works best: at 1.75 tokens per step it costs 0.276 nats/token over a causal LM, against 0.350 for forward staggering and 0.508 for one-block-delayed conditioning, and it beats meet-in-the-middle in both parallelism and cost. Under an independent judge, the seam token's cross-entropy in samples falls from 11.3 under forward staggering to 2.9 (causal: 2.6); a smaller seam remains inside the block. The schedules are not free: an ordering cost of about 0.23-0.27 nats/token remains. It barely shrinks with 4x more training, does not shrink with 4.4x more parameters, and is not removed by a discrete block-level plan.
```

## Categories

- Primary: cs.CL (Computation and Language)
- Cross-list: cs.LG (Machine Learning)

## Comments field

`N pages, 2 figures, 9 tables`

TODO(author): replace N with the page count of the PDF that Overleaf compiles.

## License

CC BY 4.0 (Creative Commons Attribution 4.0 International). This is the author's choice; confirm or change it on the arXiv form.

## Before submitting

- Fill in the author names and affiliation in `main.tex`, and remove the `TODO(author)` comments. arXiv publishes the source, comments included.
- Decide on the code-release line in the Reproducibility paragraph.
- arXiv does not run BibTeX. Download `main.bbl` from Overleaf and add it to the upload next to `main.tex`.
