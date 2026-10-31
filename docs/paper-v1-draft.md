# Paper v1 draft — reverse-staggered block decoding

> 写作说明（中文）：这是 arXiv v1 骨架，2026-10-31 凌晨起草。理论部分已完成并有穷举验证；2026-10-31 13:05 已填入 3 seed、长训练、宽模型、混合 mask 与交叉评委样本结果。剩余 TODO：接入块级潜变量系统（决定主张能否升级）、LatLM 式等待对照、温度扫描。主张必须与最终数字一致，不要提前写成"消除了代价"。

## Title candidates

- *Write the Seam Last: Reverse-Staggered Decoding for Block-Parallel Language Models*
- *Where Parallel Decoding Loses Information: Ordering Blocks so Dropped Dependencies Are Distant*

## Abstract (draft)

Block-parallel and pipelined text generation start a block before its predecessor is finished, so some token pairs can no longer condition on each other. Existing staggered schemes (line-parallel LMs, row-pipelined image AR) keep left-to-right order inside a unit, and therefore drop exactly the strongest dependency: the first token of a unit and the last tokens of the unit before it. Recent work identifies this seam as the dominant cost and mitigates it by waiting or deferred commitment. We observe that the seam can instead be removed by ordering: generating each block's tokens in reverse (or in reversed segments) while staggering block starts by *s* steps makes the seam token the last one written, after the previous block is complete. We prove that under this schedule every dropped dependency spans at least L − g + 1 + g⌈(s−g+1)/g⌉⁺ tokens (L + s for token-level reversal), whereas the forward schedule with the same parallelism and the same number of dropped pairs drops dependencies at distance 1. In a controlled study where schedules differ only in the attention mask (3 seeds), dropping the same 216 pairs per window costs 0.016 nats/token when they are distant versus 0.350 when they are adjacent. At 1.75 tokens/step, segment-reversed staggering costs 0.276 nats/token versus 0.350 for forward staggering and 0.508 for one-block-delayed conditioning; with all blocks decoded in parallel it beats a non-causal landing-target scheme (0.466 vs 0.498). Almost all of the remaining cost is an ordering (learnability) cost, which by the chain rule is not information loss: it shrinks slowly with training (0.248 → 0.226 over 4× steps) while the forward schedule's information cost stays fixed, so the gap widens; at 3x capacity the ordering cost is unchanged (0.272 vs 0.256) while the forward cost grows (0.389 vs 0.353). Reverse staggering is therefore the best causal schedule at every scale we test, but not free relative to a causal LM.

## 1. Introduction

- Parallel generation must drop some dependencies; the question is *which*.
- Seam problem documented across modalities: ZipAR (image rows), LCLM / LatLM (text lines), PSF (block-parallel latent planning), our block-latent LM (first-token cost 2.8 nats under one-block-delayed feedback).
- Contribution:
  1. Reverse-staggered and segment-reversed schedules; causal, streaming-compatible KV cache, training = inference.
  2. Exact lower bound on dropped-dependency distance; tight, verified exhaustively.
  3. Mask-only controlled study separating information cost (dropped pairs) from ordering cost.
  4. Sample-quality evaluation with an independent judge; comparison with landing targets (PSF-style) at equal parallelism. TODO: waiting (LatLM-style).

## 2. Setup

Sequence = prompt + blocks of length L. A schedule assigns each generated token a time τ; token i may read token k iff k is in the prompt, or τ(k) < τ(i) and block(k) ≤ block(i). Tokens that read neither each other form *dropped pairs*. Serial steps = number of distinct rounds in the induced partial order.

Schedules: causal; delayed (block t reads ≤ t−2 plus own prefix); forward-s; reverse-s (token reversal); reverse-s-g (segment reversal, segments of length g in reverse order, left-to-right inside); landing (all blocks start together, tokens after the first also read the next block's first token — PSF-style, non-causal).

## 3. Distance guarantee

**Lemma.** For reverse-s-g with g | L and s ≥ 0, every dropped pair has distance at least
d_min = L − g + 1 + g · max(0, ⌈(s − g + 1)/g⌉), and the bound is attained.
For forward-s (s < L) dropped pairs exist at distance 1.

*Proof sketch.* Write j = qg + r. For a pair (t−1, j′), (t, j) with k = q − q′, the pair is dropped iff kg − (r − r′) ≥ s, and its distance is L + kg + (r − r′). Dropping requires k ≥ 0; minimising over r − r′ ≥ −(g − 1) gives the bound. Pairs further apart have distance > L. Verified exhaustively for L ∈ {8, 16}, all g | L, s = 0..L (`tests/test_seam_schedules.py`).

Special cases: token reversal (g=1): L + s. Half-block segments with s = g = L/2: L + 1 with left-to-right order inside both halves (the block head becomes a fill-in-the-middle span). Fully parallel (s=0, g=1): L, causal, unlike landing targets.

## 4. Controlled study: schedules differ only in the mask

Two-stream (XLNet-style) Transformer: content stream ordered by generation time, predictions from position-only queries, so each prediction reads exactly the tokens the schedule allows. Same model (~10M params, d=256, 6 layers), data (TinyStories, prepared split), optimiser and budget for every arm. Metric: natural-corpus CE on steady-state blocks (block ≥ 2), per in-block slot, and per window.

Results (5000 steps, seeds 42/43/44, Δ paired by seed, mean ± sd; causal steady CE 1.977 ± 0.006):

| schedule | tokens/step | min dropped distance | Δ vs causal | first-token CE | last-token CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| reverse-16-g8 (order only) | 1.00 | — | 0.264 ± 0.007 | 2.02 | 2.14 |
| **reverse-8-g8** | 1.75 | 17 | **0.276 ± 0.009** | 2.01 | 2.14 |
| reverse-16 (order only) | 1.00 | — | 0.314 ± 0.008 | 1.29 | 5.39 |
| reverse-8 | 1.75 | 24 | 0.330 ± 0.004 | 1.30 | 5.43 |
| forward-8 | 1.75 | 1 | 0.350 ± 0.007 | 5.24 | 1.97 |
| delayed | 1.75 | 1 | 0.508 ± 0.008 | 5.50 | 2.06 |
| **reverse-0** | 7.00 | 16 | **0.466 ± 0.014** | 1.38 | 5.81 |
| landing (non-causal) | 7.00 | 2 | 0.498 ± 0.009 | 5.83 | 1.46 |
| forward-0 | 7.00 | 1 | 0.570 ± 0.006 | 5.83 | 2.02 |

Decomposition: information cost = Δ(reverse-s) − Δ(reverse-16), i.e. 0.016 (g=1) and 0.012 (g=8); forward-8 has natural order, so its 0.350 is all information cost. Segment length is non-monotonic (g=4: 0.409 order-only, seed 42).

Longer training (seed 42, 20k steps, checkpoints within one run):

| step | causal CE | reverse-16 | reverse-16-g8 | reverse-8-g8 | forward-8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 2.041 | 0.299 | 0.248 | 0.265 | 0.350 |
| 10k | 1.881 | 0.289 | 0.245 | 0.262 | 0.356 |
| 15k | 1.788 | 0.280 | 0.234 | 0.253 | 0.359 |
| 20k | 1.748 | 0.270 | 0.226 | 0.246 | 0.355 |

Where the ordering cost lives (reverse-16-g8, 20k, nats per block vs causal): the first token written (8 positions ahead of the visible context) +3.34, the rest of the right half +2.79, the left half written last as infill −2.52; net +3.61 = 0.226/token. With exact conditionals these terms cancel.

Capacity (seed 42, 31.5M params, d=512, 8 layers, 5000 steps; small-model seed-42 values in parentheses): causal CE 1.772 (1.977); reverse-16-g8 0.272 (0.256); reverse-8-g8 0.296 (0.271); forward-8 0.389 (0.353).

Mixing causal-mask steps into the same model hurts (reverse-8-g8: 0.271 at p=0, 0.291 at p=0.25, 0.351 at p=0.5), mostly through the infilled left half, so the ordering cost is not a representation-sharing deficit.

## 5. Sample quality

512 dev prompts, ancestral sampling (T=1) in each schedule's own order, judged by an independently trained causal model (seed 43; self-judging is biased by the judge's own entropy). Seed-42 models, 5000 steps:

| schedule | tokens/step | judge CE | judge CE at seam |
| --- | ---: | ---: | ---: |
| real text | — | 1.97 | 2.04 |
| causal | 1.0 | 2.61 | 2.56 |
| reverse-16-g8 | 1.0 | 3.22 | 2.90 |
| **reverse-8-g8** | 1.75 | **3.26** | **2.89** |
| reverse-8 | 1.75 | 3.59 | 7.12 |
| forward-8 | 1.75 | 3.60 | 11.29 |
| delayed | 1.75 | 3.97 | 11.43 |
| landing | 7.0 | 3.74 | 6.15 |
| reverse-0 | 7.0 | 3.78 | 7.69 |
| forward-0 | 7.0 | 4.04 | 11.53 |

The seam failure of forward staggering (11.3) disappears under segment reversal (2.9). The ordering cost shows up in samples more than in likelihood (3.26 vs 2.61). At full parallelism landing and reverse-0 are tied in samples (3.74 vs 3.78) despite the likelihood gap. Cross-judged (20k-step models judged by the wide causal model; wide models judged by the 20k causal model): judge CE causal 2.34 / 2.42, reverse-8-g8 3.04 / 3.21, forward-8 3.33 / 3.49; seam 2.64 / 2.83 vs 11.29 / 11.81. Token reversal (g=1) is clearly worse in samples (3.30, seam 5.78), so segment reversal with g = L/2 is the recommended variant. Reverse samples repeat fewer 4-grams (0.005 vs 0.009 causal, 0.012 real); a temperature sweep is needed to separate entropy from disfluency.

## 6. Related work

- Staggered / wavefront parallelism: ZipAR (2412.04062), Diagonal Decoding (2503.14070), PAR (CVPR 2025), WPP in video coding.
- Line/block-parallel text: LCLM (2609.07129), LatLM (2610.23951), PSF (2610.12898).
- Block diffusion pipelines: TeCoP (2610.27583), FlowBlock (2607.17652), MultiBD (2606.29215), AsyncLane (2606.08411), Lafa (2610.28559).
- Reversal for shorter dependency lags: Sutskever et al. 2014 (1409.3215) reverse the *source* so the first target words sit close to the first source words; we reverse *within decoder blocks* so the seam tokens are written right after the tokens they depend on. Use this as the motivating analogy in the introduction.
- Generation order: order-agnostic decoding (2606.00997), information-theoretic parallel decoding (2602.00286), FIM training (Bavarian et al. 2022), semi-autoregressive NMT (EMNLP 2018).
- Speculative / multi-token: blockwise parallel decoding (Stern et al. 2018), EAGLE, MTP.

## 7. Limitations (to keep honest)

- Small scale (TinyStories, ~10M params); TODO scale-up.
- Ordering cost of reverse schedules (~0.23–0.27 nats/token) does not shrink with 3x capacity or 4x training here; untested whether a block-level latent plan removes it.
- Reverse order emits text block-wise, not token-wise: worse streaming latency for chat.
- Not compatible with pretrained left-to-right LMs without retraining or fine-tuning.
- Tokens/step is not wall-clock speed.
