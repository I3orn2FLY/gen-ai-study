# Section 01 — Attention and the transformer

**Roadmap:** Phase 1 · **Tier 1 (Core)** · 8 parts

The transformer is the shared substrate for everything downstream: the DiT in Phase 10, the
CLIP text encoder in Phase 4, the video backbone in Phase 15, and the VLM in Phase 16 are all
this same block under different conditioning. Building it once here means Phase 10 is a
reparameterization rather than a new architecture.

It is also the densest interview material in the field.

## Parts

| # | Part | Status |
|---|---|---|
| 1 | [Scaled dot-product attention](part1-scaled-dot-product.md) — the operation, the √d derivation, why not recurrence | in progress |
| 2 | Multi-head attention | not written |
| 3 | Causal masking and the AR factorization | not written |
| 4 | The block: residual stream, MLP, LayerNorm | not written |
| 5 | Position: none → sinusoidal → learned → RoPE | not written |
| 6 | Encoder-decoder → decoder-only | not written |
| 7 | Complexity and the memory-bandwidth view | not written |
| 8 | Train a GPT-2-style model on TinyShakespeare | not written |

Parts are written one at a time (`TEACHING.md`) — how part N goes determines part N+1.

## Files

```
attention.py       implementations (Kenessary)
check_part1.py     correctness checks + scaling ablation (boilerplate)
experiments/       run outputs — plots and notes, tracked
```

Run: `python 01-attention/check_part1.py`

Everything in this section runs on CPU in seconds; no GPU needed until part 8.

## Added to requirements

`torch`, `matplotlib` (ablation plots).

## Interview writeup

*Added at the end of the section (`TEACHING.md` stage 9).*
