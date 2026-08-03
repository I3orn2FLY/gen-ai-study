# Section 01 — Attention and the transformer

**Roadmap:** Phase 1 · **Tier 1 (Core)** · 8 lessons

Everything downstream is this same block wearing a different hat: the DiT in Phase 10, the
CLIP text encoder in Phase 4, the video backbone in Phase 15, the VLM in Phase 16. Build it
properly once here and Phase 10 becomes a reparameterization instead of a new architecture.

It's also the densest interview material in the field.

## Lessons

Each lesson is one sitting (~1–3h), split into short parts you can read one at a time.

| # | Lesson | Status |
|---|---|---|
| 1 | [Scaled dot-product attention](lesson1/) — the operation, the √d derivation, why not RNNs | **ready** |
| 2 | Multi-head attention | not written |
| 3 | Causal masking and the AR factorization | not written |
| 4 | The block: residual stream, MLP, LayerNorm | not written |
| 5 | Position: none → sinusoidal → learned → RoPE | not written |
| 6 | Encoder-decoder → decoder-only | not written |
| 7 | Complexity and the memory-bandwidth view | not written |
| 8 | Train a GPT-2-style model on TinyShakespeare | not written |

Lessons get written one at a time (`TEACHING.md`) — how lesson N goes decides lesson N+1.

## Files

```
lesson1/           the lesson, in 6 short parts
attention.py       implementations (Kenessary)
check_lesson1.py   checks + ablation (boilerplate)
experiments/       run outputs — plots and notes, tracked
```

Run: `python 01-attention/check_lesson1.py`

Everything here runs on CPU in seconds. No GPU needed until lesson 8.

## Added to requirements

`torch`, `matplotlib` (ablation plots).

## Interview writeup

*Added when the section closes (`TEACHING.md` stage 9).*
