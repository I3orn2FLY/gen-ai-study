# Lesson 1 — Scaled dot-product attention

*Roadmap Phase 1, steps 1–2 · ~2 hours total · runs on CPU*

Eight short parts. Read in order.

| # | Part | Time |
|---|---|---|
| 1 | [Where it came from](1-where-it-came-from.md) | ~3 min |
| 2 | [Why not RNNs](2-why-not-rnns.md) | ~4 min |
| 3 | [**Query, key, value**](3-query-key-value.md) — a dict lookup with three things relaxed | ~7 min |
| 4 | [The operation](4-the-operation.md) | ~4 min |
| 5 | [Where the scores actually live](5-where-scores-live.md) — real shapes, end to end | ~6 min |
| 6 | [**Why √d**](6-why-sqrt-d.md) — the one that matters | ~6 min |
| 7 | [The PyTorch you need](7-pytorch-you-need.md) | ~4 min |
| 8 | [Your task](8-your-task.md) | ~45 min doing |

~34 min reading, then you write code. Terms get defined where they appear — no part assumes
you remember jargon from another one.

---

## The one-paragraph version

Attention is a Python dict lookup with three things relaxed: matching is a dot product instead
of equality, retrieval is a blend of *all* values instead of one, and the keys are learned
(`x @ W_K`) instead of written by hand. Query = what you're asking for, key = the label a token
is filed under, value = what it hands over — key and value are different objects, which is the
whole point. `softmax(QKᵀ/√d_k)V`. The `√d_k` is there because the
variance of a `d`-term dot product is `d`, and unscaled scores saturate the softmax — which
zeroes its Jacobian, which means no gradient reaches `W_Q` and `W_K`, which means the model
can't learn where to look. It's `d_head`, not `d_model`. And it only fixes the scale at
initialization, not during training — that's what QK-norm is for.

---

## Figures

All diagrams are generated, not stock — `python 01-attention/make_figures.py` rebuilds
`lesson1/figures/`. Edit the script if a diagram is wrong or unclear.

## Run it

```bash
python 01-attention/check_lesson1.py
```

---

## Interview questions this covers

1. **Why are Q and K separate matrices?** Sharing them makes the score matrix symmetric —
   relationships would lose direction — and the diagonal `‖xW‖²` would dominate every row, so
   every token would mostly attend to itself.
2. **Self-attention vs cross-attention?** Same operation; only where Q, K, V come from
   changes. Self → square score matrix, cross → rectangular `(T_q, T_k)`.
3. **Why is there a √d in attention?** Variance of a `d`-term dot product is `d` → unscaled
   logits saturate softmax → saturated softmax has ~zero Jacobian → no gradient to `W_Q`/`W_K`,
   at init.
4. **`d_model` or `d_head`?** `d_head`. The dot product lives inside one head.
5. **What if you divided by `d`?** Attention goes uniform. Gradients survive, selectivity dies.
6. **Why did transformers beat RNNs?** Parallelism first — `O(1)` vs `O(n)` sequential steps.
   Path length second. And attention is *more* expensive in FLOPs past `n ≈ d`; saying it's
   "more efficient" is wrong.
7. **Does √d solve softmax saturation?** No — at init only. Logits drift up during training,
   which is what QK-norm and logit soft-capping are for.

---

## Papers

- **Bahdanau, Cho, Bengio (2014)** — *NMT by Jointly Learning to Align and Translate.*
  Attention's origin, inside an RNN. §3.1 is the mechanism.
- **Luong, Pham, Manning (2015)** — *Effective Approaches to Attention-based NMT.*
  Dot-product vs additive, compared head-to-head.
- **Vaswani et al. (2017)** — *Attention Is All You Need.* §3.2.1 and **footnote 4** for the
  scaling. **Table 1** for part 2's comparison.
- Optional: Lilian Weng, *Attention? Attention!* — good consolidation read, **after** you
  implement it.
