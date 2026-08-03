# Lesson 1 — Scaled dot-product attention

*Roadmap Phase 1, steps 1–2 · ~2 hours total · runs on CPU*

Eight short parts. Read in order.

| # | Part | Time |
|---|---|---|
| 1 | [The scoring function](1-the-scoring-function.md) — what a score is and where it's computed | ~5 min |
| 2 | [Why not recurrence](2-why-not-rnns.md) | ~4 min |
| 3 | [**The forward pass**](3-the-forward-pass.md) — the whole model, real shapes, where attention sits | ~6 min |
| 4 | [Query, key, value](4-query-key-value.md) — a dict lookup with three things relaxed | ~7 min |
| 5 | [The operation](5-the-operation.md) | ~4 min |
| 6 | [**Why √d**](6-why-sqrt-d.md) — the one that matters | ~6 min |
| 7 | [The PyTorch you need](7-pytorch-you-need.md) | ~4 min |
| 8 | [Your task](8-your-task.md) | ~45 min doing |

Parts 1 and 3 both anchor on a concrete forward pass with real shapes, so no formula shows up
without a home. History is used as structure, not as the subject.

~34 min reading, then you write code. Terms get defined where they appear — no part assumes
you remember jargon from another one.

---

## The one-paragraph version

Inside every transformer block sits an attention op. It projects the input `x` `(T, d)` three
ways — `Q = x W_Q`, `K = x W_K`, `V = x W_V` — builds a `(T, T)` **score** table `S = QKᵀ/√d`,
softmaxes each row into weights, and returns `A V`: each position replaced by a weighted blend
of all positions. The scores are a temporary, rebuilt and discarded every forward pass; only
the three `W` matrices are learned. It's a Python dict lookup with three things relaxed —
matching is a dot product not equality, retrieval blends *all* values not one, and the keys are
learned not hand-written. `softmax(QKᵀ/√d_k)V`. The `√d_k` is there because the
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
