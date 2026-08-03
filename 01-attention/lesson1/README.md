# Lesson 1 — Scaled dot-product attention

*Roadmap Phase 1, steps 1–2 · ~2 hours total · runs on CPU*

Six short parts. Read in order.

| # | Part | Time |
|---|---|---|
| 1 | [Where it came from](1-where-it-came-from.md) | ~3 min |
| 2 | [Why not RNNs](2-why-not-rnns.md) | ~4 min |
| 3 | [The operation](3-the-operation.md) | ~4 min |
| 4 | [**Why √d**](4-why-sqrt-d.md) — the one that matters | ~6 min |
| 5 | [The PyTorch you need](5-pytorch-you-need.md) | ~4 min |
| 6 | [Your task](6-your-task.md) | ~45 min doing |

~20 min reading, then you write code.

---

## The one-paragraph version

Attention is a soft dictionary lookup: score the query against every key, softmax the scores,
return the value-weighted average. `softmax(QKᵀ/√d_k)V`. The `√d_k` is there because the
variance of a `d`-term dot product is `d`, and unscaled scores saturate the softmax — which
zeroes its Jacobian, which means no gradient reaches `W_Q` and `W_K`, which means the model
can't learn where to look. It's `d_head`, not `d_model`. And it only fixes the scale at
initialization, not during training — that's what QK-norm is for.

---

## Run it

```bash
python 01-attention/check_lesson1.py
```

---

## Interview questions this covers

1. **Why is there a √d in attention?** Variance of a `d`-term dot product is `d` → unscaled
   logits saturate softmax → saturated softmax has ~zero Jacobian → no gradient to `W_Q`/`W_K`,
   at init.
2. **`d_model` or `d_head`?** `d_head`. The dot product lives inside one head.
3. **What if you divided by `d`?** Attention goes uniform. Gradients survive, selectivity dies.
4. **Why did transformers beat RNNs?** Parallelism first — `O(1)` vs `O(n)` sequential steps.
   Path length second. And attention is *more* expensive in FLOPs past `n ≈ d`; saying it's
   "more efficient" is wrong.
5. **Does √d solve softmax saturation?** No — at init only. Logits drift up during training,
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
