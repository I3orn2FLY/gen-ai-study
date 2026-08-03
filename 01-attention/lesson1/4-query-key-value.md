# 4 · Query, key, value

*~7 min. Lesson 1 — **not written yet**.*

> **Placeholder.** Parts are written one at a time, after the previous one is right.
> Below is what this part will contain.

Part 3 had three tensors appear out of nowhere. What are they, and why three?

**Will cover**

- A Python dict already has all three — and key $\neq$ value is the whole point
- Attention as that dict with three things relaxed: exact match → dot-product score,
  one winner → softmax blend, hand-written keys → learned $K = xW_K$
- Two things to un-learn: *"the key is what I need"* (no — that's the value) and
  *"key = the encoder"* (only in cross-attention)
- Where the names really come from: key-value stores → Memory Networks (which introduced the
  K/V split) → Vaswani. Deliberate, not arbitrary
- Why $W_Q \neq W_K$: sharing them makes $S = (xW)(xW)^\top$ symmetric, and
  $S_{ii} = \lVert x_i W\rVert^2$ dominates every row

*Figures ready: `fig3-dict-to-attention.png`, `fig4-qkv-projections.png`, `fig5-symmetry.png`*
