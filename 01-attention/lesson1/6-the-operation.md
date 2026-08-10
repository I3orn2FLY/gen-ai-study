# 6 · The operation

*~4 min. Lesson 1 — **not written yet**.*

> **Placeholder.** Parts are written one at a time, after the previous one is right.
> Below is what this part will contain.

Part 2 batched scoring over the keys, but still one query at a time — the worst possible shape
for a GPU.

**Will cover**

- The loop rewritten as matrices: $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$
- Side-by-side with part 1's three steps — nothing new, just the loops removed
- Anchored on part 4's numbers first ($T=7$, $d=64$), then generalized to
  $(B, H, L_q, d_k)$
- `dim=-1` not `dim=-2`, and why the wrong one trains fine while being broken
- The real attention matrix as a heatmap
- $L_q \neq L_k$ and $d_k \neq d_v$ — don't assume square

*Figures ready: `fig6-three-steps.png`, `fig7-attention-heatmap.png`*
