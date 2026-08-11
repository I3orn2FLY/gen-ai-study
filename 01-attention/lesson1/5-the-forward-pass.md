# 5 · The forward pass — where attention sits

*~6 min. Lesson 1 — **not written yet**.*

> **Placeholder.** Parts are written one at a time, after the previous one is right.
> Below is what this part will contain.

Before any formula: where does this thing actually live?

**Will cover**

- Your `(T, vocab) → (T, feat) → (T, vocab2)` sketch, graded — one-hot vs embedding lookup,
  `feat_num` $= d_{model}$, and why source and target lengths differ
- A concrete decoder-only GPT: *"The cat sat because it was tired"*, $T = 7$, $d = 64$,
  2 blocks. Every shape through the stack
- Zooming into the attention box: the $(7,7)$ score table, what its rows and columns mean
- What happens to the scores afterwards — discarded, rebuilt every block, every forward pass.
  Only $W_Q, W_K, W_V$ persist
- Sizes made real: $T = 2048$ → 4.2M score entries per block per pass
- Self vs cross attention: one mechanism, three wirings
- **Causal masking** — part 4 showed the decoder's recurrence enforced "no looking forward" for
  free, and that attention has to be told. Padding masks alongside it

*Figure ready: `figures/fig11-forward-pass.png`*
