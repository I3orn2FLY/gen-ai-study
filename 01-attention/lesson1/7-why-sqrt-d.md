# 7 · Why √d

*~6 min — the one to slow down for. Lesson 1 — **not written yet**.*

> **Placeholder.** Parts are written one at a time, after the previous one is right.
> Below is what this part will contain.

Part 4 walked past one line: `scores = scores / √64`. Why is it there?

**Will cover**

- The derivation: $\mathbb{E}[q \cdot k] = 0$ and $\mathrm{Var}(q \cdot k) = d$, so
  $\mathrm{std} = \sqrt{d}$ — score scale grows with head width automatically
- Why large scores are a failure, not confidence: the softmax Jacobian
  $\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$ goes to **zero** when $p$ is
  near one-hot → no gradient reaches $W_Q, W_K$, at initialization
- Dividing by the standard deviation, not the variance — and what dividing by $d$ does instead
  (uniform attention: gradients survive, selectivity dies)
- $\sqrt{d_{head}}$, not $\sqrt{d_{model}}$
- **What it doesn't fix**: only the scale at init. Logits drift during training anyway, which
  is what QK-norm and logit soft-capping are for

*Figures ready: `fig8-saturation.png`, `fig9-softmax-jacobian.png`*
