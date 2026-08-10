# 9 · Your task

*~3 min read, ~45 min doing. Lesson 1 — **not written yet**.*

> **Placeholder.** Parts are written one at a time, after the previous one is right.
> Below is what this part will contain.

**Will cover**

- The three functions to implement in `attention.py`: `causal_mask`,
  `scaled_dot_product_attention`, `attention_entropy` (~25–40 lines total)
- The 18 checks in `check_lesson1.py` and what each one catches
- Common mistakes written down in advance, split **Wrong** (silently bad results) vs
  **Fragile** (works here, breaks later)
- The ablation: entropy and softmax Jacobian vs head dimension, scaled / unscaled / over-scaled
- Three predictions to make *before* running it
- Deliberate sabotage: set the scale to $1$ and to $1/d$, watch the two opposite failures
