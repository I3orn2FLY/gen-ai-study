# 9 · The PyTorch you need

*~4 min. Lesson 1 — **not written yet**.*

> **Placeholder.** Parts are written one at a time, after the previous one is right.
> Below is what this part will contain.

Every line of part 5's attention box, as an idiom. Not the solution — the vocabulary it's
written in.

**Will cover**

- Batched matmul: `q @ k.transpose(-2, -1)`, and why not `.T` or `permute`
- The `einsum` equivalent, since papers use it constantly
- Softmax dim, and how the wrong one fails silently
- `masked_fill` before softmax; why `-inf` not `0`, and why not `-1e9` in fp16
- **The polarity trap**: `masked_fill` True = remove, `F.scaled_dot_product_attention`
  True = keep
- Fully-masked rows: `NaN` by hand, `0.0` from the fused kernel
- `view` vs `reshape` and contiguity
