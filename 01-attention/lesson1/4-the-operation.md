# 4 · The operation

*~4 min. Lesson 1, part 4 of 8.*

The whole thing:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

That's it. The rest of this part is reading it properly.

---

## Same three steps as part 3, now batched

Part 3 built this one query at a time: score against every key, softmax, weighted average of
values. The formula is that, done for all queries at once with matrices instead of loops.

---

## Three steps, with shapes

Take:

```
Q: (B, H, L_q, d_k)      B = batch, H = heads
K: (B, H, L_k, d_k)      L = sequence length, d = dimension
V: (B, H, L_k, d_v)
```

![the three steps and their shapes](figures/fig6-three-steps.png)

**Step 1 — scores.**

```
S = Q Kᵀ / √d_k                    → (B, H, L_q, L_k)
```

`S[b, h, i, j]` = how much query `i` wants key `j`.

**Step 2 — weights.**

```
A = softmax(S, dim=-1)             → (B, H, L_q, L_k)
```

Each row now sums to 1.

**Step 3 — output.**

```
O = A V                            → (B, H, L_q, d_v)
```

---

## `dim=-1`. Not `dim=-2`.

Softmax goes over the **key** dimension.

Think of it as: each query has one unit of attention to spend, and it splits that across the
keys.

`dim=-2` would make keys compete against each other across different queries. That quantity
means nothing. The rows won't sum to 1.

**And it will not crash.** The loss still goes down a bit. The model is just quietly broken.
This is the single most common attention bug — a one-character typo.

---

## What it looks like

Real attention weights for *"The cat sat because it was tired"* — `it` finds `cat`:

![attention heatmaps, unmasked and causal](figures/fig7-attention-heatmap.png)

Every **row** is one query's distribution over the keys, and every row sums to 1. The right
panel is the causal mask from lesson 3 — the upper triangle is exactly zero, so no token can
see the future.

## Two shapes that don't have to match

**`L_q` and `L_k` can differ.** Queries from one sequence, keys/values from another — that's
cross-attention. Phase 9 conditions images on text exactly this way.

**`d_k` and `d_v` can differ** too, though in practice everyone sets them equal.

Your implementation shouldn't assume either is square.

---

Next: the one part of that formula that isn't obvious.

**→ [5 · Where the scores actually live](5-where-scores-live.md)**
