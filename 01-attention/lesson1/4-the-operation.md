# 4 · The operation

*~4 min. Lesson 1, part 4 of 7.*

The whole thing:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

That's it. The rest of this part is reading it properly.

---

## The "soft" part

Part 3 covered what Q, K, V are. Now the other half of the metaphor: why *soft*.

A Python dict lookup is hard:

```python
d[key]          # exact match. one winner, everyone else gets nothing.
```

Attention never picks a winner:

1. Compare the query against **every** key (dot product)
2. Turn those scores into probabilities (softmax)
3. Return the **weighted average of all values**

Nothing is retrieved. Everything is blended, in proportion to how well it matched.

That's not a compromise — it's the requirement. A hard lookup has no useful derivative, so
you can't learn *what to look up*. A weighted average does. Softness is what makes the
mechanism trainable.

---

## Three steps, with shapes

Take:

```
Q: (B, H, L_q, d_k)      B = batch, H = heads
K: (B, H, L_k, d_k)      L = sequence length, d = dimension
V: (B, H, L_k, d_v)
```

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

## Two shapes that don't have to match

**`L_q` and `L_k` can differ.** Queries from one sequence, keys/values from another — that's
cross-attention. Phase 9 conditions images on text exactly this way.

**`d_k` and `d_v` can differ** too, though in practice everyone sets them equal.

Your implementation shouldn't assume either is square.

---

Next: the one part of that formula that isn't obvious.

**→ [5 · Why √d](5-why-sqrt-d.md)**
