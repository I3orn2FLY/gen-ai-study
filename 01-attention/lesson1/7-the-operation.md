# 7 · The operation, generalized

*~5 min. Lesson 1, part 7 of 10.*

## The problem

Part 4's attention box quietly hardwired three assumptions: queries and keys came from the same
seven rows, so the table was square; one sentence at a time, no batch; one width everywhere. None
of them is part of the mechanism — part 1's translator already broke the first, scoring French
queries against English keys — and part 10 will check your implementation against a PyTorch op
(`F.scaled_dot_product_attention`) that assumes none of them.

So: the same operation, assumptions removed. Nothing new happens here — score, normalize, blend,
the three steps standing since part 1 — but the shapes come loose.

## The general form

$$\mathrm{Attention}(Q, K, V) \;=\; \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

> Three inputs, no assumptions about where they came from.
> $Q$ — $(L_q, d_k)$: $L_q$ queries.
> $K$ — $(L_k, d_k)$: $L_k$ keys, sharing the queries' width $d_k$ — a dot product needs one.
> $d_k$ is part 2's shared width $d$, renamed to say *whose* width it is.
> $V$ — $(L_k, d_v)$: one value per key, same count, width free.
> $\sqrt{d_k}$ — part 4's ÷8 in general form. Named here, explained in part 8.

The three steps, with the shape at each:

$$\underbrace{\;QK^\top\;}_{(L_q,\ L_k)} \;\longrightarrow\;
\underbrace{\;\mathrm{softmax}\;}_{(L_q,\ L_k),\ \text{rows sum to 1}} \;\longrightarrow\;
\underbrace{\;AV\;}_{(L_q,\ d_v)}$$

In: $L_q$ questions. Out: $L_q$ answers, each a $d_v$-wide blend of values. $L_k$ lives only in
the intermediate table. Read the constraints straight off the formula:

- $Q$ and $K$ **must share $d_k$** — the dot product pairs their coordinates.
- $K$ and $V$ **must share $L_k$** — key $j$ and value $j$ are one entry's label and payload
  (part 6's dict: every entry files exactly one of each).
- **Nothing ties $L_q$ to $L_k$, or $d_k$ to $d_v$.** Square-and-one-width was part 4's choice,
  not a law. ($d_k \neq d_v$ is part 6's split taken to its conclusion: the matching space and
  the payload space don't even need the same size.)

## The two cases you already own

| | $Q$ from | $K, V$ from | Table |
|---|---|---|---|
| **Self-attention** — part 4 | the sentence | the same sentence | square, $(7, 7)$ |
| **Cross-attention** — part 1, named at last | 4 decoder steps | 3 encoder states | $(4, 3)$ |

**Cross-attention** is the proper name for what part 1's translator did all along: queries from
one side, keys and values from the other. Part 2's promissory note — $E = QK^\top$,
$(T_y \times d)(d \times T_x)$ — was exactly this rectangle; the general form just stops caring
where the two sides came from.

Batching is a prefix, not a change: stack $B$ sentences ($B$ — the batch size) along a leading
dim — $Q\,(B, L_q, d_k)$ and so on — and `@` contracts the last two dims while broadcasting the
rest, so the same line of code runs the whole batch. Part 9 drills that mechanic.

![three steps, and the shape at each one](figures/fig6-three-steps.png)

## The bug that trains fine

The softmax needs to be told its dimension, and the formula's meaning fixes it: over the **keys**
— the last dim, `dim=-1`. Each query spends one unit of attention across all keys; each row of
$A$ sums to 1; each output row is a genuine weighted average of values.

Write `dim=-2` instead and nothing complains. Shapes are identical and everything stays
differentiable, so the loss still goes down. But now each *column* sums to 1: each **key** splits
one unit of budget across the queries. An output row is no longer a weighted average of anything
— its total weight is how much of the keys' budgets *that query* captured, so a query that wins
many keys gets a huge row and one that wins none gets almost nothing. The op computes something
else entirely, and nothing crashes to tell you.

Shape bugs that crash are gifts. The expensive ones train. A loss curve *alone* can't catch this
— the buggy run still descends; only a comparison can. That is why part 10's success criterion
is *match `F.scaled_dot_product_attention` to within $10^{-5}$*: an exactness check against a
reference, not an eyeball on a curve.

## What the table looks like

![attention weights on the running sentence](figures/fig7-attention-heatmap.png)

A weight table for the running sentence — constructed for illustration (the vectors are random,
with "it" nudged toward "cat"), not from a trained model, but this is exactly the object a
trained model produces. Each row is one query's distribution over keys. The left panel is the
unmasked op; the red box is the row for "it" putting 0.99 of its budget on "cat" — part 6's
directional relationship, drawn. The right panel is the same scores under part 4's causal mask:
the upper triangle is exactly zero and every surviving row still sums to 1, because the $-\infty$
went in *before* the softmax.

---

```
Attention(Q, K, V) = softmax(QKᵀ/√d_k) V
    (L_q, d_k)(d_k, L_k) → (L_q, L_k) → rows to proportions → (L_q, d_v)
    self-attention: both sides one sentence · cross-attention: two sides — same op
    softmax dim = keys. The wrong dim trains. That's what makes it dangerous
```

One symbol in that formula is still unexplained, and it's been tagging along since part 4: the
$\sqrt{d_k}$. It exists because of a failure you can see in a five-line experiment — and it's
the part to slow down for.

**→ [8 · Why √d](8-why-sqrt-d.md)**
