# 2 · Additive or multiplicative

*~5 min. Lesson 1, part 2 of 10.*

## The problem

Part 1 left a hole: $e_{ij} = \mathrm{score}(s_{i-1}, h_j)$ was never opened up. It has a hard job
— eat two vectors that aren't even the same width ($d_s = 1000$, $d_h = 2000$) and return one
number saying how well they go together.

Two answers arrived within a year of each other. One is more general. The other is in every model
you'll ever load, and the reason isn't "it worked better".

---

## Additive — a small neural network

$$\mathrm{score}(q, k) \;=\; v^{\top} \tanh\!\big(W\,[\,q;\,k\,]\big)$$

One pair in, one scalar out — same type as part 1, so no $T_x$ anywhere yet.

| | Shape | |
|---|---|---|
| $q$ | $(d_s,)$ | the query — one decoder state |
| $k$ | $(d_h,)$ | the key — one encoder state |
| $[\,q;\,k\,]$ | $(d_s + d_h,)$ | stacked end to end |
| $W$ | $(d_a,\; d_s{+}d_h)$ | **learned** |
| $\tanh(W[q;k])$ | $(d_a,)$ | elementwise |
| $v$ | $(d_a,)$ | **learned** |
| $v^{\top}\tanh(\cdot)$ | $()$ | **scalar** — this is $e_{ij}$ |

> $d_a$ — how wide the hidden layer is. A knob on the scoring function, unrelated to $d_s$ or
> $d_h$. Bahdanau used 1000.

A one-hidden-layer MLP eating a (query, key) pair. $W$ and $v$ are its only parameters, shared
across every step, position and sentence.

### The concatenation is secretly two projections

Split $W$ down the middle: $W = [\,W_q \mid W_k\,]$, shapes $(d_a, d_s)$ and $(d_a, d_h)$. Then

$$W\,[\,q;\,k\,] \;=\; W_q\,q \;+\; W_k\,k$$

Concatenate-then-multiply **is** multiply-separately-then-add. Bahdanau writes it the second way,
and batching over the keys becomes obvious ($H \in \mathbb{R}^{T_x \times d_h}$ is the encoder
states as rows):

```python
K_proj = H @ W_k.T          # (T_x, d_h) @ (d_h, d_a)  ->  (T_x, d_a)
q_proj = W_q @ s_prev       # (d_a, d_s) @ (d_s,)      ->  (d_a,)
Z      = torch.tanh(q_proj + K_proj)    # broadcast     ->  (T_x, d_a)
e_i    = Z @ v              # (T_x, d_a) @ (d_a,)      ->  (T_x,)
```

Two things fall out. **`K_proj` doesn't depend on the step** — encoder states don't change while
decoding, so project the keys once before the loop and reuse them. Section 03 hits the same idea
again for a much bigger payoff. And **query and key never have to be the same width**: $W_q$ and
$W_k$ only have to agree on their *output* size. Bahdanau leaned on that.

---

## Multiplicative — just a dot product

$$\mathrm{score}(q, k) \;=\; q^{\top} k \;=\; \sum_{m=1}^{d} q_m k_m$$

> $d$ — **the width $q$ and $k$ share.** Writing this at all forces them to have one. Part 1's
> $d_s \neq d_h$ is exactly the case where no such $d$ exists.

No parameters at all. Batched over the keys it's one line, `e_i = K @ q`, with
$K \in \mathbb{R}^{T_x \times d}$ the keys as rows — the job $H$ did above, renamed because we're
writing the general form now.

Which should bother you. In the additive form the learned $W$ was where "how do these two relate"
lived. Delete it and you're asking two vectors, made by two different networks doing two different
jobs, to line up under a plain sum of products. Why would they?

### Why it isn't absurd

**It doesn't even typecheck everywhere.** Bahdanau's model **fails**: 2000-wide encoder states
against a 1000-wide decoder. You can't drop a dot product into his architecture at all.

Luong built differently — stacked LSTMs, 4 layers of 1000 cells on both sides, decoder seeded from
the encoder's final state. So $d = 1000$ everywhere: same width by design, shared origin, and the
two spaces aren't strangers. **Everything below is his architecture, not Bahdanau's.**

**And the relation is still learned — by the RNNs.**

$$\frac{\partial e_{ij}}{\partial q} \;=\; k, \qquad \frac{\partial e_{ij}}{\partial k} \;=\; q$$

Say the model should have attended to word 2 and didn't. The loss pushes $e_{i2}$ up, and the
instruction coming back is: **move the query toward $h_2$, move $h_2$ toward the query.** Those are
directions in the state spaces, so they flow into the encoder and decoder weights. Over training,
one loss shapes both networks into a geometry where *relevant* means *aligned*.

> You need a learned comparison when you **can't change** the things being compared — frozen
> features, someone else's encoder. When both sides are trainable, the comparison can be fixed.

### Luong wasn't sure either

He proposed **three** scoring functions and measured them:

| | Form | |
|---|---|---|
| dot | $q^{\top} k$ | no parameters |
| general | $q^{\top} W_a\, k$ | $W_a \in \mathbb{R}^{d \times d}$ — a learned matrix in between |
| concat | $v_a^{\top}\tanh\!\big(W_a[\,q;k\,]\big)$ | Bahdanau's form. Different $W_a$: $(d_a, 2d)$, plus $v_a \in \mathbb{R}^{d_a}$ |

The middle row is the obvious hedge against the worry above — if the spaces might not line up, put
a learned matrix between them. It shipped as a peer of the other two.

**No form won.** *Dot* suited one attention variant, *general* another, and `concat` — Bahdanau's
own form — underperformed in a way he flagged as suspicious rather than settled.

That's the whole empirical basis. Nobody proved a bare inner product was enough; it performed
comparably and cost nothing, so it survived.

*(Luong et al., 2015. **Origin tag: Empirical / efficiency** — not a fix for a failure. The tidy
story you'll hear, "the dot product is the natural similarity measure", was attached afterwards.)*

---

## Why the dot product won

Not for the usual reason. Additive batches over keys fine — the code above is one broadcast add.
The difference is what each has to **materialize** on the way to those $T_x$ numbers.

![what each scoring function materializes](figures/fig13-additive-vs-dot.png)

| | intermediate, per step |
|---|---|
| additive | $Z$, shape $(T_x, d_a)$ — then reduced to $(T_x,)$ |
| dot product | none — scores fall straight out |

At $d_a = 1000$, additive touches a thousand values for every score it produces, and $\tanh$ is
elementwise work no matmul kernel can swallow. Over a whole translation, $T_y$ steps:

$$\underbrace{T_y \cdot T_x \cdot d_a}_{\text{additive}} \qquad\text{versus}\qquad
\underbrace{T_y \cdot T_x}_{\text{dot product}}$$

A factor of $d_a$. For 50 words into 50: $50 \cdot 50 \cdot 1000 = 2.5$ million values against
2500. Same scores either way.

The $\tanh$ sitting *between* $q$ and $k$ forces that extra dimension — and it's why there's no
matrix product to factor out of the additive form. The two vectors never meet as a product at all.

**The technique that won is the one that fits the hardware, not the one with more expressive
power.** And note which way it points: every later architecture has more positions attending to
more positions, so $T_y \cdot T_x$ only grows.

### What it cost

**Expressiveness.** $v^{\top}\tanh(W_q q + W_k k)$ is *nonlinear* in the pair; $q^{\top}k$ is a sum
of products. Additive can express interactions the dot product can't reach. It lost on cost, not
quality.

**Scale.** With roughly independent, zero-mean, unit-variance components,
$\mathrm{Var}(q^{\top}k) = d$, so scores have standard deviation $\sqrt{d}$ — growing with the
width. At $d = 1000$ that's a spread of about $\pm 31$, wide enough that softmax collapses onto one
position and stops passing gradient back. Additive is immune: $\tanh$ bounds its input.

Luong didn't diagnose it this way, and it isn't why he preferred one form over another. It gets
found and patched later — part 8, the one to slow down for.

---

## The prize you can't collect yet

At one step you hold one query and all $T_x$ keys, so it's a matrix–**vector** product,
$e_i = K s_{i-1}$. But hold **all** the queries at once and every score in the translation is a
single matmul:

$$E \;=\; Q K^{\top}, \qquad (T_y \times d)(d \times T_x) \;=\; (T_y \times T_x)$$

> $Q \in \mathbb{R}^{T_y \times d}$ — every decoder state as a row. $E_{ij} = e_{ij}$.

No loop, no intermediate — exactly what GPUs are built for. But you don't hold them: $s_i$ needs
$c_i$ needs $s_{i-1}$, so the queries arrive one at a time and $Q$ never exists as a matrix.

**A promissory note.** The only way to cash it is to delete the recurrence.

**→ [3 · Why not recurrence](3-why-not-rnns.md)**
