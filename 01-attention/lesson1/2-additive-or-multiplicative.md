# 2 · Additive or multiplicative

*~6 min. Lesson 1, part 2 of 10.*

## The problem

Part 1 never opened up `score()`. It has a genuinely awkward job: take a decoder state and an
encoder state — two vectors made by two different networks, doing two different jobs, not even the
same length — and say how much they have to do with each other. In one number.

Two answers showed up within a year of each other. The first is the one you'd think of. The second
is the one that's still running.

---

## Additive — train a little network to judge

If you don't know how to compare two things, learn it. Stick them end to end, run them through a
small neural net, read off one number.

$$\mathrm{score}(q, k) \;=\; v^{\top} \tanh\!\big(W\,[\,q;\,k\,]\big)$$

That's a one-hidden-layer MLP. Nothing fancier.

| | Shape | |
|---|---|---|
| $q$ | $(d_s,)$ | the query — one decoder state |
| $k$ | $(d_h,)$ | the key — one encoder state |
| $[\,q;\,k\,]$ | $(d_s + d_h,)$ | glued into one long vector |
| $W$ | $(d_a,\; d_s{+}d_h)$ | **learned** |
| $\tanh(\cdot)$ | $(d_a,)$ | elementwise, shape unchanged |
| $v$ | $(d_a,)$ | **learned** |
| out | $()$ | **one number** — this is $e_{ij}$ |

> $d_a$ — how wide the hidden layer is. Your choice, nothing to do with $d_s$ or $d_h$. Bahdanau
> used 1000.

$W$ and $v$ are the only parameters, and one copy is shared by every step, every position, every
sentence.

### Gluing them is the same as projecting them separately

Worth seeing, because it's how you'd actually implement it. Cut $W$ down the middle,
$W = [\,W_q \mid W_k\,]$. Then

$$W\,[\,q;\,k\,] \;=\; W_q\,q \;+\; W_k\,k$$

Same arithmetic, regrouped — glue-then-multiply is multiply-separately-then-add. Bahdanau writes
it the second way, and now the code is obvious ($H$ is the encoder states stacked as rows;
`s_prev` is the query $s_{i-1}$ — part 1's rule, the state finished last step):

```python
K_proj = H @ W_k.T          # (T_x, d_h) @ (d_h, d_a)  ->  (T_x, d_a)
q_proj = W_q @ s_prev       # (d_a, d_s) @ (d_s,)      ->  (d_a,)
Z      = torch.tanh(q_proj + K_proj)    # broadcast     ->  (T_x, d_a)
e_i    = Z @ v              # (T_x, d_a) @ (d_a,)      ->  (T_x,)
```

Two freebies. **`K_proj` doesn't change during decoding**, so you compute it once before the loop —
section 03 rediscovers that idea for a much bigger prize. And **the two sides can be different
widths**, since $W_q$ and $W_k$ only have to agree on what comes *out*. Bahdanau needed that:
1000-wide decoder, 2000-wide encoder.

---

## Multiplicative — just multiply them

$$\mathrm{score}(q, k) \;=\; q^{\top} k \;=\; \sum_{m=1}^{d} q_m k_m$$

Pair up the numbers, multiply, add everything. No network, no parameters, nothing to train.

**What that actually measures:** think of the two vectors as arrows. The dot product is large and
positive when they point the same way, near zero when they're unrelated, negative when they point
opposite ways. It's an alignment meter.

> $q_m$, $k_m$ — the $m$-th number in each vector.
> $d$ — the width they **share**. Writing this down at all forces them to have one.

So the whole scoring function becomes one line, `e_i = K @ q`, where $K$ is the keys as rows.

And it should bother you. In the additive version, $W$ was the part that knew how a French decoder
state relates to an English encoder state. Delete it and you're just hoping the two networks
happen to produce arrows that line up. Why on earth would they?

### Two reasons it isn't crazy

**One: you can't always do it anyway.** Multiplying pairwise means both vectors need the same
length. Bahdanau's don't — 2000 against 1000 — so the dot product isn't even an option in his
model. Luong's (Luong et al., 2015 — the second of the two papers, a year after Bahdanau) is built
the other way: stacked LSTMs, 4 layers of 1000 cells on both sides. Same size by design — and both
stacks train jointly against one loss, so they're not strangers to begin with. **Everything below
is Luong's setup.**

**Two: the networks learn to line up.** Nobody told them to point the same way — but the loss does.
When the model should have looked at word 2 and didn't, the gradient's instruction is literally
*"swing the query toward $h_2$, and swing $h_2$ toward the query"*:

$$\frac{\partial e_{ij}}{\partial q} \;=\; k, \qquad \frac{\partial e_{ij}}{\partial k} \;=\; q$$

Those are directions in the two state spaces, so the correction lands in the encoder and decoder
weights. Do that a few million times and the two networks end up in a shared geometry where
*relevant* just means *pointing the same way*.

> The general version: you need a **learned** comparison when you can't change the things being
> compared — frozen features, someone else's pretrained encoder. When both sides are still
> trainable, the comparison itself can be fixed.

### Luong wasn't sure either

He didn't propose the dot product as *the* answer. He proposed three and measured them:

| | Form | |
|---|---|---|
| dot | $q^{\top} k$ | no parameters |
| general | $q^{\top} W_a\, k$ | a learned matrix in between — the obvious hedge |
| concat | $v_a^{\top}\tanh(W_a[\,q;k\,])$ | Bahdanau's form |

> Luong's paper writes $W_a, v_a$ where we wrote $W, v$ — and `general`'s $W_a$ is $(d, d)$ while
> `concat`'s is $(d_a, 2d)$: same letter in the paper, two different matrices.

The middle one is exactly the worry above, patched: if the two spaces might not line up, put
something learnable between them. It shipped alongside the other two.

**None of them won on quality.** Dot did better in one setting, general in another, and `concat`
— Bahdanau's own — underperformed; the paper's own comment is that it "does not yield good
performances and more analysis should be done."

That's the whole empirical basis. Nobody proved a bare dot product was enough. It did about as
well and cost nothing, so it stuck.

*(Luong et al., 2015. **Origin tag: Empirical / efficiency** — not a fix for a failure. Given
that table, the tidy line you'll hear today — "the dot product is the natural similarity
measure" — is confidence the experiments didn't have.)*

---

## Why the dot product won anyway

Not on quality — on cost. It's cheaper, and not by a little.

**No parameters.** Additive carries $W$ and $v$ — about two million numbers to store, initialize,
compute gradients for, and update every single batch. The dot product carries **zero**.

**Much less arithmetic.** For 50 words into 50, with everything 1000 wide, counting
multiply-adds (the tanh and the additions ride along and don't change the story):

| | | multiply-adds |
|---|---|---|
| additive | project the keys | 50 M |
| | project the queries | 50 M |
| | score all pairs (the $v$ dot) | 2.5 M |
| | **total** | **102.5 M** |
| dot product | all of it | **2.5 M** |

Roughly **40× the work** for the same 2500 numbers. Look where it goes: almost all of it is the
projections. That's not an implementation detail you could optimize away — it *is* additive's idea.
"Map both sides into a shared space, then compare there" costs two big matrix multiplies. The dot
product skips straight to comparing.

**A third, smaller advantage.** Additive has to hold a $(T_x, d_a)$ scratch tensor on the way to
$T_x$ scores, because the $\tanh$ sits *between* $q$ and $k$. The dot product holds nothing.

![what each scoring function materializes](figures/fig13-additive-vs-dot.png)

**The one that won is the one that fits the hardware, not the one with more expressive power.**
That pattern decides a lot of this roadmap.

### What it cost

**Expressiveness.** Additive runs the pair through a nonlinearity, so it can express relationships
a plain sum of products can't reach. It lost on cost, not on quality — don't let the outcome
convince you it was also the better function.

**Scale.** Adding up $d$ products means the scores get bigger as the model gets wider, and softmax
misbehaves when its inputs are spread too far apart. Additive doesn't have this problem — $\tanh$
keeps things bounded. This one is a real bomb and it goes off later: **part 8** is where it gets
diagnosed and defused, and it's the part to slow down for.

---

## The prize you can't collect yet

One decoder step: one query against all $T_x$ keys, so scoring is a matrix times a vector.

But if you had **all** the queries at once, every score in the whole translation would be a single
matrix multiply:

$$E \;=\; Q K^{\top}, \qquad (T_y \times d)(d \times T_x) \;=\; (T_y \times T_x)$$

> $Q$ — every decoder state as a row. $E_{ij} = e_{ij}$, the entire score table.

No loop, no scratch tensor — the single thing GPUs are best at. But you don't have all the queries.
$s_i$ needs $c_i$ needs $s_{i-1}$, so they show up one at a time and $Q$ never exists.

**A promissory note.** The only way to cash it is to delete the recurrence.

**→ [3 · Why not recurrence](3-why-not-rnns.md)**
