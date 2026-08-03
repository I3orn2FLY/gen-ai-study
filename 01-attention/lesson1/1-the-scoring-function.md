# 1 · The scoring function

*~6 min. Lesson 1, part 1 of 8.*

## The problem

Sequence-to-sequence models used to be a straight pipeline:

```
"the cat sat on the mat"  →  [RNN encoder]  →  [512 numbers]  →  [RNN decoder]  →  "le chat..."
```

Spot it? A 4-word sentence and a 40-word sentence both get **the same 512 numbers**.
Everything squeezes through one vector, and translation quality collapsed on long inputs.

The obvious fix is to stop squeezing: keep one vector per input word, and let the consumer
pick which ones it needs at each output step.

Which raises the actual question:

> **Given this position, which other positions should it pull information from?**

"Pick" isn't differentiable, so you can't learn it. What you *can* do is put a number on every
candidate — how relevant is this one to what I need right now — and blend by those numbers.

**That number is the score.** It's the whole of this lesson: how it's computed, how it becomes
weights, and how those weights mix information.

![the fixed-vector bottleneck, and attention removing it](figures/fig1-bottleneck.png)

*(Bahdanau et al., 2014 — and note the recurrence stayed. This was an addition to the RNN, not
a replacement for it. **Origin tag: Fix** — named failure, targeted response.)*

---

## Where the score is computed

Running example: translate "the cat sat" → "le chat s'assit".

### The encoder runs once

It reads the source one word at a time, carrying a vector forward and updating it each step.
Keep that vector at every position instead of only the last one:

$$h_1,\; h_2,\; h_3$$

> $h_j \in \mathbb{R}^{d}$ — the encoder's state after reading source word $j$: a summary of
> the source as of position $j$. One per source word, so $T_x = 3$ of them.
> $d$ is the width, $512$ in the paper.

Keeping all $T_x$ instead of just $h_{T_x}$ *is* the fix. Everything below is about choosing
between them.

### The decoder runs once per output word

At step $i$ it holds $s_{i-1}$ and does three things before emitting anything.

> $s_{i-1} \in \mathbb{R}^{d}$ — the decoder's state: a vector holding everything generated so
> far. The subscript is $i-1$ because when producing word $i$, the state you have is the one
> left over from word $i-1$. This is the thing doing the asking.

**1 · Score every candidate.**

$$e_{ij} \;=\; \mathrm{score}\!\left(s_{i-1},\, h_j\right) \qquad j = 1, \dots, T_x$$

> $e_{ij} \in \mathbb{R}$ — **one number.** How relevant source word $j$ is to what the decoder
> needs right now.

Collect them across $j$ and step $i$ has a vector $e_i \in \mathbb{R}^{T_x}$, here shape
$(3,)$. **These are the scores.**

**2 · Normalize into weights.** Scores are unbounded reals. To use them as mixing proportions
they have to be positive and sum to 1 — which is what softmax does:

$$\alpha_{ij} \;=\; \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

> $\alpha_{ij} \in \mathbb{R}$ — the weight on source word $j$ at step $i$. The denominator
> sums over **every** source position, which is exactly what forces $\sum_j \alpha_{ij} = 1$.

Concretely $\alpha_i = [0.02,\ 0.95,\ 0.03]$: *while producing this French word, 95% of my
attention is on "cat".*

**3 · Blend.**

$$c_i \;=\; \sum_{j=1}^{T_x} \alpha_{ij}\, h_j$$

> $c_i \in \mathbb{R}^{d}$ — the **context vector**. A weighted average of the $h_j$, so it
> lands back in the same space they live in.

The decoder emits word $i$ from $s_{i-1}$ **and** $c_i$, instead of from one frozen vector.

### So where is it, exactly

Inside the decoder loop, recomputed at every step, once per source position. Step $i$ builds a
fresh $e_i$ of shape $(T_x,)$; the sentence has $T_y = 4$ output words, so that's $4$ of them —
a $T_y \times T_x = 4 \times 3$ table over the whole translation.

And watch its lifetime: built → softmaxed into $\alpha$ → used to weight the $h_j$ → **dropped**.
Scores are never stored. That stays true in the transformer; part 3 puts these same three steps
inside a modern block.

---

## Query and key

Two names for the two sides of $\mathrm{score}(\cdot,\cdot)$.

**Query** — what the thing doing the looking is after.
Here $s_{i-1}$. *"I'm about to emit a French word, what do I need?"*

**Key** — what a thing being looked at advertises about itself.
Here each $h_j$. *"I'm the word 'cat', position 2."*

One query, one key, one number out.

Now notice: $h_j$ appears **twice** — once inside $\mathrm{score}$ (step 1) and once in the
weighted sum (step 3). One vector, two jobs: *how you get found* and *what you contribute*.
Splitting those apart produces the third name, **value**, in part 4.

---

## Two ways to compute the score

A real design choice, and the winner won for a reason worth internalizing.

### Additive — a small neural network

$$\mathrm{score}(q, k) \;=\; v^{\top} \tanh\!\big(W\,[\,q;\,k\,]\big)$$

| Piece | Type | What it does |
|---|---|---|
| $[\,q;\,k\,]$ | $\mathbb{R}^{2d}$ | concatenate — stack the two vectors into one of length $2d$ |
| $W$ | $\mathbb{R}^{d_a \times 2d}$ | **learned** matrix → a hidden vector of size $d_a$ (its own hyperparameter) |
| $\tanh$ | — | elementwise squash |
| $v^{\top}$ | $\mathbb{R}^{d_a}$ | **learned** vector; dot it down to one number |

A one-hidden-layer MLP eating a (query, key) pair. It has its own parameters $W$ and $v$, and
it runs **once per pair** — $T_y \times T_x = 12$ tiny forward passes for this short sentence.

### Multiplicative — just a dot product

$$\mathrm{score}(q, k) \;=\; q^{\top} k \;=\; \sum_{m=1}^{d} q_m k_m$$

No parameters. No hidden layer. A dot product is large when two vectors point the same way,
which is already the "do these match?" question.

### Why the dot product won

It scored about as well and is **much** faster — but not for the reason usually given.

Those 12 dot products aren't 12 operations. Stack all the queries into one matrix and all the
keys into another, and the entire score table falls out of **one matmul**:

$$E \;=\; Q K^{\top}$$

> $Q \in \mathbb{R}^{T_y \times d}$ — the $T_y$ decoder states as rows.
> $K \in \mathbb{R}^{T_x \times d}$ — the $T_x$ encoder states as rows.
> $E \in \mathbb{R}^{T_y \times d} \cdot \mathbb{R}^{d \times T_x} = \mathbb{R}^{T_y \times T_x}$,
> and $E_{ij} = e_{ij}$ — every score at once.

The additive version can't collapse that way — its $\tanh$ sits *between* the two vectors, so
there's no matrix product to factor out.

Matmul is the operation GPUs are built for. **The technique that won is the one that maps onto
the hardware, not the one with more expressive power.**

*(Luong et al., 2015. **Origin tag: Empirical / efficiency** — this was not fixing a failure.)*

This pattern decides a lot of this roadmap. FlashAttention, GQA, and MoE all win the same way.

---

## What cheap scoring unlocked

Once scoring is one matmul, the recurrence stops being the helper and becomes the bottleneck —
so it gets deleted (Vaswani et al., 2017).

But the RNN had quietly been doing three other jobs, and all three needed replacing:

| Removing the RNN broke | Patched with | Covered in |
|---|---|---|
| Score scale grows with $d$ | $1/\sqrt{d}$ scaling | part 6 |
| One attention pattern isn't enough | multi-head | lesson 2 |
| No notion of word order at all | positional encoding | lesson 5 |

The honest framing of the transformer: **not three good ideas, but one idea and three repairs
it forced.**

It also moves where scores live. No decoder loop any more — every position scores every other
simultaneously, in one matmul, inside every block. **Part 3 traces exactly where.**

---

## The chain

```
fixed-vector bottleneck
    → score every candidate, mix by weight
    → make scoring a dot product, so it's one matmul
    → which makes the RNN removable
    → which breaks 3 things
    → fix those 3 things
```

**→ [2 · Why not recurrence](2-why-not-rnns.md)**
