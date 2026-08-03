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

## Notation

Every symbol used below. Nothing here is assumed.

| Symbol | Type | What it is |
|---|---|---|
| $T_x$ | scalar | **source** length. $T_x = 3$ for "the cat sat" |
| $T_y$ | scalar | **target** length. $T_y = 4$ for "le chat s'assit" |
| $d$ | scalar | hidden width, same for both RNNs. $d = 512$ in the paper |
| $h_j$ | $\mathbb{R}^{d}$ | **encoder state** at source position $j$ |
| $s_{i-1}$ | $\mathbb{R}^{d}$ | **decoder state** just before emitting target word $i$ |
| $e_{ij}$ | $\mathbb{R}$ | the **score**: how relevant $h_j$ is to $s_{i-1}$ |
| $\alpha_{ij}$ | $\mathbb{R}$ | the score turned into a weight, $\sum_j \alpha_{ij} = 1$ |
| $c_i$ | $\mathbb{R}^{d}$ | **context vector** — the blend handed to the decoder |

Two of these are worth spelling out, because "state" is doing a lot of quiet work.

**Encoder state $h_j$.** An RNN reads the source one word at a time, carrying a vector forward
and updating it at each step. $h_j$ is that vector after reading word $j$ — a $d$-dimensional
summary of the source, as of position $j$. There are $T_x$ of them, one per source word. In the
old pipeline only $h_{T_x}$ survived; that discard *is* the bottleneck.

**Decoder state $s_{i-1}$.** The same idea on the output side: a $d$-dimensional vector holding
everything generated so far. The subscript is $i-1$ because when producing word $i$, the state
you have is the one left over from word $i-1$. It's the decoder's "where am I in this
sentence" — and it's what does the asking.

---

## Where the score is computed

The encoder runs **once**, producing $T_x$ vectors:

$$h_1,\; h_2,\; \dots,\; h_{T_x} \qquad h_j \in \mathbb{R}^{d}$$

The decoder runs **once per output word**. At step $i$ it holds $s_{i-1}$, and before emitting
anything it does three things.

**1 · Score every candidate.** One number per source position:

$$e_{ij} \;=\; \mathrm{score}\!\left(s_{i-1},\, h_j\right) \;\in\; \mathbb{R}
\qquad j = 1, \dots, T_x$$

Collect them and you have a vector $e_i \in \mathbb{R}^{T_x}$ — **these are the scores.**
With $T_x = 3$:

$$e_i = \big[\,e_{i1},\; e_{i2},\; e_{i3}\,\big] \qquad \text{shape } (3,)$$

**2 · Normalize into weights.** Scores are unbounded reals; you need them to sum to 1 so they
can act as mixing proportions. That's exactly softmax:

$$\alpha_{ij} \;=\; \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
\qquad\Longrightarrow\qquad \sum_{j=1}^{T_x} \alpha_{ij} = 1$$

The denominator sums over **all** source positions, which is what forces the row to total 1.
Concretely $\alpha_i = [0.02,\ 0.95,\ 0.03]$ — *while producing this French word, 95% of my
attention is on "cat".*

**3 · Blend.** Weighted sum of the encoder states, back in $\mathbb{R}^{d}$:

$$c_i \;=\; \sum_{j=1}^{T_x} \alpha_{ij}\, h_j \;\in\; \mathbb{R}^{d}$$

Then the decoder emits word $i$ from $s_{i-1}$ **and** $c_i$, instead of from one frozen vector.

### So where is it, exactly

Inside the decoder loop, recomputed at every step, once per source position. Step $i$ builds a
fresh $e_i$ of shape $(T_x,)$; over the whole sentence that's $T_y$ of them — a
$T_y \times T_x = 4 \times 3$ table.

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
| $W$ | $\mathbb{R}^{d_a \times 2d}$ | **learned** matrix → a hidden vector of size $d_a$ |
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

Those 12 dot products aren't 12 operations. Stack the queries as rows of
$Q \in \mathbb{R}^{T_y \times d}$ and the keys as rows of $K \in \mathbb{R}^{T_x \times d}$,
and the entire score table is **one matmul**:

$$E \;=\; Q K^{\top} \;\in\; \mathbb{R}^{T_y \times T_x}
\qquad\text{where } E_{ij} = e_{ij}$$

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
