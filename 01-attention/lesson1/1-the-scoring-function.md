# 1 · The scoring function

*~5 min. Lesson 1, part 1 of 8.*

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

Concretely, with shapes. Translate **"the cat sat"** (3 words) → **"le chat s'assit"**
(4 words), `d = 512`.

**The encoder runs once,** one vector per source word:

```
h₁, h₂, h₃           each (512,)      "the", "cat", "sat"
```

**The decoder runs one step per output word.** At step `i` it holds state `s_{i-1}`, also
`(512,)`. Before emitting anything:

```
1.  compare its state against every encoder vector
      e_i1 = score(s_{i-1}, h₁)     → one number
      e_i2 = score(s_{i-1}, h₂)     → one number
      e_i3 = score(s_{i-1}, h₃)     → one number
    e_i = [e_i1, e_i2, e_i3]          (3,)      ←←← THE SCORES

2.  turn them into weights that sum to 1
    α_i = softmax(e_i)                (3,)      e.g. [0.02, 0.95, 0.03]

3.  mix the encoder vectors with those weights
    c_i = α_i1·h₁ + α_i2·h₂ + α_i3·h₃  (512,)   the "context vector"

4.  emit French word i using s_{i-1} AND c_i
```

**That's the location.** Inside the consumer's loop, recomputed every step, once per candidate.
Step `i` produces a fresh `(3,)` score vector; four output words give four of them — a `(4, 3)`
table over the sentence.

`α_i = [0.02, 0.95, 0.03]` reads as: *while producing this French word, 95% of my attention is
on "cat".*

Watch what happens to the scores: built → softmaxed → used to mix → **dropped**. Never stored.
That stays true in the transformer, and part 3 puts these same three steps inside a modern
block.

---

## Query and key

Two names for the two sides of the comparison.

**Query** — what the thing doing the looking is after.
Here: `s_{i-1}`, the decoder state. *"I'm about to emit a French word, what do I need?"*

**Key** — what a thing being looked at advertises about itself.
Here: each `h_j`. *"I'm the word 'cat', position 2."*

One query + one key → the scoring function → one number.

Notice `h_j` is doing **two jobs**: it's what gets scored against, *and* it's what gets
averaged. Splitting those jobs apart produces the third name — *value* — in part 4.

---

## Two ways to compute the score

A real design choice, and the winner won for a reason worth internalizing.

### Additive — a small neural network

```
score(q, k) = vᵀ tanh(W[q; k])
```

| Piece | What it does |
|---|---|
| `[q; k]` | glue the two vectors end to end — 512 + 512 = length 1024 |
| `W` | a **learned** matrix; multiply to get a hidden vector |
| `tanh` | squash it |
| `vᵀ` | a **learned** vector; dot it down to a single number |

A one-hidden-layer MLP eating a (query, key) pair. It has its own learned parameters and runs
**once per pair** — 4 output words × 3 source words = 12 tiny forward passes for this one
short sentence.

### Multiplicative — just a dot product

```
score(q, k) = qᵀk        # multiply elementwise, add it up. one number.
```

No parameters. No hidden layer. A dot product is large when two vectors point the same way,
which is already the "do these match?" question.

### Why the dot product won

It scored about as well and is **much** faster — but not for the reason usually given.

Those 12 dot products aren't 12 operations. Stack the queries into a matrix, stack the keys
into a matrix, and **one matmul produces the entire `(4, 3)` score table at once**. The
additive version can't collapse that way: its nonlinearity sits *between* the two vectors.

Matmul is the operation GPUs are built for. So the technique that won is the one that maps
onto the hardware — **not** the one with more expressive power.

*(Luong et al., 2015. **Origin tag: Empirical / efficiency** — this was not fixing a failure.)*

**This pattern decides a lot of this roadmap.** FlashAttention, GQA, and MoE all win the same
way. Recognizing it is worth more than any single one of them.

---

## What cheap scoring unlocked

Once scoring is one matmul, the recurrence stops being the helper and becomes the bottleneck —
so it gets deleted (Vaswani et al., 2017).

But the RNN had quietly been doing three other jobs, and all three needed replacing:

| Removing the RNN broke | Patched with | Covered in |
|---|---|---|
| Score scale blows up with dimension | `1/√d` scaling | part 6 |
| One attention pattern isn't enough | multi-head | lesson 2 |
| No sense of word order at all | positional encoding | lesson 5 |

The honest framing of the transformer: **not three good ideas, but one idea and three repairs
it forced.**

It also moves where scores live. No decoder loop any more — every position computes its scores
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
