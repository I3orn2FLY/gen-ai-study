# 1 · Where attention came from

*~3 min. Lesson 1, part 1 of 7.*

Most people say "attention replaced RNNs." That's backwards. Attention was invented **inside**
an RNN, three years before the transformer.

Worth getting right — it's a cheap way to sound like you actually read the papers.

---

## 2014 — Bahdanau: attention as a patch on RNNs

Translation models back then were encoder–decoder:

```
"the cat sat on the mat"  →  [RNN encoder]  →  [512 numbers]  →  [RNN decoder]  →  "le chat..."
```

See the problem? A 4-word sentence and a 40-word sentence both get **the same 512 numbers**.
Everything has to squeeze through that one vector. Translation quality fell off a cliff on
long sentences.

Bahdanau's fix: stop squeezing. Let the decoder look back at *every* encoder state, and take a
weighted average of them — with the weights learned.

**The RNN stayed.** Attention was an add-on, not a replacement.

> **Origin tag: Fix.** A concrete failure (the bottleneck), a targeted response.

### The two words you now need

Those learned weights come from comparing two things, and both have names:

**Query** — what the thing doing the looking is after.
Here: the decoder's current state. *"I'm about to emit a French noun, what do I need?"*

**Key** — what a thing being looked at advertises about itself.
Here: each encoder state. *"I'm the word 'cat', position 2."*

You feed a query and a key into a scoring function and get one number out: **how well do these
two match?** Big number = this is what I was looking for.

Do that for every (query, key) pair and you get a grid of match scores. Normalize each row to
sum to 1, and those are your weights.

One detail that matters later: in Bahdanau's version the encoder state is used **both** for
scoring *and* as the thing being averaged. One vector doing two jobs. Splitting those two jobs
apart is where the third name — *value* — comes from, and that's part 3.

---

## 2015 — Luong: use a dot product

Bahdanau's scoring function was a tiny neural network:

```
score(q, k) = vᵀ tanh(W[q; k])
```

Reading it right to left:

| Piece | What it does |
|---|---|
| `[q; k]` | glue the two vectors end to end — if each is length 512, this is length 1024 |
| `W` | a **learned** matrix; multiply to get a hidden vector |
| `tanh` | squash it |
| `vᵀ` | a **learned** vector; dot it down to a single number |

So: a one-hidden-layer MLP that eats a (query, key) pair and outputs one score. It has real
learned parameters (`W` and `v`), and you run it **once per pair**. 40 words attending to 40
words = 1600 tiny forward passes.

Luong tried the boring thing instead:

```
score(q, k) = qᵀk        # multiply elementwise, add it all up. one number.
```

No parameters. No hidden layer. Just a dot product — and dot products are big when two vectors
point the same way, which is exactly the "do these match?" question.

It worked about as well and was **much** faster. Why so much faster? Those 1600 dot products
aren't 1600 separate operations — stack the queries into a matrix, stack the keys into a
matrix, and **one matmul produces the entire grid of scores at once**. Matmul is the one thing
GPUs are built to do.

> **Origin tag: Empirical / efficiency.** Not more expressive. Just a better fit for the
> hardware.

Remember this pattern — it decides a lot of things later in this roadmap. FlashAttention, GQA,
and MoE all win the same way.

---

## 2017 — Vaswani: drop the RNN

"Attention Is All You Need" did **not** invent attention. It noticed something else:

*Once you have attention, the RNN is the part slowing you down.*

So they deleted it. But the RNN was quietly doing jobs nobody had noticed, and all three had
to be rebuilt:

| Removing the RNN broke | Patched with | Covered in |
|---|---|---|
| Logit scale blows up with dimension | `1/√d` scaling | part 5 |
| One attention pattern isn't enough | multi-head | lesson 2 |
| No sense of word order at all | positional encoding | lesson 5 |

That's the honest framing of the transformer: **not three good ideas, but one idea and three
repairs it forced.**

---

## The chain

```
fixed-vector bottleneck
    → attention (Bahdanau)
    → dot-product attention is cheap (Luong)
    → so the RNN can go (Vaswani)
    → which breaks 3 things
    → fix those 3 things
```

Next: why dropping the RNN was worth that trouble.

**→ [2 · Why not RNNs](2-why-not-rnns.md)**
