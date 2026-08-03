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

---

## 2015 — Luong: use a dot product

Bahdanau scored a **query** against a **key** with a small neural net.

*(Those two words get defined properly in part 3. For now: a query is what one position is
looking for; a key is what another position advertises about itself.)*

```
score(q, k) = vᵀ tanh(W[q; k])        # an MLP, per query-key pair
```

Luong tried the boring thing instead:

```
score(q, k) = qᵀk                     # just a dot product
```

It worked about as well and was **much** faster. Why? A whole matrix of dot products is one
matmul, and matmul is the one thing GPUs are built to do.

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
