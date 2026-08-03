# 3 · Query, key, value

*~5 min. Lesson 1, part 3 of 7.*

Three names that sound profound and aren't. Let's kill the mystery before using them.

---

## The names are borrowed from databases

That's the whole metaphor:

| Database | Attention |
|---|---|
| **query** — what you're looking for | what this token wants to know |
| **key** — what's in the index | what a token advertises about itself |
| **value** — what you get back | what a token actually hands over |

You search with a **query**, it's matched against **keys**, and you receive **values**.

Vaswani introduced these names without much justification. They're a metaphor, not a
derivation. Don't over-think them.

---

## Where they actually come from

Mechanically, they're just three linear layers applied to the same input.

Say token embeddings come in as `x` of shape `(L, d_model)`:

```
Q = x @ W_Q          "what am I looking for?"
K = x @ W_K          "what am I, as a search target?"
V = x @ W_V          "what do I contribute if selected?"
```

Three learned weight matrices. That's it. Same input, three different views of it.

This is **self-attention**: Q, K, V all come from the same sequence.

In **cross-attention**, Q comes from one sequence and K, V from another — that's how Phase 9
makes an image attend to a text prompt. Same math, different source.

---

## A concrete example

Sentence: **"The cat sat because it was tired."**

Take the token **"it"**. To represent it usefully, the model needs to know what "it" refers to.

- **"it" emits a query**: *"I'm a pronoun, I need a noun that could be tired."*
- **"cat" emits a key**: *"I'm an animate singular noun."*
- Those match → high dot product → high attention weight.
- **"cat" emits a value**: the actual features about cats that get mixed into "it"'s
  representation.

The output for "it" becomes mostly "cat"'s value, plus a bit of everything else.

---

## Why three matrices and not fewer?

Good question to be asked, so here's the reasoning.

### Why not use the same matrix for Q and K?

Suppose `W_Q = W_K = W`. Then the score matrix is:

```
S = (xW)(xW)ᵀ
```

That's **symmetric**. `S[i,j] = S[j,i]`, always.

Two problems:

**1. Relationships are directional.** "it" badly wants to look at "cat". "cat" has little
reason to look at "it". A symmetric matrix can't express that.

**2. Every token would mostly attend to itself.** The diagonal is `S[i,i] = ‖xᵢW‖²`, a squared
norm — almost always the biggest number in its row. Attention would collapse to "each token
looks at itself," which is a very expensive way to do nothing.

Separate `W_Q` and `W_K` break the symmetry. That's what buys directional relationships.

### Why is V separate from K?

Because **how you get found** and **what you deliver** are different jobs.

A token might be a great search target for one reason and carry useful information for a
totally different reason. Think of a library card catalogue: the index entry ("Physics,
1687") is not the book.

Tying them would force a token to advertise exactly what it contains. Separating them lets it
be findable by one property and useful for another.

---

## For this lesson: you don't build the projections

Important, so you don't go looking for them.

In `attention.py`, `q`, `k`, and `v` **arrive as tensors already**. Somebody upstream already
did the `x @ W_Q` part.

You're implementing the operation that consumes them, nothing more.

The `W_Q`/`W_K`/`W_V` layers show up in **lesson 2**, when we wrap this in multi-head
attention. Splitting it this way keeps each piece small.

---

## Sanity check

Before moving on, you should be able to say:

- Q, K, V are three linear projections of the same input (in self-attention)
- Sharing `W_Q` and `W_K` would make attention symmetric and self-dominated
- K is the address, V is the payload
- The database names are a metaphor, not a mechanism

**→ [4 · The operation](4-the-operation.md)**
