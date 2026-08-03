# 3 · Query, key, value

*~7 min — the names, properly. Lesson 1, part 3 of 8.*

The names come from **key-value stores** — dictionaries, hash maps, databases. Not loosely.
Attention is a dictionary lookup with three specific things relaxed, and if you follow which
three, every name earns itself.

Start with an actual dict.

---

## Step 0 — a Python dict already has all three

```python
memory = {
    "cat": [0.2, 0.9, 0.1],      # ← key: "cat"   value: [0.2, 0.9, 0.1]
    "sat": [0.7, 0.1, 0.4],
    "mat": [0.3, 0.3, 0.8],
}

memory["cat"]        # ← "cat" here is the QUERY
# [0.2, 0.9, 0.1]    # ← you get back the VALUE
```

Three roles, already distinct:

| Role | In the dict | Job |
|---|---|---|
| **query** | the thing in the brackets | what I'm asking for |
| **key** | the thing on the left of `:` | the label something is filed under |
| **value** | the thing on the right of `:` | the actual content you receive |

**Look at that middle row and that bottom row.** `"cat"` is a 3-letter string. Its value is a
3-number vector. They are *not the same object*. You **search by** one and **receive** the
other.

That's the distinction to hang on to. Everything else is detail.

Now break the dict three times.

---

## Break 1 — exact match is too strict

```python
memory["kitten"]     # KeyError
```

A dict compares by equality. Miss by one character, get nothing.

**Fix:** replace equality with a **score**. Make keys vectors instead of strings, and measure
match by dot product — big when two vectors point the same way.

```
score(query, key) = q · k        # "how well do these match?", as a number
```

Now `"kitten"` scores 0.8 against `"cat"` instead of failing.

---

## Break 2 — one winner takes everything

A dict returns one value. But "kitten" is a bit like "cat" and a bit like "dog," and you'd
like some of each.

**Fix:** score against **every** key, softmax the scores into weights summing to 1, and return
the **weighted average of all values**.

```
weights = softmax([q·k₁, q·k₂, q·k₃, ...])
output  = w₁v₁ + w₂v₂ + w₃v₃ + ...
```

Nothing is retrieved. Everything is blended, in proportion to match quality.

This is also what makes it *learnable*: `d["cat"]` has no useful derivative, a weighted average
does. You can't gradient-descend your way to a better hash lookup.

---

## Break 3 — you had to write the keys by hand

In a dict, *you* decide that this vector is filed under `"cat"`.

**Fix:** learn them. Each token produces its own query, key, and value by multiplying its
embedding by a learned matrix:

```
Q = x @ W_Q        what I'm looking for
K = x @ W_K        what I'm filed under
V = x @ W_V        what I hand over if selected
```

Three learned matrices. Same input `x`, three different views of it.

![x projected into query, key and value](figures/fig4-qkv-projections.png)

**That's attention.** A dict where matching is soft, retrieval is a blend, and the index is
learned instead of written. All three breaks together:

![a dict lookup relaxed three times](figures/fig3-dict-to-attention.png)

---

## Two things worth un-learning

If you half-remember this from somewhere, these are the two spots it usually goes wrong.

### ❌ "the key is the thing I need"

No — **the key is the label, the value is the thing you need.**

This is the single most important split, and the dict makes it obvious: you look up by
`"cat"`, you receive `[0.2, 0.9, 0.1]`. Nobody wants the string `"cat"`. They want what's
filed under it.

Same in a library: you search the **card catalogue entry** (key), you walk away with the
**book** (value). Same in web search: the page is indexed by keywords (key), you read the page
(value).

### ❌ "key = the encoder"

True in one setting, and it's the setting attention was invented in — so this is a reasonable
thing to have absorbed. But it's a special case, not the definition:

| | Query from | Key & value from |
|---|---|---|
| **Self-attention** (this lesson) | the sequence | **the same sequence** |
| **Cross-attention** (Bahdanau; Phase 9) | decoder / image | encoder / text |

"Key comes from the encoder" describes *cross*-attention. In self-attention a token is both
searcher and search target.

### ✅ "query is a filter"

That one's good, keep it. A query is a request pattern: *"I want something that looks like
this."* The only thing to add is that the query is **learned and per-token**, not a filter you
write by hand.

---

## Where the names actually come from

You're right to expect names to mean something. This lineage is real:

**1970s–, key-value stores.** Hash maps, associative arrays, `SELECT ... WHERE`. Query, key,
value have meant exactly this for fifty years.

**2014 — Bahdanau.** Query and key exist here, but **there is no separate value**. The
encoder state `h_j` is used both to compute the score *and* as the thing being averaged. One
vector, two jobs.

**2014–15 — Memory Networks** (Weston; Sukhbaatar et al.). This is where the split happens.
Their memory stores each fact **twice**, under two different embeddings: one used for matching
against the query, one used for the output. Miller et al. then named the idea outright:
**"Key-Value Memory Networks"** (2016).

**2017 — Vaswani.** Inherits the vocabulary and makes all three learned projections of the
same input. The paper defines it in one line: *the output is a weighted sum of the **values**,
where each weight comes from a compatibility function of the **query** with the corresponding
**key***.

So the names are inherited, not invented — and the key/value split was a deliberate design
decision by people who had a reason for it.

**Where honesty is required:** the names describe the *mechanism's structure*, not the meaning
of any learned vector. You cannot open a trained model and read a key as "I am a plural noun."
The names tell you what each projection is *for*. Interpretations of what individual heads
learn are reverse-engineered after the fact, and often wrong.

---

## Why three matrices and not fewer?

Now the names are earned, this is answerable — and it's a standard interview question.

### Why not share W_Q and W_K?

If `W_Q = W_K = W`, the whole score matrix is:

```
S = (xW)(xW)ᵀ
```

Which is **symmetric**: `S[i,j] = S[j,i]`, always. Two problems:

**Relationships have direction.** In *"The cat sat because it was tired"*, the token **"it"**
badly needs to look at **"cat"**. "cat" has little reason to look at "it". Symmetric scores
can't express that.

**Everything would attend to itself.** The diagonal is `S[i,i] = ‖xᵢW‖²` — a squared norm,
almost always the biggest number in its row. Every token's top match would be itself.

![shared vs separate projections](figures/fig5-symmetry.png)

The left panel is the failure: a bright diagonal and almost nothing else. Every token's best
match is itself.

Separate matrices break the symmetry. That's what buys directional relationships.

### Why is V separate from K?

Because **being findable** and **being useful** are different jobs.

Back to the catalogue: a book is indexed by title and author, but that's not what you read.
Tying `K` and `V` would force every token to advertise exactly what it contains. Splitting
them lets a token be found for one reason and contribute something else — which is precisely
what Memory Networks discovered was worth doing.

---

## In this lesson you don't build the projections

So you don't go looking for them: in `attention.py`, `q`, `k`, and `v` **arrive as tensors
already**. Somebody upstream did the `x @ W_Q` part.

You implement the operation that consumes them. `W_Q`/`W_K`/`W_V` show up in lesson 2, wrapped
into multi-head attention.

---

## Sanity check

You should be able to say, without looking:

- A dict lookup with three things relaxed: soft matching, blended retrieval, learned keys
- **Key = the label you search by. Value = what you receive.** Not the same thing
- Sharing `W_Q` and `W_K` → symmetric scores → no direction, and self-attention dominates
- The names came from key-value stores via Memory Networks; the K/V split was deliberate

**→ [4 · The operation](4-the-operation.md)**
