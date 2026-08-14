# 5 · What deleting it cost

*~5 min. Lesson 1, part 5 of 10.*

## The problem

Part 4's model runs and trains, every position at once. Looks finished.

But part 3 deleted the recurrence *knowing* it was doing three jobs, and nobody has gone back to
check the wreckage against the model that actually got built. This part is that audit. Every cost
points at a tensor you have already traced — scaled up to real lengths, but never to machinery
you haven't met.

Start with the claim you'll hear most often, which is false.

---

## Attention is not cheaper

Vaswani's Table 1, for one layer — *layer* in part 3's sense, the attention op alone against one
recurrent layer, not part 4's whole block. He writes $n$ where we write $T$, and gives both
architectures one shared width $d$:

| | Work per layer | Steps in order | Longest path |
|---|---|---|---|
| Self-attention | $T^2 d$ | $1$ | $1$ |
| Recurrent | $T d^2$ | $T$ | $T$ |

Look at the first column and ask when one beats the other:

$$T^2 d \;<\; T d^2 \qquad\Longleftrightarrow\qquad T \;<\; d$$

**Attention is cheaper only while the sentence is shorter than the model is wide.** Part 4's toy
was safely inside that: $T = 7$ against $d = 64$. Real settings aren't — at $d = 512$ and
$T = 1024$, attention does about **twice** the arithmetic of the RNN it replaced.

So "transformers won because attention is more efficient" is wrong, and a good way to lose an
interview. The two ideas to hold apart have names:

> **Total work** — how much arithmetic gets done: the left column. Attention does *more* past
> $T \approx d$.
> **Critical path** — how much of it has to wait in a queue: the middle column, $1$ against $T$.
> This is part 3's throughput argument, and it is the whole victory.
> The right column is part 3's *path length* — any position to any other in one hop. Real, and
> part 3 already priced it: a bonus, not the decider.

A GPU would much rather do twice the work in one go than half the work in a queue. Part 3 argued
this from inside Bahdanau's decoder; Table 1 is the same argument with the deletion done. (And no
conflict with part 2's "the dot product is cheaper" — that compared two scoring functions; this
compares attention against the recurrence it replaced.)

---

## Which of the three jobs survive

Part 3's list, checked off against the built model:

| What recurrence gave you | Survives? | |
|---|---|---|
| It knew word order | **no** | proven below, on part 4's own ops |
| Any length, same memory | **no** | counted below |
| Length doesn't change the parameter count | **yes** | read part 4's shapes |

The survivor first, because it's what made the deletion viable at all. Walk part 4's parameters:
the embedding table $(1000, 64)$, the three projections at $(64, 64)$, the output projection
$(64, 1000)$ — and the MLP and LayerNorm weights, which act on **one row at a time**, so their
shapes come from widths you chose. The only ops that touch two positions, $QK^\top$ and $AV$,
carry no weights at all.

**$T$ appears in none of those shapes.** Train on 7-word sentences, run on 700-word ones — same
weights. If this job had broken too, short-sentence training would never transfer and the whole
design would be dead on arrival. That it holds is what made losing the other two survivable.

One exception, and it's instructive: part 4's **position table** stores one learned vector per
position, so a maximum length *is* baked into that single tensor — train it at length 7 and
position 8 has no vector. The attention machinery is length-blind; the patch chosen for word
order isn't. Position signals that are *computed* instead of stored close that gap — part of
what lesson 5 weighs.

### Memory stops being flat

Running forward, an RNN holds one state of fixed width no matter how long the sentence — that was
job two. Part 4's model instead holds *everything at once*, and two different things grow:

| Tensor | Size | Growth |
|---|---|---|
| $K$ (likewise $Q$, $V$, $x$) | $T \times d$ | **linear** — every position stays reachable |
| $E$ (likewise $A$) | $T \times T$ | **quadratic** — one score per pair |

At part 4's $T = 7$ that's 49 score entries — nothing. At $T = 2048$ it's $2048^2 =$ **4,194,304
entries — each, for $E$ and again for $A$ — per block, per forward pass**, and blocks stack. This
is why the **context window** — the longest input a model can take at once — is a hard limit every
model ships with, and why it stayed small for years.

*(Part 1 called scores throwaway, and they are — never parameters, never kept between sentences.
But they occupy memory from the moment they're built until the backward pass has used them.
Temporary isn't free at this size. The engineering that pays this bill down — computing attention
without ever materializing the full table, caching $K$ and $V$ during generation — is section 03's
entire subject; nothing in this lesson depends on it.)*

### Nothing knows the order of the words

Job one — and now you can *prove* it broke, on the model in hand. Set aside the mask and the
position vectors for a moment, so $x$ is the bare word embeddings — then shuffle its rows: the
words, re-ordered.

- $Q = xW_Q$, $K = xW_K$, $V = xW_V$ act row-by-row → their rows shuffle the same way
- $E = QK^\top$ → rows *and* columns shuffle to match
- softmax runs per row → $A$ shuffles the same way
- $\text{out} = AV$ → its rows follow

Chase any single word through that: it ends up in a new row, carrying **exactly the numbers it
would have had anyway**. At no point did any operation ask for a row's index. *"The cat sat"* and
*"sat cat The"* produce the same three vectors, just re-ordered.

> The property's name: permutation **equivariance**. Permute the input and the outputs permute to
> match — they move, but none of them changes value. (Not "invariance" — that would mean they
> don't even move. The distinction is a standard interview trap.)

Two honest footnotes. The causal mask *does* consult indices — masking $j > i$ is nothing but an
index rule — so the masked op isn't perfectly order-blind; but "past vs future" is all it knows,
never *which word sits where*. And Bahdanau never faced any of this: his $h_j$ came out of an RNN
that read the sentence in order. That's the thing that got deleted.

This is why part 4 added position vectors to the embeddings. That line was the repair — already
installed, already in your trace. Lesson 5 is about which position signals work best, not about
whether you need one. You just proved you do.

### The fourth job — already paid for

Part 3's list had three entries. The decoder's recurrence was quietly doing a fourth thing:
$s_i$ was built from $s_{i-1}$, so seeing the future was structurally impossible. Attention has no
such scruple — which part 4 discovered mid-build, the moment training-in-parallel met
predict-the-next-token. The **causal mask** was motivated and installed right there. It's on this
list only so the ledger is complete.

---

## The ledger

```
delete the recurrence
    word order        broke    → position vectors — installed in part 4; designs, lesson 5
    flat memory       broke    → T·d keys + T² scores; the standing constraint (section 03 pays it)
    params vs length  SURVIVED → every weight acts per row; one asterisk, the stored position table
    no peeking ahead  broke    → causal mask — installed in part 4
```

Every line is either already repaired in the model you've built, or is a permanent constraint
you now know the exact shape of. No IOUs.

One thing on the audit *is* still open, and it's not a cost — it's a question. Part 4 conjured
three matrices, $W_Q, W_K, W_V$, and never justified them. Part 1's $h_j$ played every role at
once and worked fine. Why three?

**→ [6 · Query, key, value](6-query-key-value.md)**
