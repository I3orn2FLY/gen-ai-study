# 4 · What deleting it cost

*~4 min. Lesson 1, part 4 of 10.*

## The problem

Part 3 deleted the recurrence for parallelism. That's the standard story and it usually stops
there, one sentence short of the interesting part: recurrence was doing three jobs, and the
replacement doesn't do all three. Before building anything on top of attention, count what it
owes.

Start with the claim you'll hear most often, which is false.

---

## The trap in the complexity table

Vaswani's Table 1, for one layer, sequence length $n$ and width $d$:

| | Complexity per layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |

Read the first column carefully.

$$n^2 d \;\lessgtr\; n d^2 \qquad\Longleftrightarrow\qquad n \;\lessgtr\; d$$

**Attention is cheaper only while the sentence is shorter than the model is wide.** Past
$n \approx d$ it does strictly *more* arithmetic than the recurrence it replaced. At $d = 512$
and $n = 1024$, attention is doing about twice the FLOPs.

So "transformers won because attention is more efficient" is **false**, and it's a good way to
lose an interview. Attention won while being *more expensive*, because its extra work is parallel
and the RNN's cheaper work is serialized. A GPU would rather do 2× the FLOPs at once than 1× of
them in a queue. **Total work versus critical path** — that distinction explains most of what this
roadmap covers later.

---

## What the deletion cost

Take the three jobs from the top of this part, one at a time.

| Job | Survives? | |
|---|---|---|
| Order, structurally | **no** | nothing left in the computation refers to position |
| Unbounded length in bounded memory | **no** | see below — the bounded state is gone |
| Parameters independent of length | **yes** | the projection matrices don't depend on $n$ either |

Two of three, and the third is why the trade was tolerable at all.

### Job 2 — memory stops being bounded

A recurrence holds one state of width $d$ no matter how long the sentence: $O(d)$, flat. Attention
holds every position at once, and two different things now grow:

| | Size | Why |
|---|---|---|
| The keys themselves | $O(n\,d)$ — **linear** | every position must stay addressable, so nothing can be discarded |
| The score table $E$ | $O(n^2)$ — **quadratic** | one number per ordered pair of positions |

Part 2's $E$ was $T_y \times T_x$, two different lengths. Once queries and keys come from the same
sequence, $T_y = T_x = n$ and it's square. About a million entries at $n = 1024$; 67 million at
$n = 8192$ — and that is one attention operation, in a model that stacks many layers of them.

That is why the **context window** — the longest input a model can take at once — stayed small for
years, and it's a debt with a repayment schedule:

| Debt | Paid by | Where |
|---|---|---|
| $O(n^2)$ **memory** | FlashAttention — computes the same result without ever storing $E$ | section 03 |
| $O(n^2)$ **compute** | sliding-window, sparse, linear attention | section 03 |
| $O(nd)$ keys, recomputed for every generated word | KV caching | section 03 |

Named only so you know the bill gets paid. Nothing below assumes them.

*(One reconciliation with part 1, which said scores are "dropped" after use. Both are true: scores
are never **parameters** and never persist between sentences — but they do have to exist in memory
during the forward pass, and be kept for the backward pass. Transient is not the same as free.
That gap is exactly what FlashAttention attacks.)*

### Job 1 — nothing knows what order the words were in

This is the startling one, and it's already visible in what you have. Look at where order could
possibly enter a context vector:

$$c_i \;=\; \sum_{j=1}^{T_x} \alpha_{ij}\, h_j$$

$\alpha_{ij}$ is computed from the *content* of $h_j$ — the score function never sees $j$ itself.
And a sum doesn't care what order its terms come in. So permute the source words, carry each $h_j$
along with its word, and $c_i$ comes out **identical**. Attention contributes exactly nothing to
the model's sense of order.

In Bahdanau that's harmless, because order got in somewhere else: the encoder RNN built each $h_j$
by reading the sentence in sequence. **That is the thing you just deleted.** Take it away and
nothing anywhere in the model refers to position — a model with no notion of sequence, for
sequence modelling. That's the hole **positional encoding** fills, in lesson 5.

*(The standard name for the general property is permutation **equivariance**, not invariance —
once every position is a query rather than one decoder state, permuting the input permutes the
outputs along with it rather than leaving them alone. The version derivable here, with a single
query, is genuine invariance of $c_i$. Part 5 has the machinery for the general statement; the
part that holds either way is that no computation inside attention depends on where a token sits.)*

### Two problems that were never recurrence's fault

One attention pattern per layer turns out not to be enough (**multi-head**, lesson 2), and the
dot-product score scale from part 2 grows with $d$ (**$1/\sqrt d$ scaling**, part 8). Neither is a
job the recurrence had been doing — they're new bills from the new design. Worth keeping straight,
because the usual telling lumps all five together as "the transformer's ideas."

---

## The chain

```
delete the recurrence
    → 2 of its 3 jobs break
         word order        → positional encoding    (lesson 5)
         bounded memory    → O(nd) keys + O(n²) scores, paid off in section 03
         params vs length  → survives untouched
    → 2 new problems appear
         one pattern per layer → multi-head         (lesson 2)
         score scale grows     → 1/√d               (part 8)
```

Five items, and only the first is what "the transformer's ideas" usually refers to. Being able to
say which of them is a repair, which is a new bill, and which was never broken is most of what
separates a real answer from a recited one.

Next: with the recurrence gone, what does the model actually look like? Real tensors, real
shapes, and where the attention block sits inside it.

**→ [5 · The forward pass](5-the-forward-pass.md)**
