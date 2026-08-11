# 4 · What deleting it cost

*~4 min. Lesson 1, part 4 of 10.*

## The problem

Part 3 deleted the recurrence for parallelism. The story usually stops there, one sentence short
of the interesting part: recurrence was doing three jobs, and the replacement doesn't do all
three. Before building anything on top of attention, count what it owes.

Start with the claim you'll hear most often, which is false.

---

## The trap in the complexity table

Vaswani's Table 1, for one layer, sequence length $n$ and width $d$:

| | Work per layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-attention | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n d^2)$ | $O(n)$ | $O(n)$ |

Read the first column carefully:

$$n^2 d \;\lessgtr\; n d^2 \qquad\Longleftrightarrow\qquad n \;\lessgtr\; d$$

**Attention is cheaper only while the sentence is shorter than the model is wide.** Past
$n \approx d$ it does strictly *more* arithmetic than the recurrence it replaced. At $d = 512$ and
$n = 1024$, it's about twice the FLOPs.

So "transformers won because attention is more efficient" is **false**, and it's a good way to lose
an interview. Attention won while being *more expensive*, because its extra work is parallel and
the RNN's cheaper work is stuck in a queue. A GPU would rather do 2× the work at once than 1× of it
in sequence.

**Total work versus critical path.** That distinction explains most of what this roadmap covers
later.

---

## Which jobs survive

| Job | Survives? | |
|---|---|---|
| Order, for free | **no** | nothing left in the computation refers to position |
| Any length, fixed memory | **no** | see below |
| Parameters don't grow with length | **yes** | the projection matrices don't depend on $n$ either |

Two of three, and the third is why the trade was tolerable.

### Memory stops being fixed

A recurrence holds one state of width $d$ no matter how long the sentence — flat. Attention holds
everything at once, and two different things now grow:

| | Size | Why |
|---|---|---|
| The keys | $O(nd)$ — **linear** | every position stays addressable, so nothing can be thrown away |
| The score table $E$ | $O(n^2)$ — **quadratic** | one number per pair of positions |

Part 2's $E$ was $T_y \times T_x$, two different lengths. Once queries and keys come from the same
sequence, both are $n$ and it's square: about a million entries at $n = 1024$, 67 million at
$n = 8192$. And that's one attention operation, in a model that stacks many layers of them.

That's why the **context window** — the longest input a model can take at once — stayed small for
years. It's a debt with a repayment schedule:

| Debt | Paid by | Where |
|---|---|---|
| $O(n^2)$ memory | FlashAttention — same result, never stores $E$ | section 03 |
| $O(n^2)$ compute | sliding-window, sparse, linear attention | section 03 |
| $O(nd)$ keys, redone for every generated word | KV caching | section 03 |

Named so you know the bill gets paid. Nothing below assumes them.

*(Part 1 said scores get "dropped" after use. Both are true: they're never parameters and never
persist between sentences — but they do sit in memory during the forward pass and have to survive
until the backward pass. Transient isn't free, and that gap is what FlashAttention attacks.)*

### Nothing knows what order the words were in

This one is already visible in what you have. Look at where order could possibly enter a context
vector:

$$c_i \;=\; \sum_{j=1}^{T_x} \alpha_{ij}\, h_j$$

$\alpha_{ij}$ is computed from the *content* of $h_j$ — the score function never sees $j$ itself.
And a sum doesn't care what order its terms come in. So shuffle the input words, carry each $h_j$
along with its word, and $c_i$ comes out **identical**. Attention contributes nothing at all to
the model's sense of order.

In Bahdanau that's harmless, because order got in elsewhere: the encoder RNN built each $h_j$ by
reading the sentence in sequence. **That's the thing you just deleted.** Take it away and nothing
anywhere refers to position — a model with no notion of sequence, for sequence modelling. That's
the hole **positional encoding** fills, in lesson 5.

*(The general name is permutation **equivariance**, not invariance: once every position is a query
instead of one decoder state, shuffling the input shuffles the outputs to match rather than leaving
them alone. What's derivable here, with a single query, is genuine invariance of $c_i$. Part 5 has
the machinery for the general statement. Either way, nothing inside attention depends on where a
token sits.)*

---

## Two problems that were never recurrence's fault

One attention pattern per layer turns out not to be enough (**multi-head**, lesson 2), and the
dot-product score scale from part 2 grows with $d$ (**$1/\sqrt d$ scaling**, part 8). Neither is a
job the recurrence had been doing — they're new bills from the new design.

---

```
delete the recurrence
    → 2 of its 3 jobs break
         order              → positional encoding   (lesson 5)
         fixed memory       → O(nd) keys + O(n²) scores, paid in section 03
         params vs length   → survives untouched
    → 2 new problems appear
         one pattern per layer → multi-head         (lesson 2)
         score scale grows     → 1/√d               (part 8)
```

Five items, and only the first is what "the transformer's ideas" usually means. Being able to say
which is a repair, which is a new bill, and which was never broken is most of what separates a real
answer from a recited one.

Next: with the recurrence gone, what does the model actually look like? Real tensors, real shapes,
and where the attention block sits.

**→ [5 · The forward pass](5-the-forward-pass.md)**
