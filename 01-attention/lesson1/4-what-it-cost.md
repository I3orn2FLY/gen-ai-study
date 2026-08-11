# 4 · What deleting it cost

*~4 min. Lesson 1, part 4 of 10.*

## The problem

The RNN is gone. What's left — stacked self-attention, no recurrence anywhere — is the
**transformer**, and part 5 builds one.

The usual story stops at "and it's faster." That's one sentence short of the interesting part.
Part 3 listed three things the recurrence was quietly doing for free. The replacement doesn't do
all three, and there's a fourth nobody counts. Before building anything on top of attention, add
up what it owes.

Start with the claim you'll hear most often, which is false.

---

## Attention is not cheaper

Vaswani's Table 1, for one layer, sentence length $n$ and model width $d$:

| | Work per layer | Steps in order | Longest path |
|---|---|---|---|
| Self-attention | $n^2 d$ | $1$ | $1$ |
| Recurrent | $n d^2$ | $n$ | $n$ |

Look at the first column and ask when one beats the other:

$$n^2 d \;<\; n d^2 \qquad\Longleftrightarrow\qquad n \;<\; d$$

**Attention is cheaper only while the sentence is shorter than the model is wide.** Once it's
longer, attention does strictly *more* arithmetic than the RNN it replaced. At $d = 512$ and
$n = 1024$, about twice as much.

So "transformers won because attention is more efficient" is wrong, and it's a good way to lose an
interview. Attention won **while being more expensive**, because its extra work all happens at
once and the RNN's cheaper work happens in a queue. A GPU would much rather do twice the work in
one go than half of it in sequence.

That distinction — how much work in total, versus how long the longest chain is — explains most of
what this roadmap covers later.

---

## Which of the three survive

| What recurrence gave you | Survives? | |
|---|---|---|
| It knew word order | **no** | nothing left in the computation refers to position |
| Any length, same memory | **no** | see below |
| Length doesn't change the parameter count | **yes** | attention's weights act on one position at a time, so their shapes come from the width, not the length |

Two out of three. Don't skim past the survivor: if the parameter count grew with sentence length,
you couldn't train on short sentences and run on long ones, and the whole thing would stop being
reusable. That one holding is what made losing the other two worth it.

### Memory stops being flat

Running forwards, an RNN holds one state — 2000 numbers in Bahdanau's encoder — no matter how long
the sentence is. Attention holds everything at once, and two separate things now grow:

| | Size | Why |
|---|---|---|
| The keys themselves | $n \times d$ — **grows linearly** | every position has to stay reachable, so nothing can be dropped |
| The score table $E$ | $n \times n$ — **grows quadratically** | one number for every pair of positions |

Part 2's $E$ was $T_y \times T_x$, two different lengths. Once queries and keys come from the same
sentence they're both $n$, so it's square: about a million entries at $n = 1024$, and 67 million at
$n = 8192$. That's for *one* attention operation, in a model that stacks many.

Which is why the **context window** — the longest input a model can take at once — stayed small for
years. It's a debt, and here's the repayment schedule:

| Debt | Paid by | Where |
|---|---|---|
| the $n \times n$ table taking up memory | FlashAttention — same answer, never stores it | section 03 |
| the $n^2$ arithmetic | sliding-window, sparse, linear attention | section 03 |
| redoing all the keys for every new word you generate | KV caching | section 03 |

Listed so you know the bill gets paid. Nothing below depends on them.

*(Part 1 said scores get thrown away after use. Both are true — they're never parameters and never
kept between sentences, but they do sit in memory while the pass runs and have to stay there until
the backward pass. Temporary isn't the same as free, and that gap is exactly what FlashAttention
goes after.)*

### Nothing knows what order the words were in

You can already see this in what you have. Look at where order could possibly get into a context
vector:

$$c_i \;=\; \sum_{j=1}^{T_x} \alpha_{ij}\, h_j$$

$\alpha_{ij}$ comes from the *content* of $h_j$ — the scoring function is never shown $j$ itself.
And a sum doesn't care what order you add things in. So shuffle the input words, keep each $h_j$
attached to its own word, and $c_i$ comes out **exactly the same**. Attention contributes nothing
whatsoever to the model's sense of order.

In Bahdanau that was harmless, because order got in somewhere else — the encoder RNN built each
$h_j$ by reading the sentence in sequence. **That's the thing you just deleted.** Take it away and
nothing anywhere in the model refers to position: a model with no notion of sequence, doing
sequence modelling. That's the hole **positional encoding** fills, in lesson 5.

*(The proper name for the general property is permutation **equivariance**, not invariance — once
every position is a query rather than one decoder state, shuffling the input shuffles the outputs
to match rather than leaving them alone. What you can derive here, with a single query, is genuine
invariance of $c_i$. Part 5 has the machinery for the general version. Either way: nothing inside
attention depends on where a word sits.)*

### And the fourth one, on the decoder side

Not on the list of three, because it's a property of the decoder rather than the encoder — but it
broke the same way.

$s_i$ was built from $s_{i-1}$, so word $i$ physically could not consult word $i+1$. The model
couldn't cheat by reading the answer, and nobody had to arrange that. It was free.

Attention has no such scruple. Every position sees every other one, including the ones it's
supposed to be predicting. Part 3 said teacher forcing can't make an RNN parallel; the flip side is
that once you *can* train every position at once, you have to explicitly forbid looking forward.
That's a **causal mask**, and it's in part 5.

---

## Two problems that were never the RNN's job

One attention pattern per layer turns out not to be enough — that's **multi-head**, in lesson 2.
And the dot-product score scale from part 2 grows with the width — that's **$1/\sqrt d$ scaling**,
in part 8. Neither of these is a job the recurrence had been doing. They're new bills from the new
design.

---

```
delete the recurrence
    → 2 of its 3 jobs break
         word order       → positional encoding   (lesson 5)
         flat memory      → n·d keys + n² scores, paid off in section 03
         params vs length → survives untouched
    → plus one it did on the decoder side
         can't peek ahead → causal mask           (part 5)
    → and 2 new problems appear
         one pattern per layer → multi-head       (lesson 2)
         scores grow with width → 1/√d            (part 8)
```

Six items, and only the first is what people usually mean by "the transformer's ideas". Being able
to say which is a repair, which is a fresh bill, and which was never broken is most of what
separates a real answer from a recited one.

Next: with the RNN gone, what does the model actually look like? Real tensors, real shapes, and
where the attention block sits in it.

**→ [5 · The forward pass](5-the-forward-pass.md)**
