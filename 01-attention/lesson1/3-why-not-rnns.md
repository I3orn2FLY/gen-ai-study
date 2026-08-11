# 3 · Why not recurrence

*~6 min. Lesson 1, part 3 of 10.*

## The problem

Part 2 ended holding a promissory note. The whole score table for a translation is one matmul:

$$E \;=\; Q K^{\top}$$

but $Q$ never exists, because $s_i$ needs $c_i$ needs $s_{i-1}$ — the queries arrive one at a
time. To collect, you have to delete the recurrence.

That should feel reckless. The recurrence *is* the model. So before attacking it: what was it
actually giving you?

---

## What recurrence was for

Three jobs, and it did all of them without being asked. Write the recurrence generically:

$$h_t \;=\; \mathrm{cell}(h_{t-1},\, x_t)$$

> $x_t$ — the embedding of the $t$-th input word, a vector.
> $h_t \in \mathbb{R}^{d}$ — the state after reading it. **In this part $d$ is whatever width the
> state happens to be**; part 1's $d_h = 2000$ was specifically Bahdanau's two concatenated
> directions, which is a detail we don't need here.
> $h_0$ — a starting state, either zeros or a learned vector. Nothing depends on the choice.
> $\mathrm{cell}$ — the recurrent unit (plain RNN, LSTM or GRU), holding one weight matrix reused
> at every $t$. Part 1's decoder cell $f$ is one of these with an extra input.

**Order, structurally.** The index $t$ is in the wiring. Nothing has to *represent* word order
because nothing can escape it — feed the words differently and you get a different computation,
necessarily.

**Unbounded length in bounded memory.** One state vector of fixed width $d$, however long the
sentence. Position 400 is processed with exactly the machinery that processed position 4.

**Parameters that don't grow with length.** The same cell every step. A 5-word sentence and a
500-word sentence use identical parameters.

That is a genuinely good design. Hold onto these three — the last section of this part is the
bill, and one of them survives, which is worth knowing in advance.

---

## An RNN is a deep network

Unroll the loop and stop reading it as a loop:

$$h_0 \;\to\; h_1 \;\to\; h_2 \;\to\; \cdots \;\to\; h_n$$

> $n$ — the sequence length. Part 1's $T_x$ and $T_y$ are the source and target versions of this
> same quantity; when only one sequence is in play, $n$ is the name.

with a fresh $x_t$ injected at every arrow. Read as a feedforward stack, **step $t$ is layer $t$**.
Three differences from an ordinary deep network:

| | Ordinary deep net | Unrolled RNN |
|---|---|---|
| Weights | one matrix per layer | **one matrix, shared by every layer** |
| Input | enters at layer 1 | enters at **every** layer |
| Depth | you choose it | **the sentence length chooses it** |

Otherwise the math is the same — and therefore so is the failure mode. "Vanishing gradients in
deep networks" and "vanishing gradients in RNNs" are not two results, they are one result, found
in the same window (Hochreiter 1991; Bengio, Simard, Frasconi 1994).

Keep this frame. It makes the next section obvious, and it puts attention in the same family as
every other shortcut connection in deep learning rather than off on its own.

---

## Argument 1 — path length

> **Path length** — the number of computation steps a signal has to pass through to get from
> position $i$ to position $j$. Not distance in the sentence; distance *in the network*.
>
> *(Index warning: in this section $i$ and $j$ are two positions in **one** sequence. Part 1 used
> $i$ for a decoder step and $j$ for a source position. Same letters, and from here on the
> single-sequence reading is the one that matters.)*

In an RNN, position 1 reaches position 7 by being carried through every state in between:

$$h_1 \to h_2 \to h_3 \to h_4 \to h_5 \to h_6 \to h_7$$

Six steps. In general $O(|i - j|)$ — it grows with how far apart the words are. With attention,
position 7 scores directly against position 1 and pulls from it in a single hop: $O(1)$,
regardless of distance.

![path length: walking versus jumping](figures/fig2-path-length.png)

### Why length hurts

Backpropagating from step $n$ to step 1 multiplies together the Jacobian of every layer in
between — which, per the table above, means every *step*:

$$\frac{\partial h_n}{\partial h_1} \;=\; \prod_{t=2}^{n} \frac{\partial h_t}{\partial h_{t-1}}$$

> $\dfrac{\partial h_t}{\partial h_{t-1}} \in \mathbb{R}^{d \times d}$ — the Jacobian of one
> recurrent step: how each component of the new state responds to each component of the old one.
> The product runs $t = 2 \ldots n$, so there are $n-1$ of them.

A product of $n-1$ matrices behaves roughly like $\sigma^{\,n-1}$, where $\sigma$ is a typical
singular value of one such Jacobian. The exponent is the sentence length, and you don't control
that. It can fail in two directions, and they are **not** equally bad:

| Direction | $\sigma$ | Fix |
|---|---|---|
| Exploding | $>1$ | **gradient clipping** — rescale the gradient vector $g$ whenever $\lVert g \rVert$ exceeds a chosen threshold $\tau$. One line, standard by 2013 |
| Vanishing | $<1$ | *nothing analogous* |

You can shrink a gradient that came back too large. You cannot restore one that already reached
numerical zero — the direction is gone, not just the magnitude. So when people say "the RNN
gradient problem," they essentially always mean **vanishing**.

### Gating is a shortcut, not a removal

This is what LSTMs and GRUs were invented for. The LSTM keeps a separate memory channel:

$$c_t \;=\; f_t \odot c_{t-1} \;+\; i_t \odot \tilde{c}_t$$

> **Letter clash, worth flagging:** this $c_t$ is the LSTM's **cell state**, nothing to do with
> part 1's context vector $c_i$. And $f_t$ below is a *vector*, not part 1's cell function $f$.
> Both collisions are standard in the literature; you'll meet them again.
>
> $c_t \in \mathbb{R}^{d}$ — the cell state, a memory vector running alongside $h_t$.
> $f_t \in (0,1)^{d}$ — the **forget gate**: how much of the old memory to keep, per component.
> $i_t \in (0,1)^{d}$ — the **input gate**: how much of the new candidate to admit.
> $\tilde c_t \in \mathbb{R}^{d}$ — the **candidate**, a $\tanh$ of the input and previous state.
> $\odot$ — elementwise product.

When $f_t \approx 1$ the update is **additive**, so $\partial c_t / \partial c_{t-1} \approx I$ —
the identity, not a decaying matrix. The product above stops shrinking along that channel.
(The forget gate wasn't in the original 1997 LSTM; Gers et al. added it in 2000.)

That is an additive shortcut through the unrolled depth — the same move highway networks and
ResNet make on ordinary depth in 2015. So three things belong to one family:

| | Shortcut across | Wiring |
|---|---|---|
| LSTM gating | unrolled time | learned gates, still one hop per step |
| Residual connections | layers | fixed at build time |
| Attention | positions, in one hop | **computed from the content**, per example |

Gating and residuals make the path *easier*. Attention **deletes** it: $i$ reaches $j$ with no
intermediate state at all. And its shortcuts aren't wiring — $\alpha_{ij}$ is recomputed from the
sentence every forward pass, so which connections exist depends on the input.

**Now the trap.** This is the argument everyone reaches for, and it is *not* the one that
mattered. Gated RNNs were already good enough at long range that translation quality wasn't what
was holding the field back. Path length is a real advantage and a satisfying story. It didn't
decide anything.

---

## Argument 2 — parallelism

This is the one. And to see it properly, notice that **the model you already have is partly
parallel** — and that the parallel part is the attention.

One Bahdanau decoder step, shapes from part 2. $H \in \mathbb{R}^{T_x \times d_h}$ is the encoder
states stacked as rows. The question in the last column is *the only one that matters for
wall-clock*: does this line need something from an **earlier decoder step**?

| Step | Shape | Needs an earlier decoder step? |
|---|---|---|
| $K_{\text{proj}} = H W_k^{\top}$ | $(T_x, d_a)$ | no — computed **once**, before the loop |
| $Z = \tanh(W_q s_{i-1} + K_{\text{proj}})$ | $(T_x, d_a)$ | no |
| $e_i = Z v$ | $(T_x,)$ | no |
| $\alpha_i = \mathrm{softmax}(e_i)$ | $(T_x,)$ | no |
| $c_i = \alpha_i^{\top} H$ | $(d_h,)$ | no |
| $s_i = \mathrm{cell}(s_{i-1},\, y_{i-1},\, c_i)$ | $(d_s,)$ | **yes — $s_{i-1}$** |

Rows 2–5 do have to run in order relative to *each other* — that's an ordinary five-deep
computation graph, and every deep net has one. What matters is that none of them reaches back to
a previous **word**. Their work spreads across all $T_x$ source positions at once. (Two of them
are reductions over $T_x$ rather than elementwise, which costs $\log T_x$ depth on a GPU, not
$T_x$.) **Attention is already the parallel part.** The last row is the RNN, and it is the only
thing in the step that serializes over words.

Now count a whole training example:

$$\underbrace{O(T_x)}_{\text{encoder RNN}} \;+\; \underbrace{O(T_y)}_{\text{decoder RNN}}
\qquad\text{sequential steps, with } O(1) \text{ attention inside each}$$

Each of those steps is one small matrix–vector product. A GPU with thousands of cores runs it,
idles through the rest, and does it again — $T_x + T_y$ times.

So the question isn't "is attention parallel?" It already is. The question is: **what if the
parallel part were the whole model?**

> **Self-attention** — attention where the queries and the keys come from the *same* sequence, so
> every position scores against every other instead of against a separate encoder. Part 5 builds
> it; here it only means "attention with no RNN wrapped around it."
>
> **Layer** — one attention operation over a whole sequence, of the kind a model stacks many of.
> Also part 5.

| Per layer | Sequential operations |
|---|---|
| Recurrence | $O(n)$ — state $t$ needs state $t-1$ |
| Self-attention, no recurrence | $O(1)$ — every position scores every other at once |

The score computation for one such layer over a 50-word sentence is a single
$(50 \times d)(d \times 50)$ matmul, every output element independent — the shape the hardware
was built for.

### Why teacher forcing doesn't rescue the RNN

Worth being precise, because it's the detail that makes this decisive. During training the whole
target sentence is known in advance, so you might think the decoder could process all positions at
once. It can't: $s_i$ still needs $s_{i-1}$. Knowing the *inputs* ahead of time doesn't help when
the *states* form a chain.

That is exactly what *Attention Is All You Need* (Vaswani et al., 2017 — the paper that performs
the deletion) names: the sequential nature of recurrence "precludes parallelization within
training examples" (§1). And what its abstract claims is time, not quality — more parallelizable,
significantly less time to train.

> **Recurrence wasn't replaced for learning badly. It was replaced for training slowly.**

*(**Origin tag: Fix** — a named failure with a targeted response. The failure was a training-time
constraint, not a modelling one.)*

---

---

## The chain so far

```
scores are one matmul — but only if all the queries exist at once
    → attention is already parallel; the RNN around it is not
    → is the recurrence load-bearing?  path length: nice, not decisive
                                       parallelism: decisive
    → delete it
```

The deletion is decided. What it costs hasn't been counted yet, and the bill is larger than the
"attention is more efficient" story admits — it starts with that claim being false.

**→ [4 · What deleting it cost](4-what-it-cost.md)**
