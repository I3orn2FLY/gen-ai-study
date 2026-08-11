# 3 · Why not recurrence

*~5 min. Lesson 1, part 3 of 10.*

## The problem

Part 2's promissory note: every score in a translation is one matmul, $E = QK^{\top}$ — except $Q$
never exists, because the queries arrive one at a time. To collect, delete the recurrence.

That should feel reckless. The recurrence *is* the model. So first: what was it giving you?

---

## What recurrence was for

$$h_t \;=\; \mathrm{cell}(h_{t-1},\, x_t)$$

> $x_t$ — the embedding of input word $t$. $h_t \in \mathbb{R}^{d}$ — the state after reading it,
> where $d$ is just "however wide the state is". $h_0$ is zeros or learned; doesn't matter.
> $\mathrm{cell}$ — the recurrent unit (RNN, LSTM, GRU), one weight matrix reused every step. It's
> the same kind of object part 1 called $f$; renamed here because $f_t$ is about to mean something
> else. A decoder cell takes an extra input or two, but the shape of the argument is identical.

Three jobs, all done without being asked:

- **Order, for free.** The index $t$ is in the wiring. Nothing has to *represent* word order
  because nothing can escape it.
- **Any length, fixed memory.** Running forwards, one state of width $d$ carries the whole
  sentence, however long it is. (Training is different — backprop needs every intermediate state
  kept, so that's $n$ of them. The claim is about the model's design, not its training bill.)
- **Parameters don't grow with length.** Same cell every step, so 5 words and 500 words use
  identical parameters.

Remember these three — part 4 is the bill, and one of them survives.

---

## An RNN is a deep network

Unroll it and stop reading it as a loop: $h_0 \to h_1 \to \cdots \to h_n$, with a fresh $x_t$
entering at every arrow. ($n$ is the sequence length; part 1's $T_x$ and $T_y$ are the source and
target versions of it.) Read as a feedforward stack, **step $t$ is layer $t$**:

| | Ordinary deep net | Unrolled RNN |
|---|---|---|
| Weights | one matrix per layer | **one matrix shared by all layers** |
| Input | enters at layer 1 | enters at **every** layer |
| Depth | you pick it | **the sentence length picks it** |

The math is otherwise the same, so the failure mode is the same. "Vanishing gradients in deep
nets" and "vanishing gradients in RNNs" aren't two results — they're one, found in the same window
(Hochreiter 1991; Bengio, Simard, Frasconi 1994).

---

## Argument 1 — path length

> **Path length** — how many computation steps a signal passes through to get from one position to
> another. Not distance in the sentence; distance *in the network*.

In an RNN, word 1 reaches word 7 through everything in between: six steps, growing with the gap.
Attention scores them against each other directly — one hop, however far apart.

![path length: walking versus jumping](figures/fig2-path-length.png)

### Why length hurts

Backprop from step $n$ to step 1 multiplies the Jacobian of every layer between — which, per the
table above, means every *step*:

$$\frac{\partial h_n}{\partial h_1} \;=\; \prod_{t=2}^{n} \frac{\partial h_t}{\partial h_{t-1}}$$

> $\partial h_t / \partial h_{t-1} \in \mathbb{R}^{d\times d}$ — how each part of the new state
> responds to each part of the old one. There are $n-1$ of them.

That product behaves like $\sigma^{\,n-1}$, where $\sigma$ is a typical singular value of one. The
exponent is the sentence length, which you don't control. It fails two ways, and only one is hard:

| | $\sigma$ | Fix |
|---|---|---|
| Exploding | $>1$ | **gradient clipping** — shrink it when it gets too big. Standard by 2013 |
| Vanishing | $<1$ | *nothing like it* |

You can shrink a gradient that came back too large. You can't restore one that hit numerical zero
— the direction is gone, not just the size. "The RNN gradient problem" means **vanishing**.

### Gating shortens the path, it doesn't remove it

This is what LSTMs and GRUs were for — a memory channel running alongside the state:

$$c_t \;=\; f_t \odot c_{t-1} \;+\; i_t \odot \tilde{c}_t$$

> $c_t$ — the **cell state**. $f_t \in (0,1)^{d}$ — the **forget gate**, how much old memory to keep
> per component. $i_t \odot \tilde c_t$ — **input gate** times **candidate**, how much new to let
> in. $\odot$ is elementwise.
>
> Three letters here are reused from part 1, and they're reused in the literature too, so it's
> worth naming once: $c_t$ is a cell state, not a context vector; $f_t$ is a gate vector, not a
> cell function; $i_t$ is a gate vector, not a decoder step index. Subscript $t$ means "gate".

When $f_t \approx 1$ the update is **additive**, so $\partial c_t/\partial c_{t-1} \approx I$ — the
identity, not a decaying matrix, and the product stops shrinking. (The forget gate wasn't in the
1997 LSTM; Gers et al. added it in 2000.)

That's a shortcut through the unrolled depth — the same move highway networks and ResNet make on
ordinary depth in 2015. Three of a kind:

| | Shortcut across | Wiring |
|---|---|---|
| LSTM gating | unrolled time | learned gates, one hop per step |
| Residual connections | layers | fixed when you build the model |
| Attention | positions, one hop | **computed from the content**, per sentence |

Gating and residuals make the path easier. Attention **deletes** it, and its shortcuts aren't
wiring — $\alpha_{ij}$ is recomputed every forward pass.

**Now the trap.** This is the argument everyone reaches for and it's *not* the one that mattered.
Stacked LSTMs held the state of the art in machine translation right up to 2017 — Google's
production translation system was one. Long-range quality was not what had the field stuck. Path
length is a real advantage and a satisfying story. It decided nothing.

---

## Argument 2 — parallelism

This is the one. To see it, notice the model you already have is **partly parallel** — and the
parallel part is the attention.

One Bahdanau decoder step, shapes from part 2. Every line takes $s_{i-1}$ as an input — that isn't
the problem. The problem is which line *produces* the state the **next** step needs, because that's
what forces the steps into a queue:

| Step | Shape | Parallel over the $T_x$ keys? | Feeds the next step? |
|---|---|---|---|
| $K_{\text{proj}} = H W_k^{\top}$ | $(T_x, d_a)$ | yes — and done once, before the loop | no |
| $Z = \tanh(W_q s_{i-1} + K_{\text{proj}})$ | $(T_x, d_a)$ | yes | no |
| $e_i = Z v$ | $(T_x,)$ | yes | no |
| $\alpha_i = \mathrm{softmax}(e_i)$ | $(T_x,)$ | yes (a sum, so $\log T_x$ deep, not $T_x$) | no |
| $c_i = \alpha_i^{\top} H$ | $(d_h,)$ | yes (same) | no |
| $s_i = \mathrm{cell}(s_{i-1},\, y_{i-1},\, c_i)$ | $(d_s,)$ | — | **yes — it makes $s_i$** |

Only the last row puts anything into the chain $s_0 \to s_1 \to \cdots$. Everything above it is
one sweep across the source positions whose depth doesn't grow with $T_x$. **Attention is already
the parallel part.** The RNN cell is the only thing that serializes over words.

Count a training example: $T_x$ sequential steps for the encoder plus $T_y$ for the decoder, each a
small matrix–vector product. A GPU with thousands of cores runs one, idles, repeats.

So the question isn't "is attention parallel?" It already is. It's: **what if the parallel part
were the whole model?** And notice what that fixes — part 2 couldn't batch the queries because each
$s_{i-1}$ had to be computed first. Take the RNN out and a position's query comes from its own
input: the word embedding, or whatever the previous layer produced there. Every query exists before
you start.

> **Self-attention** — attention where queries and keys come from the *same* sequence, so every
> position scores every other instead of scoring a separate encoder.
> **Layer** — from here on, one attention operation over a whole sequence, the kind a model stacks
> many of. (Narrower than "layer" in "4 stacked LSTM layers" a moment ago — same word, new unit.)
> Both are part 5's; here they just mean "attention with no RNN around it."

| Per layer | Sequential operations |
|---|---|
| Recurrence | $O(n)$ — state $t$ needs state $t-1$ |
| Self-attention, no recurrence | $O(1)$ — every position scores every other at once |

The score computation for one such layer over a 50-word sentence is a single
$(50 \times d)(d \times 50)$ matmul, every output independent — the shape the hardware was built
for.

### Teacher forcing doesn't rescue the RNN

This is the detail that makes it decisive. In training the whole target sentence is known up front,
so you'd think the decoder could do all positions at once. It can't: $s_i$ still needs $s_{i-1}$.
Knowing the *inputs* ahead of time doesn't help when the *states* form a chain.

That's exactly what *Attention Is All You Need* (Vaswani et al., 2017) says: recurrence "precludes
parallelization within training examples" (§1). And what its abstract claims is time, not quality —
more parallelizable, significantly less time to train.

> **Recurrence wasn't replaced for learning badly. It was replaced for training slowly.**

*(**Origin tag: Fix** — a training-time constraint, not a modelling one.)*

---

```
scores are one matmul — but only if all the queries exist at once
    → attention is already parallel; the RNN around it is not
    → is recurrence load-bearing?  path length: nice, not decisive
                                   parallelism: decisive
    → delete it
```

The deletion is decided. What it costs hasn't been counted — and the bill starts with the claim
you'll hear most often being false.

**→ [4 · What deleting it cost](4-what-it-cost.md)**
