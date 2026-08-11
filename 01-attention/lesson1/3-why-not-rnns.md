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
> $\mathrm{cell}$ — the recurrent unit (RNN, LSTM, GRU), one weight matrix reused every step.

Three jobs, all done without being asked:

- **Order, for free.** The index $t$ is in the wiring. Nothing has to *represent* word order
  because nothing can escape it.
- **Any length, fixed memory.** One state of width $d$, however long the sentence.
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

In an RNN, word 1 reaches word 7 by being carried through everything between: six steps, growing
with the gap. Attention scores word 7 against word 1 directly and pulls in one hop, however far
apart they are.

![path length: walking versus jumping](figures/fig2-path-length.png)

### Why length hurts

Backprop from step $n$ to step 1 multiplies the Jacobian of every layer between — which, per the
table above, means every *step*:

$$\frac{\partial h_n}{\partial h_1} \;=\; \prod_{t=2}^{n} \frac{\partial h_t}{\partial h_{t-1}}$$

> $\partial h_t / \partial h_{t-1} \in \mathbb{R}^{d\times d}$ — how each part of the new state
> responds to each part of the old one. There are $n-1$ of them.

That product behaves like $\sigma^{\,n-1}$, where $\sigma$ is a typical singular value of one. The
exponent is the sentence length, which you don't control. Two ways to fail, not equally bad:

| | $\sigma$ | Fix |
|---|---|---|
| Exploding | $>1$ | **gradient clipping** — shrink the gradient when it gets too big. One line, standard by 2013 |
| Vanishing | $<1$ | *nothing like it* |

You can shrink a gradient that came back too large. You can't restore one that hit numerical zero
— the direction is gone, not just the size. So "the RNN gradient problem" almost always means
**vanishing**.

### Gating shortens the path, it doesn't remove it

This is what LSTMs and GRUs were for. The LSTM runs a memory channel alongside the state:

$$c_t \;=\; f_t \odot c_{t-1} \;+\; i_t \odot \tilde{c}_t$$

> $c_t$ — the **cell state**. (Not part 1's context vector; the letters collide here and in the
> literature.) $f_t \in (0,1)^{d}$ — the **forget gate**, how much old memory to keep per
> component. $i_t \odot \tilde c_t$ — **input gate** times **candidate**, how much new to let in.
> $\odot$ is elementwise.

When $f_t \approx 1$ the update is **additive**, so $\partial c_t/\partial c_{t-1} \approx I$ — the
identity, not a decaying matrix, and the product stops shrinking. (The forget gate wasn't in the
1997 LSTM; Gers et al. added it in 2000.)

That's a shortcut through the unrolled depth — the same move highway networks and ResNet make on
ordinary depth in 2015. Three things in one family:

| | Shortcut across | Wiring |
|---|---|---|
| LSTM gating | unrolled time | learned gates, still one hop per step |
| Residual connections | layers | fixed when you build the model |
| Attention | positions, one hop | **computed from the content**, per sentence |

Gating and residuals make the path easier. Attention **deletes** it — no intermediate state at all
— and its shortcuts aren't wiring, since $\alpha_{ij}$ is recomputed every forward pass.

**Now the trap.** This is the argument everyone reaches for and it's *not* the one that mattered.
Gated RNNs were already good enough at long range that quality wasn't the blocker. Path length is
a real advantage and a satisfying story. It decided nothing.

---

## Argument 2 — parallelism

This is the one. To see it, notice the model you already have is **partly parallel** — and the
parallel part is the attention.

One Bahdanau decoder step, shapes from part 2. The question that sets wall-clock time is whether a
line waits on an **earlier word**:

| Step | Shape | Waits on an earlier word? |
|---|---|---|
| $K_{\text{proj}} = H W_k^{\top}$ | $(T_x, d_a)$ | no — done once, before the loop |
| $Z = \tanh(W_q s_{i-1} + K_{\text{proj}})$ | $(T_x, d_a)$ | no |
| $e_i = Z v$ | $(T_x,)$ | no |
| $\alpha_i = \mathrm{softmax}(e_i)$ | $(T_x,)$ | no |
| $c_i = \alpha_i^{\top} H$ | $(d_h,)$ | no |
| $s_i = \mathrm{cell}(s_{i-1},\, y_{i-1},\, c_i)$ | $(d_s,)$ | **yes — $s_{i-1}$** |

Rows 2–5 run in order relative to each other, of course — that's an ordinary computation graph. What
matters is that none reaches back to a previous *word*; their work spreads across all $T_x$ input
positions at once. **Attention is already the parallel part.** The last row is the RNN, and it's
the only thing that serializes over words.

Count a training example: $T_x$ sequential steps for the encoder plus $T_y$ for the decoder, each a
small matrix–vector product. A GPU with thousands of cores runs one, idles, repeats.

So the question isn't "is attention parallel?" It already is. It's: **what if the parallel part
were the whole model?**

> **Self-attention** — attention where queries and keys come from the *same* sequence, so every
> position scores every other instead of scoring a separate encoder.
> **Layer** — one attention operation over a whole sequence, the kind a model stacks many of.
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

*(**Origin tag: Fix**. The failure was a training-time constraint, not a modelling one.)*

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
