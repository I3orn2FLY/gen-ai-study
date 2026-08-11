# 3 · Why not recurrence

*~6 min. Lesson 1, part 3 of 9.*

## The problem

Part 2 ended holding a promissory note. The whole score table for a translation is one matmul:

$$E \;=\; Q K^{\top}$$

but $Q$ never exists, because $s_i$ needs $c_i$ needs $s_{i-1}$ — the queries arrive one at a
time. To collect, you have to delete the recurrence.

That should feel reckless. The recurrence *is* the model. So before attacking it: what was it
actually giving you?

---

## What recurrence was for

Three jobs, and it did all of them without being asked. Write the recurrence as

$$h_t \;=\; f(h_{t-1},\, x_t)$$

> $x_t \in \mathbb{R}^{d_e}$ — the embedding of the $t$-th input word.
> $h_t \in \mathbb{R}^{d_h}$ — the state after reading it.
> $f$ — the recurrent cell (a plain RNN, LSTM or GRU), holding one weight matrix reused at
> every $t$.

**Order, structurally.** The index $t$ is in the wiring. Nothing has to *represent* word order
because nothing can escape it — feed the words differently and you get a different computation,
necessarily.

**Unbounded length in bounded memory.** One state vector of fixed width $d_h$, however long the
sentence. Position 400 is processed with exactly the machinery that processed position 4.

**Parameters that don't grow with length.** The same $f$ every step. A 5-word sentence and a
500-word sentence use identical parameters.

That is a genuinely good design. Hold onto these three, because the last section of this part is
the bill — and one of them survives the deletion, which is worth knowing in advance.

---

## An RNN is a deep network

Unroll the loop and stop reading it as a loop:

$$h_0 \;\to\; h_1 \;\to\; h_2 \;\to\; \cdots \;\to\; h_n$$

> $n$ — the sequence length. In this part it stands for whichever sequence is under discussion;
> part 1's $T_x$ and $T_y$ are the source and target versions of the same quantity.

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

> $\dfrac{\partial h_t}{\partial h_{t-1}} \in \mathbb{R}^{d_h \times d_h}$ — the Jacobian of one
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

> $c_t \in \mathbb{R}^{d_h}$ — the **cell state**, a memory vector running alongside $h_t$.
> $f_t \in (0,1)^{d_h}$ — the **forget gate**: how much of the old memory to keep, per component.
> $i_t \in (0,1)^{d_h}$ — the **input gate**: how much of the new candidate to admit.
> $\tilde c_t \in \mathbb{R}^{d_h}$ — the **candidate**, a $\tanh$ of the input and previous state.
> $\odot$ — elementwise product. All four are the same width as the state.

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
states stacked as rows:

| Step | Shape | Must wait for something? |
|---|---|---|
| $K_{\text{proj}} = H W_k^{\top}$ | $(T_x, d_a)$ | computed **once**, before the loop |
| $Z = \tanh(W_q s_{i-1} + K_{\text{proj}})$ | $(T_x, d_a)$ | no — all $T_x$ rows independent |
| $e_i = Z v$ | $(T_x,)$ | no |
| $\alpha_i = \mathrm{softmax}(e_i)$ | $(T_x,)$ | no |
| $c_i = \alpha_i^{\top} H$ | $(d_h,)$ | no |
| $s_i = f(s_{i-1},\, y_{i-1},\, c_i)$ | $(d_s,)$ | **yes — $s_{i-1}$** |

Every row but the last is one parallel sweep over the source positions. **Attention is already
the parallel part.** The last row is the RNN, and it is the only serial thing in the step.

Now count a whole training example:

$$\underbrace{O(T_x)}_{\text{encoder RNN}} \;+\; \underbrace{O(T_y)}_{\text{decoder RNN}}
\qquad\text{sequential steps, with } O(1) \text{ attention inside each}$$

Each of those steps is one small matrix–vector product. A GPU with thousands of cores runs it,
idles through the rest, and does it again — $T_x + T_y$ times.

So the question isn't "is attention parallel?" It already is. The question is: **what if the
parallel part were the whole model?**

| Per layer | Sequential operations |
|---|---|
| Recurrence | $O(n)$ — state $t$ needs state $t-1$ |
| Self-attention, no recurrence | $O(1)$ — every position scores every other at once |

> **Self-attention** — attention where the queries and the keys come from the *same* sequence, so
> each position scores against every other rather than against a separate encoder. Part 5 does it
> properly; here it only means "attention with no RNN wrapped around it."
>
> **Layer** — one attention operation plus the small per-position network after it, stacked $L$
> deep the way convolutional blocks stack. Part 4 builds one.

Then one layer over a 50-word sentence is a single $(50 \times d)(d \times 50)$ matmul with every
output element independent — the shape the hardware was built for.

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

And the $n^2$ isn't only arithmetic: the score matrix $E$ is an $n \times n$ tensor that has to
exist for every attention operation in the model. About a million entries at $n = 1024$; 67
million at $n = 8192$ — and a real model runs many such operations per layer, stacked over many
layers. That is why context windows stayed small for years, and it is a debt with a repayment
schedule:

| Debt | Paid by | Where |
|---|---|---|
| $O(n^2)$ **memory** | FlashAttention — computes the same result without ever storing $E$ | section 03 |
| $O(n^2)$ **compute** | sliding-window, sparse, linear attention | section 03 |
| Recomputing every earlier position for each new output word | KV caching | section 03 |

Those are named here only so you know the bill gets paid. None of them is assumed below.

---

## What the deletion cost

Now check the three jobs from the top of this part, one at a time.

| Job | Survives? | |
|---|---|---|
| Order, structurally | **no** | nothing in the computation refers to position → **positional encoding**, lesson 5 |
| Unbounded length in bounded memory | **no** | the fixed-width state becomes $n$ keys you must hold at once → that's the $n^2$ bill above |
| Parameters independent of length | **yes** | the projection matrices don't depend on $n$ either |

Two of three, and the third is why the trade was tolerable at all.

**Row 1 is the startling one.** Look at what a context vector is: $c_i = \sum_j \alpha_{ij} h_j$ —
a weighted sum over the source positions. A sum doesn't care what order its terms come in, and
$\alpha_{ij}$ is computed from the *content* of $h_j$, never from $j$ itself. So permuting the
input permutes the outputs along with it and changes nothing else:

$$\mathrm{Attn}(P X) \;=\; P\,\mathrm{Attn}(X)$$

> $P$ — a permutation matrix: $PX$ is $X$ with its rows reordered.

This is called **permutation equivariance**, and the concrete version is worse than it sounds:
in *"dog bites man"* and *"man bites dog"*, the representation attention computes for **dog** is
the same vector in both. Each word's output depends only on which words are present, not on where
any of them sit. A model with no notion of sequence, for sequence modelling — that's the hole
positional encoding exists to fill.

*(Careful with the wording: attention is permutation **equivariant**, not invariant. The outputs
do move — they follow the permutation. What's invariant is any single position's output with
respect to reordering the things it attends over.)*

Two further problems appeared that recurrence had never been responsible for: one attention
pattern per layer turns out not to be enough (**multi-head**, lesson 2), and the dot-product score
scale from part 2 grows with $d$ (**$1/\sqrt d$ scaling**, part 7). Those aren't recurrence's
jobs breaking — they're new bills from the new design. Worth keeping straight, because the
usual telling lumps all of it together.

---

## The chain

```
scores are one matmul — but only if all the queries exist at once
    → attention is already parallel; the RNN around it is not
    → is the recurrence load-bearing?  path length: nice, not decisive
                                       parallelism: decisive
    → delete it
    → 2 of its 3 jobs break: word order, and bounded memory (the n² bill)
    → 2 new problems appear: one pattern per layer, and score scale
```

Next: with the recurrence gone, what does the model actually look like? Real tensors, real
shapes, and where the attention block sits inside it.

**→ [4 · The forward pass](4-the-forward-pass.md)**
