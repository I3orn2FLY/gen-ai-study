# 3 · Why not recurrence

*~5 min. Lesson 1, part 3 of 9.*

## The problem

Part 2 ended holding a promissory note. The whole score table for a translation is one matmul:

$$E \;=\; Q K^{\top}$$

but $Q$ never exists, because $s_i$ needs $c_i$ needs $s_{i-1}$ — the queries arrive one at a
time. To collect, you have to delete the recurrence.

That should feel reckless. The recurrence *is* the model. It's what makes the encoder read word
order, what carries context along the sentence, what a decade of sequence modelling was built
on. "Delete it so my matmul is bigger" is not an argument.

So: what was recurrence actually giving you, and is any of it worth keeping?

Two answers get given, and they are usually said in the same breath as if they were one. They
are not, and only one of them decided the outcome.

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

It's a gradient argument. Backpropagating from step $n$ to step 1 multiplies together the
Jacobian of every step in between:

$$\frac{\partial h_n}{\partial h_1} \;=\; \prod_{t=2}^{n} \frac{\partial h_t}{\partial h_{t-1}}$$

> $\dfrac{\partial h_t}{\partial h_{t-1}}$ — the Jacobian of one recurrent step: the matrix of
> partial derivatives saying how each component of the new state responds to each component of
> the old one. One per step, $n-1$ of them multiplied.

A product of $n$ matrices behaves roughly like $\sigma^n$, where $\sigma$ is a typical singular
value. Below 1, it collapses toward zero — the **vanishing gradient**: word 1 gets no learning
signal from an error at word 40. Above 1, it explodes. There's no comfortable setting, because
the exponent is the sentence length and you don't control that.

This is the classic problem, and it's why LSTMs and GRUs exist — their gating gives the gradient
a more direct route along the state chain and pushes the usable range out a long way. **It does
not remove the path.** Signal still traverses $O(|i-j|)$ steps; the steps are just gentler.

Attention makes the path length 1 and the question disappears.

**Now the trap.** This is the argument everyone reaches for, and it is *not* the one that
mattered. LSTMs were already good enough at long range that translation quality wasn't the thing
holding the field back. Path length is a real advantage and a satisfying story. It didn't decide
anything.

---

## Argument 2 — parallelism

This is the one.

> **Sequential operations** — the length of the longest chain of steps that *must* happen in
> order, one after another, inside a single layer. It's what sets wall-clock time, because
> nothing on the chain can start before the previous link finishes.

| Per layer | Sequential operations |
|---|---|
| Recurrence | $O(n)$ — state $t$ needs state $t-1$ |
| Attention | $O(1)$ — every position scores every other at once |

An RNN over a 50-word sentence is 50 dependent steps, and no hardware can compress them. A GPU
with thousands of cores runs one small matvec at a time and idles through the rest, then does it
again 49 times.

Attention over the same sentence is one $(50 \times d) \times (d \times 50)$ matmul. Every
output element is independent, so all of them go at once, on hardware designed for exactly that
shape.

That's the argument. Vaswani et al. put it in the abstract of *Attention Is All You Need*:
recurrence "precludes parallelization within training examples." Not "recurrence learns badly" —
**recurrence is slow to train**, and training throughput is what converts compute and data into
model quality.

*(**Origin tag: Fix** — a named failure with a targeted response. The failure was a training-time
constraint, not a modelling one.)*

---

## The trap in the complexity table

Here's where the usual telling goes wrong. Vaswani's Table 1, for one layer, sequence length $n$
and width $d$:

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
lose an interview. Attention won while being *more expensive*, because its extra work is
parallel and the RNN's cheaper work is serialized. A GPU would rather do 2× the FLOPs all at
once than 1× of them in a queue.

That distinction — total work versus critical path — is worth keeping. It explains most of what
this roadmap covers later.

---

## What else the RNN was doing

Parallelism was the reason to delete it. But the recurrence had quietly been holding down three
other jobs, and pulling it out broke all of them:

| What broke | Patched with | Covered in |
|---|---|---|
| Nothing represents word order any more | positional encoding | lesson 5 |
| One attention pattern per layer isn't enough | multi-head | lesson 2 |
| Dot-product score scale grows with $d$ (part 2) | $1/\sqrt{d}$ scaling | part 7 |

The first one is the startling one. Strip the recurrence and attention is a **set** operation:
$\sum_j \alpha_{ij} v_j$ doesn't care what order the $j$'s come in. Shuffle the words of the
input and every output is identical. A model with no notion of sequence, for sequence
modelling — that's the hole positional encoding exists to fill.

So the honest framing of the transformer: **not three good ideas, but one idea and three repairs
it forced.**

---

## What it cost

Look at that $n^2$ again. It isn't only arithmetic: the score matrix $E$ is an $n \times n$
tensor that has to exist, per head, per layer.

At $n = 1024$ that's about a million entries per head; at $n = 8192$, 67 million. Memory grows
quadratically with context length, which is why context windows were small for years and why so
much engineering since has gone into this one number:

| Debt | Paid by | Where |
|---|---|---|
| $O(n^2)$ **memory** | FlashAttention — never materialize $E$ | section 03 |
| $O(n^2)$ **compute** | sliding-window, sparse, linear attention | section 03 |
| Recomputing everything each generated token | KV cache | section 03 |

Deleting the recurrence traded a *sequential* bottleneck for a *quadratic* one. That was a good
trade — parallel and quadratic beats serial and linear on this hardware — but it was a trade,
and the bill still arrives.

---

## The chain

```
scores are one matmul — but only if all the queries exist at once
    → the recurrence prevents that
    → is recurrence load-bearing?  path length: nice, not decisive
                                   parallelism: decisive
    → delete it
    → now nothing knows word order, and n² memory is your problem
```

Next: with the recurrence gone, what does the model actually look like? Real tensors, real
shapes, and where the attention block sits inside it.

**→ [4 · The forward pass](4-the-forward-pass.md)**
