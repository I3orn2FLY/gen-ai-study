# 3 · Why not recurrence

*~4 min. Lesson 1, part 3 of 10.*

## The problem

Part 2 ended with something you can see but can't have. Every score in a whole translation is one
matrix multiply, $E = QK^\top$ — except $Q$ never exists, because the decoder produces its states
one at a time.

The only way to get it is to throw out the RNN.

Which should sound reckless. The RNN *is* the model. So before touching it: what is it actually
doing for you?

---

## What the recurrence was giving you

An RNN is just this, over and over:

$$h_t \;=\; \mathrm{cell}(h_{t-1},\, x_t)$$

> $x_t$ — the embedding of input word $t$. $h_t$ — the state after reading it. $\mathrm{cell}$ —
> the recurrent unit (plain RNN, LSTM, GRU), one set of weights reused at every step. It's the same
> kind of thing part 1 called $f$.

Three things fall out of that shape for free:

- **It knows word order.** The step number $t$ is baked into the wiring. Nothing has to *represent*
  order because nothing can avoid it.
- **Any length, same memory.** One state vector carries the sentence, whether it's 5 words or 500.
- **Length doesn't change the parameter count.** Same cell every step, so a short sentence and a
  long one use exactly the same weights.

Remember those three. Part 5 is the invoice, and one of them survives.

---

## An RNN is secretly a deep network

Unroll the loop and stop reading it as a loop:

$$h_0 \;\to\; h_1 \;\to\; h_2 \;\to\; \cdots \;\to\; h_n$$

with a new word entering at every arrow. ($n$ is the sentence length; part 1's $T_x$ and $T_y$ are
the input and output versions of it.) Squint and **step $t$ is layer $t$** — it's a deep
feedforward network wearing a disguise:

| | Ordinary deep net | Unrolled RNN |
|---|---|---|
| Weights | one matrix per layer | **one matrix shared by all layers** |
| Input | enters at layer 1 | enters at **every** layer |
| Depth | you choose it | **the sentence length chooses it** |

That's not a cute analogy, it's the reason the two fields found the same problem at the same time.
"Vanishing gradients in deep networks" and "vanishing gradients in RNNs" are one result, not two
(Hochreiter 1991; Bengio, Simard, Frasconi 1994).

---

## Argument 1 — the long way round

For word 1 to influence word 7, its information has to be carried through every state in between.
Six hops. Twenty words apart, twenty hops. Attention just scores the two against each other
directly — **one hop, no matter the distance.**

![path length: walking versus jumping](figures/fig2-path-length.png)

> **Path length** — how many computation steps something passes through to get from one position to
> another. Distance in the network, not in the sentence.

### Why the long way hurts

It's a gradient problem. To send a learning signal from step $n$ back to step 1, backprop
multiplies together one matrix per step in between:

$$\frac{\partial h_n}{\partial h_1} \;=\; \prod_{t=2}^{n} \frac{\partial h_t}{\partial h_{t-1}}$$

> Each factor says how much the state at one step responds to the state at the previous one.
> There are $n-1$ of them.

Multiplying $n-1$ similar matrices is like raising one number to the power $n-1$. If that number is
a bit below 1, the result collapses toward zero. A bit above 1 and it explodes. And the exponent is
the sentence length, which you don't control.

Only one direction is a real problem:

| | Fix |
|---|---|
| Exploding | **gradient clipping** — if the gradient comes back too big, scale it down. One line, standard by 2013 |
| Vanishing | *nothing equivalent* |

You can shrink something that's too large. You can't recover something that already went to zero —
the *direction* is gone, not just the size. So "the RNN gradient problem" means vanishing.

### Gating shortens the trip, it doesn't cancel it

This is what LSTMs and GRUs were built for. An LSTM keeps a second vector alongside the state, a
memory line:

$$c_t \;=\; f_t \odot c_{t-1} \;+\; i_t \odot \tilde{c}_t$$

> $c_t$ — the **cell state**, the memory line. $f_t$ — the **forget gate**, numbers between 0 and 1
> saying how much old memory to keep. $i_t \odot \tilde c_t$ — how much new material to let in.
> $\odot$ multiplies elementwise. (Letters get reused here and in every paper: this $c_t$ isn't
> part 1's context vector, and $f_t$ isn't part 1's cell function.)

When the forget gate sits near 1, that line reads $c_t \approx c_{t-1} + \text{something}$ —
addition, not multiplication. Nothing shrinks. The gradient gets a road with no tolls on it. (That
gate wasn't in the original 1997 LSTM; Gers et al. added it in 2000.)

So gating, residual connections and attention are all the same idea: give the signal a shortcut so
it doesn't have to survive a long chain. The first two make the trip easier; attention deletes the
trip.

**Now the catch.** This is the argument everyone reaches for, and it's *not* the one that mattered.
Stacked LSTMs held the state of the art in translation right up to 2017 — Google's production
system was one. Long-range quality was not what had the field stuck. Path length is real, it's
satisfying, and it decided nothing.

---

## Argument 2 — the queue

Here's the one. And the way in is to notice that the model you already have is **half parallel
already**, and the parallel half is the attention.

Take one decoder step. The thing that costs wall-clock time isn't how much arithmetic there is,
it's which line has to wait for the *previous word* to finish:

| Step | Shape | Spread over the $T_x$ keys? | Makes the next step wait? |
|---|---|---|---|
| $K_{\text{proj}} = H W_k^{\top}$ | $(T_x, d_a)$ | yes — and done once, before the loop | no |
| $Z = \tanh(W_q s_{i-1} + K_{\text{proj}})$ | $(T_x, d_a)$ | yes | no |
| $e_i = Z v$ | $(T_x,)$ | yes | no |
| $\alpha_i = \mathrm{softmax}(e_i)$ | $(T_x,)$ | yes | no |
| $c_i = \alpha_i^{\top} H$ | $(d_h,)$ | yes | no |
| $s_i = \mathrm{cell}(s_{i-1},\, y_{i-1},\, c_i)$ | $(d_s,)$ | — | **yes — it produces $s_i$** |

Only the last row puts anything into the chain $s_0 \to s_1 \to s_2 \to \cdots$. Everything above
it fans out across all the input positions at once, and doesn't get slower as the sentence gets
longer. **Attention is already the parallel part.** The RNN cell is the bottleneck.

Add it up over one training example: $T_x$ steps for the encoder, $T_y$ for the decoder, each one a
small matrix-times-vector. A GPU with thousands of cores does one, sits idle, does the next.

So the real question isn't "can attention be parallel". It already is. It's: **what if the parallel
part were the whole model?**

And notice what that fixes. Part 2 couldn't batch the queries because each one had to wait for the
previous state. Take the RNN out and a position's query comes from its own input — the word
embedding, or whatever the layer below produced there. They all exist before you start.

> **Self-attention** — attention where the queries and keys come from the *same* sentence, so every
> position scores every other one instead of scoring a separate encoder.
> **Layer** — one attention operation over a whole sentence, the kind you stack many of. (Narrower
> than "layer" in "4 stacked LSTM layers"; same word, new unit.)
> Both get built in part 4. Here they just mean "attention with no RNN wrapped around it."

| Per layer | Steps that must happen in order |
|---|---|
| Recurrence | $n$ — state $t$ needs state $t-1$ |
| Self-attention | $1$ — every position scores every other at once |

Scoring one such layer over a 50-word sentence is a single $(50 \times d)(d \times 50)$ matrix
multiply, every entry independent. That is exactly the shape GPUs are built for.

### And teacher forcing doesn't save the RNN

Worth being precise, because this is the part that settles it. During training you already know the
whole target sentence, so you might think the decoder could do all positions at once. It can't:
$s_i$ still needs $s_{i-1}$. Knowing the *inputs* in advance doesn't help when the *states* form a
chain.

That's exactly the complaint in *Attention Is All You Need* (Vaswani et al., 2017): recurrence
"precludes parallelization within training examples" (§1). And what the abstract promises is time,
not quality — more parallelizable, significantly less time to train.

> **The RNN wasn't replaced for learning badly. It was replaced for training slowly.**

*(**Origin tag: Fix** — a training-time constraint, not a modelling one.)*

---

```
every score is one matmul — but only if all the queries exist at once
    → attention is already parallel; the RNN around it is not
    → is the RNN load-bearing?   long-range quality:  fine, LSTMs handled it
                                 training throughput: no, and that's the blocker
    → delete it
```

Decided. Delete it — and realize there is now no model left to run. Building its replacement
comes first; the bill comes right after.

**→ [4 · The forward pass](4-the-forward-pass.md)**
