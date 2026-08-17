# 4 · The forward pass

*~6 min. Lesson 1, part 4 of 10.*

## The hole

Part 3 ended on a decision: the RNN goes. So delete it — encoder RNN, decoder RNN, the loop.

What's left is attention, and attention was never a model — only a component. The RNN supplied
its queries (decoder states), its keys (encoder states), and the loop it ran inside. There's a
hole where the model used to be.

This part fills the hole by pushing one sentence through the replacement, shape by shape:

*"The cat sat because it was tired"*

> $T$ — the sequence length: $T = 7$ here, with a vocabulary of 1000 words. One sequence now, not
> a source and a target — part 1's $T_x$ and $T_y$, and part 3's $n$, all collapse into this one
> number.

## The job comes first

Translation is off the table: the two deleted RNNs *were* the machinery relating a source
sentence to a target one. One stack over one sentence can do the one job that needs nothing
else — **predict the next word**. Position $t$'s output answers: given words $1 \dots t$, what
is word $t{+}1$?

And it answers at *every position at once* — the whole point of the deletion. A 7-word sentence
yields seven predictions in a single pass: after "The", predict "cat"; after "The cat", predict
"sat"; and so on.

Your sketch — $(T, \text{vocab}) \to (T, \text{feat}) \to (T, \text{vocab}_2)$ — had the right
skeleton. Its three details get fixed where the pipeline reaches them, starting with the last:
there is no second vocabulary. $\text{vocab}_2 = \text{vocab}$, shifted one step.

![the whole model, one block, and inside the attention box](figures/fig11-forward-pass.png)

The figure is the whole machine. The rest of this part walks it, entry to exit.

## In: words become a table of numbers

The sketch's first fix: the input is ids, not one-hot. Each word becomes an integer — a
**token id**, which is why the job is properly called *next-token prediction* — so the input is
$(7,)$: seven integers in $[0, 1000)$. Multiplying a one-hot row by a matrix just returns one row
of that matrix, so nobody builds the one-hot; the first layer is an **embedding lookup**,
`x = Emb[ids]` — rows selected from a learned $(1000, 64)$ table.

> $d_{\text{model}}$ — the model's working width, and the sketch's "feat", its second fix: 64
> here, and it's part 2's shared width $d$. Every stage from entry to exit keeps it.

Each row also gets a **position vector** added: row $t$ of a second learned table, one row per
position up to a chosen maximum length, same width. It has to ride *inside* the vector because
everything downstream reads a row's *contents*, never its index — part 5 proves nothing in the
model can tell the rows' order apart without it. Which position signals work best is lesson 5's
design question.

The sentence is now $x$, shape $(7, 64)$: one row per word, in the model's width.

## The attention box

$x$ flows into the attention op — parts 1 and 2's machinery, with a change of cast. There is no
decoder asking about an encoder anymore: queries and keys both come from the same sentence, so
**every position asks, and every position is asked about**. That's **self-attention**. ("Key"
survives the encoder's deletion because it never meant "encoder state" — only *the side being
looked at*.)

Three projections, attention's only parameters:

$$Q = xW_Q \qquad K = xW_K \qquad V = xW_V$$

> $W_Q, W_K, W_V$ — learned, each $(64, 64)$. (Capitals — these are not part 2's $W_q, W_k$,
> which lived inside the additive score.) Rows of $Q$ are queries, rows of $K$ are keys —
> part 1's words. $V$ holds the **values**: what a position hands over once it's picked. Part 1
> flagged that $h_j$ did two jobs — deciding *whether* it gets picked, and being *what you get*.
> This is the split: $K$ decides, $V$ delivers. Why that's worth three matrices is part 6's
> question.

Part 2's promissory note gets cashed here. All seven queries exist at once — no recurrence making
them wait — so every score in the sentence is one matmul:

$$E = QK^\top \qquad (7, 64)(64, 7) = (7, 7)$$

> Part 2's table $E$, gone square. Row $i$ is position $i$ asking; column $j$ is position $j$
> being asked about. $E_{ij}$ is part 1's $e_{ij}$ — all 49 of them in one shot, no loop.

## The cheat in that table

Look at row 3. Its job is to predict word 4 — and its scores cover *every* position, including
position 4, where the answer is sitting. Attend there, pass word 4's row of $V$ through, and the
loss falls to zero by copying — no language learned.

The RNN never had this problem: $s_i$ was built from $s_{i-1}$, so the future physically wasn't
wired in. "Can't peek ahead" came free with the recurrence. You deleted it, so you enforce it by
hand — **before** softmax, overwrite the future's scores:

$$E_{ij} \leftarrow -\infty \quad \text{for every } j > i$$

$\exp(-\infty) = 0$, so the future gets weight exactly 0 and each row renormalizes over positions
$\le i$ automatically. Not zero *scores* — a score of 0 is a legitimate opinion, $\exp(0) = 1$, a
full vote — and not zeroing the weights after softmax, which would leave rows summing to less
than 1. What survives is the lower triangle:

```
         asked about →
  row 1:  ■ · · · · · ·
  row 2:  ■ ■ · · · · ·
  row 3:  ■ ■ ■ · · · ·
  row 7:  ■ ■ ■ ■ ■ ■ ■
```

That's the **causal mask**. (Same trick, different target: a **padding mask** puts $-\infty$ in
the *columns* of the filler tokens that pad a batch's shorter sentences, so nothing attends to
filler.)

Now the box can finish. The full sequence: divide $E$ by $\sqrt{d} = 8$ — part 8 is entirely
about why — mask, softmax each row, blend:

$$A = \mathrm{softmax}(\text{each row}) \qquad \text{out} = AV \qquad (7,7)(7,64) = (7,64)$$

> $A$ — the weight table. Its rows are part 1's $\alpha_i$: non-negative, summing to 1. Row $i$
> of $\text{out}$ is a weighted average of the rows of $V$ — part 1's context vector $c_i$,
> except every position gets one, all computed together.

$E$ and $A$ are built, used, and discarded — every block, every forward pass. They're
activations, not parameters; only the three $W$'s persist. Part 5 weighs what that $(T, T)$ habit
costs at real lengths.

## Closing the block

Attention moved information *across* positions. The second half of the unit lets each position
**process what it gathered, by itself**: an **MLP** — two linear layers with a nonlinearity
between — applied to each row independently, returning to width 64.

Wire the two halves together and you have the unit the whole model stacks (the figure's middle
panel). Part 3's "layer" was the attention op alone; wrapped with its MLP, the stackable unit is
called a **block**:

LayerNorm → attention → add the result back to $x$ · LayerNorm → MLP → add back again

The add-backs are **residual connections** — ResNet's move: each half computes a *correction* to
its input, not a replacement, so stacking blocks doesn't re-earn what earlier ones built. **LayerNorm**
rescales each row to a steady size before each half — batchnorm's job, done per row, no batch
statistics. Neither is this lesson's subject.

The whole block in one sentence: **every position looks at every other position, then each
position thinks by itself.**

Our model stacks two of them. $x$ enters block 1 as $(7, 64)$ and leaves block 2 as $(7, 64)$ —
same shape, each row now carrying what it chose to gather from the sentence.

## Out: the table becomes predictions

One step left: turn row $t$ — 64 numbers — into a verdict over 1000 words. A final LayerNorm
steadies the scale, then the **output projection**: one last learned matrix. Not another MLP —
the MLPs live inside blocks and return to width 64. This is a single multiply, the model's exit:

$$\text{logits} = xW_{\text{out}} \qquad (7, 64)(64, 1000) = (7, 1000)$$

> **logits** — one raw, unbounded score per vocabulary word, per position. $x$ here is the final
> LayerNorm's output, still $(7, 64)$; $W_{\text{out}}$ is learned, like everything else here.
> Softmax row $t$ and you have a probability distribution for word $t{+}1$ — softmax doing
> part 1's scores-to-proportions job one last time.

And the model **stops there**. No id is picked inside it. Training grades every position at
once — the loss reads the probability each row gave to the word that actually came next (row 7's
"next word" is whatever followed the sentence in the training text). Generation uses only the
last row, and the picking happens in a loop *around* the model: take the distribution's argmax
every time ("greedy") or sample from it — section 03 compares the pickers — append the id, run
the whole pass again.

## What you built

| Stage | Shape | What happens |
|---|---|---|
| token ids | $(7,)$ | seven integers |
| embedding + position | $(7, 64)$ | row-select from the $(1000, 64)$ table, add position vectors |
| block 1 | $(7, 64)$ | mix across positions, then think per position |
| block 2 | $(7, 64)$ | same structure, its own weights |
| LayerNorm | $(7, 64)$ | steady the scale before the readout |
| output projection | $(7, 1000)$ | one learned $(64, 1000)$ matrix → logits |

**This is a transformer**: an embedding, a stack of identical blocks, a projection. Nothing else.
(Strictly, the stripped-down decoder-only variant of Vaswani et al.'s 2017 design; lesson 2
starts restoring the rest.)

It runs, it trains, every position in parallel. What nobody has done yet is count what the
deletion actually cost — starting with the claim you'll hear most often, which is false.

**→ [5 · What deleting it cost](5-what-it-cost.md)**
