# 4 · The forward pass

*~6 min. Lesson 1, part 4 of 10.*

## The problem

Part 3 ended on a decision: the RNN goes. So delete it — encoder RNN, decoder RNN, the loop.

What's left is attention — which was never a model, only a component: the RNN supplied its queries
(decoder states), its keys (encoder states), and the loop it ran inside. There's a hole where the
model used to be.

You sketched a candidate shape for what fills it:

$$(T, \text{vocab}) \;\to\; (T, \text{feat}) \;\to\; (T, \text{vocab}_2)$$

The skeleton is right. This part builds it, shape by shape.

> $T$ — the sequence length. One sentence now, not a source and a target, so part 1's $T_x$ and
> $T_y$ — and part 3's $n$ — all collapse into this one number. Running example:
> *"The cat sat because it was tired"* — $T = 7$, and a vocabulary of 1000 words.

---

## The skeleton, graded

What's right about it: a table with one row per word goes in, gets transformed, and a table with
one prediction per word comes out. No row waits for another row — part 3's parallelism demand,
made structural.

Three details to fix.

**The input is ids, not one-hot.** Each word becomes an integer — a **token id** — so the input is
just $(7,)$: seven integers in $[0, 1000)$. Multiplying a one-hot row by a $(1000, 64)$ matrix
returns one row of that matrix, so nobody builds the one-hot at all. Layer 1 is an **embedding
lookup**: `x = Emb[ids]`, a row-select from a learned table.

**"feat" has a name.**

> $d_{\text{model}}$ — the model's working width. Every stage between entry and exit keeps it.
> Here $d_{\text{model}} = 64$, and it's part 2's shared width $d$: queries and keys will inherit
> it directly.

Each row also gets a **position vector** added — a learned table with one row per position, up to
a chosen maximum length, same width. That's the whole mechanism; which position signals work
best is lesson 5's design question. Part 5 proves nothing downstream can tell the rows' order
apart without one.

**There is no second vocabulary.** Translation needed two sequences, and the deleted RNNs *were*
the machinery relating them. One stack over one sentence gets the one job that needs nothing else:
**predict the next token**. Row $t$ of the output answers: given tokens $1 \dots t$, what is token
$t{+}1$? So $\text{vocab}_2 = \text{vocab}$, shifted by one.

---

## The stack

![the whole model, one block, and inside the attention box](figures/fig11-forward-pass.png)

| Stage | Shape | What happens |
|---|---|---|
| token ids | $(7,)$ | seven integers |
| embedding + position | $(7, 64)$ | row-select from the $(1000, 64)$ table, add position vectors |
| block 1 | $(7, 64)$ | mix across positions, then think per position — below |
| block 2 | $(7, 64)$ | same structure, its own weights |
| LayerNorm | $(7, 64)$ | steady the scale before the readout |
| output projection | $(7, 1000)$ | times a learned $(64, 1000)$ matrix |

**This is a transformer**: an embedding, a stack of identical blocks, a projection. Nothing else.
(Strictly, the stripped-down decoder-only variant of Vaswani et al.'s 2017 design; lesson 2
starts restoring the rest.)

> **logits** — the $(7, 1000)$ output: one raw, unbounded score per vocabulary word, per position.
> Softmax row $t$ and you have a probability distribution for token $t{+}1$ — the same
> scores-to-proportions job softmax did in part 1.

---

## Inside a block

Two halves, one sentence: **every position looks at every other position, then each position
thinks by itself.** (Part 3's *layer* was the attention op alone; wrapped with its second half,
the stackable unit is called a **block**.)

Attention is the first half — the *only* place in the whole model where rows interact. The second
half is an **MLP**: two linear layers with a nonlinearity between, applied to each row
independently. Mix, then process what you gathered.

The wiring around them (middle panel of the figure): LayerNorm → attention → add the result back
to the input, LayerNorm → MLP → add back again. The add-backs are **residual connections** — same
move as ResNet: each half computes a *correction* to $x$, not a replacement, so stacking many
blocks doesn't have to re-earn what earlier ones built. LayerNorm rescales each row to a steady
size before each half — batchnorm's job, done per row, no batch statistics. Neither is this
lesson's subject.

---

## Inside the attention box

The block hands attention $x$, shape $(7, 64)$. It makes three projections:

$$Q = xW_Q \qquad K = xW_K \qquad V = xW_V$$

> $W_Q, W_K, W_V$ — learned, each $(64, 64)$: from $d_{\text{model}}$ to $d_{\text{model}}$ here.
> These three matrices are attention's **only** parameters.
> Rows of $Q$ are queries, rows of $K$ are keys — part 1's words, except now **every position is
> both**: it asks, and it is asked about.
> $V$ — the **values**: what a position hands over once it's picked. Part 1 flagged that $h_j$ did
> two jobs — deciding *whether* it gets picked and being *what you get*. This is the split: $K$
> decides, $V$ delivers. Why splitting is worth three matrices is part 6's question.

Now part 2's promissory note gets cashed. All the queries exist at once — no recurrence is making
them wait — so every score in the sentence is one matmul:

$$E = QK^\top \qquad (7, 64)(64, 7) = (7, 7)$$

> Part 2's table $E$, gone square. Row $i$ is position $i$ asking; column $j$ is position $j$
> being asked about. $E_{ij}$ is part 1's $e_{ij}$, all 49 of them in one shot, no loop.

This is **self-attention**: queries and keys from the same sentence. Part 1's habit pays off here
— "key" never meant "encoder state", it meant *the side being looked at*, and now there is no
encoder.

Three small steps finish the box: divide $E$ by $\sqrt{d} = 8$ (part 8 is entirely about why),
apply the mask below, then softmax each row:

$$A = \mathrm{softmax}(\text{each row}) \qquad \text{out} = AV \qquad (7,7)(7,64) = (7,64)$$

> $A$ — the weight table. Its rows are part 1's $\alpha_i$: non-negative, each row summing to 1.
> Row $i$ of $\text{out}$ is a weighted average of all rows of $V$ — part 1's context vector
> $c_i$, except every position gets one, all computed together.

The residual adds $\text{out}$ back into $x$, and the block moves on to its MLP.

$E$ and $A$ are built, used, and discarded — every block, every forward pass. They're activations,
not parameters; only $W_Q, W_K, W_V$ persist. Part 5 weighs what that table costs at real sentence
lengths.

---

## The mask

There's a cheat sitting in plain sight. Training runs all 7 positions at once — that was the whole
point of the deletion. Row $t$'s job is to predict token $t{+}1$. But row $t$'s query scores
*every* position, including $t{+}1$. **The answer is in the input, one row down.** The loss can
be driven to zero by copying it — no language required.

The RNN never had this problem. $s_i$ was built from $s_{i-1}$, so the future physically wasn't
wired in — "can't peek ahead" came free with the recurrence. You deleted it, so now you enforce it
by hand.

Before the softmax, overwrite the future's scores:

$$E_{ij} \leftarrow -\infty \quad \text{for every } j > i$$

$\exp(-\infty) = 0$, so after softmax those positions get weight exactly 0, and each row
renormalizes over positions $\le i$ automatically. Not zero *scores* — a score of 0 is a
legitimate opinion ($\exp(0) = 1$, a full vote) — and not zeroing $A$ afterwards either, which
would leave rows summing to less than 1. $-\infty$, before. What survives is the lower triangle:

```
         asked about →
  row 1:  ■ · · · · · ·
  row 2:  ■ ■ · · · · ·
  row 4:  ■ ■ ■ ■ · · ·
  row 7:  ■ ■ ■ ■ ■ ■ ■
```

That's the **causal mask**. Same trick, different target: a training batch pads its shorter
sentences with a filler token, and a **padding mask** puts $-\infty$ in the filler's *columns* so
nothing attends to it.

---

```
delete the RNN → no model left
    → embed the ids, add position vectors            (T, d)
    → stack blocks: mix across positions, then think per position
         Q = xW_Q   K = xW_K   V = xW_V
         E = QKᵀ  → /√d → mask → softmax → A → out = AV
    → project to logits                              (T, vocab)
    → this is the transformer
```

It runs, it trains, every position in parallel. What nobody has done yet is count what the
deletion actually cost — starting with the claim you'll hear most often, which is false.

**→ [5 · What deleting it cost](5-what-it-cost.md)**
