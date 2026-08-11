# 1 · The scoring function

*~4 min. Lesson 1, part 1 of 10.*

## The problem

Translation models used to be a straight pipeline:

```
"the cat sat on the mat"  →  [RNN encoder]  →  [ one fixed vector ]  →  [RNN decoder]  →  "le chat..."
```

A 4-word sentence and a 40-word sentence get **the same one vector**. Everything squeezes through
it, and quality collapsed on long inputs.

The obvious fix: stop squeezing. Keep one vector per input word, and let the decoder pick which
ones it needs at each step. Which raises the real question — **given where I am now, which input
words should I look at?**

"Pick" isn't differentiable. But you can put a number on every candidate — how relevant is this
one right now — and blend by those numbers.

**That number is the score.**

![the fixed-vector bottleneck, and attention removing it](figures/fig1-bottleneck.png)

*(Bahdanau et al., 2014. The recurrence stayed — this was an addition to the RNN, not a
replacement. **Origin tag: Fix**.)*

---

## The encoder

Running example: "the cat sat" → "le chat s'assit". So $T_x = 3$ input words and $T_y = 4$ output
words — both change from sentence to sentence.

The encoder reads the source word by word. Keep its state at **every** position, not just the last:
$h_1, h_2, h_3$.

> $h_j \in \mathbb{R}^{d_h}$ — the encoder's state at word $j$. $d_h$ is how wide it is.

Bahdanau read the sentence twice, once each way, and glued the halves:

$$h_j \;=\; \left[\,\overrightarrow{h}_j \,;\, \overleftarrow{h}_j\,\right]$$

Each half is 1000 wide, so $d_h = 2000$. So $h_j$ isn't "the source up to word $j$" — it's the
**whole sentence, centred on word $j$**.

Keeping all of them instead of only the last one *is* the fix. The rest is about choosing between
them.

---

## Starting the decoder

The decoder is a second RNN, so it needs a state to start from — and nothing has been generated
yet.

$$s_0 \;=\; \tanh\!\left(W_s\, \overleftarrow{h}_1\right)$$

> $\overleftarrow{h}_1$ — the **backward** encoder's state at word 1. Reading right-to-left, by the
> time it gets there it has consumed the whole sentence. It's the one half-vector that has seen
> everything.
> $W_s \in \mathbb{R}^{d_s \times d_h/2}$ — a small learned matrix. $d_s$ is the decoder's width:
> 1000, against the encoder's 2000. They don't have to match, and part 2 turns on that.

Nothing circular — it's a learned summary of the source. Other implementations just average the
$h_j$. It's a bootstrap detail, not a mechanism.

The decoder also needs a previous *word*, and there isn't one, so a reserved token
$y_0 = \texttt{<sos>}$ ("start of sequence") goes in front of every target sentence.

---

## One decoder step

At step $i$ the decoder holds $s_{i-1}$ and does three things.

> $s_{i-1} \in \mathbb{R}^{d_s}$ — the decoder's state, holding everything generated so far. It's
> $i-1$ because when producing word $i$, the state you have is left over from word $i-1$.

**1 · Score every candidate.**

$$e_{ij} \;=\; \mathrm{score}\!\left(s_{i-1},\, h_j\right) \qquad j = 1, \dots, T_x$$

> $e_{ij}$ — **one number**: how relevant input word $j$ is right now. Unbounded, so $-3.7$ and
> $12.0$ are both fine.
> **Query** — what the thing looking wants. Here $s_{i-1}$. *"I'm about to write a French word;
> what do I need?"*
> **Key** — what a thing being looked at advertises. Here $h_j$. *"I'm 'cat', position 2."*

$\mathrm{score}$ stays a black box until part 2, but notice its shape: one query, one key, **one
scalar**. No $T_x$ inside it. The $T_x$ comes from running it once per input word and stacking the
answers into $e_i$, shape $(T_x,)$.

**2 · Turn scores into weights.** Raw scores can't multiply anything. Doubling all of them would
double the output while the *preferences* between words stayed the same — output size tracking
something meaningless. Mixing proportions must be positive and sum to 1:

$$\alpha_{ij} \;=\; \frac{\exp(e_{ij})}{\sum_{j'=1}^{T_x} \exp(e_{ij'})}$$

> $\alpha_{ij} \in [0,1]$ — the **weight** on input word $j$ at step $i$. The denominator runs over
> every input position ($j'$ is just the summation index), forcing the row to sum to 1.

Scores and weights are two things one softmax apart. Everyone conflates them — papers and library
code both call $\alpha$ "attention scores". Ask which side someone means.

$\alpha_i = [0.02,\ 0.95,\ 0.03]$ reads: *while writing this word, 95% of my attention is on
"cat".*

**3 · Blend.**

$$c_i \;=\; \sum_{j=1}^{T_x} \alpha_{ij}\, h_j \qquad\qquad
s_i \;=\; f\!\left(s_{i-1},\, y_{i-1},\, c_i\right)$$

> $c_i \in \mathbb{R}^{d_h}$ — the **context vector**, a weighted average of the $h_j$.
> $f$ — the decoder's RNN cell: where it was, what it last said, what it just looked at.
> $y_{i-1}$ — the previous output word. In training this is the **true** previous word, not the
> model's guess, so one bad prediction doesn't wreck the sentence. That's **teacher forcing**.

Word $i$ comes from $s_i$, $y_{i-1}$ and $c_i$ — instead of from one frozen vector.

---

## The loop

```
encoder runs once:    h₁  h₂  h₃

s₀ = tanh(W_s ←h₁)                                                   ← from the encoder
step 1:   e₁ⱼ = score(s₀, hⱼ)  →  α₁  →  c₁  →  s₁ = f(s₀, y₀, c₁)  →  emit "le"
step 2:   e₂ⱼ = score(s₁, hⱼ)  →  α₂  →  c₂  →  s₂ = f(s₁, y₁, c₂)  →  emit "chat"
step 3:   e₃ⱼ = score(s₂, hⱼ)  →  α₃  →  c₃  →  s₃ = f(s₂, y₂, c₃)  →  emit "s'"
step 4:   e₄ⱼ = score(s₃, hⱼ)  →  α₄  →  c₄  →  s₄ = f(s₃, y₃, c₄)  →  emit "assit"
```

![the decoder loop, and where the query comes from](figures/fig12-decoder-loop.png)

No chicken-and-egg: at step $i$ you attend with a state you finished computing at step $i-1$.

But follow that chain — green in the figure — and see the cost: $s_1 \to c_2 \to s_2 \to c_3$. You
can't compute $c_2$ before $s_1$, and $s_1$ needs $c_1$. **The decoder can't be parallelized over
$i$**, in training or inference. That's structural, and it's part 3's subject.

---

## Two things worth carrying

**Scores are activations, not parameters.** Built → softmaxed → used → dropped. Nothing in the
optimizer touches them; training updates the *function* that makes them. (Dropped isn't free
though — they sit in memory during the pass, and part 4 shows that becoming the dominant cost.)

**$h_j$ does two jobs.** It appears in the score *and* in the weighted sum — *how you get found*
and *what you contribute once found*. No reason those must be the same vector. Splitting them
gives a third name and a third projection, in part 6.

Also: bind "key" to **the side being looked at**, not to "the encoder". Part 5 builds attention
where both sides come from the same sequence, and "key = encoder" stops meaning anything there.

---

```
fixed-vector bottleneck
    → keep every encoder state
    → score each one against the decoder's state, mix by weight
    → ??? — nobody has said how score() actually works
```

**→ [2 · Additive or multiplicative](2-additive-or-multiplicative.md)**
