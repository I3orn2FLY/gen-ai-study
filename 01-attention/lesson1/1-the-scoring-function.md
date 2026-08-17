# 1 · The scoring function

*~6 min. Lesson 1, part 1 of 10.*

## The problem

Here's how translation worked before 2014. An RNN — the **encoder** — reads the English sentence
and boils it down to one vector. A second RNN — the **decoder** — reads that vector and writes out
French.

```
"the cat sat on the mat"  →  [encoder]  →  [ one vector ]  →  [decoder]  →  "le chat..."
```

See the problem? That middle vector is the same size for a 4-word sentence and a 40-word one.
Everything squeezes through it, and on long sentences quality fell apart — naming that fixed-size
vector as the bottleneck is literally how the fix's paper opens.

So stop squeezing. Keep a vector for **every** input word, and let the decoder look at whichever
ones it needs at each step.

That sounds easy until you ask what "look at whichever ones it needs" means in code. You can't
write an `if` — a hard pick has no gradient, so backprop can't train it. What you *can* do is give
every input word a **relevance number**, then take a weighted average of their vectors.

That mechanism is **attention**. The number is the **score**, and this lesson is about where it
comes from and what it does.

![the fixed-vector bottleneck, and attention removing it](figures/fig1-bottleneck.png)

*(Bahdanau et al., 2014 — and the RNNs stayed. This was a patch on the existing model, not a new
one. **Origin tag: Fix** — a named failure, a targeted response.)*

---

## The setup

Running example: "the cat sat" → "le chat s'assit". Three words in, four *tokens* out —
"s'assit" gets split into "s'" + "assit"; output pieces don't have to align with words, a fact
section 02 makes precise.

> $T_x = 3$ — the number of input words. $T_y = 4$ — the number of output words. Both change from
> sentence to sentence.

The encoder reads the input word by word, as usual. The only change: **keep its state at every
position**, not just the final one.

> $h_j \in \mathbb{R}^{d_h}$ — the encoder's state at input word $j$. One per word. $d_h$ is how
> many numbers are in it.

The paper this comes from — Bahdanau, Cho and Bengio, 2014 — actually read the sentence twice,
forwards and backwards, and glued the halves together:

$$h_j \;=\; \left[\,\overrightarrow{h}_j \,;\, \overleftarrow{h}_j\,\right]$$

Each half is 1000 numbers wide, so $d_h = 2000$. Worth seeing why he bothered: reading only
forwards, $h_j$ knows words 1 through $j$ and nothing after. Reading both ways, $h_j$ knows the
**whole sentence, from word $j$'s point of view.** That matters twice below.

---

## Where the decoder starts

The decoder is an RNN too, so it needs a state to begin from — and it hasn't generated anything
yet. Chicken and egg. Bahdanau's answer:

$$s_0 \;=\; \tanh\!\left(W_s\, \overleftarrow{h}_1\right)$$

> $\overleftarrow{h}_1$ — the **backward** encoder's state at word 1. It read right-to-left, so by
> the time it got to word 1 it had seen everything. It's the one half-vector that has read the
> entire sentence.
> $W_s$ — a small learned matrix, shape $(d_s, d_h/2)$, trained with everything else.
> $d_s = 1000$ is the decoder's width — **not** the same as the encoder's 2000. Part 2 turns on
> that mismatch.

So the decoder starts from a learned summary of the input. Nothing mysterious. Any reasonable
summary works here — it's a detail, not a mechanism.

At every step the decoder also eats the word it just produced — that arrives as $y_{i-1}$ in
step 3 below — and at step 1 there is no previous word. So every target sentence gets a made-up
token `<sos>` ("start of sequence") stuck on the front, with a learned embedding like any other
word.

---

## One step of the decoder

The decoder is about to produce output word $i$. It's holding $s_{i-1}$ — the state it finished
computing last step.

> $s_{i-1} \in \mathbb{R}^{d_s}$ — everything generated so far, plus the input summary it started
> from ($s_0$), rolled into one vector. It's $i-1$ and not $i$ because $s_i$ doesn't exist yet;
> that's what this step is for.

**Step 1 — rate every input word.**

$$e_{ij} \;=\; \mathrm{score}\!\left(s_{i-1},\, h_j\right) \qquad j = 1, \dots, T_x$$

One number per input word: how relevant it is to what the decoder needs right now. It can be
anything — $-3.7$, $12.0$.

The two arguments have names you'll see everywhere from here on:

> **Query** — what the thing doing the looking wants. Here $s_{i-1}$. *"I'm about to write a French
> word. What do I need?"*
> **Key** — what a thing being looked at says about itself. Here each $h_j$. *"I'm the state for
> 'cat'"* — and because $h_j$ read both ways, that state knows its whole sentence. The second
> place bidirectionality pays.

We won't open up `score()` until part 2. But notice its shape now: **one query, one key, one number
out.** No $T_x$ inside it. The $T_x$ shows up because you run it once per input word and collect
the answers into $e_i$, a vector of length $T_x$.

**Step 2 — turn ratings into proportions.**

You can't build the weighted average from raw scores. Two things are wrong with them. They can be negative, and a
negative weight would *subtract* an input word from the mix, which isn't a thing you want. And they
have no fixed scale — double them all and the output doubles, while the *preferences* between words
haven't changed at all.

Fix both: make them positive, then make them add to 1. Exponentiate — always positive, and it keeps
the ordering — then divide by the total. That's softmax.

$$\alpha_{ij} \;=\; \frac{\exp(e_{ij})}{\sum_{j'=1}^{T_x} \exp(e_{ij'})}$$

> $\alpha_{ij}$ — a number in $[0,1]$: the **weight** on input word $j$ at step $i$. The bottom adds
> up over every input word ($j'$ is just the counter), which is what makes the row total 1. Collect
> the row into $\alpha_i$, shape $(T_x,)$.

Scores and weights are two different objects, one softmax apart. Everyone mixes up the names —
you'll see $\alpha$ called "weights" in one paper and "scores" in the next. Ask which side of the
softmax someone means.

Reading $\alpha_i = [0.02,\ 0.95,\ 0.03]$: *while writing this French word, 95% of my attention is
on "cat".*

**Step 3 — mix, then step.**

$$c_i \;=\; \sum_{j=1}^{T_x} \alpha_{ij}\, h_j \qquad\qquad
s_i \;=\; f\!\left(s_{i-1},\, y_{i-1},\, c_i\right)$$

> $c_i$ — the **context vector**: a weighted average of the encoder states, so it comes out the
> same shape as one of them.
> $f$ — the decoder's RNN cell. Three inputs: where it was, what it last said, what it just looked
> at.
> $y_{i-1}$ — the previous output word. During training this is the **correct** previous word, not
> whatever the model guessed, so one bad prediction doesn't derail the sentence. That trick is
> called **teacher forcing**.

Output word $i$ is then predicted from $s_i$, $y_{i-1}$ and $c_i$ — instead of from one frozen
vector for the whole sentence. That's the fix.

---

## The loop

```
encoder runs once:    h₁  h₂  h₃          decoder runs T_y = 4 steps:

s₀ = tanh(W_s ←h₁)                                                   ← from the encoder
step 1:   e₁ⱼ = score(s₀, hⱼ)  →  α₁  →  c₁  →  s₁ = f(s₀, y₀, c₁)  →  emit "le"
step 2:   e₂ⱼ = score(s₁, hⱼ)  →  α₂  →  c₂  →  s₂ = f(s₁, y₁, c₂)  →  emit "chat"
step 3:   e₃ⱼ = score(s₂, hⱼ)  →  α₃  →  c₃  →  s₃ = f(s₂, y₂, c₃)  →  emit "s'"
step 4:   e₄ⱼ = score(s₃, hⱼ)  →  α₄  →  c₄  →  s₄ = f(s₃, y₃, c₄)  →  emit "assit"
```

![the decoder loop, and where the query comes from](figures/fig12-decoder-loop.png)

> $y_0$ — the `<sos>` embedding. This is the job it was invented for.

No circularity — at step $i$ you're asking with a state you finished computing at step $i-1$.

But look at the green chain in that figure: $s_1 \to c_2 \to s_2 \to c_3$. You can't work out $c_2$
until $s_1$ exists, and $s_1$ needed $c_1$ first. **The decoder has to run one word at a time**, in
training and in inference. A GPU can't help — thousands of cores sit idle while the steps happen
one after another. Hold onto that — it's what part 3 is about.

---

## Two things to carry forward

**Scores are throwaway.** Computed → softmaxed → used → gone. They aren't parameters; the optimizer
never sees them. Training changes the *function* that produces them. (Throwaway isn't free, though
— they sit in memory while the pass runs, and part 5 shows how fast that grows.)

**$h_j$ is doing two jobs at once.** It appears in the score, deciding *whether* word $j$ gets
picked. Then it appears in the weighted sum, being *what you get* when it is picked. Those are
different jobs, and there's no law saying one vector has to do both. Splitting them is where the
third name comes from — part 6.

One habit worth forming now: "key" means **the side being looked at**, not "the encoder". Looks
like a pointless distinction here, where the keys obviously are encoder states. Part 4 builds
attention where both sides come from the same sentence, and "key = encoder" stops making sense.

---

```
one fixed vector for the whole sentence
    → keep every encoder state instead
    → rate each one against the decoder's state, mix by the ratings
    → ??? — nobody has said how the rating is computed
```

**→ [2 · Additive or multiplicative](2-additive-or-multiplicative.md)**
