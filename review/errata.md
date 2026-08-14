# Errata

Every factual error found in material that had already been written, and what replaced it.

**Why this file exists.** The expensive failure mode in this curriculum is not a confusing
explanation — that gets flagged the moment it's read. It's a *wrong fact that reads fluently*,
survives review, and gets built on for ten lessons. This log makes corrections visible instead
of silent, so a claim that was wrong once doesn't quietly come back.

Rules:

- A correction is logged **whether or not** anyone noticed it in the material. Errors Claude
  finds while writing a later part count.
- Record where the claim had already **propagated** — other parts, figures, checker comments,
  quiz questions. That's the actual work; the fix itself is usually one line.
- If a wrong claim was ever **quizzed**, add the corrected version to `review/questions.md` as
  its own question. A wrong answer that was rehearsed needs overwriting, not just deleting.

---

## 2026-08-10 — Section 01, lesson 1, part 1

Four errors, all introduced when part 1 was first written and all found while writing part 2.
None had been quizzed; nothing beyond part 1 existed yet, so propagation was limited to part 1
and one figure.

| # | Claimed | Actually | How it was found |
|---|---|---|---|
| 1 | $h_j$ is "a summary of the source as of position $j$" | Bahdanau's encoder is **bidirectional** — $h_j = [\overrightarrow{h}_j ; \overleftarrow{h}_j]$, so $h_j$ sees the *whole* sentence, centred on word $j$ | Writing the $s_0 = \tanh(W_s \overleftarrow{h}_1)$ explanation, which is unstatable without the backward pass |
| 2 | "$d$ is the width, $512$ in the paper" | $512$ is **Vaswani's** $d_{\text{model}}$, not Bahdanau's anything. Bahdanau: $1000$ units per direction, so $d_h = 2000$, and the decoder is $d_s = 1000$ | Same pass. The single symbol $d$ was hiding the fact that the two widths differ |
| 3 | Additive scoring "runs once per pair — 12 tiny forward passes" | It **batches over the keys fine**: $W[\,q;k\,] = W_q q + W_k k$, so it's one broadcast add. The real cost is the materialized $(T_x, d_a)$ intermediate | Kenessary asked for the shapes. Writing them out made the claim collapse |
| 4 | `fig1-bottleneck` labelled the seq2seq bottleneck "512 numbers" | Unsourced. Sutskever and Cho both used $1000$; $512$ came from nowhere | Audit prompted by #2 |

**Consequences beyond the fix.** Error 2 caused a real notation bug, not just a wrong number:
part 1 wrote both the encoder and decoder states as width $d$, which made $W_s$'s shape wrong
and hid the mismatched-width property that is the *entire* advantage of additive scoring. Part 1
now carries $d_h$ and $d_s$ separately, and part 2 uses the mismatch as an argument.

Error 3 was the one with teeth — it was a *cost* claim, and cost claims are how this course
justifies which technique won. It has been replaced by the intermediate-tensor argument, which
is both true and the one that actually scales to self-attention.

**Rule added in response** (`TEACHING.md` § Writing the material, and `CLAUDE.md`): verify claims
rather than recalling them; write the tensor shapes before writing a sentence about cost; prefer
the symbol to the number unless the number teaches something.

---

## 2026-08-11 — Section 01, lesson 1, part 3

Found the day part 3 was written, while Kenessary was reading it. Not quizzed. Propagation was
limited to part 3 itself; `README.md`'s interview-question 6 states the parallelism claim without
attributing it to a location, so it needed no change.

| # | Claimed | Actually | How it was found |
|---|---|---|---|
| 5 | Vaswani et al. put it "in the **abstract**": recurrence "precludes parallelization within training examples" | That sentence is in **§1, the Introduction**. The abstract makes the weaker, related claim — the model is "more parallelizable and requiring significantly less time to train" | Fetched the paper. Both quotes are now used, each attributed to the right place |
| 6 | The sequential-operations table compared "Recurrence $O(n)$" against **"Attention $O(1)$"** | $O(1)$ holds only for **self-attention with the recurrence removed**. The attention taught up to that point sits inside two RNNs, so a training example costs $O(T_x) + O(T_y)$ sequential steps | Kenessary: "at that point it is still sequential for both encoder and decoder" |

**Consequences beyond the fix.** Error 6 is the serious one — a **forward reference disguised as
a fact**. Part 3's whole job is to *earn* the deletion of recurrence, and an unqualified
"Attention — $O(1)$" row assumes the deletion has already happened, which is the thing being
argued for. Part 3 now derives the argument from the model actually in hand: within one Bahdanau
decoder step every operation except $s_i = f(s_{i-1}, y_{i-1}, c_i)$ is parallel over the source
positions, so *attention is already the parallel part* and the RNN is the only serial one. The
counterfactual — "what if the parallel part were the whole model?" — then belongs to the reader
rather than to Vaswani.

**Structural defect fixed in the same pass** (not a factual error, logged because it caused the
same misreading twice): part 3 criticised recurrence for two sections before ever saying what
recurrence was *for*. The three jobs it did — order for free, unbounded length in bounded memory,
parameters independent of length — appeared only at the end, *after* the decision to delete it.
They are now stated second, before any criticism, and the closing section is explicitly the bill
for that list.

**Related claim checked and confirmed correct:** Bahdanau's abstract motivates attention by the
**fixed-length vector bottleneck** — it does not mention vanishing gradients or long-range
dependencies. Part 1's framing stands.

---

## 2026-08-11 — Section 01, lesson 1, part 3 (second pass)

Found by the first **cold-read audit** — an agent reading parts 1–3 with no context, reporting
unearned machinery, undefined symbols, and non-derivable claims. Both errors below were
introduced *the same day*, by the rewrite that fixed errors 5 and 6. Not quizzed; caught before
Kenessary re-read the part.

| # | Claimed | Actually | How it was found |
|---|---|---|---|
| 7 | "Shuffle the words of the input and every output is **bit-identical**" | Attention is permutation **equivariant**, not invariant: $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$. The outputs *do* move — they follow the permutation. What's invariant is one position's output with respect to reordering the things it attends over. ("Bit-identical" is also false in floating point, where reordering a sum changes the result) | Cold-read audit |
| 8 | "the recurrence had been holding down three other jobs, and pulling it out **broke all three**" | Two of the three break. **Parameters independent of length survives** — attention's projections don't depend on $n$ either. And the table listed as the "three broken jobs" didn't match the three jobs stated 200 lines earlier: two of its rows (multi-head, $1/\sqrt d$) were never recurrence's responsibility | Cold-read audit |

**Why 7 matters more than it looks.** Permutation equivariance is the standard statement and the
one an interviewer will expect; "invariance" is the common wrong version. The corrected part now
states the equivariance identity, gives the concrete form (*dog bites man* / *man bites dog* — the
representation of **dog** is the same vector in both), and carries an explicit note about the
distinction, because the wrong version is easy to re-derive from a half-memory.

**Why 8 matters.** It was a *self-consistency* failure — the part contradicted its own list
inside one document. The "one idea and three repairs" framing had been inherited from an earlier
draft and was never re-checked when the "what recurrence was for" section was added. Part 3 now
walks the three jobs one at a time and separates them from the two problems the new design
introduced on its own.

**Outstanding from the same audit — parts 1 and 2, not yet fixed:**

| Where | Problem |
|---|---|
| 2, dot-product section | $e_i = K\,s_{i-1}$ with $K \in \mathbb{R}^{T_x \times d}$ requires $d_h = d_s$ — which the same part declares impossible for Bahdanau 100 lines earlier. It silently switches to Luong's architecture while keeping Bahdanau's symbols |
| 2 | $d_a = 1000$ in one paragraph, $d_a = 64$ in the next, unexplained. The 67M figure depends entirely on the substitution |
| 2 | $d$ used ~12 times and never defined; $K$ used 120 lines before its definition; $H$, $W_a$, $v_a$ never defined; $W_a$ silently changes shape between two rows of the same table |
| 2 | The central "why the dot product won" verdict rests on self-attention, *head* and *layer* — all unearned in part 2. Fixable by arguing over $T_y$ queries instead of $n$, which needs nothing beyond Bahdanau |
| 1 | "Query" used at L76, defined at L189. Same for "the transformer", used as an established object and never introduced |
| 1 | $k$ is both the softmax summation index and the key vector; $T_x$, $T_y$ introduced by value, never defined as symbols |
| 1 vs 3 | Self-attention promised in "part 4" by part 1 and "part 5" by part 3; *value* promised for part 5 but used in part 3 |

The last row is why part 3 now writes the permutation argument with $c_i = \sum_j \alpha_{ij} h_j$
rather than $\sum_j \alpha_{ij} v_j$ — $h_j$ is defined, $v_j$ isn't yet, and $v$ already means
part 2's learned additive vector.

---

## 2026-08-11 — Section 01, lesson 1, second cold-read audit

Run against the state that had just been fixed. Everything below was found by the audit except
#9a, which was found by running the arithmetic in a shell before committing.

| # | Claimed | Actually |
|---|---|---|
| 9a | "$50\cdot50\cdot1000$ = **2.5 billion** intermediate values" | **2.5 million.** Off by $1000\times$, in the one sentence whose job was to make the magnitude vivid |
| 9b | The permutation argument, derived from $c_i = \sum_j\alpha_{ij}h_j$ and stated as $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$ | **That equation gives invariance of $c_i$, not equivariance.** Equivariance needs one output *per position*, i.e. self-attention, which part 3 has only glossed. Correct claim #7 → wrong derivation for it |
| 10 | "the fixed-width state becomes $n$ keys you must hold at once → that's the $n^2$ bill above" | Holding $n$ keys is $O(nd)$ — **linear**. The $n^2$ is the score table, a different object with a different cause. The arrow equated a linear cost with a quadratic one |
| 11 | "a layer is one attention op **plus the per-position network after it**" … "one layer is **a single** $(50\times d)(d\times 50)$ matmul" | Self-contradiction three lines apart, and the definition leaned on a feed-forward block that doesn't exist yet in this lesson |
| 12 | Part 1: "Scores are **never stored**." Part 3: $E$ "**has to exist**", and FlashAttention's whole point is not storing it | Both true in different senses — not *parameters*, not persisted between sentences, but they do occupy memory during the pass and must survive to the backward pass. Unreconciled, the memory section reads as unmotivated |
| 13 | "in `concat` it's **part 1's** $W$" | $W$ is introduced in part 2. Part 1 has no $W$, only $W_s$ |
| 14 | "the gap only widens. **Part 3 puts a number on it.**" | Part 3 puts numbers on $n^2d$ vs $nd^2$ and on $n\times n$ memory — never on the additive-vs-dot gap. Promise never cashed |

Symbol collisions also fixed: $f$ was the decoder cell, a generic recurrent cell, *and* the forget
gate $f_t$; $c$ was both context vector and LSTM cell state; $d_h$ meant Bahdanau's concatenated
bidirectional width in part 1 and a plain state width in part 3; $E$ went from $T_y\times T_x$ to
$n \times n$ without a note; $i$/$j$ switched from (decoder step, source position) to two
positions in one sequence; $d_e$, $L$, $h_0$, $X$ and $\mathrm{Attn}(\cdot)$ were used undefined.

**The pattern worth naming.** Errors 7–8 were introduced by the fix for 5–6; 9b was introduced by
the fix for 7. Three rounds, each one clean on the thing it targeted and wrong somewhere new.
A *revision* is as likely to introduce an error as a first draft, which is the argument for the
audit running after every pass rather than once at the end.

Also worth recording: the audit's own report contained a stale finding — it flagged the 2.5
billion figure that had already been corrected while it was reading. **Audit output is evidence,
not verdict.** Check each finding against the current file before acting on it.

---

## 2026-08-11 — Section 01, lesson 1, third cold-read audit (after shortening)

Run against parts 1–4 after a 30% length cut, specifically looking for arguments that lost a step.
It found real damage, plus things the earlier passes had missed.

| # | Problem | Fix |
|---|---|---|
| 15 | Part 3's parallelism table asked "waits on an earlier word?" and answered **no** for four rows that all consume $s_{i-1}$ — the very dependency the last row is bolded for. The patch sentence "none reaches back to a previous word" was false as written | The right question is which line *produces* what the next step needs. Only the cell does. Table is now two columns: parallel over the $T_x$ keys, and feeds the next step |
| 16 | Part 2's cost argument said additive "touches a thousand values per score" — but $q^\top k$ also touches $d \approx d_a$ values. As FLOPs the comparison is **wrong**; the two are comparable | It was always a *memory* argument. Now stated: a matmul reduces in registers and writes one number; additive has to write out and read back $T_x d_a$ pre-activations because the $\tanh$ sits between $q$ and $k$. The underbrace is labelled "values materialized" |
| 17 | Part 2: Luong's dot product "performed comparably" at $d = 1000$, and 40 lines later $d = 1000$ gives $\pm 31$ spread that "collapses softmax". Both stated flatly | Added the missing clause: real RNN states aren't independent or unit-variance, so $\sqrt d$ is a worst case, not what his model did |
| 18 | Part 4: "the projection matrices don't depend on $n$ either" — no projection matrices exist in the model built so far; the dot-product form has no parameters at all | Restated without them |
| 19 | Part 1's key advertises *"I'm 'cat', **position 2**"* — while part 4 argues the score function never sees position | Advertisement is now content-only |
| 20 | Part 3: "one state of width $d$, however long the sentence" ignores that backprop retains every intermediate state, so training memory is $O(nd)$ — the same regime part 3 argues in | Scoped to the forward pass, with the training bill stated |
| 21 | Part 1's softmax argument justified normalizing but asserted positivity — nine lines after saying $-3.7$ is a fine score | Both halves argued: a negative weight would subtract a word from the blend; exp is positive and order-preserving |
| 22 | "Gated RNNs were already good enough at long range" — load-bearing (it's what makes path length non-decisive) with no support | Replaced with the checkable version: stacked LSTMs held translation SOTA up to 2017, Google's production system among them |
| 23 | Part 3's $O(1)$ claim needs all queries to exist at once — which part 2 had just said in bold they don't. Never bridged | Added: without the RNN, a position's query comes from its own embedding or the previous layer, so they all exist up front |

**The gap none of the three audits had caught until now:** the decoder's recurrence also made it
*impossible to see the future*, and attention has no such constraint. Training a decoder on all
positions at once requires an explicit **causal mask**. It wasn't in the broken-jobs list, wasn't
a forward promise, and wasn't in part 5's outline — which listed padding masks only. Now in both.

**Also fixed:** $f$/$f_t$ and $i$/$i_t$ collisions (the $c_t$ one was flagged, which made the
silence on the other two conspicuous); $\mathrm{cell}$ used at two arities and never tied to
part 1's $f$; "layer" silently narrowed from "4 stacked LSTM layers" to "one attention operation";
"transformer" used argumentatively while undefined; $m$ and $\alpha_i$'s shape.

**Note on the length rule.** The 700–1200 word ceiling written into `CLAUDE.md` earlier the same
day was set from a bad estimate and turned out to be unreachable without cutting content. It is now
1500, set by measuring. Part 3 sits at 1655 and is logged as over in `PROGRESS.md` rather than
trimmed into incoherence.

---

## 2026-08-13 — Section 01, lesson 1, parts 4–5 (regenerated after the 4↔5 swap)

Parts 4 and 5 were swapped and regenerated the same day (Kenessary: old part 4 was foreshadowing
problems whose solutions lived in future material). The cold-read audit ran on the fresh prose,
before Kenessary read it. Not quizzed; no propagation beyond the two new files.

| # | Claimed | Actually | How it was found |
|---|---|---|---|
| 24 | Part 5: "**$T$ appears in none of those shapes**" — every parameter is length-independent, so train-short/run-long works unconditionally | Part 4's own position mechanism is a **learned table with one row per position** — a parameter whose shape contains the maximum length. Train at length 7 and position 8 has no vector. True claim: every *per-row* weight is length-blind; the stored position table is the one exception | Cold-read audit |
| 25 | Part 5's equivariance proof: "shuffle the rows of $x$ (the words, re-ordered)" with only the mask set aside | Part 4's $x$ already has position vectors added, so shuffling its rows is *not* re-ordering the words — re-ordered words would get different position vectors, which is exactly the symmetry-breaking the section says doesn't exist yet. The proof holds only for the bare embeddings; mask *and* position vectors must both be set aside | Cold-read audit |
| 26 | Part 5: "$2048^2$ = 4,194,304 entries per block" | The part's own table two lines up lists **both $E$ and $A$** at $T \times T$ — the materialized score tables are ~8.4M entries per block, double the stated figure | Cold-read audit |
| 27 | Part 5: "**Critical path** — … the middle **and right** columns: 1 against $T$, and it's the whole victory" | The middle column (sequential steps) is part 3's decisive throughput argument; the right column is part 3's *path length*, which part 3 explicitly priced as **not** decisive. Folding both into "the victory" un-drew part 3's central distinction | Cold-read audit |

**The pattern, again.** All four were introduced by a regeneration that was clean on its target
(removing forward-promises) and wrong somewhere new — same shape as errors 7–8 and 9b. The audit
after every pass continues to earn its keep: all four were caught before first read.

**#24 is the keeper.** The correction is more instructive than the original claim — "the attention
machinery is length-blind, the stored position patch isn't" is a real interview distinction, and it
is now the honest hook for lesson 5 instead of a hand-waved "positional encoding comes later."
