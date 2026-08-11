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
