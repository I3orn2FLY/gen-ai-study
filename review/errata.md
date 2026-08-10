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
