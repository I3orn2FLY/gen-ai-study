# Question bank

Every quiz question, with grade and date. Three to five pulled at the start of roughly every
third session, weighted toward **shaky**, **blank**, and **confidently wrong**
(`TEACHING.md` § The review layer).

Grades: `solid` · `shaky` · `blank` · `wrong` — `wrong` meaning *answered confidently and
incorrectly*, which is the worst one and gets re-asked until it isn't.

**⚠ = a misconception he actually held.** These are not hypothetical distractors; they were said
out loud and corrected. Ask these first.

---

## Section 01 · Lesson 1 — parts 1–3

| # | Question | Answer in one line | Grade | Last asked |
|---|---|---|---|---|
| 1 | ⚠ Why did transformers replace RNNs? | **Parallelism** — $O(1)$ vs $O(n)$ sequential ops. A throughput argument, not a quality one. Path length is real but wasn't decisive | — | — |
| 2 | ⚠ What problem was Bahdanau attention introduced to solve? | The **fixed-length vector bottleneck** in seq2seq — *not* vanishing gradients, which gating had addressed in 1997 | — | — |
| 3 | ⚠ Does attention fix the vanishing-gradient problem? | Only across the encoder–decoder boundary. Both RNNs keep their own $O(n)$ chains until the recurrence itself is deleted | — | — |
| 4 | Is attention more computationally efficient than recurrence? | No — $O(n^2d)$ vs $O(nd^2)$, so it's *more* expensive past $n \approx d$. It won on critical path, not total work | — | — |
| 5 | What's the difference between a score and an attention weight? | $e_{ij}$ is pre-softmax and unbounded; $\alpha_{ij}$ is post-softmax and sums to 1. Papers and PyTorch both call $\alpha$ "weights" — and sometimes "scores" | — | — |
| 6 | Are attention weights learned parameters? | No. They're **activations** — recomputed per input, discarded after. Only $W$, $v$ and the RNN weights are in the optimizer | — | — |
| 7 | What is the query at decoder step $i$, and why not $s_i$? | $s_{i-1}$. $s_i$ needs $c_i$, which needs the scores — circular | — | — |
| 8 | Where does $s_0$ come from? | $s_0 = \tanh(W_s\overleftarrow{h}_1)$ — the backward encoder state at position 1, the one half-vector that has read the whole sentence | — | — |
| 9 | Why can't Bahdanau's model use a dot-product score? | Widths differ: $d_h = 2000$ against $d_s = 1000$. It doesn't typecheck | — | — |
| 10 | If the dot product has no parameters, where is the relation learned? | In the two RNNs. $\partial e/\partial q = k$ pushes query and key into a shared geometry | — | — |
| 11 | Why did the dot product beat additive scoring? | Additive materializes a $(T_x, d_a)$ intermediate — $(n,n,d_a)$ in self-attention. The dot product materializes nothing | — | — |
| 12 | Did Luong show the dot product was best? | No. Three variants, no dominant one; `general` suited local attention and `concat` underperformed suspiciously | — | — |
| 13 | Why is additive scoring *more* expressive? | It's a nonlinear function of the pair; $q^\top k$ is a sum of products. It lost on cost, not quality | — | — |
| 14 | Given $W[q;k] = W_qq + W_kk$, what can be cached? | $K_{\text{proj}} = HW_k^\top$ — independent of $i$, so computed once outside the decoder loop. Ancestor of the KV cache | — | — |
| 15 | Why doesn't teacher forcing let the decoder run in parallel? | Knowing the *inputs* doesn't help when the *states* chain: $s_i$ still needs $s_{i-1}$ | — | — |
| 16 | Why is an unrolled RNN's vanishing gradient the same result as a deep net's? | Unrolled, step $t$ *is* layer $t$ — shared weights, input at every layer, depth set by the data. Same Jacobian product | — | — |
| 17 | Exploding vs vanishing — why is only one of them a real problem? | Clipping rescales a too-large gradient. Nothing restores one that reached numerical zero | — | — |
| 18 | Why does LSTM gating help the gradient? | $c_t = f_t\odot c_{t-1} + \ldots$ — with $f_t\approx 1$ the update is additive, so $\partial c_t/\partial c_{t-1}\approx I$ | — | — |
| 19 | What three things broke when the recurrence was removed? | Word order (→ positional encoding), one pattern per layer (→ multi-head), score scale growing with $d$ (→ $1/\sqrt d$) | — | — |
| 20 | Why is attention a *set* operation? | $\sum_j\alpha_{ij}v_j$ is order-invariant. Shuffle the input, outputs are identical | — | — |

---

## Quiz history

Logged in `review/log.md`. No quizzes run yet.
