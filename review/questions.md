# Question bank

Every quiz question, with grade and date. Three to five pulled at the start of roughly every
third session, weighted toward **shaky**, **blank**, and **confidently wrong**
(`TEACHING.md` § The review layer).

Grades: `solid` · `shaky` · `blank` · `wrong` — `wrong` meaning *answered confidently and
incorrectly*, which is the worst one and gets re-asked until it isn't.

**⚠ = a misconception he actually held.** These are not hypothetical distractors; they were said
out loud and corrected. Ask these first.

---

## Section 01 · Lesson 1 — parts 1–5

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
| 11 | Why did the dot product beat additive scoring? | Additive materializes a $(T_x, d_a)$ intermediate per query; the dot product materializes nothing. Over a translation that's $T_yT_xd_a$ against $T_yT_x$ — a factor of $d_a$ | — | — |
| 12 | Did Luong show the dot product was best? | No. Three variants, no dominant one; `general` suited local attention and `concat` underperformed suspiciously | — | — |
| 13 | Why is additive scoring *more* expressive? | It's a nonlinear function of the pair; $q^\top k$ is a sum of products. It lost on cost, not quality | — | — |
| 14 | Given $W[q;k] = W_qq + W_kk$, what can be cached? | $K_{\text{proj}} = HW_k^\top$ — independent of $i$, so computed once outside the decoder loop. Ancestor of the KV cache | — | — |
| 15 | Why doesn't teacher forcing let the decoder run in parallel? | Knowing the *inputs* doesn't help when the *states* chain: $s_i$ still needs $s_{i-1}$ | — | — |
| 16 | Why is an unrolled RNN's vanishing gradient the same result as a deep net's? | Unrolled, step $t$ *is* layer $t$ — shared weights, input at every layer, depth set by the data. Same Jacobian product | — | — |
| 17 | Exploding vs vanishing — why is only one of them a real problem? | Clipping rescales a too-large gradient. Nothing restores one that reached numerical zero | — | — |
| 18 | Why does LSTM gating help the gradient? | $c_t = f_t\odot c_{t-1} + \ldots$ — with $f_t\approx 1$ the update is additive, so $\partial c_t/\partial c_{t-1}\approx I$ | — | — |
| 19 | ⚠ Recurrence did three jobs. How many survive its deletion? | **Two break, one survives.** Word order breaks (→ positional encoding); bounded memory breaks ($O(nd)$ keys + $O(n^2)$ scores); parameters-independent-of-length survives. Multi-head and $1/\sqrt d$ are *new* bills, not recurrence's jobs | — | — |
| 20 | ⚠ Is attention permutation invariant or equivariant? | **Equivariant** — permute the input and the outputs permute with it, not stay put. $\alpha_{ij}$ never sees $j$ and a sum has no order, so a *single* query's output is genuinely invariant; that's the piece people over-generalize into "invariance" | — | — |
| 21 | Why does the causal mask write $-\infty$ into the scores *before* softmax, instead of zeroing weights after? | $\exp(-\infty) = 0$ and the row renormalizes over the visible positions automatically. A score of 0 is a full vote ($\exp(0)=1$), and zeroing $A$ after softmax leaves rows summing to less than 1 | — | — |
| 22 | Why do decoder-only models need a causal mask when Bahdanau's decoder needed nothing? | $s_i$ was built from $s_{i-1}$ — the future physically wasn't wired in. Attention connects every position to every position, so training all positions at once puts the answer in the input, one row down: loss → 0 by copying | — | — |
| 23 | Train a transformer on short sequences, run on long ones — what breaks, and what doesn't? | Attention/MLP weights don't — every weight acts per row, shapes come from widths. A **stored** position table does: one learned vector per position means a baked-in max length. Computed position signals close the gap | — | — |
| 24 | What persists in the attention op between two forward passes? | Only $W_Q, W_K, W_V$. $E$ and $A$ are activations — rebuilt and discarded every block, every pass ($2048^2 \approx 4.2$M entries *each* per block while the pass lives) | — | — |
| 25 | Where in a transformer block do positions interact? | Only inside attention — $QK^\top$ and $AV$ are the only cross-row ops, and neither has weights. Embedding, LayerNorm, MLP, and output projection are all per-row | — | — |
| 26 | ⚠ What produces the $(T, \text{vocab})$ logits — the block's MLP? | No — a single learned $(d_{\text{model}}, \text{vocab})$ matrix applied once after the last block. Every MLP lives *inside* a block and returns to width $d_{\text{model}}$ | — | — |
| 27 | ⚠ Where does argmax happen in a transformer? | Never inside the model — the pass ends at logits. Training consumes the softmax distributions (probability of the true next token; argmax has no gradient). Generation picks/samples one id, from the **last row only** | — | — |
| 28 | Why is the output projection "the same move as attention scoring"? | Each column of $W_{\text{out}}$ is a learned word vector, so row $t$'s logits are vocab-many dot-product scores against position $t$'s representation, then softmax — dot product as compatibility, softmax as scores-to-proportions | — | — |
| 29 | ⚠ Row $t$ of the output is "structurally" step $t$ — so why do inputs need position vectors at all? | Structural position survives everywhere (rows keep their slots), but only *outside* consumers read indices (`logits[t]`, the loss). Inside, every op is content-only — $QK^\top$ and $AV$ contain no $t$ (mask excepted, past-vs-future only) — so order must be injected into the contents, or into the op itself (RoPE) | — | — |

---

## Quiz history

Logged in `review/log.md`. No quizzes run yet.
