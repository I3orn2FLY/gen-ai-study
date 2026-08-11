# Lesson 1 — plan

Filled **before** each part is written; edited when feedback arrives, then the prose is regenerated
from it. `CLAUDE.md` § How material gets written. Not reading material.

---

## Claims

Every fact in this lesson that can't be re-derived on the spot. `✅` means the source was actually
opened. Anything not checked is a liability, and the list gets built from a cold-read audit's
enumeration, never from memory — the first version of this table missed two thirds of them.

| # | Claim | Source | In | Checked |
|---|---|---|---|---|
| 1 | Encoder is bidirectional; $h_j = [\overrightarrow{h}_j;\overleftarrow{h}_j]$ | Bahdanau §3.1 | 1 | ✅ 08-11 |
| 2 | 1000 units per direction → $d_h = 2000$; decoder $d_s = 1000$ | Bahdanau A.1.2/A.2 | 1, 2 | ✅ 08-11 |
| 3 | $s_0 = \tanh(W_s \overleftarrow{h}_1)$, $W_s \in \mathbb{R}^{n\times n}$ | Bahdanau A.2 | 1 | ✅ 08-11 |
| 4 | Alignment model $e_{ij} = v_a^\top\tanh(W_a s_{i-1} + U_a h_j)$; query is $s_{i-1}$ | Bahdanau A.1.2 | 1, 2 | ✅ 08-11 |
| 5 | Alignment hidden width $n' = 1000$ (our $d_a$) | Bahdanau A.1.2 | 2 | ✅ 08-11 |
| 6 | Motivation is the **fixed-length vector bottleneck** — not gradients | Bahdanau abstract | 1 | ✅ 08-11 |
| 7 | Luong's three forms: `dot`, `general`, `concat` | Luong §3.1 | 2 | ✅ 08-11 |
| 8 | Luong: 4 stacked LSTM layers, 1000 cells | Luong §4 | 2 | ✅ 08-11 |
| 9 | No form dominated: `dot` suited global, `general` local, `concat` underperformed | Luong §3.1 | 2 | ✅ 08-11 |
| 10 | Table 1: self-attn $n^2d$ / 1 / 1; recurrent $nd^2$ / $n$ / $n$ | Vaswani Table 1 | 4 | ✅ 08-11 |
| 11 | "precludes parallelization within training examples" — **§1, not the abstract** | Vaswani §1 | 3 | ✅ 08-11 |
| 12 | Abstract: "more parallelizable and requiring significantly less time to train" | Vaswani abstract | 3 | ✅ 08-11 |
| 13 | Additive ≈157M multiply-adds vs dot 2.5M at $T_x{=}T_y{=}50$, $d{=}d_a{=}1000$; 3M params vs 0 | derived, arithmetic run | 2 | ✅ 08-11 |
| 14 | Vanishing gradients: Hochreiter 1991, Bengio–Simard–Frasconi 1994 | — | 3 | ⬜ recalled |
| 15 | Gradient clipping standard by 2013 | — | 3 | ⬜ recalled |
| 16 | Forget gate added by Gers et al. 2000, not in the 1997 LSTM | — | 3 | ⬜ recalled |
| 17 | Stacked LSTMs held translation SOTA to 2017, incl. Google production | — | 3 | ⬜ recalled |

⬜ rows are dates and attributions. None carries an argument — if one can't be confirmed before the
lesson closes, drop the date and keep the mechanism.

---

## Parts 1–4 — written

Retrofitted from the prose after the fact, which is the wrong order and is why they needed
regenerating. Kept so the term order stays checkable.

| Part | Teaches | Introduces | May name (never load-bearing) |
|---|---|---|---|
| **1** The scoring function | scores → weights → context vector, inside the decoder loop | $T_x$, $T_y$, $h_j$, $d_h$, $s_i$, $d_s$, $s_0$, $W_s$, $f$, $y_{i-1}$, $e_{ij}$, $\alpha_{ij}$, $j'$, $c_i$, query, key, teacher forcing, `<sos>` | value (part 6), self-attention (part 5) |
| **2** Additive or multiplicative | how `score()` is computed, and why the dot product won | $W$, $v$, $d_a$, $W_q$, $W_k$, $H$, $m$, $d$ (shared width), $K$, $Q$, $E$, $W_a$, $v_a$ | KV caching (03), $1/\sqrt d$ (part 8) |
| **3** Why not recurrence | recurrence's three jobs; path length loses, parallelism decides | $\mathrm{cell}$, $x_t$, $h_0$, $n$, path length, $c_t$, $f_t$, $i_t$, $\odot$, self-attention (gloss), layer (gloss) | positional encoding (lesson 5) |
| **4** What deleting it cost | the FLOPs trap; which jobs survive; the four repairs | transformer, context window, the $nd$ / $n^2$ split, permutation invariance of $c_i$, causal mask (named) | multi-head (lesson 2), FlashAttention, KV cache (03) |

**Symbol collisions, declared once:** $c$ is both context vector (part 1) and LSTM cell state
(part 3); $f$ is both the decoder cell (part 1) and the forget gate $f_t$ (part 3); $i$ is both the
decoder step and the input gate $i_t$. All three are standard in the literature, so they're flagged
in place rather than renamed.

---

## Part 5 — The forward pass

Next to write. Plan first this time.

- **opens on** — the recurrence is deleted and there is now no model. Attention was always a
  component inside something; what's the something?
- **teaches** — where the attention block sits in a stack, and every tensor shape around it
- **intuition** — a transformer block is "every position looks at every other position, then each
  position thinks by itself" — mixing across positions, then processing within one. Attention is
  only the first half
- **figure** — `fig11-forward-pass.png` exists; check it still matches before embedding
- **trace** — *"The cat sat because it was tired"*, $T = 7$, $d = 64$, 2 blocks. Every shape from
  token ids through to logits, including the $(7,7)$ score table and what its rows and columns mean
- **introduces** — embedding lookup vs one-hot, $d_{\text{model}}$, block, residual connection,
  self-attention proper, causal mask, logits
- **may name** — multi-head (lesson 2), positional encoding (lesson 5), $1/\sqrt d$ (part 8).
  Part 4 promised the causal mask *here*, so it must be delivered, not deferred again
- **claims** — GPT-2 small's actual widths if quoted at all; otherwise keep symbols. Fetch before
  writing any number
- **earns** — parts 6–8 may then use: self-attention, layer, block, $Q/K/V$ as tensors, the
  $(T,T)$ score table

Open question to settle in the plan, not in prose: parts 6 and 7 both look like they cover the
operation. If part 5 has already traced $QK^\top$ with real shapes, 6 (query/key/value) and 7 (the
operation) may need merging into one part. Decide before writing 6.
