# Lesson 1 — ledger

Two lists, both maintained as parts are written. Not reading material.

- **Claims** — every assertion in this lesson that can't be re-derived on the spot. `checked` means
  the source was actually opened, not recalled. Anything not `checked` is a liability.
- **Terms** — where each term is first defined. Nothing may be *used* in part $N$ that isn't
  defined at part $\le N$.

---

## Claims

| # | Claim | Source | Used in | Checked |
|---|---|---|---|---|
| 1 | Encoder is bidirectional; $h_j = [\overrightarrow{h}_j;\overleftarrow{h}_j]$ | Bahdanau §3.1 | 1 | ✅ 08-11 |
| 2 | 1000 units per direction → $d_h = 2000$; decoder $d_s = 1000$ | Bahdanau A.1.2/A.2 | 1, 2 | ✅ 08-11 |
| 3 | $s_0 = \tanh(W_s \overleftarrow{h}_1)$, $W_s \in \mathbb{R}^{n\times n}$ | Bahdanau A.2 | 1 | ✅ 08-11 |
| 4 | Alignment model $e_{ij} = v_a^\top\tanh(W_a s_{i-1} + U_a h_j)$; query is $s_{i-1}$ | Bahdanau A.1.2 | 1, 2 | ✅ 08-11 |
| 5 | Alignment hidden width $n' = 1000$ (our $d_a$) | Bahdanau A.1.2 | 2 | ✅ 08-11 |
| 6 | Motivation is the **fixed-length vector bottleneck** — not gradients | Bahdanau abstract | 1 | ✅ 08-11 |
| 7 | Luong's three forms: `dot`, `general` $q^\top W_a k$, `concat` | Luong §3.1 | 2 | ✅ 08-11 |
| 8 | Luong: 4 stacked LSTM layers, 1000 cells, 1000-d embeddings | Luong §4 | 2 | ✅ 08-11 |
| 9 | Luong scores with the **current** state, not $s_{i-1}$ | Luong §3 | 2 | ✅ 08-11 |
| 10 | No form dominated: `dot` suited global, `general` local, `concat` underperformed | Luong §3.1 | 2 | ✅ 08-11 |
| 11 | Table 1: self-attn $O(n^2d)$ / $O(1)$ / $O(1)$; recurrent $O(nd^2)$ / $O(n)$ / $O(n)$ | Vaswani Table 1 | 3 | ✅ 08-11 |
| 12 | "precludes parallelization within training examples" — **§1, not the abstract** | Vaswani §1 | 3 | ✅ 08-11 |
| 13 | Abstract claims "more parallelizable and requiring significantly less time to train" | Vaswani abstract | 3 | ✅ 08-11 |
| 14 | Vanishing gradients: Hochreiter 1991 (thesis), Bengio–Simard–Frasconi 1994 | — | 3 | ⬜ recalled |
| 15 | Gradient clipping standard by 2013 (Pascanu, Mikolov, Bengio) | — | 3 | ⬜ recalled |
| 16 | LSTM 1997; **forget gate added by Gers et al. 2000**, not in the original | — | 3 | ⬜ recalled |
| 17 | Highway networks and ResNet are 2015 | — | 3 | ⬜ recalled |
| 18 | Sutskever and Cho both used 1000-wide context vectors | — | fig1 | ⬜ recalled |

⬜ items are attributions and dates. None carries an argument — if one can't be confirmed before
lesson 1 closes, drop the date and keep the mechanism.

---

## Terms

| Term | Defined in |
|---|---|
| encoder / decoder state, $h_j$, $s_i$ | 1 |
| the fixed-vector bottleneck | 1 |
| score $e_{ij}$ vs weight $\alpha_{ij}$ | 1 |
| context vector $c_i$ | 1 |
| query, key | 1 |
| teacher forcing, `<sos>` | 1 |
| activation vs parameter | 1 |
| additive / multiplicative scoring, $d_a$ | 2 |
| $Q$, $K$, $E = QK^\top$ (named, **not** yet usable) | 2 |
| path length | 3 |
| sequential operations | 3 |
| unrolled RNN as a deep network | 3 |
| self-attention (**named only** — earned in 5) | 3 |
| value $v_j$, multi-head, positional encoding, KV cache | *forward promises — flagged, never load-bearing* |

The last row is the one to police. A part may **name** something later as a promise; it may not
rest an argument on it. Part 3's original "Attention — $O(1)$" table row broke this and is
errata #6.
