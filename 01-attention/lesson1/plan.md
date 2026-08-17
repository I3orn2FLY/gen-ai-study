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
| 10 | Table 1: self-attn $n^2d$ / 1 / 1; recurrent $nd^2$ / $n$ / $n$ | Vaswani Table 1 | 5 | ✅ 08-11 |
| 11 | "precludes parallelization within training examples" — **§1, not the abstract** | Vaswani §1 | 3 | ✅ 08-11 |
| 12 | Abstract: "more parallelizable and requiring significantly less time to train" | Vaswani abstract | 3 | ✅ 08-11 |
| 13 | Additive ≈102.5M multiply-adds (50M keys + 50M queries + 2.5M $v$-dots) vs dot 2.5M at $T_x{=}T_y{=}50$, uniform $d{=}d_a{=}1000$ → ≈41×; ≈2M params vs 0 | derived, re-run 08-17 (old 157M/60×/3M silently used Bahdanau's $d_h{=}2000$ — errata #19) | 2 | ✅ 08-17 |
| 14 | Vanishing gradients: Hochreiter 1991, Bengio–Simard–Frasconi 1994 | — | 3 | ⬜ recalled |
| 15 | Gradient clipping standard by 2013 | — | 3 | ⬜ recalled |
| 16 | Forget gate added by Gers et al. 2000, not in the 1997 LSTM | — | 3 | ⬜ recalled |
| 17 | Stacked LSTMs held translation SOTA to 2017, incl. Google production | — | 3 | ⬜ recalled |
| 18 | K/V split — "different encodings in the addressing and output stages" — explicit in Key-Value Memory Networks, Miller et al. 2016 | abstract | 6 | ✅ 08-17 |

⬜ rows are dates and attributions. None carries an argument — if one can't be confirmed before the
lesson closes, drop the date and keep the mechanism.

---

## Parts 1–3 — written

Retrofitted from the prose after the fact, which is the wrong order and is why they needed
regenerating. Kept so the term order stays checkable.

| Part | Teaches | Introduces | May name (never load-bearing) |
|---|---|---|---|
| **1** The scoring function | scores → weights → context vector, inside the decoder loop | $T_x$, $T_y$, $h_j$, $d_h$, $s_i$, $d_s$, $s_0$, $W_s$, $f$, $y_{i-1}$, $e_{ij}$, $\alpha_{ij}$, $j'$, $c_i$, query, key, teacher forcing, `<sos>` | value (part 6), self-attention (part 4) |
| **2** Additive or multiplicative | how `score()` is computed, and why the dot product won | $W$, $v$, $d_a$, $W_q$, $W_k$, $H$, $m$, $d$ (shared width), $K$, $Q$, $E$, $W_a$, $v_a$ | KV caching (03), $1/\sqrt d$ (part 8) |
| **3** Why not recurrence | recurrence's three jobs; path length loses, parallelism decides | $\mathrm{cell}$, $x_t$, $h_0$, $n$, path length, $c_t$, $f_t$, $i_t$, $\odot$, self-attention (gloss), layer (gloss) | positional encoding (lesson 5) |

**Parts 4 and 5 swapped 2026-08-13.** "What deleting it cost" was written as part 4 — before the
transformer existed on the page — and Kenessary's read confirmed the consequence: five of its seven
sections ended in a pointer to future material. It read as foreshadowing because structurally it was.
The forward pass now comes first (part 4); the bill is audited against the built model (part 5).
The old part 4 text is regenerated, not just renumbered.

**Re-audit 08-17 (all written parts, one cold reader each, accuracy + followability), incorporated:**
part 1 — "attention" now named in the body (it never was); encoder/decoder attached to the RNNs in
prose; "s'assit" token split stated; `<sos>`→$y_0$ connected in the loop; argmax claim narrowed to
"no gradient"; PyTorch-naming claim fixed (errata #21); Bahdanau named in prose; $s_{i-1}$ gloss
covers $i{=}1$; bidirectionality's second payoff marked. part 2 — cost table corrected to uniform
Luong dims (errata #19: 102.5M vs 2.5M, ≈41×, ≈2M params); Luong init claim dropped, shared-origin
argument rebuilt (errata #20); query↔`s_prev` binding stated in prose; Luong introduced at first
use; `concat` verdict now quotes the paper; "won on quality" vs "won anyway" pivot marked;
$W_a/v_a$ notation note added. part 3 — Argument 1 signposted as non-decisive up front; its opening
example rebuilt on earned machinery (decoder→$h_j$ hop, not self-attention); cell's 3-arg decoder
form noted at definition; broadcast note on the step table; "time, not quality" overreach dropped
(the abstract claims quality too). part 5 — fourth job announced in the opening; verdict table
reordered to match proof order; train-on-700 claim carries its exception inline; shape-limit vs
cost-limit ceilings connected; Vaswani table intro de-fragmented; bidirectional gloss fixed
(errata #22); stalling parenthetical trimmed. Reading times updated (parts run ~6–7 min at
1450–1700 words; ceiling stays subordinated to flow per 08-17 instruction).

**Symbol collisions, declared once:** $c$ is both context vector (part 1) and LSTM cell state
(part 3); $f$ is both the decoder cell (part 1) and the forget gate $f_t$ (part 3); $i$ is both the
decoder step and the input gate $i_t$. All three are standard in the literature, so they're flagged
in place rather than renamed.

---

**Feedback rule this reorder encodes (Kenessary, 2026-08-13):** don't raise a problem whose
solution lives in future material — problem and mechanism arrive together, in the same part or an
adjacent one. A far-future repair gets either cut (raise it where it's solved) or a one-line sketch
of the answer at the point the tension appears. This extends the no-forward-references rule from
*terms* to *problems*.

## Part 4 — The forward pass

- **opens on** — part 3 deleted the recurrence and there is now no model. Attention was always a
  component inside something; what's the something? His own sketch
  $(T,\text{vocab}) \to (T,\text{feat}) \to (T,\text{vocab}_2)$ is graded as the opening move:
  skeleton right, three details wrong (ids + embedding lookup, not one-hot; feat $= d_{\text{model}}$;
  one vocabulary, shifted by one, not two)
- **teaches** — the transformer: where the attention op sits in a stack, and every tensor shape
  around it
- **intuition** — a transformer block is "every position looks at every other position, then each
  position thinks by itself" — mixing across positions, then processing within one. Attention is
  only the first half
- **figure** — `fig11-forward-pass.png` — re-checked 08-13 against this block: T=7, d=64, 2 blocks,
  (7,7) table, mask before softmax, (7,1000) output. Matches
- **trace** — *"The cat sat because it was tired"*, $T = 7$, $d_{\text{model}} = 64$, vocab 1000,
  2 blocks. Every shape from token ids through to logits, including the $(7,7)$ score table and
  what its rows and columns mean
- **introduces** — transformer (defined by building it), $T$ (one sequence; $T_x$/$T_y$/part 3's
  $n$ collapse), token id, embedding lookup vs one-hot, $d_{\text{model}}$ (relation to part 2's
  $d$: equal here), $x$ the $(T,d)$ activation, block, residual connection (anchored to ResNet —
  known DL), LayerNorm (one-line gloss, known DL), MLP-half of the block, logits, output
  projection, self-attention proper, $W_Q, W_K, W_V$, $V$ = value with a one-line working
  definition (full treatment part 6), $A$ = softmaxed score table whose rows are part 1's
  $\alpha_i$, **causal mask** — motivated *and* delivered here: next-token training on all
  positions at once + attention sees everything ⇒ $-\infty$ above the diagonal, before softmax.
  Padding mask in one paragraph alongside. Position: the "+ position" box gets its one-line
  mechanism inline (a per-position vector added to the embedding), full design space lesson 5 —
  problem and sketch of the solution in the same breath, per the rule above
- **may name** — $1/\sqrt d$ (part 8, same sitting — named in the trace, explained there).
  Multi-head is NOT named; lesson 2 raises it where it's solved
- **claims** — none external. GPT-2 widths not quoted; every number is arithmetic run in a shell
  ($2048^2 = 4{,}194{,}304$; $\sqrt{64} = 8$; $7\times7 = 49$). One-hot-times-matrix = row-selection
  is derivable on the spot
- **symbols** — $E$ stays the raw score table (part 2), now square; $V$ means value only — vocabulary
  size is written as the word "vocab", never the letter, to avoid a new collision
- **earns** — parts 5–8 may then use: transformer, self-attention, block, $Q/K/V$ as $(T,d)$
  tensors, $W_Q/W_K/W_V$, the $(T,T)$ score table $E$, $A$, causal mask, logits, $T$, $d_{\text{model}}$
- **cold-read audit 08-13, incorporated** — task switch translation→next-token justified in place
  (deleted RNNs were the two-sequence machinery); "block" bridged to part 3's "layer" (layer = the
  attention op alone, block = attention + MLP wrapped); LayerNorm glossed off batchnorm (per-row,
  no batch statistics); position table honest about its chosen max length; transformer attribution
  qualified as the stripped decoder-only variant of Vaswani 2017; copy-shortcut claim replaced
  with the derivable version (loss reaches zero by copying)
- **feedback 08-17 (Kenessary, reading part 4)** — the output projection was read as the block's
  MLP producing $(7,1000)$, with an imagined argmax→ids step inside the forward pass; the "times
  a learned $(64,1000)$ matrix" cell didn't land, and the stack table's `LayerNorm → output
  projection` rows mirror the block's `LayerNorm → MLP` wiring, inviting the conflation. Fix in
  the logits blockquote: it is **one matrix multiply**, not a block MLP (those return to width
  $d_{\text{model}}$), and **no id is picked inside the model** — training reads the probability
  given to the true next token; generation samples one id, from the last row only. Full answer
  (incl. columns-of-$W_{\text{out}}$-as-word-vectors and why "projection") in `scratch-qa.md`;
  ⚠ questions added to `review/questions.md`
- **feedback 08-17 (Kenessary, second — structural)** — "no coherent story", struggling to read,
  can't point at why; and: don't focus on the 1500 ceiling, make the text smooth. Diagnosis: the
  part was **answer-first and outside-in** — the stack table landed before *block* or *projection*
  meant anything; then two nested unpackings (block → attention box), each holding IOUs; the mask
  applied in the box but explained a section later; the *task* (next-token prediction) buried as
  detail three of the sketch grading. **Regenerated as one pass of one sentence through the
  machine, in data order**: the hole → the job (next-token, every position at once) → ids /
  embedding / position in → the attention box → the cheat visible the moment $E$ exists → causal
  mask lands there, padding mask one line → MLP + residuals + LayerNorm close the block → final
  LayerNorm → output projection → logits and their two consumers → the reveal (stack table +
  "this is a transformer" as *summary*, not introduction). Sketch grading compressed to one line;
  its three corrections land where the pipeline reaches them. Figure moved up to serve as the map
  the part then walks. $W_{\text{out}}$ added to introduces (defined at its formula). Ceiling
  subordinated to flow — if a regenerated part runs long, split it, never sand sentences down.
  **If part 5 gives him the same struggle, it gets the same diagnosis-and-restructure, not edits.**
- **feedback 08-17 (Kenessary, third — reading the ending)** — (a) "how do we not argmax in
  generation — isn't it greedy, or beam search?" The claim was about the *boundary*, not the
  picker: prose now says the picking happens in a loop around the model (greedy argmax / sample;
  pickers are section 03's subject). (b) "output logits are structurally per-step — why do inputs
  need position vectors at all?" The right observation: structural position survives everywhere,
  but only *outside* consumers (loss, generation loop) read indices; every op inside is
  content-only — $QK^\top$/$AV$ contain no $t$ (mask excepted, past-vs-future only) — so the
  index must be smuggled into the contents, or into the op itself (RoPE — lesson 5). Position
  paragraph now carries the one-line reason; swap-two-words derivation in `scratch-qa.md`;
  ⚠ question 29 added. Part 5's order-blindness section is this question's proof — it arrived
  on schedule
- **cold-read audit 08-17, incorporated** — "next *word*" until token id exists (then the job gets
  its proper name, next-token prediction); the quoted sentence loses its trailing period (word-level
  tokenization was silently dropping it); all three sketch fixes now labeled where they land; row 7's
  target honesty (whatever followed in the training text); $x$ re-bound at the logits formula (final
  LayerNorm's output); residual "correction to *its input*", not to $x$; the mask triangle shows
  row 3 — the worked example — instead of row 4; $W_Q/W_K$ casing disambiguated from part 2's
  additive-score $W_q/W_k$

## Part 5 — What deleting it cost (regenerated from old part 4)

- **opens on** — the model from part 4 runs and trains. But part 3 deleted recurrence knowing it
  was doing three jobs; nobody has checked the wreckage against the *built* model. Start from the
  claim most people would offer — "and it's cheaper" — which is false
- **teaches** — the total-work vs critical-path distinction, audited on the model in hand
- **intuition** — attention won *while doing more arithmetic*, because its work happens at once
  and the RNN's happens in a queue
- **figure** — none; part 4's figure is the referent, plus Vaswani Table 1 quoted as a table
- **trace** — every cost pointed at a tensor from part 4's trace: the $(T,T)$ table at $T=2048$
  ($4{,}194{,}304$ entries per block), the $(T,d)$ keys, the mask already applied
- **content, per the problem-with-solution rule** —
  (a) FLOPs trap: $T^2 d$ vs $T d^2$, crossover at $T=d$ — complete argument, delivered here.
  Vaswani's Table 1 writes $n$ for our $T$; noted once.
  (b) which of part 3's three jobs survive: order breaks, flat memory breaks, params-vs-length
  survives — with the survivor argued from part 4's actual weight shapes.
  (c) memory: keys $Td$ linear, score table $T^2$ quadratic — both facts complete in themselves;
  "context window" defined here. Engineering repairs get **one line naming section 03**, no
  repayment-schedule table.
  (d) order-blindness *proven* on part 4's ops (no index appears anywhere in emb-lookup → QKV →
  $E$ → $A$ → out except the mask), equivariance stated properly now that every position has an
  output; the mask's index-dependence flagged honestly as past-vs-future only; the one-line fix
  (position vector into the embedding) already seen in part 4 — lesson 5 is elaboration, not rescue.
  (e) causal mask: retrospective only — it was motivated and paid in part 4; one line connecting it
  to the RNN's free version.
  (f) multi-head and √d as "new bills": **cut.** √d is named where it appears (part 4's trace) and
  explained in part 8; multi-head is raised in lesson 2 where it's solved
- **introduces** — context window, the $Td$ / $T^2$ split, permutation equivariance (proper
  statement), total work vs critical path as named concepts
- **may name** — FlashAttention / KV cache / sparse attention in the single section-03 line
- **claims** — rows 10–12 of the table above (all fetched); everything else derived
- **earns** — parts 6–10 may rely on: the crossover argument, $T^2$ memory as the standing
  constraint, equivariance
- **cold-read audit 08-13, incorporated** — the survivor row now carries the **position-table
  exception** (a stored table has one row per position ⇒ max length baked in; length-blindness
  holds for every per-row weight — this is lesson 5's honest hook); critical path = the
  *middle* column only, with the right column tied back to part 3's path length (bonus, not
  decider — part 3's verdict preserved); $E$ **and** $A$ both counted at $T=2048$ (4.2M each);
  equivariance proof sets aside mask *and* position vectors before shuffling; Table 1's "layer"
  pinned to part 3's sense (the attention op, not a block) with one shared width $d$ noted;
  "model card" jargon cut; opening overclaim ("nothing about a future model") softened

**6/7 merge question — settled 08-17: keep them separate, re-scope 7.** Part 4 already traced
$QK^\top$ with real shapes, so 7's old "loops removed" framing is spent — but what remains splits
into two different kinds of part: 6 is a *why* part (why three matrices), 7 is a *shapes* part
(the op beyond the square self-attention case). Merged, they'd be one oversized part; Kenessary
asked parts stay small (08-17). The old stub 7's $(B, H, L_q, d_k)$ is cut — $H$ is multi-head,
lesson 2's machinery, unearned here; the batch dim stays.

## Part 6 — Query, key, value

- **opens on** — a cheaper model: delete the three matrices and attend with $x$ directly,
  $Q = K = V = x$. It typechecks, it runs, it saves parameters. What breaks?
- **teaches** — why the three projections exist; query/key/value as *roles*, not tensors
- **intuition** — a Python dict: a request matched against labels to fetch contents. Attention is
  that dict with three relaxations — exact match → dot-product score, one winner → softmax blend,
  hand-written labels → learned projections
- **figure** — `fig3-dict-to-attention.png`, `fig4-qkv-projections.png`, `fig5-symmetry.png`,
  each eyeballed before shipping
- **trace** — $S = xx^\top$ on part 4's $(7, 64)$ rows: symmetric ($S_{ij} = S_{ji}$ — "it→cat"
  forced equal to "cat→it"), and the diagonal $S_{ii} = \lVert x_i \rVert^2$ is the row max once
  LayerNorm has equalized norms (Cauchy–Schwarz) — every position mostly votes for itself.
  Shared $W$ ($Q = K = xW$) fixes neither: still symmetric, diagonal still
  $\lVert x_i W \rVert^2$
- **introduces** — the three roles properly (query = what I'm looking for, key = what I
  advertise, value = what I hand over); un-learns two live errors: "the key is what I need"
  (no — that's the value) and "key = the encoder side" (only in cross-attention)
- **may name** — cross-attention (part 7, adjacent)
- **claims** — row 18 (fetched). Naming lineage stated softly: the names follow key-value
  stores, and the addressing/output split predates the transformer by a year — no claim about
  what Vaswani personally borrowed
- **earns** — parts 7–10 use query/key/value as roles; README interview Q1 (why
  $W_Q \neq W_K$) becomes answerable
- **cold-read audit 08-17, incorporated** — the symmetry break was **false for the causal
  model** (the mask discards the mirror entry $E_{ji}$, so no two-needs averaging ever occurs);
  break one rebuilt on the mask-proof argument: with $Q = K$ the only expressible question is
  "similar to me" (nearest-neighbor lock), symmetry demoted to a scoped aside for unmasked ops.
  Cauchy–Schwarz hedge tightened ("comparable" → equal norms, which LayerNorm delivers at init);
  "mostly attends to itself" → "largest single weight is its own"; figure's shared-$W$ diagonal
  labeled an instance, not a theorem; the three-views box softened from conclusion to setup;
  linguistic features marked as illustration. README interview Q1 rewritten to match

## Part 7 — The operation, generalized

- **opens on** — part 4's box quietly assumed queries and keys are the same 7 rows: square
  table, one sentence, no batch. Part 1's translator was never square — French queries against
  English keys. The op needs its general form, which is also what part 10's checker
  (`F.scaled_dot_product_attention`) implements
- **teaches** — $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^\top / \sqrt{d_k})\,V$ with
  fully general shapes
- **intuition** — part 1's three steps — score, normalize, blend — survive unchanged; only the
  assumption about where $Q$ and $K$ come from is dropped
- **figure** — `fig6-three-steps.png`, `fig7-attention-heatmap.png`, eyeballed before shipping
- **trace** — part 4's $(7, 64)$ self case first, then general: $Q\,(L_q, d_k)$,
  $K\,(L_k, d_k)$, $V\,(L_k, d_v)$ → scores $(L_q, L_k)$ → out $(L_q, d_v)$; part 1's
  translation as the worked rectangle; a batch dim broadcast in front
- **introduces** — **cross-attention** (the rectangular case, named), $L_q$, $L_k$, $d_k$,
  $d_v$ — why $K$ and $V$ share $L_k$ but may differ in width; `dim=-1` and why the wrong
  softmax dim trains fine while being broken
- **may name** — `F.scaled_dot_product_attention` (parts 9–10). Multi-head NOT named (lesson 2)
- **claims** — none external
- **earns** — parts 8–10 may use the canonical formula and the general shapes
- **cold-read audit 08-17, incorporated** — the `dim=-2` consequence had its direction backwards
  (an output row's scale swings with how much of the keys' budgets *that query* captures, not
  with key popularity) — corrected; $B$ glossed at first use and fig6 (batched shapes) moved to
  the batching paragraph; loss-curve claim made precise (the buggy run descends regardless; only
  a comparison or exactness check catches it); "layers learn to cope" causal story cut; $d_k$
  tied to part 2's $d$
