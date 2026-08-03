# Part 1 — Scaled dot-product attention

**Roadmap:** Phase 1, steps 1–2 · **Time:** ~2 hours · **Runs on:** CPU

---

## 1 · Theory

### 1.1 Where attention came from — and what it actually fixed

Attention is older than the transformer by three years, and it was not invented to replace
recurrence. Getting this history right matters, because "attention replaced RNNs" is the
answer everyone gives and it's chronologically backwards.

**Bahdanau et al., 2014** — attention as a **Fix**, inside an RNN. Neural machine translation
at the time was encoder–decoder: a recurrent encoder read the source sentence and compressed
it into one fixed-size vector; a recurrent decoder generated from that vector. The bottleneck
is obvious once stated — a 40-word sentence and a 4-word sentence get the same 512 numbers.
Performance degraded sharply with sentence length. Bahdanau's fix: let the decoder, at each
output step, look back at *all* encoder states and take a weighted average, with weights
produced by a small learned MLP scoring compatibility between the decoder state and each
encoder state. **The recurrence stayed.** Attention was a supplement to it.

**Luong et al., 2015** — the dot-product form, as an **Empirical/efficiency** choice. Bahdanau's
scoring function was additive: `score(q, k) = vᵀ tanh(W[q; k])` — an MLP per query–key pair.
Luong compared alternatives and found plain `score(q, k) = qᵀk` worked about as well and was
dramatically cheaper, because a matrix of dot products is one matmul, and matmul is the
operation GPUs are built for. This is a recurring pattern worth naming: **a technique wins
because it maps onto the hardware, not because it is more expressive.** It comes back for
FlashAttention, for GQA, and for MoE.

**Vaswani et al., 2017** — removing the recurrence, as a **Fix**. The contribution of "Attention
Is All You Need" was not attention. It was the observation that once you have attention, the
recurrence is the part holding you back, and the scaffolding needed to stand without it:
the √d scaling (§1.3), multi-head (part 2), and positional encoding (part 3) — each of which
is a *cost incurred* by removing recurrence, not an independent good idea.

So the chain is: bottleneck → attention → attention makes recurrence removable → removing
recurrence breaks three things → fix those three things.

### 1.2 Why remove the recurrence — two arguments

RNNs are argued against here and never built (`ROADMAP.md` §13). Two reasons, and they are
different reasons that get conflated.

**Argument 1 — path length and gradient signal.**

For a token at position `i` to influence the representation at position `j`, information must
traverse some number of sequential transformations. Call it the *path length*.

- **RNN:** `O(|i − j|)`. Information from step 1 reaching step 100 passes through 99 recurrent
  updates.
- **Attention:** `O(1)`. Every position attends to every other position directly, in one op.

Path length matters because backpropagation through a path of length `n` multiplies `n`
Jacobians together. If the typical singular value of those Jacobians is `σ`, the gradient
scales roughly as `σⁿ` — vanishing for `σ < 1`, exploding for `σ > 1`. Only `σ ≈ 1` survives,
and that is a knife edge. The LSTM's cell state exists precisely to build an *additive*,
near-identity path through time that dodges this; it mitigates the problem, and does not
remove it, because the gating still attenuates.

**Argument 2 — parallelism, which is the one that actually decided it.**

Per layer, an RNN needs `O(n)` **sequential** operations: step `t` cannot start until step
`t−1` finishes. This is not a constant-factor issue you can fix with a bigger GPU. During
training, where the whole target sequence is already known, the RNN *still* has to walk the
sequence one step at a time. Self-attention needs `O(1)` sequential ops — the entire sequence
is processed as a batched matmul.

From Vaswani Table 1 (`n` = sequence length, `d` = representation dim, `k` = conv kernel):

| Layer type | Complexity per layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-attention | `O(n²·d)` | **`O(1)`** | **`O(1)`** |
| Recurrent | `O(n·d²)` | `O(n)` | `O(n)` |
| Convolutional | `O(k·n·d²)` | `O(1)` | `O(log_k n)` |

**Read the first column honestly, because this is a standard interview trap.** Self-attention
is *not* cheaper in FLOPs. It's `O(n²d)` against the RNN's `O(nd²)` — so attention is cheaper
only when `n < d`, and more expensive when `n > d`. With `d = 512` that crossover is around
512 tokens, and modern context lengths are far past it. Attention won despite costing more
arithmetic, because the arithmetic is *parallel* and the RNN's is not. A GPU would rather do
100× the FLOPs all at once than 1× serially.

That `n²` is a real debt, and the roadmap keeps paying it: KV-cache memory (Phase 2, 3),
FlashAttention's IO analysis (Phase 1, step 8), and every long-context method in Phase 2.

### 1.3 The operation

Given queries, keys, and values:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

Read it as a **soft dictionary lookup**. A hard lookup takes a key, finds the matching entry,
returns its value. Here, the query is compared against every key by dot product, the
comparisons are turned into a probability distribution by softmax, and the answer is the
value-weighted average under that distribution. Nothing is retrieved; everything is blended,
in proportion to match quality. "Soft" is what makes it differentiable, which is what makes it
learnable.

Three steps, with shapes. Take `Q: (B, H, L_q, d_k)`, `K: (B, H, L_k, d_k)`, `V: (B, H, L_k, d_v)`:

1. **Scores.** `S = Q Kᵀ / √d_k` → `(B, H, L_q, L_k)`. Entry `S[b,h,i,j]` is how much query `i`
   wants key `j`.
2. **Weights.** `A = softmax(S, dim=-1)` → `(B, H, L_q, L_k)`, each row summing to 1.
3. **Output.** `O = A V` → `(B, H, L_q, d_v)`.

**Softmax goes over the key dimension, `dim=-1`.** Each query gets one unit of attention mass
to distribute across the keys. Softmax over `dim=-2` would instead make *keys compete against
each other across queries* for a fixed budget — a quantity with no interpretation, whose rows
do not sum to 1. It is a one-character bug, it does not crash, the loss still goes down some,
and the model is quietly broken. This is the single most common attention bug.

Note `L_q` and `L_k` may differ (that's cross-attention: queries from one sequence, keys and
values from another — Phase 9 conditions images on text this way). `d_k` and `d_v` may also
differ, though in practice they're equal.

### 1.4 Why √d_k — the derivation

This is the part worth being able to derive on a whiteboard.

**Setup.** Take one query `q ∈ ℝ^d` and one key `k ∈ ℝ^d`. Assume their components are
independent, mean 0, variance 1. (An idealization — discussed honestly in §1.6.)

**Mean of the dot product.** With `q ⋅ k = Σᵢ qᵢkᵢ`, independence gives

```
E[q ⋅ k] = Σᵢ E[qᵢ kᵢ] = Σᵢ E[qᵢ] E[kᵢ] = 0
```

**Variance.** The terms `qᵢkᵢ` are independent across `i`, so variances add:

```
Var(q ⋅ k) = Σᵢ Var(qᵢ kᵢ)
```

For a single term, using `Var(X) = E[X²] − E[X]²` and independence:

```
Var(qᵢ kᵢ) = E[qᵢ² kᵢ²] − (E[qᵢ kᵢ])²
           = E[qᵢ²] E[kᵢ²] − 0
           = Var(qᵢ) · Var(kᵢ)        (since the means are 0, E[X²] = Var(X))
           = 1 · 1 = 1
```

Summing over `d` terms:

```
Var(q ⋅ k) = d          →      std(q ⋅ k) = √d
```

**So the logits entering softmax have standard deviation √d, and their scale grows with the
head dimension.** With `d = 64`, logits routinely land at ±8 and gaps between them are of the
same order. Softmax of logits separated by 8 is essentially one-hot: `e⁸ ≈ 3000`.

**Why near-one-hot is a failure and not just "confident".** The answer is in the gradient.
Writing `p = softmax(z)`, the Jacobian is

```
∂pᵢ/∂zⱼ = pᵢ (δᵢⱼ − pⱼ)
```

Suppose `p` is nearly one-hot at index `m`: `p_m ≈ 1`, all others `≈ 0`. Then every entry of
that Jacobian goes to zero:

- `i ≠ m`: `pᵢ ≈ 0` kills the whole term.
- `i = m, j = m`: `p_m(1 − p_m) ≈ 1 · 0 = 0`.
- `i = m, j ≠ m`: `p_m(0 − pⱼ) ≈ −pⱼ ≈ 0`.

**Zero Jacobian means no gradient reaches the query and key projections.** The model cannot
learn *where* to look, because the derivative of "where to look" with respect to the
parameters that decide it has vanished. And this happens *at initialization* — the moment when
the attention pattern is random and being wrong should be maximally informative. The network
starts out confidently arbitrary and cannot correct itself.

**The fix.** Divide by `√d_k`. Since `std(q ⋅ k) = √d`, dividing by `√d` gives unit variance
logits regardless of head dimension. The softmax stays in its responsive regime, gradients
flow, and — the part people miss — **the behaviour becomes independent of `d`**, so widening
heads doesn't silently change the temperature of every attention distribution in the network.

**Why `√d` and not `d`?** `√d` is the *standard deviation*; `d` is the variance. Dividing by
`d` over-normalizes: logits shrink toward 0, softmax approaches uniform, and attention loses
selectivity — every position attends everywhere equally. Less catastrophic than saturation
(gradients still flow) but the mechanism stops doing its job. You want variance 1, so divide
by the standard deviation.

Vaswani says this outright in footnote 4 — dot-product attention underperformed additive
attention at large `d_k`, and they hypothesized saturation as the cause. Note the word
*hypothesize*: the scaling was a reasoned guess that worked, not a derived-then-verified
theorem. It has held up.

### 1.5 It's √d_head, not √d_model

`d_k` is the **per-head** dimension. In a model with `d_model = 512` and `H = 8` heads,
`d_k = 64`, and the scaling is `1/√64 = 0.125`, not `1/√512`.

The derivation says why: the dot product being scaled is taken over the `d_k` components of a
single head. `d_model` never appears. Using `√d_model` would over-shrink by a factor of `√H`
and push attention toward uniform — the failure at the other end of §1.4.

This is a favourite interview probe because it separates "memorized the formula" from
"followed the argument."

### 1.6 What this doesn't fix — be honest about it

The unit-variance assumption is true at initialization with standard schemes, and roughly
maintained by LayerNorm feeding the projections. **It is not true later in training.** Nothing
stops the learned `W_Q` and `W_K` from growing, and in large models they do. Attention logits
in big training runs drift upward over time and the softmax saturates anyway — the same
failure the `√d` was introduced to prevent, arriving by a different route.

This is why **QK-norm** exists (Phase 2, step 9): normalize the queries and keys themselves
before the dot product, so logit scale is controlled throughout training rather than only at
step 0. Gemma 2's logit soft-capping addresses the same drift.

The honest summary: **`√d` fixes the initialization-time scale, not the training-time
trajectory.** Saying so is stronger than claiming it solved saturation, and it sets up
Phase 2 properly.

---

## 2 · Primitives

Runnable on toy tensors. These are the ops the task is written in — not the solution.

**Batched matmul.** `@` contracts the last two dims and broadcasts everything to the left:

```python
import torch
q = torch.randn(2, 8, 10, 64)     # (batch, heads, seq, head_dim)
k = torch.randn(2, 8, 10, 64)
scores = q @ k.transpose(-2, -1)  # (2,8,10,64) @ (2,8,64,10)
print(scores.shape)               # torch.Size([2, 8, 10, 10])
```

Use `transpose(-2, -1)`, not `.T` (which errors on >2 dims in recent torch) and not
`permute(0, 1, 3, 2)` (correct, but hardcodes rank — breaks the moment you drop the head dim).

**`einsum` says the same thing**, and is worth reading fluently because papers and reference
implementations use it constantly:

```python
scores = torch.einsum('bhqd,bhkd->bhqk', q, k)   # identical result
```

Repeated index `d` is summed over; indices in the output are kept. It names the contraction
instead of making you track which transpose achieves it.

**Softmax picks a dimension, and the choice is silent if wrong:**

```python
s = torch.randn(2, 3)
print(s.softmax(dim=-1).sum(dim=-1))   # tensor([1., 1.])  ← rows sum to 1
print(s.softmax(dim=-2).sum(dim=-1))   # tensor([1.4, 1.6]) ← garbage, no error
```

**Masking with `-inf`, before the softmax:**

```python
L = 5
causal = torch.tril(torch.ones(L, L, dtype=torch.bool))   # True on/below diagonal
s = torch.randn(L, L)
s = s.masked_fill(~causal, float('-inf'))                 # note the ~
print(s.softmax(-1)[0])   # tensor([1., 0., 0., 0., 0.])  ← exactly zero, not merely small
```

Why `-inf` and not `0`: softmax of 0 is `e⁰ = 1`, an *above-average* score. Why not `-1e9`:
it works in fp32 but is `-inf` in fp16 (max ≈ 65504) and can produce `NaN`. If you need a
finite sentinel, use `torch.finfo(scores.dtype).min`.

**Two opposite mask conventions — this will bite you:**

```python
import torch.nn.functional as F
# masked_fill:  True means REMOVE
# F.scaled_dot_product_attention(attn_mask=bool):  True means KEEP
```

Verified on this machine: passing the lower-triangular `True`-means-keep mask to
`F.scaled_dot_product_attention` gives exactly the same result as `is_causal=True`. So
converting between your mask and torch's means a `~`. Getting the polarity backwards gives an
*anti-causal* model that sees only the future — and it still trains.

**A fully-masked row is `NaN`** in a naive implementation (softmax of all `-inf`), while
torch's fused kernel returns zeros. If you compare against `F.scaled_dot_product_attention`
with a mask that empties a row, expect a mismatch there; it's a real edge case, not your bug.

**`view` vs `reshape` and contiguity** — this matters in part 2, but meet it now:

```python
x = torch.randn(2, 3, 4).transpose(1, 2)
print(x.is_contiguous())      # False — transpose returns a view with permuted strides
# x.view(2, -1)               # RuntimeError: view size is not compatible ...
print(x.reshape(2, -1).shape) # works — silently copies when it has to
```

`reshape` is `view` when it can be and a copy when it can't. Convenient, and it hides a
memory allocation from you.

---

## 3 · Task

Implement three functions in `01-attention/attention.py`. Signatures, docstrings, and shapes
are given; the bodies are yours.

| Function | What it does |
|---|---|
| `causal_mask(seq_len, device)` | Boolean mask, `True` = position may be attended to |
| `scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)` | The operation from §1.3, returning both output and weights |
| `attention_entropy(attn_weights)` | Entropy in nats of each attention distribution — the measuring instrument for the ablation |

Roughly 25–40 lines total. Use `torch.finfo`, `masked_fill`, `softmax`, `@` or `einsum`. Do
not call `F.scaled_dot_product_attention` — that's what you're being checked against.

Return the attention weights alongside the output because part of this part is *looking* at
them. Real implementations don't (the fused kernel never materializes the full matrix — that's
the point of FlashAttention), which is exactly why attention visualizations are harder to get
out of production code than you'd expect.

### Success criteria

Run `python 01-attention/check_part1.py`. It checks:

1. **Matches `F.scaled_dot_product_attention` to within `1e-5`** (fp32) on: no mask, an
   explicit boolean mask, and a causal mask.
2. **Rows sum to 1.** `attn.sum(-1)` is all ones — catches softmax on the wrong dim.
3. **Causality is semantic, not just shaped.** Perturbing tokens at positions `> i` leaves
   output `i` bit-identical. A shape check cannot catch an off-by-one in the mask diagonal;
   this can.
4. **Broadcasting.** Works with `L_q ≠ L_k` and with a mask of shape `(L_q, L_k)` against
   4-D inputs.
5. **Entropy is right.** Uniform attention over `L` keys gives exactly `ln L` nats;
   one-hot gives `0` and not `NaN`.

### Common mistakes — read these before you start

Written down in advance so a wrong number is diagnosable instead of mysterious.

**Wrong (produces incorrect results, usually silently):**

- **`softmax(dim=-2)`.** The rows won't sum to 1. Loss still decreases. Check #2 catches it.
- **Scaling by `√d_model` instead of `√d_head`**, or by `d` instead of `√d`. Output is a valid
  tensor with the wrong temperature. Check #1 catches it; §1.5 says why.
- **Scaling after the softmax.** Softmax isn't scale-equivariant — `softmax(z)/c ≠ softmax(z/c)`.
- **Masking after the softmax.** Rows no longer sum to 1, *and* the masked positions already
  polluted the normalizer, so future information leaked into the denominator.
- **Mask polarity inverted.** `masked_fill` fills where `True`; your mask marks *keep* with
  `True`. You need `~mask`. Symptom: perfectly reasonable-looking training and a model that
  cheats.
- **Off-by-one on the diagonal.** `tril(diagonal=-1)` forbids self-attention; `triu` gives you
  the future instead of the past. Check #3 exists for this.
- **`0 · log 0 = NaN` in the entropy.** `p·log p → 0` as `p → 0` mathematically, but
  `0 * float('-inf')` is `NaN` in floating point. Clamp before the log, don't patch after.

**Fragile (correct here, breaks elsewhere):**

- **`-1e9` instead of `-inf`.** Fine in fp32, overflows in fp16 — and this box is fp16-only for
  mixed precision. Use `torch.finfo(scores.dtype).min`.
- **Hardcoding 4-D.** `permute(0,1,3,2)` and `scores.softmax(3)` break on unbatched or
  5-D input. `-1` and `-2` don't.
- **Dividing by `math.sqrt(q.shape[-1])` computed on a Python int** is fine; dividing by a
  tensor you built on the wrong device is a silent CPU–GPU sync.
- **Multiplying by `1/√d` vs dividing by `√d`** — numerically equal here, but scaling `q`
  *before* the matmul (`q * d**-0.5 @ kᵀ`) is what fused kernels do to save a full
  `(L_q, L_k)` elementwise pass. Worth knowing; not required.

---

## 4 · Run and observe

`check_part1.py` ends by running the ablation and writing
`01-attention/experiments/part1_scaling.png`:

**Left panel — the mechanism failing.** Mean attention entropy against head dimension
`d ∈ {4, 16, 64, 256, 1024}`, with and without the `1/√d` scaling, on random unit-variance
`q`, `k`. Unscaled entropy should collapse toward 0 as `d` grows (attention becomes one-hot);
scaled entropy should sit flat near `ln L` regardless of `d`. **This is the derivation in
§1.4 as a picture, and it is the thing to look at.**

**Right panel — the consequence.** The largest softmax Jacobian entry `max pᵢ(1 − pᵢ)` under
both conditions. Watch it go to zero in the unscaled case: that is the gradient disappearing.

**Predict both curves before you run it.** Specifically: what should unscaled entropy be at
`d = 4`, and why isn't the unscaled curve flat-zero everywhere?

Then break it on purpose — set the scale to `1.0` in your own function and re-run check #1 to
see how far off the outputs are, and to `1/d` to see the opposite failure.

---

## 5 · References

- **Bahdanau, Cho, Bengio (2014)**, *Neural Machine Translation by Jointly Learning to Align
  and Translate* — attention's origin, inside an RNN. §3.1 is the mechanism.
- **Luong, Pham, Manning (2015)**, *Effective Approaches to Attention-based NMT* — dot-product
  vs. additive scoring, compared empirically.
- **Vaswani et al. (2017)**, *Attention Is All You Need* — §3.2.1 and **footnote 4** for the
  scaling argument; **Table 1** for the complexity comparison in §1.2.
- Optional: Lilian Weng, *Attention? Attention!* — a good consolidation read once you've
  implemented it, not before.

---

## 6 · Interview framing

Questions this part answers, phrased the way they get asked:

1. *Why is there a √d in attention?* — Derive it. Variance of a `d`-term dot product is `d`;
   unscaled logits saturate the softmax; a saturated softmax has zero Jacobian, so the query
   and key projections receive no gradient at initialization.
2. *`d_model` or `d_head`?* — `d_head`. The dot product being scaled lives inside one head.
3. *What if you divided by `d` instead?* — Uniform attention. Gradients survive, selectivity
   doesn't.
4. *Why did transformers replace RNNs?* — Lead with sequential-op count and parallelism, not
   FLOPs. Attention costs *more* arithmetic past `n ≈ d`; it wins because the arithmetic is
   parallel. Path length is the second argument, not the first.
5. *Does √d solve softmax saturation?* — No: at initialization only. Logits drift up during
   training, which is what QK-norm and logit soft-capping address. Knowing this is the
   difference between having read the paper and having trained something.
