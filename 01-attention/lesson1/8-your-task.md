# 8 · Your task

*~3 min to read, ~45 min to do. Lesson 1, part 8 of 8.*

Implement three functions in `01-attention/attention.py`. Signatures and shapes are already
there. Bodies are yours.

**~25–40 lines total.**

| Function | What it does |
|---|---|
| `causal_mask(seq_len, device)` | Bool mask. `True` = allowed to attend here |
| `scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)` | The attention box from part 3 |
| `attention_entropy(attn_weights)` | Entropy of each attention row, in nats |

Don't call `F.scaled_dot_product_attention` — that's what you're checked against.

**Why return the attention weights too?** Because part of this lesson is *looking* at them.
Real implementations don't return them — the fused kernel never even builds the full matrix,
which is exactly what FlashAttention is about. That's why attention heatmaps are annoyingly
hard to get out of production code.

---

## Then run

```bash
python 01-attention/check_lesson1.py
```

18 checks. Grouped:

1. **Matches `F.scaled_dot_product_attention`** to 1e-5 — no mask, bool mask, causal
2. **Rows sum to 1** — catches softmax on the wrong dim
3. **Causality is real, not just shaped** — changes tokens after position 5, requires output
   before it to be *bit-identical*, and requires the tail to actually move. A shape check
   can't catch an off-by-one diagonal. This can.
4. **Broadcasting** — `L_q ≠ L_k`, a 2-D mask against 4-D input, 3-D input with no batch dim
5. **Entropy** — uniform over `L` gives exactly `ln L`, one-hot gives `0` and not `NaN`

---

## Common mistakes

Written down in advance so a wrong number is diagnosable instead of mysterious.

### Wrong — silently produces bad results

- **`softmax(dim=-2)`** — rows don't sum to 1, loss still drops. Check 2 catches it.
- **Scaling by `√d_model`** instead of `√d_head`, or by `d` instead of `√d`. Valid tensor,
  wrong temperature.
- **Scaling after the softmax.** `softmax(z)/c ≠ softmax(z/c)`.
- **Masking after the softmax.** Rows stop summing to 1, *and* the masked positions already
  polluted the denominator — the future leaked in through the normalizer.
- **Mask polarity flipped.** `masked_fill` fills where `True`; your mask marks *keep* with
  `True`. You need `~`.
- **Off-by-one on the diagonal.** `tril(diagonal=-1)` blocks self-attention. `triu` gives you
  the future. Check 3 is for this.
- **`0 · log 0 = NaN`** in the entropy. Mathematically `p log p → 0`, but in floating point
  `0 * -inf` is `NaN`. Clamp before the log — don't patch after.

### Fragile — works here, breaks later

- **`-1e9` instead of `-inf`.** Fine in fp32. This box is fp16-only for mixed precision.
  Use `torch.finfo(scores.dtype).min`.
- **Hardcoding 4-D.** `permute(0,1,3,2)` and `softmax(3)` die on 3-D input. `-1`/`-2` don't.
- **Scaling `q` before the matmul** (`q * d**-0.5 @ kᵀ`) vs dividing the scores after — same
  numbers, but the first is what fused kernels do, because it avoids a full `(L_q, L_k)`
  elementwise pass. Not required. Worth knowing.

---

## Then look at the picture

Once all 18 pass, the script runs the ablation and writes
`01-attention/experiments/lesson1_scaling.png`.

It sweeps head dim `d ∈ {4, 16, 64, 256, 1024}` in three conditions: scaled by `1/√d`,
unscaled, and over-scaled by `1/d`.

**Left panel** — mean attention entropy. **Right panel** — the softmax Jacobian trace
`Σ p(1−p)`, i.e. how much gradient survives.

*(Neat trick in there: it never reimplements attention. Your function always divides by `√d`,
so to get the unscaled version it passes in `q · √d` and lets the two cancel.)*

### Predict before you run

Three questions, answer them first:

1. What should the **unscaled** entropy curve do as `d` grows?
2. Why isn't the unscaled curve flat at zero even at `d = 4`?
3. The over-scaled `1/d` case — does it kill gradients, or something else?

Then run it and see if you were right. Getting #3 wrong is the useful one.

### Break it on purpose

Set your scale to `1.0`, re-run check 1, see how far off you land. Then `1/d`. Seeing the two
opposite failure modes is what makes the mechanism stick.

---

**Done?** Tell me and I'll review. Then: lesson 2, multi-head attention.
