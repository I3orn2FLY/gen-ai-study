# 7 · The PyTorch you need

*~4 min. Lesson 1, part 7 of 8.*

The ops your implementation is made of. Toy tensors, run them if you want.

Not the solution — the vocabulary the solution is written in.

---

## Batched matmul

`@` contracts the last two dims and broadcasts everything to the left.

```python
import torch
q = torch.randn(2, 8, 10, 64)     # (batch, heads, seq, head_dim)
k = torch.randn(2, 8, 10, 64)

scores = q @ k.transpose(-2, -1)  # (2,8,10,64) @ (2,8,64,10)
print(scores.shape)               # torch.Size([2, 8, 10, 10])
```

Use `transpose(-2, -1)`:

- `.T` errors on >2 dims in recent torch
- `permute(0, 1, 3, 2)` works but hardcodes 4 dims — breaks on 3-D input

`einsum` says the same thing. Worth being able to read, papers use it constantly:

```python
scores = torch.einsum('bhqd,bhkd->bhqk', q, k)   # identical
```

Repeated index `d` gets summed. Indices in the output stay.

---

## Softmax dim — silent if wrong

```python
s = torch.randn(2, 3)
print(s.softmax(dim=-1).sum(dim=-1))   # tensor([1., 1.])      ✅
print(s.softmax(dim=-2).sum(dim=-1))   # tensor([1.4, 1.6])    ❌ no error
```

---

## Masking, before the softmax

```python
L = 5
keep = torch.tril(torch.ones(L, L, dtype=torch.bool))   # True on/below diagonal
s = torch.randn(L, L)

s = s.masked_fill(~keep, float('-inf'))                 # note the ~
print(s.softmax(-1)[0])   # tensor([1., 0., 0., 0., 0.])
```

Those zeros are **exactly** zero, not just small.

**Why `-inf` and not `0`?** `e⁰ = 1`. Zero is an above-average score, not a "no."

**Why not `-1e9`?** Fine in fp32. In fp16 the max is 65504, so `-1e9` is just `-inf` anyway,
and can give you `NaN`. If you want a finite value: `torch.finfo(scores.dtype).min`.

---

## ⚠️ Two opposite mask conventions

This one will get you:

```python
x.masked_fill(mask, val)              # True = REMOVE
F.scaled_dot_product_attention(..., attn_mask=m)   # True = KEEP
```

I checked this on this machine: passing a lower-triangular True-means-keep mask to
`F.scaled_dot_product_attention` gives **exactly** the same result as `is_causal=True`.

So converting between your mask and torch's means a `~`.

Get the polarity backwards and you build a model that only sees the **future**. It trains fine.
It just cheats.

---

## Edge case: a fully-masked row

Softmax of all `-inf` → `NaN` in a hand-written version.

torch's fused kernel returns **zeros** instead.

So if you ever compare against `F.scaled_dot_product_attention` with a row that's completely
masked out, expect a mismatch. That's real, not your bug. (The checker avoids it.)

---

## `view` vs `reshape`

Matters properly in lesson 2, but meet it now:

```python
x = torch.randn(2, 3, 4).transpose(1, 2)
print(x.is_contiguous())       # False — transpose just permutes strides
# x.view(2, -1)                # RuntimeError
print(x.reshape(2, -1).shape)  # works — copies when it has to
```

`reshape` = `view` when possible, silent copy when not. Convenient, and it hides a memory
allocation from you.

**→ [8 · Your task](8-your-task.md)**
