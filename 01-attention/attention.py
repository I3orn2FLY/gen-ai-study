"""Scaled dot-product attention, from scratch.

Section 01, part 1. See `part1-scaled-dot-product.md`.

Implement the three functions below. Do not import
`torch.nn.functional.scaled_dot_product_attention` here — it is the reference
`check_part1.py` compares against.
"""

from __future__ import annotations

import torch


def causal_mask(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Boolean mask permitting each position to attend to itself and the past.

    Convention: ``True`` means "this position MAY be attended to". This matches
    ``F.scaled_dot_product_attention``'s boolean ``attn_mask`` and is the opposite
    of ``Tensor.masked_fill``'s convention, which fills where ``True``.

    Args:
        seq_len: Sequence length L.
        device:  Device to allocate the mask on.

    Returns:
        Bool tensor of shape (L, L). ``mask[i, j]`` is True iff j <= i.
        For L = 3:

            [[ True, False, False],
             [ True,  True, False],
             [ True,  True,  True]]
    """
    raise NotImplementedError


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V.

    Args:
        q: (..., L_q, d_k)  queries.  Leading dims are typically (batch, heads),
           but the implementation must not assume how many there are.
        k: (..., L_k, d_k)  keys.
        v: (..., L_k, d_v)  values.
        attn_mask: Optional bool tensor broadcastable to (..., L_q, L_k).
           ``True`` = may attend, ``False`` = forbidden. Shapes like (L_q, L_k)
           must broadcast against 4-D q/k/v.
        dropout_p: Dropout probability applied to the attention weights (after
           softmax, before multiplying by V). Only active under ``self.training``
           semantics — here, apply it whenever ``dropout_p > 0``. The correctness
           checks all use 0.0.

    Returns:
        (output, attn_weights)
            output:       (..., L_q, d_v)
            attn_weights: (..., L_q, L_k), each row summing to 1 (pre-dropout).

    Notes:
        - Scale by √d_k where d_k = q.shape[-1] — the PER-HEAD dimension.
        - Mask BEFORE the softmax, with a value that is safe in fp16.
        - Softmax over the key dimension.
        - Return the pre-dropout weights, so the entropy measurement in the
          ablation reflects the attention distribution rather than the noise.
    """
    raise NotImplementedError


def attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """Shannon entropy, in nats, of each attention distribution.

        H(p) = -Σ_j p_j log p_j

    High entropy (→ ln L_k) means attention is spread across keys; low entropy
    (→ 0) means it has collapsed onto one key. This is the instrument used to
    show the √d_k scaling doing its job.

    Args:
        attn_weights: (..., L_q, L_k), rows summing to 1.

    Returns:
        (..., L_q) tensor of per-query entropies, in nats.

    Notes:
        - p log p → 0 as p → 0, but ``0 * float('-inf')`` is NaN in floating
          point. Handle the zeros; a one-hot distribution must return exactly 0.
        - Masked positions are exactly 0 and are the common case here, not an
          edge case.
    """
    raise NotImplementedError
