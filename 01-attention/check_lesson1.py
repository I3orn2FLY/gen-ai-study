"""Checks and ablation for Section 01, lesson 1.

Boilerplate — this is plumbing, not the mechanism. Run it after implementing
`attention.py`:

    python 01-attention/check_lesson1.py

Five correctness checks, then the scaling ablation, which writes
`01-attention/experiments/lesson1_scaling.png`.
"""

from __future__ import annotations

import math
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from attention import attention_entropy, causal_mask, scaled_dot_product_attention

torch.manual_seed(0)
OUT_DIR = pathlib.Path(__file__).parent / "experiments"
TOL = 1e-5

_results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    _results.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def close(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs expected {tuple(b.shape)}"
    err = (a - b).abs().max().item()
    return err < TOL, f"max abs err {err:.2e}"


# ---------------------------------------------------------------------------
# 1 · Matches F.scaled_dot_product_attention
# ---------------------------------------------------------------------------
def check_matches_reference() -> None:
    print("\n1 · Against F.scaled_dot_product_attention (fp32, tol 1e-5)")
    B, H, L, D = 2, 4, 16, 32
    q, k, v = (torch.randn(B, H, L, D) for _ in range(3))

    out, attn = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    check("no mask", *close(out, ref))

    # Random bool mask, kept row-wise non-empty so both implementations agree.
    # (A fully masked row is NaN in a naive impl and 0.0 in torch's fused kernel
    #  — a real divergence, not a bug in your code. See lesson1 part 5.)
    mask = torch.rand(B, H, L, L) > 0.3
    mask[..., 0] = True
    out, _ = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    check("explicit bool mask", *close(out, ref))

    cm = causal_mask(L)
    check("causal_mask matches torch.tril",
          torch.equal(cm, torch.tril(torch.ones(L, L, dtype=torch.bool))),
          f"dtype={cm.dtype}, shape={tuple(cm.shape)}")

    out, _ = scaled_dot_product_attention(q, k, v, attn_mask=cm)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    check("causal", *close(out, ref))


# ---------------------------------------------------------------------------
# 2 · Attention weights are a distribution over keys
# ---------------------------------------------------------------------------
def check_rows_sum_to_one() -> None:
    print("\n2 · Attention weights sum to 1 over the key dim")
    q, k, v = (torch.randn(2, 3, 7, 16) for _ in range(3))

    _, attn = scaled_dot_product_attention(q, k, v)
    sums = attn.sum(dim=-1)
    check("unmasked", torch.allclose(sums, torch.ones_like(sums), atol=1e-6),
          f"min {sums.min():.6f}, max {sums.max():.6f}")

    _, attn = scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(7))
    sums = attn.sum(dim=-1)
    check("causal", torch.allclose(sums, torch.ones_like(sums), atol=1e-6),
          f"min {sums.min():.6f}, max {sums.max():.6f}")

    upper = attn[0, 0].triu(diagonal=1)
    check("future weights are exactly zero", bool((upper == 0).all()),
          f"max above diagonal: {upper.max():.2e}")


# ---------------------------------------------------------------------------
# 3 · Causality, semantically
# ---------------------------------------------------------------------------
def check_causality_semantic() -> None:
    print("\n3 · Causality: perturbing the future cannot change the past")
    B, H, L, D = 1, 2, 12, 16
    cut = 5
    q, k, v = (torch.randn(B, H, L, D) for _ in range(3))
    out_a, _ = scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(L))

    # Replace everything strictly after `cut` with different values.
    k2, v2 = k.clone(), v.clone()
    k2[:, :, cut + 1:] = torch.randn_like(k2[:, :, cut + 1:])
    v2[:, :, cut + 1:] = torch.randn_like(v2[:, :, cut + 1:])
    out_b, _ = scaled_dot_product_attention(q, k2, v2, attn_mask=causal_mask(L))

    prefix_delta = (out_a[:, :, :cut + 1] - out_b[:, :, :cut + 1]).abs().max().item()
    suffix_delta = (out_a[:, :, cut + 1:] - out_b[:, :, cut + 1:]).abs().max().item()
    check(f"positions 0..{cut} unchanged", prefix_delta == 0.0, f"delta {prefix_delta:.2e}")
    # Guards against a mask so aggressive that nothing depends on the input at all.
    check("positions after the cut DID change", suffix_delta > 1e-3, f"delta {suffix_delta:.2e}")

    # Off-by-one probe: position i must still see itself.
    _, attn = scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(L))
    diag = attn[0, 0].diagonal()
    check("each position attends to itself", bool((diag > 0).all()), f"min diag {diag.min():.2e}")


# ---------------------------------------------------------------------------
# 4 · Shape flexibility
# ---------------------------------------------------------------------------
def check_broadcasting() -> None:
    print("\n4 · Broadcasting and non-square attention")
    B, H, Lq, Lk, D, Dv = 2, 3, 5, 9, 16, 24
    q = torch.randn(B, H, Lq, D)
    k = torch.randn(B, H, Lk, D)
    v = torch.randn(B, H, Lk, Dv)

    out, attn = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    check(f"cross-attention L_q={Lq} != L_k={Lk}, d_v={Dv}", *close(out, ref))
    check("attn weights shape", attn.shape == (B, H, Lq, Lk), f"{tuple(attn.shape)}")

    mask2d = torch.rand(Lq, Lk) > 0.3
    mask2d[:, 0] = True
    out, _ = scaled_dot_product_attention(q, k, v, attn_mask=mask2d)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask2d)
    check("2-D mask broadcast against 4-D input", *close(out, ref))

    q3, k3, v3 = q[0], k[0], v[0]
    out, _ = scaled_dot_product_attention(q3, k3, v3)
    ref = F.scaled_dot_product_attention(q3, k3, v3)
    check("3-D input (no batch dim)", *close(out, ref))


# ---------------------------------------------------------------------------
# 5 · Entropy
# ---------------------------------------------------------------------------
def check_entropy() -> None:
    print("\n5 · attention_entropy")
    L = 8
    uniform = torch.full((1, 1, 4, L), 1.0 / L)
    h = attention_entropy(uniform)
    check(f"uniform over {L} keys == ln {L} = {math.log(L):.4f}",
          torch.allclose(h, torch.full_like(h, math.log(L)), atol=1e-5),
          f"got {h.flatten()[0].item():.6f}")
    check("shape drops the key dim", h.shape == (1, 1, 4), f"{tuple(h.shape)}")

    onehot = torch.zeros(1, 1, 3, L)
    onehot[..., 2] = 1.0
    h = attention_entropy(onehot)
    check("one-hot == 0 and not NaN",
          bool(torch.isfinite(h).all()) and torch.allclose(h, torch.zeros_like(h), atol=1e-6),
          f"got {h.flatten().tolist()}")

    causal_probs = torch.tril(torch.ones(L, L))
    causal_probs = causal_probs / causal_probs.sum(-1, keepdim=True)
    h = attention_entropy(causal_probs)
    expected = torch.tensor([math.log(i + 1) for i in range(L)])
    check("causal-uniform rows give ln(i+1)", torch.allclose(h, expected, atol=1e-5),
          f"row 0 -> {h[0]:.4f}, row {L-1} -> {h[-1]:.4f}")


# ---------------------------------------------------------------------------
# Ablation: does 1/√d actually keep the softmax responsive?
# ---------------------------------------------------------------------------
def ablation() -> None:
    """Sweep head dim, with and without the scaling.

    Trick worth understanding: this never reimplements attention. Your function
    always divides by √d, so to get UNSCALED scores we pre-multiply q by √d and
    let the two cancel:

        (q·√d) @ kᵀ / √d  ==  q @ kᵀ

    Same idea with q/√d to get the over-scaled 1/d case.
    """
    print("\nAblation: attention entropy and softmax gradient vs head dimension")
    dims = [4, 16, 64, 256, 1024]
    B, H, L = 4, 4, 64
    rows = []

    for d in dims:
        q = torch.randn(B, H, L, d)
        k = torch.randn(B, H, L, d)
        v = torch.randn(B, H, L, d)
        row = {"d": d}
        for label, qq in (
            ("scaled  (1/√d)", q),
            ("unscaled (1/1)", q * math.sqrt(d)),
            ("over    (1/d)", q / math.sqrt(d)),
        ):
            _, attn = scaled_dot_product_attention(qq, k, v)
            # Trace of the softmax Jacobian, Σ_j p_j(1 - p_j), averaged over rows.
            # Uniform over L keys -> 1 - 1/L ~= 1.  One-hot -> 0.  This is exactly
            # the quantity lesson1 part 4 argues vanishes, and unlike a `max` it is not
            # rescued by one lucky row somewhere in the batch.
            row[label] = (
                attention_entropy(attn).mean().item(),
                (attn * (1 - attn)).sum(-1).mean().item(),
            )
        rows.append(row)

    labels = ["scaled  (1/√d)", "unscaled (1/1)", "over    (1/d)"]
    header = f"{'d':>6} | " + " | ".join(f"{lab:>24}" for lab in labels)
    print("\n" + header)
    print(" " * 8 + "| " + " | ".join(f"{'entropy    Jacobian tr':>24}" for _ in labels))
    print("-" * len(header))
    for r in rows:
        cells = " | ".join(f"{r[lab][0]:>10.4f}  {r[lab][1]:>12.2e}" for lab in labels)
        print(f"{r['d']:>6} | {cells}")
    print(f"\nln(L) = ln({L}) = {math.log(L):.4f}  <- maximum possible entropy (uniform attention)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — table only.  pip install matplotlib)")
        return

    OUT_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"scaled  (1/√d)": "o-", "unscaled (1/1)": "s--", "over    (1/d)": "^:"}

    for lab in labels:
        ax1.plot(dims, [r[lab][0] for r in rows], styles[lab], label=lab)
        ax2.plot(dims, [r[lab][1] for r in rows], styles[lab], label=lab)

    ax1.axhline(math.log(L), color="grey", lw=0.8, ls="-.", label=f"ln L = {math.log(L):.2f}")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("head dimension $d_k$")
    ax1.set_ylabel("mean attention entropy (nats)")
    ax1.set_title("Attention collapses onto one key as $d_k$ grows")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("head dimension $d_k$")
    ax2.set_ylabel(r"mean softmax Jacobian trace  $\sum_j p_j(1-p_j)$")
    ax2.set_title("...and the gradient goes with it")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Lesson 1 ablation — why the $1/\\sqrt{d_k}$", y=1.0)
    fig.tight_layout()
    path = OUT_DIR / "lesson1_scaling.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nWrote {path}")


def main() -> None:
    print("=" * 74)
    print("Section 01 · Lesson 1 — scaled dot-product attention")
    print("=" * 74)
    for fn in (check_matches_reference, check_rows_sum_to_one, check_causality_semantic,
               check_broadcasting, check_entropy):
        try:
            fn()
        except NotImplementedError:
            print(f"  [ .. ] {fn.__name__}: not implemented yet")
            return
        except Exception as exc:  # noqa: BLE001 — a raised error is a failed check
            check(f"{fn.__name__} raised", False, f"{type(exc).__name__}: {exc}")

    passed = sum(ok for _, ok, _ in _results)
    print("\n" + "=" * 74)
    print(f"{passed}/{len(_results)} checks passed")
    print("=" * 74)
    if passed == len(_results):
        ablation()
    else:
        print("\nFix the failures above before running the ablation "
              "— it uses your implementation to make its point.")


if __name__ == "__main__":
    main()
