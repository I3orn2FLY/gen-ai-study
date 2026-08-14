"""Generate the diagrams used by lesson 1.

Boilerplate. Run once; the PNGs are committed so the markdown renders anywhere.

    python 01-attention/make_figures.py

Writes to 01-attention/lesson1/figures/.
"""

from __future__ import annotations

import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = pathlib.Path(__file__).parent / "lesson1" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED = "#1a1a1a", "#6b7280"
BLUE, ORANGE, GREEN, RED = "#2563eb", "#ea580c", "#16a34a", "#dc2626"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.labelcolor": INK, "figure.facecolor": "white"})


def box(ax, x, y, w, h, label, fc="white", ec=INK, fs=10, weight="normal", tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, weight=weight, color=tc or INK, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                 lw=lw, linestyle=ls, mutation_scale=14, zorder=1))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", name)


# ---------------------------------------------------------------- part 1
def fig_bottleneck():
    fig, ax = blank((10, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)

    ax.text(0.1, 4.3, "2014 — the fixed-vector bottleneck", fontsize=12, weight="bold")
    words = ["the", "cat", "sat", "on", "the", "mat"]
    for i, w in enumerate(words):
        box(ax, 0.3 + i * 0.85, 3.2, 0.75, 0.45, w, fc="#eff6ff", ec=BLUE, fs=9)
        arrow(ax, 0.67 + i * 0.85, 3.2, 0.67 + i * 0.85, 2.95, color=MUTED)
    box(ax, 0.3, 2.45, 5.1, 0.5, "RNN encoder", fc="#dbeafe", ec=BLUE, fs=10)
    arrow(ax, 5.4, 2.7, 6.05, 2.7, color=RED, lw=2.2)
    box(ax, 6.05, 2.42, 1.5, 0.56, "one fixed\nvector", fc="#fee2e2", ec=RED, fs=9, weight="bold")
    ax.text(6.8, 2.3, "everything squeezes\nthrough here", ha="center", va="top",
            fontsize=8.5, color=RED, style="italic")
    arrow(ax, 7.55, 2.7, 8.2, 2.7, color=MUTED, lw=2.2)
    box(ax, 8.2, 2.45, 1.5, 0.5, "RNN decoder", fc="#dbeafe", ec=BLUE, fs=9)

    ax.plot([0.1, 9.9], [1.98, 1.98], color="#d1d5db", lw=1)
    ax.text(0.1, 1.72, "with attention — the decoder reads every encoder state", fontsize=12,
            weight="bold")
    for i, w in enumerate(words):
        box(ax, 0.3 + i * 0.85, 1.02, 0.75, 0.45, w, fc="#eff6ff", ec=BLUE, fs=9)
    box(ax, 8.2, 1.02, 1.5, 0.45, "decoder", fc="#dbeafe", ec=BLUE, fs=9)
    weights = [0.05, 0.55, 0.16, 0.04, 0.06, 0.14]
    for i, wt in enumerate(weights):
        ax.annotate("", xy=(8.2, 1.15), xytext=(0.67 + i * 0.85, 1.02),
                    arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.5 + 5 * wt,
                                    alpha=0.35 + 0.6 * wt,
                                    connectionstyle="arc3,rad=0.28"))
    ax.text(9.9, 1.78, "thicker line = higher attention weight\nno bottleneck",
            ha="right", va="bottom", fontsize=9, color=GREEN, style="italic")
    save(fig, "fig1-bottleneck.png")


# ---------------------------------------------------------------- part 3
def fig_path_length():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    n = 7
    for ax, mode in zip(axes, ("rnn", "attn")):
        ax.set_axis_off(); ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-1.15, 1.5)
        for i in range(n):
            c = ORANGE if mode == "rnn" else BLUE
            fill = "#fff7ed" if mode == "rnn" else "#eff6ff"
            box(ax, i - 0.28, -0.22, 0.56, 0.44, f"$x_{i+1}$", fc=fill, ec=c, fs=9)
        if mode == "rnn":
            for i in range(n - 1):
                arrow(ax, i + 0.3, 0, i + 0.7, 0, color=ORANGE, lw=1.6)
            ax.set_title("RNN — information walks", fontsize=11, weight="bold", color=ORANGE)
            ax.text((n - 1) / 2, -0.85, f"$x_1 \\to x_{n}$  =  {n-1} sequential steps\n"
                    "gradient multiplies " + f"{n-1} Jacobians", ha="center", fontsize=9.5,
                    color=INK)
        else:
            for i in range(1, n):
                r = 0.5 + i * 0.09
                ax.annotate("", xy=(i, 0.25), xytext=(0, 0.25),
                            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.1, alpha=0.75,
                                            connectionstyle=f"arc3,rad=-{r/3:.2f}"))
            ax.set_title("Attention — information jumps", fontsize=11, weight="bold", color=BLUE)
            ax.text((n - 1) / 2, -0.85, f"$x_1 \\to x_{n}$  =  1 step\n"
                    "every pair is directly connected", ha="center", fontsize=9.5, color=INK)
    fig.tight_layout()
    save(fig, "fig2-path-length.png")


# ---------------------------------------------------------------- part 6
def fig_dict_to_attention():
    fig, ax = blank((10, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2)
    ax.text(0.1, 4.95, "a dict lookup, relaxed three times", fontsize=12.5, weight="bold")

    rows = [
        ("Python\ndict", "exact match", "one value", "hand-written keys", "#f3f4f6", MUTED),
        ("break 1", "score, not equality", "one value", "hand-written keys", "#fff7ed", ORANGE),
        ("break 2", "score, not equality", "softmax → blend all", "hand-written keys",
         "#fefce8", "#ca8a04"),
        ("break 3\n= attention", "score, not equality", "softmax → blend all",
         "$K = x\\,W_K$  (learned)", "#eff6ff", BLUE),
    ]
    for x, lab in ((3.2, "matching"), (5.95, "retrieval"), (8.6, "the index")):
        ax.text(x, 4.5, lab, ha="center", fontsize=9.5, weight="bold", color=MUTED)
    for i, (name, match, ret, idx, fc, ec) in enumerate(rows):
        y = 3.55 - i * 1.02
        ax.text(1.65, y + 0.3, name, fontsize=10, weight="bold", color=ec,
                ha="right", va="center", linespacing=1.4)
        box(ax, 1.95, y, 2.5, 0.6, match, fc=fc, ec=ec, fs=9)
        box(ax, 4.75, y, 2.4, 0.6, ret, fc=fc, ec=ec, fs=9)
        box(ax, 7.45, y, 2.3, 0.6, idx, fc=fc, ec=ec, fs=9)
        if i:
            for x in (3.2, 5.95, 8.6):
                arrow(ax, x, y + 1.02, x, y + 0.62, color="#d1d5db", lw=1.0)
    save(fig, "fig3-dict-to-attention.png")


def fig_qkv():
    fig, ax = blank((9, 3.9))
    ax.set_xlim(0, 9); ax.set_ylim(0, 3.9)
    ax.text(0.1, 3.65, "one input, three learned views of it", fontsize=12, weight="bold")
    box(ax, 0.3, 1.55, 1.5, 0.75, "$x$\ntoken embeddings", fc="#f3f4f6", ec=INK, fs=9)
    specs = [(2.95, "$W_Q$", "$Q$  query", "what I'm looking for", BLUE, "#eff6ff"),
             (1.55, "$W_K$", "$K$  key", "what I'm filed under", GREEN, "#f0fdf4"),
             (0.15, "$W_V$", "$V$  value", "what I hand over", ORANGE, "#fff7ed")]
    for y, wm, name, desc, c, fc in specs:
        arrow(ax, 1.85, 1.93, 2.75, y + 0.35, color=c, lw=1.5)
        box(ax, 2.75, y, 1.05, 0.7, wm, fc="white", ec=c, fs=11)
        arrow(ax, 3.85, y + 0.35, 4.55, y + 0.35, color=c, lw=1.5)
        box(ax, 4.55, y, 1.45, 0.7, name, fc=fc, ec=c, fs=10, weight="bold")
        ax.text(6.15, y + 0.35, desc, fontsize=9.5, va="center", color=MUTED, style="italic")
    ax.text(0.3, 1.15, "learned matrices →", fontsize=9, color=MUTED)
    save(fig, "fig4-qkv-projections.png")


def fig_symmetry():
    torch.manual_seed(3)
    L, d = 8, 32
    x = torch.randn(L, d)
    W = torch.randn(d, d) / math.sqrt(d)
    Wq, Wk = torch.randn(d, d) / math.sqrt(d), torch.randn(d, d) / math.sqrt(d)
    shared = ((x @ W) @ (x @ W).T / math.sqrt(d)).softmax(-1).numpy()
    sep = ((x @ Wq) @ (x @ Wk).T / math.sqrt(d)).softmax(-1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, m, title, sub in (
        (axes[0], shared, "$W_Q = W_K$  (shared)",
         "symmetric · diagonal dominates\nevery token mostly attends to itself"),
        (axes[1], sep, "$W_Q \\neq W_K$  (separate)",
         "asymmetric · direction is expressible"),
    ):
        im = ax.imshow(m, cmap="magma", vmin=0, vmax=max(shared.max(), sep.max()))
        ax.set_title(title, fontsize=11, weight="bold", pad=8)
        ax.set_xlabel("key j", fontsize=9); ax.set_ylabel("query i", fontsize=9)
        ax.set_xticks(range(L)); ax.set_yticks(range(L))
        ax.tick_params(labelsize=8)
        ax.text(0.5, -0.28, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=MUTED)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("attention weights: sharing the projection forces symmetry",
                 fontsize=12, weight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig5-symmetry.png")


# ---------------------------------------------------------------- part 6
def fig_three_steps():
    fig, ax = blank((10.2, 3.2))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 3.2)
    ax.text(0.1, 2.95, "three steps, and the shape at each one", fontsize=12, weight="bold")
    stages = [
        (0.15, "$Q$", "(B, H, $L_q$, $d_k$)", "#eff6ff", BLUE),
        (2.05, "$Q K^\\top / \\sqrt{d_k}$", "(B, H, $L_q$, $L_k$)", "#f5f3ff", "#7c3aed"),
        (4.65, "softmax(·, dim=-1)", "(B, H, $L_q$, $L_k$)", "#f0fdf4", GREEN),
        (7.45, "$A V$", "(B, H, $L_q$, $d_v$)", "#fff7ed", ORANGE),
    ]
    for i, (x, lab, shp, fc, ec) in enumerate(stages):
        w = 1.4 if i in (0, 3) else 2.2
        box(ax, x, 1.5, w, 0.8, lab, fc=fc, ec=ec, fs=11)
        ax.text(x + w / 2, 1.2, shp, ha="center", fontsize=9, color=MUTED)
        if i:
            arrow(ax, x - 0.35, 1.9, x - 0.05, 1.9, color=MUTED)
    ax.text(5.75, 0.65, "softmax over the LAST dim — the keys.\n"
            "each query spends one unit of attention across all keys.",
            ha="center", fontsize=9.5, color=GREEN, style="italic")
    ax.text(1.55, 0.65, "$K$: (B,H,$L_k$,$d_k$)\n$V$: (B,H,$L_k$,$d_v$)",
            ha="center", fontsize=9, color=MUTED)
    save(fig, "fig6-three-steps.png")


def fig_heatmap():
    torch.manual_seed(0)
    toks = ["The", "cat", "sat", "because", "it", "was", "tired"]
    L, d = len(toks), 24
    q = torch.randn(L, d); k = torch.randn(L, d)
    # nudge "it" (4) toward "cat" (1) so the example matches the text
    k[1] = q[4] * 1.6 + 0.25 * torch.randn(d)
    s = q @ k.T / math.sqrt(d)
    full = s.softmax(-1).numpy()
    causal = s.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)),
                           float("-inf")).softmax(-1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, m, title in ((axes[0], full, "no mask — every token sees every token"),
                         (axes[1], causal, "causal mask — nothing sees the future")):
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(L)); ax.set_yticks(range(L))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(toks, fontsize=9)
        ax.set_xlabel("attending TO (keys)", fontsize=9.5)
        ax.set_ylabel("attending FROM (queries)", fontsize=9.5)
        ax.set_title(title, fontsize=10.5, weight="bold", pad=8)
        for i in range(L):
            for j in range(L):
                if m[i, j] > 0.28:
                    ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if m[i, j] > 0.55 else INK)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].add_patch(plt.Rectangle((0.5, 3.5), 1, 1, fill=False, ec=RED, lw=2.2))
    axes[0].annotate('"it" → "cat"', xy=(1, 4), xytext=(3.4, 5.6), fontsize=9.5, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    fig.suptitle("each ROW is one query's distribution over keys — rows sum to 1",
                 fontsize=11.5, weight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig7-attention-heatmap.png")


# ---------------------------------------------------------------- part 4
def fig_forward_pass():
    fig, ax = blank((12.6, 7.2))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 7.2)
    ax.text(0.1, 6.95, '"The cat sat because it was tired"  ·  T = 7,  $d_{model}$ = 64',
            fontsize=12, weight="bold")

    # ---- left: the stack
    ax.text(1.65, 6.55, "the model", ha="center", fontsize=10, weight="bold", color=MUTED)
    stack = [("token ids", "(7,)", "#f3f4f6", MUTED),
             ("embedding + position", "(7, 64)", "#f3f4f6", MUTED),
             ("BLOCK 1", "(7, 64)", "#eff6ff", BLUE),
             ("BLOCK 2", "(7, 64)", "#eff6ff", BLUE),
             ("LayerNorm", "(7, 64)", "#f3f4f6", MUTED),
             ("output projection", "(7, 1000)", "#f3f4f6", MUTED)]
    for i, (lab, shp, fc, ec) in enumerate(stack):
        y = 5.8 - i * 0.86
        box(ax, 0.35, y, 2.6, 0.56, lab, fc=fc, ec=ec, fs=9,
            weight="bold" if "BLOCK" in lab else "normal")
        ax.text(3.05, y + 0.28, shp, fontsize=8.5, va="center", color=MUTED,
                family="monospace")
        if i < len(stack) - 1:
            arrow(ax, 1.65, y, 1.65, y - 0.3, color=MUTED)

    # ---- middle: inside a block
    ax.text(5.6, 6.55, "inside a block", ha="center", fontsize=10, weight="bold", color=BLUE)
    blk = [("LayerNorm", "#f9fafb", MUTED), ("ATTENTION", "#fee2e2", RED),
           ("+ residual", "#f9fafb", MUTED), ("LayerNorm", "#f9fafb", MUTED),
           ("MLP", "#f9fafb", MUTED), ("+ residual", "#f9fafb", MUTED)]
    ax.add_patch(FancyBboxPatch((4.25, 0.9), 2.7, 5.15,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc="#f8fbff", ec=BLUE, lw=1.4, ls="--", zorder=0))
    for i, (lab, fc, ec) in enumerate(blk):
        y = 5.35 - i * 0.78
        box(ax, 4.5, y, 2.2, 0.52, lab, fc=fc, ec=ec, fs=9.5,
            weight="bold" if lab == "ATTENTION" else "normal")
        if i < len(blk) - 1:
            arrow(ax, 5.6, y, 5.6, y - 0.26, color=MUTED)
    ax.annotate("", xy=(4.3, 4.5), xytext=(3.0, 4.35),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(5.6, 0.6, "everything stays (7, 64)", ha="center", fontsize=8.5, color=MUTED,
            style="italic")

    # ---- right: inside attention
    ax.text(9.85, 6.55, "inside ATTENTION", ha="center", fontsize=10, weight="bold", color=RED)
    box(ax, 8.9, 5.35, 1.9, 0.5, "$x$   (7, 64)", fc="#f3f4f6", ec=MUTED, fs=9.5)
    # Q, K, V computed in PARALLEL from x
    par = [(7.75, "$Q = x\\,W_Q$", BLUE, "#eff6ff"),
           (9.25, "$K = x\\,W_K$", GREEN, "#f0fdf4"),
           (10.75, "$V = x\\,W_V$", ORANGE, "#fff7ed")]
    for x0, lab, c, fc in par:
        box(ax, x0, 4.35, 1.4, 0.52, lab, fc=fc, ec=c, fs=9)
        ax.text(x0 + 0.7, 4.18, "(7, 64)", ha="center", fontsize=8, color=MUTED,
                family="monospace")
        arrow(ax, 9.85, 5.35, x0 + 0.7, 4.89, color=c, lw=1.2)

    box(ax, 8.45, 3.15, 2.8, 0.55, "scores $= QK^\\top$", fc="#fee2e2", ec=RED, fs=10.5,
        weight="bold")
    ax.text(11.35, 3.43, "(7, 7)", fontsize=9, color=RED, weight="bold", va="center",
            family="monospace")
    arrow(ax, 8.45, 4.35, 9.2, 3.72, color=BLUE, lw=1.3)
    arrow(ax, 9.95, 4.35, 10.2, 3.72, color=GREEN, lw=1.3)

    box(ax, 8.45, 2.25, 2.8, 0.55, "/ $\\sqrt{64}$ · mask · softmax", fc="#fee2e2", ec=RED,
        fs=9.5)
    ax.text(11.35, 2.53, "(7, 7)", fontsize=8.5, color=MUTED, va="center", family="monospace")
    arrow(ax, 9.85, 3.15, 9.85, 2.82, color=RED)

    box(ax, 8.45, 1.35, 2.8, 0.55, "out $= A\\,V$", fc="#fff7ed", ec=ORANGE, fs=10.5)
    ax.text(11.35, 1.63, "(7, 64)", fontsize=8.5, color=MUTED, va="center", family="monospace")
    arrow(ax, 9.85, 2.25, 9.85, 1.92, color=RED)
    ax.annotate("", xy=(11.1, 1.9), xytext=(11.45, 4.35),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3,
                                connectionstyle="arc3,rad=0.35"))

    ax.annotate("", xy=(8.85, 5.6), xytext=(6.75, 4.6),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(9.85, 0.78, "a (7, 7) table:  row i = token i asking,\ncol j = token j asked about",
            ha="center", fontsize=9.5, color=RED, weight="bold", linespacing=1.5)
    ax.text(9.85, 0.12, "built, used, discarded — every block, every forward pass",
            ha="center", fontsize=8.5, color=MUTED, style="italic")
    save(fig, "fig11-forward-pass.png")


# ---------------------------------------------------------------- cross-attention aside
def fig_shape_trace():
    fig, ax = blank((11, 5.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.6)
    ax.text(0.1, 5.35, '"the cat sat" → "le chat s\'assit"   ·   where the score matrix appears',
            fontsize=12, weight="bold")

    # encoder column
    ax.text(1.5, 4.85, "ENCODER (source, 3 tokens)", ha="center", fontsize=9.5,
            weight="bold", color=GREEN)
    enc = [("source ids", "(3,)"), ("embedding", "(3, 512)"), ("encoder stack", "H  (3, 512)")]
    for i, (lab, shp) in enumerate(enc):
        y = 4.25 - i * 0.72
        box(ax, 0.35, y, 2.3, 0.5, lab, fc="#f0fdf4", ec=GREEN, fs=9)
        ax.text(2.75, y + 0.25, shp, fontsize=8.5, va="center", color=MUTED, family="monospace")
        if i < 2:
            arrow(ax, 1.5, y, 1.5, y - 0.22, color=GREEN)

    # decoder column
    ax.text(8.9, 4.85, "DECODER (target, 4 tokens)", ha="center", fontsize=9.5,
            weight="bold", color=BLUE)
    dec = [("target ids", "(4,)"), ("embedding + self-attn", "D  (4, 512)")]
    for i, (lab, shp) in enumerate(dec):
        y = 4.25 - i * 0.72
        box(ax, 7.75, y, 2.3, 0.5, lab, fc="#eff6ff", ec=BLUE, fs=9)
        ax.text(10.15, y + 0.25, shp, fontsize=8.5, va="center", color=MUTED,
                family="monospace")
        if i < 1:
            arrow(ax, 8.9, y, 8.9, y - 0.22, color=BLUE)

    # projections
    for x, lab, src, c, fc in ((3.55, "$K = H\\,W_K$", "(3, 64)", GREEN, "#f0fdf4"),
                               (5.15, "$V = H\\,W_V$", "(3, 64)", GREEN, "#f0fdf4"),
                               (6.9,  "$Q = D\\,W_Q$", "(4, 64)", BLUE, "#eff6ff")):
        box(ax, x, 2.45, 1.35, 0.5, lab, fc=fc, ec=c, fs=9.5)
        ax.text(x + 0.68, 2.28, src, ha="center", fontsize=8.5, color=MUTED,
                family="monospace")
    arrow(ax, 2.65, 2.85, 3.5, 2.75, color=GREEN)
    ax.annotate("", xy=(5.5, 2.4), xytext=(2.0, 2.5),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4,
                                connectionstyle="arc3,rad=0.25"))
    arrow(ax, 8.9, 3.53, 7.65, 2.98, color=BLUE)

    # the score matrix
    box(ax, 3.9, 1.15, 2.6, 0.62, "scores $= QK^\\top$", fc="#fee2e2", ec=RED, fs=11,
        weight="bold")
    ax.text(5.2, 0.93, "(4, 3)   ← 4 queries × 3 keys", ha="center", fontsize=9.5,
            weight="bold", color=RED, family="monospace")
    arrow(ax, 4.2, 2.45, 4.6, 1.79, color=RED, lw=1.6)
    arrow(ax, 7.4, 2.45, 6.2, 1.79, color=RED, lw=1.6)

    ax.text(0.35, 1.46, "the one\nyou asked about", fontsize=9.5, color=RED, weight="bold",
            style="italic", ha="left", va="center")

    box(ax, 7.1, 1.15, 2.9, 0.62, "softmax(·/$\\sqrt{64}$) @ $V$", fc="#fff7ed", ec=ORANGE,
        fs=9.5)
    ax.text(8.55, 0.93, "(4, 64) → ... → (4, 12000)", ha="center", fontsize=8.5, color=MUTED,
            family="monospace")
    arrow(ax, 6.5, 1.46, 7.05, 1.46, color=ORANGE, lw=1.6)
    ax.annotate("", xy=(8.6, 1.79), xytext=(5.9, 2.42),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0, ls=":",
                                connectionstyle="arc3,rad=-0.3"))

    ax.text(5.5, 0.35, "self-attention: Q, K, V all from one side → square.   "
            "cross-attention: Q from the other side → rectangular.",
            ha="center", fontsize=9.5, color=INK, style="italic")
    save(fig, "fig10-shape-trace.png")


def fig_saturation():
    torch.manual_seed(0)
    L = 24
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, d in zip(axes, (4, 64, 1024)):
        q, k = torch.randn(L, d), torch.randn(L, d)
        p = (q @ k.T).softmax(-1)[0].numpy()          # UNSCALED, on purpose
        ax.bar(range(L), p, color=ORANGE if d > 4 else BLUE, width=0.85)
        ax.set_ylim(0, 1.05)
        ent = float(-(p * np.log(np.clip(p, 1e-30, None))).sum())
        ax.set_title(f"$d$ = {d}\nentropy = {ent:.2f} nats", fontsize=10.5, weight="bold")
        ax.set_xlabel("key", fontsize=9); ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("attention weight", fontsize=9.5)
    fig.suptitle("without the $1/\\sqrt{d}$: one query's attention as $d$ grows  "
                 "(max possible entropy = ln 24 = 3.18)", fontsize=11, weight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig8-saturation.png")


def fig_softmax_jacobian():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    p = np.linspace(0, 1, 400)
    ax.plot(p, p * (1 - p), color=BLUE, lw=2.2)
    ax.fill_between(p, p * (1 - p), where=(p > 0.9), color=RED, alpha=0.25)
    ax.fill_between(p, p * (1 - p), where=(p < 0.1), color=RED, alpha=0.25)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1)
    ax.annotate("saturated softmax\nlands here — no gradient", xy=(0.97, 0.03),
                xytext=(0.55, 0.16), fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlabel("$p_i$  (attention weight)", fontsize=10)
    ax.set_ylabel("$p_i(1-p_i)$", fontsize=10)
    ax.set_title("the softmax Jacobian dies at both ends", fontsize=11, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig9-softmax-jacobian.png")


def fig_decoder_loop():
    """Part 1 — the decoder loop unrolled: where s_0 comes from, and why the query
    at step i is the state from step i-1."""
    fig, ax = blank((11.2, 5.4))
    ax.set_xlim(0, 9.6); ax.set_ylim(-0.6, 5.4)

    LBLUE, LORANGE, LGREEN = "#dbeafe", "#ffedd5", "#dcfce7"
    sx = [0.5, 2.3, 4.1, 5.9, 7.7]          # left edge of each decoder-state box
    sw, sh, sy = 1.2, 0.7, 1.0
    ax_left = [1.35, 3.15, 4.95, 6.75]      # left edge of each attention node
    aw, ah, ay = 1.3, 0.7, 2.85

    # encoder states — computed once, reused at every step
    box(ax, 1.2, 4.25, 7.2, 0.68,
        "encoder states   $h_1$   $h_2$   $h_3$      (computed once, before decoding)",
        fc=LBLUE, ec=BLUE, fs=10.5)

    for i in range(4):
        cx = ax_left[i] + aw / 2
        arrow(ax, cx, 4.25, cx, ay + ah + 0.03, color=BLUE, lw=1.1, ls=(0, (3, 2)))
        box(ax, ax_left[i], ay, aw, ah, f"$e_{{{i+1}j}} \\to \\alpha_{{{i+1}}} \\to c_{{{i+1}}}$",
            fc=LORANGE, ec=ORANGE, fs=10)

    for i in range(5):
        box(ax, sx[i], sy, sw, sh, f"$s_{{{i}}}$", fc=LGREEN if i else "white",
            ec=GREEN if i else MUTED, fs=12, weight="bold")

    for i in range(4):
        arrow(ax, sx[i] + sw, sy + sh / 2, sx[i + 1], sy + sh / 2, color=GREEN, lw=1.8)
        arrow(ax, sx[i] + sw - 0.28, sy + sh, ax_left[i] + 0.18, ay, color=ORANGE, lw=1.6)
        arrow(ax, ax_left[i] + aw - 0.18, ay, sx[i + 1] + 0.32, sy + sh, color=ORANGE, lw=1.6)

    ax.text(0.98, 2.40, "query\n$s_{i-1}$", fontsize=9, color=ORANGE, ha="center", va="center")
    ax.text(2.91, 2.40, "context\n$c_i$", fontsize=9, color=ORANGE, ha="center", va="center")

    for i, w in enumerate(["le", "chat", "s'", "assit"]):
        ax.text(sx[i + 1] + sw / 2, 0.76, f"emit  “{w}”", fontsize=9.5, color=MUTED, ha="center")

    ax.annotate("$s_0=\\tanh(W_s\\,\\overleftarrow{h}_1)$\nthe encoder starts it off",
                xy=(sx[0] + sw / 2, sy), xytext=(sx[0] + sw / 2, 0.10),
                fontsize=9.5, color=MUTED, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.text(4.8, -0.45, "the green chain is strictly sequential — step $i$ cannot start "
                        "until step $i\\!-\\!1$ has finished",
            fontsize=10, color=GREEN, ha="center", weight="bold")
    ax.set_title("the query at step $i$ is the decoder state left over from step $i-1$",
                 fontsize=12, weight="bold", y=0.99)
    save(fig, "fig12-decoder-loop.png")


def fig_additive_vs_dot():
    """Part 2 — what each scoring function has to materialize on the way to T_x numbers."""
    fig, ax = blank((10.6, 5.0))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 5.0)
    cell = 0.26

    def grid(x0, y0, rows, cols, fc, ec):
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                           fc=fc, ec=ec, lw=0.7, zorder=2))

    ax.text(2.5, 4.72, "additive   $v^{\\top}\\tanh(W_q q + W_k k)$",
            fontsize=11.5, weight="bold", ha="center", color=ORANGE)
    ax.text(8.1, 4.72, "dot product   $q^{\\top}k$",
            fontsize=11.5, weight="bold", ha="center", color=GREEN)
    ax.plot([5.3, 5.3], [0.55, 4.5], color="#d1d5db", lw=1.2)

    grid(0.95, 4.05, 3, 12, "#ffedd5", ORANGE)
    ax.text(0.85, 3.79, "$T_x$", fontsize=10, ha="right", va="center", color=MUTED)
    ax.text(2.51, 4.42, "$d_a$   (1000 in Bahdanau)", fontsize=9.5, ha="center", color=MUTED)
    ax.text(2.51, 2.98, "$Z=(T_x,\\; d_a)$  materialized —\n$\\tanh$ runs on every cell",
            fontsize=10, ha="center", va="top", color=ORANGE)
    arrow(ax, 2.51, 2.52, 2.51, 2.17, color=ORANGE, lw=1.6)
    ax.text(2.72, 2.34, "$\\cdot\\, v$", fontsize=10, va="center", color=MUTED)

    grid(2.38, 1.85, 3, 1, "#dcfce7", GREEN)
    ax.text(2.9, 1.59, "$e_i=(T_x,)$", fontsize=10, va="center", color=INK)

    grid(7.97, 1.85, 3, 1, "#dcfce7", GREEN)
    ax.text(8.49, 1.59, "$e_i=(T_x,)$", fontsize=10, va="center", color=INK)
    ax.text(8.1, 3.4, "nothing in between.\none matrix–vector product,\nscores fall straight out",
            fontsize=10, ha="center", va="center", color=GREEN)

    ax.text(5.3, 0.28, "in self-attention every position is a query, so the orange block becomes "
                       "$(n,n,d_a)$ against $(n,n)$\n"
                       "— at $n=1024,\\; d_a=64$ that is 67M floats per head per layer, "
                       "against 1M",
            fontsize=10, ha="center", va="center", color=RED)
    save(fig, "fig13-additive-vs-dot.png")


if __name__ == "__main__":
    print("writing figures to", OUT)
    for fn in (fig_bottleneck, fig_path_length, fig_forward_pass, fig_dict_to_attention,
               fig_qkv, fig_symmetry, fig_three_steps, fig_heatmap, fig_shape_trace,
               fig_saturation, fig_softmax_jacobian, fig_decoder_loop,
               fig_additive_vs_dot):
        fn()
    print("done")
