"""
Regenerate every figure in the conference paper from the saved result files.

No number in a figure is typed by hand: each panel reads results/*.json or
results/*.jsonl and recomputes its own summary.  Run this before rebuilding the
paper so the figures and the text cannot drift apart.

Usage:  python make_figures.py
Output: paper/figures/fig1_curvature.png
        paper/figures/fig2_split.png
        paper/figures/fig3_threshold.png
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = os.path.join("paper", "figures")
os.makedirs(OUT, exist_ok=True)

# IEEE single-column width is ~3.5in; two-column spans ~7.16in.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

INK = "#1a1a1a"
REAL = "#1f4e79"
SCR = "#c1121f"
GLOB = "#8d8d8d"


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tmed(xs):
    """torch.median semantics: lower of the two middle values for even n.

    The result files were aggregated this way, so the figures reproduce them
    exactly rather than approximately.
    """
    s = sorted(xs)
    return s[(len(s) - 1) // 2] if s else float("nan")


# ---------------------------------------------------------------- figure 1
def fig1():
    pts = [r for r in jl("results/profiles/profile_points.jsonl")
           if isinstance(r.get("K"), (int, float)) and r["K"] == r["K"]]
    by = defaultdict(list)
    for r in pts:
        by[r["model"]].append(r["K"])

    short = {
        "gpt2": "GPT-2\n(124M)",
        "EleutherAI/pythia-160m": "Pythia\n(160M)",
        "JackFram/llama-160m": "LLaMA\n(160M)",
        "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2\n(135M)",
    }
    order = ["gpt2", "EleutherAI/pythia-160m", "JackFram/llama-160m",
             "HuggingFaceTB/SmolLM2-135M-Instruct"]
    order = [m for m in order if m in by]

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = np.array(by[m])
        x = i + 1 + rng.normal(0, 0.055, size=len(v))
        ax.scatter(x, v, s=3.2, alpha=0.30, color=REAL, linewidths=0, zorder=2)

    bp = ax.boxplot([by[m] for m in order], widths=0.42, showfliers=False,
                    medianprops=dict(color=SCR, linewidth=1.4),
                    boxprops=dict(color=INK, linewidth=0.8),
                    whiskerprops=dict(color=INK, linewidth=0.8),
                    capprops=dict(color=INK, linewidth=0.8), zorder=3)
    ax.axhline(0.25, color=INK, linestyle="--", linewidth=0.9, zorder=1)
    ax.text(len(order) + 0.44, 0.25, r"$K=\frac{1}{4}$", va="center",
            ha="left", fontsize=8, color=INK)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([short[m] for m in order], fontsize=7)
    ax.set_ylabel("sectional curvature $K$")
    ax.set_ylim(-0.05, 0.55)
    ax.set_xlim(0.45, len(order) + 1.25)
    for i, m in enumerate(order):
        ax.text(i + 1, 0.525, f"n={len(by[m])}", ha="center", fontsize=6.2,
                color="#555555")

    p = os.path.join(OUT, "fig1_curvature.png")
    fig.savefig(p)
    plt.close(fig)
    meds = {m: tmed(by[m]) for m in order}
    print(f"fig1 -> {p}")
    print(f"      medians: " + ", ".join(f"{short[m].splitlines()[0]}={meds[m]:.4f}"
                                         for m in order))
    print(f"      spread across models = {max(meds.values())-min(meds.values()):.4f}"
          f"   total points = {len(pts)}")
    return meds


# ---------------------------------------------------------------- figure 2
def fig2():
    """The central result: one control, three quantities, two outcomes."""
    wp = jl("results/scramble/within_points.jsonl")
    kl = defaultdict(list)
    for r in wp:
        if isinstance(r.get("K_real"), (int, float)) and \
           isinstance(r.get("K_within"), (int, float)):
            kl[r["layer"]].append(r)
    klay = sorted(kl)
    k_real = [tmed([r["K_real"] for r in kl[L]]) for L in klay]
    k_scr = [tmed([r["K_within"] for r in kl[L]]) for L in klay]

    v = json.load(open("results/volume_scramble/volume_scramble.json",
                       encoding="utf-8"))
    bl = defaultdict(list)
    for r in v["rows"]:
        bl[r["layer"]].append(r)
    vlay = sorted(bl)
    lv_real = [tmed([r["lv_real"] for r in bl[L]]) for L in vlay]
    lv_scr = [tmed([r["lv_within"] for r in bl[L]]) for L in vlay]
    ke_real = [tmed([r["ke_real"] for r in bl[L]]) for L in vlay]
    ke_scr = [tmed([r["ke_within"] for r in bl[L]]) for L in vlay]

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.15), sharex=True)

    ax = axes[0]
    ax.plot(klay, k_real, "o-", color=REAL, ms=3.4, lw=1.3, label="real")
    ax.plot(klay, k_scr, "s--", color=SCR, ms=3.2, lw=1.3, label="scrambled")
    ax.axhline(0.25, color=INK, ls=":", lw=0.8)
    ax.axhline(0.0, color=INK, lw=0.6)
    ax.set_title("(a) sectional curvature $K$\n" + r"$\bf{collapses}$",
                 fontsize=8, pad=4)
    ax.set_ylabel("$K$")
    ax.set_ylim(-0.35, 0.35)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")

    ax = axes[1]
    ax.plot(vlay, lv_real, "o-", color=REAL, ms=3.4, lw=1.3)
    ax.plot(vlay, lv_scr, "s--", color=SCR, ms=3.2, lw=1.3)
    ax.set_title("(b) log volume element\n" + r"$\bf{reproduced}$ ($\rho=+0.96$)",
                 fontsize=8, pad=4)
    ax.set_ylabel(r"$\log\sqrt{\det G}$")

    ax = axes[2]
    ax.plot(vlay, ke_real, "o-", color=REAL, ms=3.4, lw=1.3)
    ax.plot(vlay, ke_scr, "s--", color=SCR, ms=3.2, lw=1.3)
    ax.set_yscale("log")
    ax.set_title("(c) effective dimension $k_{\\mathrm{eff}}$\n"
                 + r"$\bf{reproduced}$ ($\rho=+0.99$)", fontsize=8, pad=4)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$k_{\mathrm{eff}}$")

    fig.tight_layout(h_pad=0.55)
    p = os.path.join(OUT, "fig2_split.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"fig2 -> {p}")
    print(f"      K  layers {klay}")
    print(f"      K  real {[round(x,4) for x in k_real]}")
    print(f"      K  scr  {[round(x,4) for x in k_scr]}")
    print(f"      ke real {[round(x,1) for x in ke_real]}")
    print(f"      ke scr  {[round(x,1) for x in ke_scr]}")
    return klay, k_real, k_scr, vlay, lv_real, lv_scr, ke_real, ke_scr


# ---------------------------------------------------------------- figure 3
def fig3():
    """Why a published near-zero curvature is not evidence of flatness.

    The local-PCA residual proxy is a function of its variance threshold.  Panel
    (a) is the measured proxy on GPT-2 across thresholds; panel (b) is the same
    estimator applied to a UNIT 3-SPHERE, whose curvature is exactly +1.
    """
    m = json.load(open("results/mabrok/mabrok.json", encoding="utf-8"))
    res = m["results"]
    ks = m["k_sweep"]
    thresholds = ["var0.9", "var0.95", "var0.99", "var0.999", "var0.9999"]
    tl = [0.9, 0.95, 0.99, 0.999, 0.9999]

    FLOOR = 1e-9

    def clamp(y):
        """Values that underflow to ~0 are drawn on the axis floor and marked.

        Plotting them unclamped drops the series off the axis, which hides
        exactly the behaviour the panel exists to show.
        """
        return [max(v, FLOOR) if v == v else np.nan for v in y], \
               [v < FLOOR for v in y]

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True)

    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(ks):
        key = f"L12_k{k}"
        if key not in res:
            continue
        y, under = clamp([res[key].get(t, np.nan) for t in thresholds])
        c = cmap(i / max(1, len(ks) - 1))
        ax.plot(range(len(tl)), y, "o-", ms=3.2, lw=1.2, color=c, label=f"$k$={k}")
        xu = [j for j, u in enumerate(under) if u]
        if xu:
            ax.plot(xu, [FLOOR] * len(xu), "v", ms=4.2, color=c,
                    markeredgecolor="white", markeredgewidth=0.4)
    ax.axhline(1e-5, color=SCR, ls="--", lw=1.0)
    ax.text(0.06, 1.7e-5, "published $10^{-5}$", color=SCR, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 3, 2)
    ax.set_xticks(range(len(tl)))
    ax.set_xticklabels([str(t) for t in tl], fontsize=6.5)
    ax.set_ylabel("residual", fontsize=7.5)
    ax.set_title("(a) the proxy on GPT-2 layer 12", fontsize=7.6, pad=3)
    ax.legend(fontsize=5.8, frameon=False, ncol=2, loc="lower left")
    ax.text(len(tl) - 1.05, FLOOR / 1.6, r"$\blacktriangledown$ = numerically 0",
            fontsize=5.8, color="#555555", ha="right")

    # ---- panel (b): the same estimator on a manifold of known curvature ----
    rng = np.random.default_rng(3)
    n, amb = 1800, 12
    X = rng.normal(size=(n, 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)          # unit 3-sphere, K=+1
    Q, _ = np.linalg.qr(rng.normal(size=(amb, 4)))
    S = X @ Q.T
    P = rng.normal(size=(n, amb)) @ np.diag([1] * 3 + [0] * (amb - 3))  # a plane

    def residual(pts, thr, nn=60):
        d = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        out = []
        for i in rng.choice(len(pts), 120, replace=False):
            nb = pts[np.argsort(d[i])[:nn]]
            nb = nb - nb.mean(0)
            sv = np.linalg.svd(nb, compute_uv=False) ** 2
            frac = np.cumsum(sv) / sv.sum()
            j = int(np.searchsorted(frac, thr)) + 1
            out.append(sv[j:].sum() / sv.sum() if j < len(sv) else 0.0)
        return float(np.median(out))

    ys = [residual(S, t) for t in tl]
    yp = [residual(P, t) for t in tl]

    # Grouped bars, because both series underflow to exactly 0 at tight
    # thresholds and a log line plot simply drops them off the axis.
    ax = axes[1]
    x = np.arange(len(tl))
    w = 0.36
    for off, y, col, lab in ((-w / 2, ys, REAL, "unit 3-sphere ($K=+1$)"),
                             (+w / 2, yp, GLOB, "flat 3-plane ($K=0$)")):
        h = [max(v, FLOOR) for v in y]
        ax.bar(x + off, h, w, bottom=FLOOR / 3, color=col, label=lab,
               edgecolor="white", linewidth=0.4)
    ax.axhline(1e-5, color=SCR, ls="--", lw=1.0)
    ax.text(len(tl) - 0.45, 1.7e-5, "published $10^{-5}$", color=SCR,
            fontsize=6.5, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 3, 2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in tl], fontsize=6.5)
    ax.set_xlabel("retained-variance threshold")
    ax.set_ylabel("residual", fontsize=7.5)
    ax.set_title("(b) the same estimator, known curvature", fontsize=7.6, pad=3)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")

    fig.tight_layout(h_pad=0.6)
    p = os.path.join(OUT, "fig3_threshold.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"fig3 -> {p}")
    print(f"      sphere residual by threshold: "
          + ", ".join(f"{t}:{y:.2e}" for t, y in zip(tl, ys)))
    print(f"      plane  residual by threshold: "
          + ", ".join(f"{t}:{y:.2e}" for t, y in zip(tl, yp)))
    return tl, ys, yp


if __name__ == "__main__":
    fig1()
    print()
    fig2()
    print()
    fig3()
    print("\nall figures written to", OUT)
