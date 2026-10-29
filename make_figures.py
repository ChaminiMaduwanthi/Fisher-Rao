"""
Regenerate every figure in the conference paper from the saved result files.

No number in a figure is typed by hand: each panel reads results/*.json and
recomputes its own summary with the project's true median.  Run this before
rebuilding the paper so the figures and the text cannot drift apart.

Usage:  python make_figures.py
Output: paper/figures/fig1_pipeline.png   workflow of the study (full width)
        paper/figures/fig2_controls.png   K by entropy band, trained model and controls
        paper/figures/fig3_spectral.png   effective dimension by layer, trained and control
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np

from fisherrao.stats import median, spearman

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = os.path.join("paper", "figures")
os.makedirs(OUT, exist_ok=True)

# IEEE single-column width is ~3.5in; two-column spans ~7.16in.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
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
# Okabe-Ito colours, with distinct line styles so the figures survive greyscale
TRAINED = dict(color="#0072B2", ls="-", marker="o", label="trained")
C1 = dict(color="#E69F00", ls="--", marker="s", label="C1 consistent scramble")
C2 = dict(color="#009E73", ls="-.", marker="^", label="C2 Gaussian readout")
C0 = dict(color="#7f7f7f", ls=":", marker="x", label="C0 inconsistent scramble")

MODELS = (("SmolLM2-135M", "SmolLM2-135M-Instruct_paper", "volume_scramble.json"),
          ("LLaMA-160M", "llama-160m_draw", "volume_scramble_llama-160m.json"),
          ("GPT-2", "gpt2_draw", "volume_scramble_gpt2.json"),
          ("Pythia-160M", "pythia-160m_draw", "volume_scramble_pythia-160m.json"))
BANDS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, float("inf")))
BAND_LABELS = ("<0.5", "0.5–1", "1–2", "2–4", ">4")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- figure 1
def fig1():
    """The workflow: measurement chain, the controls that replace the readout,
    and the validation that guards the chain."""
    fig, ax = plt.subplots(figsize=(7.0, 2.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34.5)
    ax.axis("off")

    def box(x, y, w, h, text, face="#eef3f8", edge="#1f4e79", fs=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc=face, ec=edge, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, linespacing=1.3)

    def arrow(x0, y0, x1, y1, color="#1f4e79", ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, ls=ls,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))

    ya, ha = 21.0, 12.5                      # measurement chain
    yb, hb = 1.0, 13.5                       # controls, validation, spectrum
    box(0.5, ya, 12.0, ha, "Hidden state $h$\n(layer $\\ell$,\ntoken $t$)")
    box(15.0, ya, 18.0, ha, "Final norm $n(h)$,\nreadout $U$ (512 tokens)\n"
                            "$p=\\mathrm{softmax}(U\\,n(h))$")
    box(35.5, ya, 17.0, ha, "Pullback metric (1)\n$G=J^{\\top}(\\mathrm{diag}\\,p-pp^{\\top})J$")
    box(55.0, ya, 18.0, ha, "Null directions quotiented,\n$P=I-NN^{\\top}$; slice\n"
                            "$h_0+\\mathrm{span}\\,F$, with $F$\nthe top-$k$ eigenvectors", fs=6.2)
    box(75.3, ya, 13.0, ha, "Slice curvature (3)\nautodiff, float64\n16 random planes")
    box(75.3, yb, 13.0, hb, "Spectrum of $PGP$:\n$k_{\\mathrm{eff}}$, log-volume")
    box(90.6, yb, 9.0, ya + ha - yb, "Paired\nstatistics:\nsign test $z$,\nSpearman $\\rho$,\n"
                                       "decision\nrule", face="#f2f2f2", edge="#555555")
    for x0, x1 in ((12.5, 15.0), (33.0, 35.5), (52.5, 55.0), (73.0, 75.3)):
        arrow(x0, ya + ha / 2, x1, ya + ha / 2)
    arrow(73.0, ya + 2.0, 78.5, yb + hb)                      # slice -> spectrum
    arrow(88.3, ya + ha / 2, 90.6, ya + ha / 2, color="#555555")
    arrow(88.3, yb + hb / 2, 90.6, yb + hb / 2, color="#555555")

    box(15.0, yb, 37.5, hb,
        "Controls replace the readout at the same $h_0$\n"
        "C0: $p$ from $U$, Jacobian from $\\Pi U$ (not a pullback)\n"
        "C1: $z=U\\,n(h_0)+\\Pi U\\,(n(h)-n(h_0))$\n"
        "C2: $\\mathrm{softmax}(sW\\,n(h))$, entropy matched",
        face="#fff4e0", edge="#b36b00")
    arrow(24.0, yb + hb, 24.0, ya, color="#b36b00")

    box(55.0, yb, 18.0, hb, "Validation: 7 manifolds\nof known curvature;\nidentity (2) on and off $h_0$;\n"
                            "null test; Gauss check (4)", face="#e9f5ee", edge="#1b7a4a")
    arrow(64.0, yb + hb, 64.0, ya, color="#1b7a4a", ls="--")

    p = os.path.join(OUT, "fig1_pipeline.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"fig1 -> {p}")


# ---------------------------------------------------------------- figure 2
def fig2():
    """Median K by predictive-entropy band: the trained model and the three
    controls, on the same states."""
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.95), sharex=True, sharey=True)
    lo_lim, hi_lim = -0.06, 0.53
    for ax, (name, tag, _) in zip(axes.flat, MODELS):
        rows = [r for r in load(f"results/scramble/consistent_{tag}.json")["rows"] if "error" not in r]
        gauss = {r["_i"]: r["K_gauss"] for r in load(f"results/scramble/gaussian_{tag}.json")["rows"]
                 if "K_gauss" in r}
        series = {k: [] for k in ("real", "within", "consistent", "gauss")}
        for lo, hi in BANDS:
            sub = [r for r in rows if lo <= r["entropy"] < hi]
            series["real"].append(median([r["K_real"] for r in sub]))
            series["within"].append(median([r["K_within"] for r in sub]))
            series["consistent"].append(median([r["K_consistent"] for r in sub]))
            series["gauss"].append(median([gauss[r["_i"]] for r in sub if r["_i"] in gauss]))
        x = np.arange(len(BANDS))
        ax.axhline(0.25, color=INK, ls=(0, (4, 3)), lw=0.7, zorder=1)
        w = np.array(series["within"])
        inside = w >= lo_lim
        ax.plot(x[inside], w[inside], color=C0["color"], ls=C0["ls"], marker=C0["marker"],
                ms=3.2, lw=1.0, zorder=2)
        if (~inside).any():                     # off the axis: marked on its lower edge
            ax.plot(x[~inside], np.full((~inside).sum(), lo_lim + 0.012), ls="none",
                    marker="v", ms=3.6, color=C0["color"], zorder=2)
        for key, st in (("gauss", C2), ("consistent", C1), ("real", TRAINED)):
            ax.plot(x, series[key], color=st["color"], ls=st["ls"], marker=st["marker"],
                    ms=3.0, lw=1.1, zorder=3)
        ax.set_title(f"{name} ($n$ = {len(rows)})", fontsize=7.2, pad=2.5)
        ax.set_ylim(lo_lim, hi_lim)
        ax.set_xticks(x)
        ax.set_xticklabels(BAND_LABELS, fontsize=6.2)
        ax.tick_params(axis="y", labelsize=6.5)
        print(f"fig2 {name:12s} " + " ".join(f"{k}={[round(v, 3) for v in s]}" for k, s in series.items()))
    for ax in axes[1]:
        ax.set_xlabel("predictive entropy (nats)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("median $K$", fontsize=7)
    handles = [plt.Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"], ms=3.2, lw=1.1,
                          label=s["label"]) for s in (TRAINED, C1, C2, C0)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.0)
    fig.tight_layout(h_pad=0.6, w_pad=0.4, rect=(0, 0.1, 1, 1))
    p = os.path.join(OUT, "fig2_controls.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"fig2 -> {p}")


# ---------------------------------------------------------------- figure 3
def fig3():
    """Effective dimension by layer, trained metric against the paired scramble,
    which coincides with the consistent control at the base point."""
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.55))
    for ax, (name, _, fn) in zip(axes.flat, MODELS):
        rows = load(os.path.join("results", "volume_scramble", fn))["rows"]
        layers = sorted({r["layer"] for r in rows})
        real = [median([r["ke_real"] for r in rows if r["layer"] == l]) for l in layers]
        ctrl = [median([r["ke_within"] for r in rows if r["layer"] == l]) for l in layers]
        rho = spearman([r["ke_real"] for r in rows], [r["ke_within"] for r in rows])
        ax.plot(layers, real, color=TRAINED["color"], ls="-", marker="o", ms=2.8, lw=1.1)
        ax.plot(layers, ctrl, color=C1["color"], ls="--", marker="s", ms=2.6, lw=1.1)
        ax.set_yscale("log")
        lo, hi = min(real + ctrl), max(real + ctrl)
        ax.set_yticks([t for t in (5, 10, 20, 50, 100, 200, 400) if lo / 1.3 <= t <= hi * 1.3])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{name} ($\\rho$ = {rho:.2f})", fontsize=7.2, pad=2.5)
        ax.tick_params(labelsize=6.3)
        print(f"fig3 {name:12s} layers {layers} trained {real} control {ctrl} rho {rho:.3f}")
    for ax in axes[1]:
        ax.set_xlabel("layer", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("median $k_{\\mathrm{eff}}$", fontsize=7)
    handles = [plt.Line2D([], [], color=TRAINED["color"], ls="-", marker="o", ms=3, lw=1.1, label="trained"),
               plt.Line2D([], [], color=C1["color"], ls="--", marker="s", ms=3, lw=1.1,
                          label="scrambled readout (C0 = C1 at $h_0$)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6)
    fig.tight_layout(h_pad=0.5, w_pad=0.5, rect=(0, 0.06, 1, 1))
    p = os.path.join(OUT, "fig3_spectral.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"fig3 -> {p}")


# ---------------------------------------------------------------- calibration
def sphere_calibration(tl=(0.9, 0.95, 0.99, 0.999, 0.9999)):
    """The local-PCA residual on a unit 3-sphere (K = +1) and a flat 3-plane (K = 0)
    in 12 dimensions, at each retained-variance threshold.  Seeded, so the numbers
    are reproducible; make_tables.py records them (Table 8)."""
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

    return [residual(S, t) for t in tl], [residual(P, t) for t in tl]


if __name__ == "__main__":
    fig1()
    print()
    fig2()
    print()
    fig3()
    print("\nall figures written to", OUT)
