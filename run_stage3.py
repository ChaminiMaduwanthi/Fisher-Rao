"""
STAGE 3 -- intrinsic Fisher-Rao curvature.

Part 1 validates the torch/autodiff curvature code against the three
analytically known answers already verified in validate_curvature.py.  Part 2
only runs if Part 1 passes.  That ordering is the whole point of Gate B: a sign
slip in the Riemann contraction once produced a plausible, exactly-constant,
wrong answer, and nothing but a known answer caught it.

Usage:  python run_stage3.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, fisher_metric, spectrum
from fisherrao.curvature import (
    curvature_at, sectional, scalar_curvature, volume_element, frame_invariance,
)

OUT = pathlib.Path("results/stage3")
PROMPT = ("The bank was steep and muddy after the heavy rain, so the hikers "
          "climbed down to the river")


def rule(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


# ======================================================================
# PART 1 -- GATE B: validate against analytically known curvature
# ======================================================================
def g_simplex(x):
    d = x.shape[-1]
    return torch.diag(1.0 / x) + torch.ones(d, d, dtype=x.dtype) / (1.0 - x.sum())


def g_gauss(x):
    s = x[1]
    return torch.stack([
        torch.stack([1.0 / s**2, torch.zeros((), dtype=x.dtype)]),
        torch.stack([torch.zeros((), dtype=x.dtype), 2.0 / s**2]),
    ])


def g_poincare(x):
    y = x[1]
    return torch.eye(2, dtype=x.dtype) / y**2


def validate() -> bool:
    rule("GATE B  validation ladder -- torch/autodiff implementation")
    gen = torch.Generator().manual_seed(1)
    cases = [
        ("rung 3  categorical simplex (n=4)", g_simplex, 3, +0.25,
         lambda: torch.distributions.Dirichlet(torch.ones(4, dtype=torch.float64))
                 .sample()[:3]),
        ("rung 6  univariate Gaussian", g_gauss, 2, -0.50,
         lambda: torch.tensor([torch.randn(1, generator=gen, dtype=torch.float64).item(),
                               0.5 + abs(torch.randn(1, generator=gen,
                                                     dtype=torch.float64).item())],
                              dtype=torch.float64)),
        ("rung 2  Poincare half-plane", g_poincare, 2, -1.00,
         lambda: torch.tensor([torch.randn(1, generator=gen, dtype=torch.float64).item(),
                               0.5 + abs(torch.randn(1, generator=gen,
                                                     dtype=torch.float64).item())],
                              dtype=torch.float64)),
    ]
    ok = True
    for name, g_fn, k, expect, draw in cases:
        errs = []
        for _ in range(4):
            x = draw()
            u = torch.randn(k, generator=gen, dtype=torch.float64)
            v = torch.randn(k, generator=gen, dtype=torch.float64)
            K = sectional(g_fn, x, u, v)
            errs.append(abs(K - expect))
        worst = max(errs)
        good = worst < 1e-4
        ok &= good
        print(f"  {name:<38} K = {expect:+.2f}   worst |err| = {worst:.2e}   "
              f"{'PASS' if good else 'FAIL'}")

    # scalar_curvature needs its OWN rung.  Validating sectional() does not
    # validate the Ricci contraction, and scalar R is the quantity that carries
    # the model-specific signal (06-stage3-log.md S4.2).  For constant K,
    # R must equal k(k-1)*K exactly.
    print()
    for n in (4, 6, 7):
        k = n - 1
        x = torch.distributions.Dirichlet(torch.ones(n, dtype=torch.float64)).sample()[:k]
        exp = k * (k - 1) * 0.25
        got = scalar_curvature(g_simplex, x)
        rel = abs(got - exp) / abs(exp)
        good = rel < 1e-6
        ok &= good
        print(f"  scalar R, simplex n={n} (k={k}){'':<15} R = {exp:+.2f}   "
              f"rel.err = {rel:.2e}   {'PASS' if good else 'FAIL'}")

    print(f"\n  GATE B: {'PASS -- proceeding to real activations' if ok else 'FAIL -- stopping'}")
    return ok


# ======================================================================
# PART 2 -- real Fisher-Rao curvature
# ======================================================================
def real_curvature():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    lm = LM()
    H, _ = lm.residual_stream(PROMPT)
    print(f"\n{lm.model_id}: d={lm.d}, layers={lm.n_layers}")

    K_DIM = 6          # retained subspace dimension; k_eff measured at 3-86
    TOP_K = 512        # |R| identical to top_k=2000 (1.082e-02) at 1.5x the speed
    LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]

    # -- 3.8 frame invariance: mandatory before any number is believed -----
    rule("3.8  FRAME INVARIANCE  (curvature must not depend on basis rotation)")
    inv = []
    for l in (10, 20, 30):
        fi = frame_invariance(lm, H[l, -1], K_DIM, top_k=TOP_K)
        inv.append(dict(layer=l, **fi))
        print(f"  layer {l:>2}:  R = {fi['scalar']:+.6e}   rotated = "
              f"{fi['scalar_rotated']:+.6e}   rel.diff = {fi['rel_diff']:.2e}")
    worst_inv = max(r["rel_diff"] for r in inv)
    print(f"\n  worst rel.diff = {worst_inv:.2e}   "
          f"{'PASS' if worst_inv < 1e-6 else 'FAIL -- moving-frame handling is wrong'}")
    res["frame_invariance"] = inv
    if worst_inv >= 1e-6:
        print("  Refusing to report curvature values on an unverified frame.")
        return res

    # -- the measurement -------------------------------------------------
    rule(f"INTRINSIC FISHER-RAO CURVATURE  (k = {K_DIM}, radial direction quotiented)")
    print(f"Absolute K is pinned near the ambient simplex value +0.25, so the")
    print(f"model-specific signal is the DEVIATION.  Exact references at k={K_DIM}:")
    print(f"    K_simplex = +0.25          R_simplex = k(k-1)/4 = "
          f"{K_DIM * (K_DIM - 1) * 0.25:.2f}")
    print(f"Both validated to machine precision above.\n")
    print(f"{'layer':>6} {'K median':>11} {'dK median':>11} {'dK IQR':>10} "
          f"{'scalar R':>10} {'dR':>9} {'dR rel':>8} {'log vol':>9} {'k_eff':>6}")
    rows = []
    for l in LAYERS:
        h = H[l, -1]
        # NOT merged into curvature_and_volume, deliberately.  These two use
        # DIFFERENT top_k -- 512 for curvature, 2000 (the default) for volume --
        # and merging them would silently move the layer-30 log-volume by 0.01.
        # Measured saving from merging here: 2.79s vs 2.80s, i.e. none, because
        # the Riemann tensor dominates.  Not worth changing a published number
        # for.  curvature_and_volume is the right call where one top_k serves
        # both.
        c = curvature_at(lm, h, K_DIM, top_k=TOP_K, n_planes=48)
        lv = volume_element(lm, h, K_DIM)
        ke = spectrum(fisher_metric(lm, h, top_k=2000))["k_eff"]
        print(f"{l:>6} {c['K_median']:>11.5f} {c['dK_median']:>+11.2e} "
              f"{c['dK_iqr']:>10.2e} {c['scalar_R']:>10.3f} {c['dR']:>+9.3f} "
              f"{c['dR_rel']:>+8.1%} {lv:>9.2f} {ke:>6}")
        rows.append(dict(layer=l, log_vol=lv, k_eff=ke,
                         **{k: v for k, v in c.items() if k != "K"}))
    res["curvature_by_layer"] = rows

    print(f"\n  ABSOLUTE K:  median {min(r['K_median'] for r in rows):.4f}.."
          f"{max(r['K_median'] for r in rows):.4f}, K>0 in "
          f"{min(r['frac_positive'] for r in rows):.0%}.."
          f"{max(r['frac_positive'] for r in rows):.0%} of planes")
    print(f"     -> strongly POSITIVELY curved, NOT 'essentially flat' (RQ2b).")
    print(f"        Do NOT read this as 'four orders of magnitude above Mabrok's")
    print(f"        1e-5'.  That compares against his PUBLISHED number on a")
    print(f"        different, much larger point cloud; reimplementing his proxy")
    print(f"        on THIS corpus gives 2.4e-2, not 1e-5, because a 20-token")
    print(f"        sentence is not a local neighbourhood.  The magnitude gap is")
    print(f"        NOT established (06-stage3-log.md S6, 07-stage4-log.md S1.2).")
    print(f"        The claim that survives is the CORRELATION one: on identical")
    print(f"        activations the proxy does not track the intrinsic quantity.")
    dr = [r["dR_rel"] for r in rows]
    print(f"\n  DEVIATION dR/R_ref: {min(dr):+.1%} .. {max(dr):+.1%}  "
          f"<- the model-specific signal")
    lo = min(rows, key=lambda r: r["dR_rel"]); hi = max(rows, key=lambda r: r["dR_rel"])
    print(f"     most simplex-like : layer {lo['layer']:>2} (dR/R = {lo['dR_rel']:+.1%})")
    print(f"     most anomalous    : layer {hi['layer']:>2} (dR/R = {hi['dR_rel']:+.1%})")
    dk = [r["dK_iqr"] for r in rows]
    print(f"  dK IQR spread: {min(dk):.1e} .. {max(dk):.1e}  "
          f"({max(dk)/max(min(dk),1e-30):.0f}x)  <- anisotropy varies with depth")

    # -- k sensitivity: the cutoff must not decide the answer -------------
    rule("3.12  SENSITIVITY TO THE RETAINED RANK k  (layer 20)")
    print(f"{'k':>4} {'K median':>13} {'K>0':>6} {'scalar R':>14}")
    sens = []
    for k in (3, 4, 5, 6, 7):
        c = curvature_at(lm, H[20, -1], k, top_k=TOP_K, n_planes=32)
        print(f"{k:>4} {c['K_median']:>13.4e} {c['frac_positive']:>6.0%} "
              f"{c['scalar_R']:>14.4e}")
        sens.append(dict(k=k, **{kk: v for kk, v in c.items() if kk != "K"}))
    res["k_sensitivity"] = sens
    print("\n  If the conclusions move with k, curvature is not identifiable")
    print("  from this metric and that is itself the finding (02-research-gap.md S6).")

    (OUT / "stage3.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'stage3.json'}")
    return res


if __name__ == "__main__":
    if validate():
        real_curvature()
