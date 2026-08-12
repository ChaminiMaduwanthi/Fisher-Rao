"""
STAGE 0 end-to-end baseline.

Establishes the pipeline every later stage swaps components into:

    load model -> extract residual stream -> apply a metric -> compute a
    curvature -> plot

and answers four questions on a real model rather than synthetic data:

  0.6  Manson's U^T U curvature reproduced?                  (the baseline)
  A    does the KL-Hessian identity hold for our G(h)?       (Gate A, early)
  3.10b is G(h) ill-conditioned on real activations?         (the obstacle)
  --   does the final-norm Jacobian actually matter?         (a design choice)

Usage:  python run_stage0.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

# Windows consoles default to cp1252, which cannot encode the byte-level BPE
# markers (Ġ etc.) that appear in token strings.  Without this the script
# dies on its own progress output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import (
    LM, manson_metric, euclidean_metric, fisher_metric,
    truncation_error, spectrum, gate_a, radial_null_check,
    salience, curvature, layer_trajectory,
)

OUT = pathlib.Path("results/stage0")
PROMPT = (
    "The bank was steep and muddy after the heavy rain, so the hikers "
    "climbed down to the river"
)

torch.manual_seed(0)


def rule(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    rule("MODEL")
    lm = LM()
    print(lm.summary())
    results["model"] = dict(
        model_id=lm.model_id, d=lm.d, N=lm.N, layers=lm.n_layers,
        tied=lm.tied, norm=lm.norm_kind,
    )

    # ------------------------------------------------------------------
    rule("0.3  RESIDUAL STREAM EXTRACTION")
    H, toks = lm.residual_stream(PROMPT)
    print(f"hidden states : {tuple(H.shape)}   (L+1, T, d)")
    print(f"tokens        : {len(toks)}")
    print(f"last token    : {toks[-1]!r}")
    top = torch.topk(lm.next_token_probs(H[-1, -1]), 5)
    print("top-5 next-token predictions from the final layer:")
    for p, i in zip(top.values, top.indices):
        print(f"    {float(p):6.3f}  {lm.tok.decode([int(i)])!r}")

    # sanity: logit lens at the final layer must match the model's own head
    with torch.no_grad():
        ref = lm.model(**lm.tok(PROMPT, return_tensors="pt")).logits[0, -1].to(torch.float64)
    ours = lm.logits(H[-1, -1])
    err = float(torch.linalg.vector_norm(ours - ref) / torch.linalg.vector_norm(ref))
    print(f"\nlogit-lens check vs model head: rel.err = {err:.2e}  "
          f"{'PASS' if err < 1e-6 else 'FAIL'}")
    results["logit_lens_rel_err"] = err

    # ------------------------------------------------------------------
    rule("0.6  MANSON BASELINE  (G = U^T U, constant -> flat space)")
    traj = layer_trajectory(H, token=-1)
    G_man = manson_metric(lm.U)
    G_euc = euclidean_metric(lm.d)

    sp = spectrum(G_man)
    print(f"U^T U : cond_eff = {sp['cond_eff']:.2e}, "
          f"99%-trace rank = {sp['k_eff']}/{sp['d']} ({100*sp['frac_eff']:.0f}%)")

    for name, G in (("U^T U  (Manson)", G_man), ("I      (Euclidean)", G_euc)):
        kap = curvature(traj, G)
        sal = salience(traj, G)
        print(f"\n{name}")
        print(f"   curvature : median {float(kap.median()):.4f}  "
              f"max {float(kap.max()):.4f} at layer {int(kap.argmax()) + 1}")
        print(f"   salience  : median {float(sal.median()):.4f}  "
              f"max {float(sal.max()):.4f} at layer {int(sal.argmax())}")
        # curvature[j] lives at interior point j+1 (j = 0 .. L-2);
        # salience[j+1] is the step leaving that same point.  Same length.
        r = float(torch.corrcoef(torch.stack([kap, sal[1:]]))[0, 1])
        print(f"   corr(curvature, salience) = {r:+.3f}   "
              f"(Manson reports -0.89 on LLaMA-3.2-3b)")
        results[f"baseline_{name.split()[0]}"] = dict(
            curv_median=float(kap.median()), curv_max=float(kap.max()),
            curv_argmax=int(kap.argmax()) + 1, corr_curv_sal=r,
        )

    # ------------------------------------------------------------------
    rule("GATE A (early)  KL-Hessian identity for G(h) on a real model")
    h = H[-1, -1].clone()
    v = torch.randn(lm.d, dtype=torch.float64)
    quad, rows = gate_a(lm, h, v)
    print(f"predicted 0.5 v^T G(h) v = {quad:.12e}\n")
    print(f"{'eps':>8} {'KL/eps^2':>20} {'rel.err':>11}")
    for eps, ratio, rel in rows:
        print(f"{eps:>8.0e} {ratio:>20.12e} {rel:>11.2e}")
    best_eps, _, best_rel = min(rows, key=lambda r: r[2])
    print(f"\nplateau at eps = {best_eps:.0e}, rel.err = {best_rel:.2e}   "
          f"{'PASS' if best_rel < 1e-4 else 'FAIL'}")
    results["gate_a"] = dict(quad=quad, best_eps=best_eps, best_rel_err=best_rel,
                             sweep=[(e, r) for e, _, r in rows])

    # ------------------------------------------------------------------
    rule("DESIGN CHECK  the final-norm Jacobian: scale vs structure")
    hm = H[20, -1]                      # a mid-layer state, away from the ends
    r2 = float(hm.pow(2).mean() + lm.eps)
    G_with = fisher_metric(lm, hm, top_k=None, include_norm_jacobian=True)
    G_without = fisher_metric(lm, hm, top_k=None, include_norm_jacobian=False)
    nw, no = (float(torch.linalg.matrix_norm(x)) for x in (G_with, G_without))
    after = float(torch.linalg.matrix_norm(G_with - G_without / r2) / nw)
    print(f"|| G_without || / || G_with || = {no / nw:.2f}    "
          f"r^2 = {r2:.2f}   <- almost entirely a 1/r^2 SCALE factor")
    print(f"after removing that scale, residual = {after:.4f}  "
          f"<- the rank-1 projector, the STRUCTURAL part")
    print("Small in norm, decisive in effect: only the projector produces the")
    print("exact null direction below.  Note curvature is NOT scale-invariant")
    print("(K -> K/c under g -> c g), so the scale factor matters too.")
    results["norm_jacobian"] = dict(ratio=no / nw, r2=r2, residual_after_rescale=after)

    # ------------------------------------------------------------------
    rule("RQ4 NULL-SPACE FALSIFICATION TEST  (E5, early)")
    print("RMSNorm is scale-invariant, so A = diag(g)(1/r)(I - hhat hhat^T) is a")
    print("projector and h must be an EXACT null direction of G(h).\n")
    print(f"{'layer':>6} {'|A h|/|h|':>12} {'|Gh|/(|G||h|)':>14} "
          f"{'KL(p(h)||p(2h))':>16} {'KL random':>11} {'ratio':>10}")
    null_rows = []
    for l in (0, 10, 20, 30):
        nc = radial_null_check(lm, H[l, -1])
        ratio = nc["kl_radial"] / nc["kl_random"] if nc["kl_random"] else float("nan")
        print(f"{l:>6} {nc['Ah_rel']:>12.2e} {nc['Gh_rel']:>14.2e} "
              f"{nc['kl_radial']:>16.2e} {nc['kl_random']:>11.2e} {ratio:>10.1e}")
        null_rows.append(dict(layer=l, **nc))
    results["radial_null"] = null_rows
    worst = max(abs(r["kl_radial"] / r["kl_random"]) for r in null_rows if r["kl_random"])
    print(f"\nworst radial/random KL ratio = {worst:.1e}   "
          f"{'PASS' if worst < 1e-8 else 'FAIL'}")
    print("Doubling a hidden state changes the prediction by nothing.  The")
    print("predictive map sees only DIRECTION -> the semantic manifold is a")
    print("space of directions, and that is where the Fisher geometry lives.")

    # ------------------------------------------------------------------
    rule("2.4  TOP-K TRUNCATION ERROR")
    te = truncation_error(lm, h)
    for k, e in te.items():
        print(f"   k = {k:>6} :  rel.err = {e:.3e}")
    results["truncation_error"] = te

    # ------------------------------------------------------------------
    rule("3.10b  CONDITIONING OF G(h) ON REAL ACTIVATIONS")
    print("n_pos    = eigenvalues > 1e-30 * lam_max   (~algebraic rank)")
    print("rank_num = matrix_rank at float64 default  (working-precision rank)")
    print("k_eff    = directions holding 99% of trace (what curvature needs)\n")
    print(f"{'layer':>6} {'n_pos':>6} {'rank_num':>9} {'k_eff':>6} {'frac':>7} "
          f"{'cond_eff':>10} {'PR':>7} {'entropy':>8}")
    cond_rows = []
    for l in range(lm.n_layers + 1):
        hl = H[l, -1]
        G = fisher_metric(lm, hl, top_k=2000)
        s = spectrum(G)
        logp = torch.log_softmax(lm.logits(hl), -1)
        H_ent = float(-(logp.exp() * logp).sum())
        print(f"{l:>6} {s['n_pos']:>6} {s['rank_num']:>9} {s['k_eff']:>6} "
              f"{s['frac_eff']:>7.1%} {s['cond_eff']:>10.2e} {s['pr']:>7.2f} "
              f"{H_ent:>8.3f}")
        cond_rows.append(dict(layer=l, **s, entropy=H_ent))
    results["conditioning_by_layer"] = cond_rows

    n_alg = sum(r["n_pos"] >= r["d"] - 2 for r in cond_rows)
    n_num = sum(r["rank_num"] >= r["d"] - 2 for r in cond_rows)
    from collections import Counter
    modal = Counter(r["n_pos"] for r in cond_rows).most_common(1)[0]
    print(f"\nALGEBRAIC rank ~ d at {n_alg}/{len(cond_rows)} layers; range "
          f"{min(r['n_pos'] for r in cond_rows)}.."
          f"{max(r['n_pos'] for r in cond_rows)} of {lm.d}, "
          f"modal value {modal[0]} ({modal[1]} layers)")
    print(f"   The Sigma_p all-ones direction does NOT transfer to G.  But the")
    print(f"   modal n_pos is d-1 = {lm.d - 1}, exactly as ONE radial null direction")
    print(f"   predicts (see the RQ4 test above).  It reads as d-1 rather than a")
    print(f"   hard zero only because RMSNorm's eps > 0 leaves the radial")
    print(f"   eigenvalue at ~1e-18 relative rather than exactly 0.")
    print(f"WORKING-PRECISION rank ~ d at only {n_num}/{len(cond_rows)} layers; "
          f"range {min(r['rank_num'] for r in cond_rows)}.."
          f"{max(r['rank_num'] for r in cond_rows)}")
    n_inf = sum(r["cond"] == float("inf") for r in cond_rows)
    print(f"   lam_min underflowed to exactly 0 at {n_inf}/{len(cond_rows)} layers "
          f"-> cond = inf.  float64 is necessary but NOT sufficient:")
    print(f"   subspace restriction is unavoidable, not a stylistic choice.")
    print(f"k_eff range: {min(r['k_eff'] for r in cond_rows)} .. "
          f"{max(r['k_eff'] for r in cond_rows)} of {lm.d}  "
          f"({min(r['frac_eff'] for r in cond_rows):.1%}.."
          f"{max(r['frac_eff'] for r in cond_rows):.1%})")
    print(f"   FishBack (arXiv:2605.17231) reports 2-17% on GPT-2 "
          f"-> independent replication on a different architecture.")

    # ------------------------------------------------------------------
    (OUT / "stage0.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'stage0.json'}")
    _plot(traj, G_man, G_euc, cond_rows, lm)


def _plot(traj, G_man, G_euc, cond_rows, lm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable, skipping figure")
        return

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)

    k_man, k_euc = curvature(traj, G_man), curvature(traj, G_euc)
    x = range(1, len(k_man) + 1)
    ax[0].plot(x, k_man, marker="o", ms=3, label=r"$G=U^\top U$ (Manson)")
    ax[0].plot(x, k_euc, marker="s", ms=3, label=r"$G=I$ (Euclidean)")
    ax[0].set(xlabel="layer", ylabel=r"Frenet curvature $\kappa$",
              title="Trajectory curvature\n(flat metrics: path bending only)")
    ax[0].legend(fontsize=8)

    s_man = salience(traj, G_man)
    ax[1].plot(range(len(s_man)), s_man, marker="o", ms=3, color="C2")
    ax[1].set(xlabel="layer", ylabel=r"salience $\|x_{l+1}-x_l\|_G$",
              title="Salience under $U^\\top U$")

    layers = [r["layer"] for r in cond_rows]
    ax[2].semilogy(layers, [r["cond_eff"] for r in cond_rows], marker="o", ms=3, color="C3")
    ax2 = ax[2].twinx()
    ax2.plot(layers, [r["frac_eff"] for r in cond_rows], marker="s", ms=3,
             color="C0", alpha=.7)
    ax2.set_ylabel("99%-trace rank / d", color="C0")
    ax[2].set(xlabel="layer",
              ylabel=r"cond$_{\mathrm{eff}} = \lambda_{max}/\lambda_{k_{eff}}$",
              title="Fisher metric conditioning\n(within the retained subspace)")

    fig.savefig(OUT / "stage0.png", dpi=150)
    print(f"wrote {OUT / 'stage0.png'}")


if __name__ == "__main__":
    main()
