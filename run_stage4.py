"""
STAGE 4 -- E2, the adjudication experiment, plus the layer-wise volume profile.

Runs all four instruments on IDENTICAL activations over a multi-sentence corpus:

    1. intrinsic Fisher-Rao sectional curvature   (this work)
    2. local-PCA residual proxy                   (Mabrok 2026)
    3. Frenet curvature under G = U^T U           (Manson 2025)
    4. Euclidean contextual angle                 (King et al. 2026)

and reports their pairwise correlations.  Either the proxies track the intrinsic
quantity or they do not; both are results.

Also produces the layer-wise per-dimension log-volume profile, which Stage 3
established is the k-STABLE quantity (Spearman rho 0.92-1.00 across k=3..7,
versus 0.20 for the Riemann-derived deviation).

Cost note: intrinsic curvature is ~9 s/point, so it runs on a subsample while
the cheap instruments run on everything.  Correlations are computed on the
subsample where all four exist.

Usage:  python run_stage4.py [n_sentences]
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, manson_metric, fisher_metric, spectrum
from fisherrao import corpus
from fisherrao.curvature import (curvature_at, volume_element, select_k,
                                 spectral_diagnostics)
from fisherrao.proxies import pca_curvature, second_fundamental, contextual_angle
from fisherrao.trajectory import curvature as frenet_curvature, layer_trajectory
from fisherrao.stats import spearman as _spearman

OUT = pathlib.Path("results/stage4")
K_DIM, TOP_K = 5, 512          # k=5 sits inside the cond_eff ceiling at most layers
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]
N_INTRINSIC = 360             # stratified: 40 per layer x 9 layers


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def spearman(a, b):
    """Tie-corrected Spearman -- see fisherrao/stats.py for why this is
    not argsort(argsort(x))."""
    return _spearman(a, b)
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_sent = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    res = {}

    lm = LM()
    sents = corpus.sentences()[:n_sent]
    print(lm.summary())
    print(f"\n{corpus.summary()}   using {len(sents)}")

    # ---- extract everything once -----------------------------------------
    rule("EXTRACT")
    t0 = time.time()
    streams = []
    for s in sents:
        H, toks = lm.residual_stream(s)
        streams.append((H, toks))
    n_pos = sum(H.shape[1] for H, _ in streams)
    print(f"{len(streams)} sentences, {n_pos} token positions, "
          f"{len(LAYERS)} layers analysed  ({time.time()-t0:.1f}s)")
    res["corpus"] = dict(n_sentences=len(streams), n_positions=n_pos,
                         layers=LAYERS, k=K_DIM, top_k=TOP_K)

    # ---- layer-wise volume profile (the k-stable primary quantity) --------
    rule("LAYER-WISE PER-DIMENSION LOG-VOLUME  (k-stable; the primary quantity)")
    print(f"NOTE: k is held FIXED at {K_DIM} for cross-layer comparability; k_sel is")
    print("what the cond_eff<=1e2 rule would choose, reported for diagnosis.\n")
    print(f"{'layer':>6} {'logvol/k mean':>14} {'std':>9} {'k_sel med':>10} "
          f"{'cond_eff med':>13} {'entropy':>9} {'n':>5}")
    vol_rows = []
    for l in LAYERS:
        vals, ksel, conds, ents = [], [], [], []
        for H, _ in streams:
            for t in range(H.shape[1]):
                h = H[l, t]
                # one eigendecomposition, not two -- see spectral_diagnostics
                sd = spectral_diagnostics(lm, h, K_DIM, cond_max=1e2, top_k=TOP_K)
                vals.append(sd["log_vol_per_dim"])
                ksel.append(sd["k_sel"]); conds.append(sd["cond_eff"])
                lp = torch.log_softmax(lm.logits(h), -1)
                ents.append(float(-(lp.exp() * lp).sum()))
        v = torch.tensor(vals, dtype=torch.float64)
        print(f"{l:>6} {float(v.mean()):>14.4f} {float(v.std()):>9.4f} "
              f"{int(torch.tensor(ksel).median()):>10} "
              f"{float(torch.tensor(conds).median()):>13.1f} "
              f"{float(torch.tensor(ents).mean()):>9.3f} {len(vals):>5}")
        vol_rows.append(dict(layer=l, logvol_mean=float(v.mean()),
                             logvol_std=float(v.std()),
                             k_sel_median=int(torch.tensor(ksel).median()),
                             cond_eff_median=float(torch.tensor(conds).median()),
                             entropy_mean=float(torch.tensor(ents).mean()),
                             n=len(vals)))
    res["volume_profile"] = vol_rows
    lo = min(vol_rows, key=lambda r: r["logvol_mean"])
    hi = max(vol_rows, key=lambda r: r["logvol_mean"])
    print(f"\n  highest volume: layer {hi['layer']} ({hi['logvol_mean']:.3f})   "
          f"lowest: layer {lo['layer']} ({lo['logvol_mean']:.3f})")

    # ---- E2: all four instruments on identical points ---------------------
    rule("E2  ADJUDICATION -- four instruments, identical activations")
    G_man = manson_metric(lm.U)

    # Build the sample: cheap instruments everywhere, intrinsic on a subsample.
    gen = torch.Generator().manual_seed(0)
    cheap, sample_keys = [], []
    for si, (H, toks) in enumerate(streams):
        cloud = {l: H[l] for l in LAYERS}                  # (T, d) per layer
        for l in LAYERS:
            pts = cloud[l]
            for t in range(H.shape[1]):
                cheap.append(dict(
                    sent=si, layer=l, pos=t,
                    pca=pca_curvature(pts, t, n_neighbors=min(20, H.shape[1] - 1)),
                    king=contextual_angle(H, l, t),
                ))
                sample_keys.append((si, l, t))
    # Manson's is a layer-wise trajectory quantity: one value per (sentence, pos)
    # at each interior layer.  Attach it by (sentence, pos, layer).
    man_map = {}
    for si, (H, _) in enumerate(streams):
        for t in range(H.shape[1]):
            kap = frenet_curvature(layer_trajectory(H, token=t), G_man)
            for j, val in enumerate(kap.tolist()):
                man_map[(si, j + 1, t)] = val
    for c in cheap:
        c["manson"] = man_map.get((c["sent"], c["layer"], c["pos"]), float("nan"))

    # STRATIFY by layer.  A flat random sample gave 5-6 points per layer, too few
    # for the within-layer analysis below, which is what removes the layer
    # confound.  Equal per-layer quotas fix that at no extra cost.
    by_layer = {}
    for i, c in enumerate(cheap):
        by_layer.setdefault(c["layer"], []).append(i)
    per_layer = max(1, N_INTRINSIC // len(LAYERS))
    perm = []
    for l in LAYERS:
        idxs = by_layer[l]
        pick = torch.randperm(len(idxs), generator=gen)[:per_layer].tolist()
        perm.extend(idxs[j] for j in pick)
    print(f"computing intrinsic curvature on {len(perm)} of {len(cheap)} points "
          f"({per_layer}/layer, stratified; ~3 s each)...")
    # Checkpoint as we go.  A previous run was killed at 240/360 and lost
    # 18 minutes of work; partial results are still usable, so write them out.
    ckpt = OUT / "e2_partial.json"
    done = {}
    if ckpt.exists():
        done = {int(k): v for k, v in json.loads(ckpt.read_text(encoding="utf-8")).items()}
        print(f"   resuming: {len(done)} points already computed")
    t0 = time.time()
    for n, i in enumerate(perm, 1):
        c = cheap[i]
        if i in done:
            c.update(done[i]); continue
        h = streams[c["sent"]][0][c["layer"], c["pos"]]
        cc = curvature_at(lm, h, K_DIM, top_k=TOP_K, n_planes=16)
        c["intrinsic_K"] = cc["K_median"]
        c["intrinsic_R"] = cc["scalar_R"]
        done[i] = dict(intrinsic_K=cc["K_median"], intrinsic_R=cc["scalar_R"])
        if n % 20 == 0:
            ckpt.write_text(json.dumps(done), encoding="utf-8")
            print(f"   {n}/{len(perm)}  ({time.time()-t0:.0f}s)")
    ckpt.write_text(json.dumps(done), encoding="utf-8")
    sub = [c for c in cheap if "intrinsic_K" in c]
    res["e2_sample"] = sub

    print(f"\n{'instrument':<34} {'median':>12} {'range':>26}")
    cols = {
        "1. intrinsic Fisher-Rao K (this work)": [c["intrinsic_K"] for c in sub],
        "2. local-PCA residual (Mabrok)":        [c["pca"] for c in sub],
        "3. Frenet under U^T U (Manson)":        [c["manson"] for c in sub],
        "4. Euclidean angle (King et al.)":      [c["king"] for c in sub],
    }
    for name, v in cols.items():
        vv = torch.tensor([x for x in v if x == x], dtype=torch.float64)
        print(f"{name:<34} {float(vv.median()):>12.4e} "
              f"[{float(vv.min()):>10.3e}, {float(vv.max()):>10.3e}]")

    print(f"\nPairwise Spearman rank correlations (n = {len(sub)}):")
    names = list(cols)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = spearman(cols[names[i]], cols[names[j]])
            flag = "  <-- proxies vs intrinsic" if i == 0 else ""
            print(f"   {names[i][:3]} vs {names[j][:3]}:  rho = {r:+.3f}{flag}")
    res["e2_correlations"] = {
        f"{names[i]}|{names[j]}": spearman(cols[names[i]], cols[names[j]])
        for i in range(len(names)) for j in range(i + 1, len(names))
    }

    # ---- POSITIVE CONTROL -------------------------------------------------
    # A null correlation is only informative if the intrinsic quantity HAS
    # usable variance.  If K were ~constant at +1/4 plus noise, rho ~ 0 against
    # anything would be trivially expected and E2 would prove nothing.  Test it
    # against scalar R -- the same geometry via an independent contraction, so a
    # high correlation is required for the null to mean anything.
    rule("POSITIVE CONTROL  does intrinsic K carry real signal?")
    K = [c["intrinsic_K"] for c in sub]
    t = torch.tensor(K, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([.25, .5, .75], dtype=torch.float64))
    r_ctrl = spearman(K, [c["intrinsic_R"] for c in sub])
    print(f"  K median {float(q[1]):.4f}  IQR {float(q[2]-q[0]):.4f}  "
          f"CV {float(t.std()/t.mean()):.3f}  range [{float(t.min()):.3f},"
          f"{float(t.max()):.3f}]")
    print(f"  K vs scalar R (same geometry, independent contraction): "
          f"rho = {r_ctrl:+.3f}")
    print("  " + ("PASS -- K has real variance, so the nulls above are informative"
                   if r_ctrl > 0.5 else
                   "FAIL -- K may be constant+noise; the nulls would be vacuous"))
    res["positive_control"] = dict(K_iqr=float(q[2]-q[0]),
                                   K_cv=float(t.std()/t.mean()), rho_K_vs_R=r_ctrl)

    # ---- LAYER CONFOUND ---------------------------------------------------
    # Manson's and King's quantities are both strongly layer-dependent, so a
    # pooled correlation between them may reflect shared layer variation rather
    # than pointwise agreement.  Same risk for every pair.  Recompute within
    # layers, where layer is held constant by construction.
    rule("LAYER CONFOUND  pooled vs within-layer correlations")
    for nm, key in (("intrinsic K", "intrinsic_K"), ("PCA proxy", "pca"),
                    ("Manson", "manson"), ("King", "king")):
        print(f"  {nm:<12} vs layer: rho = "
              f"{spearman([c[key] for c in sub], [c['layer'] for c in sub]):+.3f}")
    byl = {}
    for c in sub:
        byl.setdefault(c["layer"], []).append(c)
    pairs = [("intrinsic_K", "pca"), ("intrinsic_K", "manson"),
             ("intrinsic_K", "king"), ("manson", "king")]
    acc = {p: [] for p in pairs}
    for l, g in sorted(byl.items()):
        if len(g) < 8:
            continue
        for p in pairs:
            r = spearman([c[p[0]] for c in g], [c[p[1]] for c in g])
            if r == r:
                acc[p].append(r)
    print(f"\n  {'pair':<28} {'pooled':>9} {'within-layer mean':>19} {'layers':>7}")
    within = {}
    for p in pairs:
        pooled = spearman([c[p[0]] for c in sub], [c[p[1]] for c in sub])
        v = torch.tensor(acc[p], dtype=torch.float64) if acc[p] else torch.tensor([float("nan")])
        within[f"{p[0]}|{p[1]}"] = float(v.mean())
        print(f"  {p[0][:13]+' vs '+p[1][:9]:<28} {pooled:>+9.3f} "
              f"{float(v.mean()):>+19.3f} {len(acc[p]):>7}")
    res["within_layer_correlations"] = within
    print("\n  A conclusion only counts if it survives the within-layer column.")

    pca_med = float(torch.tensor([c["pca"] for c in sub]).median())
    k_med = float(torch.tensor(K).median())
    print(f"\n  Magnitudes: intrinsic K = {k_med:.3e}, local-PCA proxy = "
          f"{pca_med:.3e} (ratio {k_med/pca_med:.1f}x).")
    print(f"  NOTE this does NOT replicate Mabrok's published ~1e-5; see")
    print(f"  07-stage4-log.md S1.1(b).  The correlations, not the magnitudes,")
    print(f"  are the defensible result.")

    (OUT / "stage4.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'stage4.json'}")


if __name__ == "__main__":
    main()
