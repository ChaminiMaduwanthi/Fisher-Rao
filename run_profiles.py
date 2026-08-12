"""
E1 -- layer-wise curvature and volume profiles, across ARCHITECTURES.

Stage 4's exit criterion asks for layer-wise profiles on >= 3 models.  Until the
TLS block was removed there was exactly one (10-architecture-log.md), so every
number in this project so far describes SmolLM2-135M and nothing else.  This is
the script that changes that.

The three models now verified end to end (check_architectures.py):

    SmolLM2-135M-Instruct   RMSNorm    d=576  L=30   1 null direction
    gpt2                    LayerNorm  d=768  L=12   2 null directions
    EleutherAI/pythia-160m  LayerNorm  d=768  L=12   2 null directions

so this also tests whether K ~ +1/4 is a fact about transformers or a fact about
one RMSNorm Llama.  Two things could break it:

  * LayerNorm has a SECOND exact null direction (the all-ones, from the mean
    subtraction).  null_projector handles it; if the handling were wrong the
    curvature would be visibly poisoned rather than subtly so.
  * GPT-2 and Pythia are TIED vs UNTIED embeddings respectively, which changes
    the relationship between the unembedding rows and the residual stream.

WHAT IS REPORTED, and why these two quantities

    K       median sectional curvature over N_PLANES random 2-planes, plus the
            fraction of planes with K > 0.  The reference is the ambient simplex
            value +1/4.
    log-vol per dimension, on the SPHERE of directions.  Reported alongside
            because 08-rq3a-log.md S5.5 showed the two come apart: curvature is
            k-fragile but carries learned structure, volume is stable but
            largely reads concentration.  A cross-architecture profile is worth
            having for both, and worth trusting differently.

Layers are reported at RELATIVE depth so a 12-layer and a 30-layer model can be
put on one axis.

Usage:  python run_profiles.py [n_per_layer] [budget_seconds] [model ...]
        Checkpointed per (model, layer, point); re-run until it completes.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus
from fisherrao.metrics import fisher_metric, topk_indices
from fisherrao.curvature import (curvature_at, null_projector, volume_element,
                                 spectral_diagnostics)

OUT = pathlib.Path("results/profiles")
K_DIM, TOP_K, N_PLANES = 5, 512, 16
MODELS = ["HuggingFaceTB/SmolLM2-135M-Instruct", "gpt2", "EleutherAI/pythia-160m"]
N_DEPTHS = 9                       # relative depths sampled per model


def depths_for(n_layers: int) -> list[int]:
    """N_DEPTHS layers spread over the network, always including 1 and the last."""
    if n_layers <= N_DEPTHS:
        return list(range(1, n_layers + 1))
    step = (n_layers - 1) / (N_DEPTHS - 1)
    return sorted({max(1, round(1 + i * step)) for i in range(N_DEPTHS)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 440.0
    models = sys.argv[3:] or MODELS

    ckpt = OUT / "profile_points.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["model"], r["layer"], r["_i"])] = r
    print(f"checkpoint: {len(done)} points on file   budget {budget:.0f}s")

    t0, n_new, rows = time.time(), 0, list(done.values())
    for model_id in models:
        lm = LM(model_id)
        layers = depths_for(lm.n_layers)
        print(f"\n{model_id}: d={lm.d} L={lm.n_layers}  layers={layers}")

        states: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        for s in corpus.sentences():
            H, _ = lm.residual_stream(s)
            for l in layers:
                for t in range(H.shape[1]):
                    states[l].append(H[l, t])

        gen = torch.Generator().manual_seed(0)
        for l in layers:
            pool = states[l]
            pick = torch.randperm(len(pool), generator=gen)[:n_per].tolist()
            for i in pick:
                key = (model_id, l, i)
                if key in done:
                    continue
                if time.time() - t0 > budget:
                    print(f"   budget reached (+{n_new} new); rerun to continue",
                          flush=True)
                    report(rows, models)
                    return
                h = pool[i]
                rec = dict(model=model_id, layer=l, _i=i,
                           rel_depth=round(l / lm.n_layers, 4))
                try:
                    c = curvature_at(lm, h, K_DIM, top_k=TOP_K, n_planes=N_PLANES)
                    rec.update(K=c["K_median"], R=c["scalar_R"],
                               frac_pos=c["frac_positive"])
                    # A9 -- curvature SIGN analysis (Stage 4 task 4.4).  The
                    # median hides the sign distribution entirely: a point with
                    # 60% positive planes and one with 100% can share a median.
                    Ks = c.get("K")
                    if Ks is not None:
                        t = torch.tensor([x for x in Ks if x == x], dtype=torch.float64)
                        if len(t):
                            rec["K_min"], rec["K_max"] = float(t.min()), float(t.max())
                            rec["K_iqr"] = float(t.quantile(0.75) - t.quantile(0.25))
                            rec["n_planes_neg"] = int((t < 0).sum())
                            rec["n_planes"] = int(len(t))
                    rec["log_vol"] = volume_element(lm, h, K_DIM, top_k=TOP_K,
                                                    per_dim=True)
                    sd = spectral_diagnostics(lm, h, K_DIM, top_k=TOP_K)
                    rec["cond_eff"] = sd.get("cond_eff")
                    rec["k_sel"] = sd.get("k")
                    # A8 -- intrinsic dimension, for the hourglass overlay
                    # (Stage 4 task 4.2).  k_eff_99 is the number of directions
                    # holding 99% of the trace: the metric's own notion of how
                    # many dimensions matter here, and the right thing to put
                    # beside the curvature profile.
                    rec["k_eff_99"] = sd.get("k_eff_99", sd.get("k_eff"))
                    lp = torch.log_softmax(lm.logits(h), -1)
                    rec["entropy"] = float(-(lp.exp() * lp).sum())
                except Exception as e:                            # noqa: BLE001
                    rec["error"] = repr(e)
                rows.append(rec)
                done[key] = rec
                n_new += 1
                with ckpt.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            print(f"   layer {l:>3}  ({time.time()-t0:.0f}s, +{n_new})", flush=True)
    report(rows, models)


def _sp(a, b):
    """Spearman, tiny-n safe."""
    import torch as _t
    x = _t.argsort(_t.argsort(_t.tensor(a, dtype=_t.float64))).double()
    y = _t.argsort(_t.argsort(_t.tensor(b, dtype=_t.float64))).double()
    if len(a) < 3:
        return float("nan")
    return float(_t.corrcoef(_t.stack([x, y]))[0, 1])


def med(xs):
    t = torch.tensor([x for x in xs if x is not None and x == x],
                     dtype=torch.float64)
    return float(t.median()) if len(t) else float("nan")


def report(rows, models):
    ok = [r for r in rows if "error" not in r and "K" in r]
    print(f"\n{'='*80}\nE1 -- LAYER-WISE PROFILES, {len({r['model'] for r in ok})} "
          f"ARCHITECTURES, n={len(ok)} points\n{'='*80}")
    for m in models:
        sub = [r for r in ok if r["model"] == m]
        if not sub:
            continue
        layers = sorted({r["layer"] for r in sub})
        print(f"\n{m}")
        print(f"   {'layer':>6} {'depth':>7} {'n':>4} {'median K':>10} "
              f"{'K>0':>6} {'scalar R':>10} {'logvol/dim':>11} {'cond_eff':>9}")
        for l in layers:
            s = [r for r in sub if r["layer"] == l]
            print(f"   {l:>6} {s[0]['rel_depth']:>7.2f} {len(s):>4} "
                  f"{med([r['K'] for r in s]):>10.4f} "
                  f"{med([r['frac_pos'] for r in s]):>6.0%} "
                  f"{med([r['R'] for r in s]):>10.3f} "
                  f"{med([r['log_vol'] for r in s]):>11.4f} "
                  f"{med([r['cond_eff'] for r in s]):>9.1f}")
        K = [r["K"] for r in sub]
        print(f"   ALL LAYERS: median K = {med(K):.4f}   "
              f"IQR [{float(torch.quantile(torch.tensor(K, dtype=torch.float64), 0.25)):.4f}, "
              f"{float(torch.quantile(torch.tensor(K, dtype=torch.float64), 0.75)):.4f}]"
              f"   vs the simplex reference +0.2500")

    # ---- A8: the hourglass overlay ---------------------------------------
    print(f"\n{'='*80}\nA8 -- INTRINSIC DIMENSION vs CURVATURE, by relative depth"
          f"\n{'='*80}")
    print("   The hourglass hypothesis (Stage 4 task 4.2) says intrinsic")
    print("   dimension dips mid-network.  k_eff_99 is the metric's own count of")
    print("   directions holding 99% of the trace, so it is measured on exactly")
    print("   the same object as the curvature beside it.")
    for m in models:
        sub = [r for r in ok if r["model"] == m and r.get("k_eff_99") is not None]
        if not sub:
            continue
        layers = sorted({r["layer"] for r in sub})
        ke = [med([r["k_eff_99"] for r in sub if r["layer"] == l]) for l in layers]
        kk = [med([r["K"] for r in sub if r["layer"] == l]) for l in layers]
        lo = layers[ke.index(min(ke))]
        print(f"\n   {m}")
        print(f"      {'layer':>6} " + "".join(f"{l:>7}" for l in layers))
        print(f"      {'k_eff':>6} " + "".join(f"{v:>7.1f}" for v in ke))
        print(f"      {'K':>6} " + "".join(f"{v:>7.3f}" for v in kk))
        print(f"      minimum k_eff at layer {lo} (relative depth "
              f"{lo/max(r['layer'] for r in sub):.2f});  "
              f"range {min(ke):.1f}-{max(ke):.1f}")
        print(f"      rho(k_eff, K) across layers = {_sp(ke, kk):+.2f}")

    # ---- A9: the sign distribution ---------------------------------------
    print(f"\n{'='*80}\nA9 -- CURVATURE SIGN ANALYSIS (Stage 4 task 4.4)\n{'='*80}")
    print("   frac_pos is the fraction of sampled 2-planes with K > 0 AT A POINT.")
    print("   A median K near +1/4 is compatible with mixed signs; this is the")
    print("   check that it is not.")
    print(f"   {'model':<38} {'points 100% pos':>16} {'any plane neg':>14} "
          f"{'min plane K':>12}")
    for m in models:
        sub = [r for r in ok if r["model"] == m and r.get("n_planes")]
        if not sub:
            continue
        allpos = sum(1 for r in sub if r["n_planes_neg"] == 0)
        anyneg = sum(1 for r in sub if r["n_planes_neg"] > 0)
        mn = min(r["K_min"] for r in sub)
        print(f"   {m:<38} {allpos:>7}/{len(sub):<8} {anyneg:>14} {mn:>12.4f}")

    print(f"\n{'='*80}\nDOES K ~ +1/4 HOLD ACROSS ARCHITECTURES?\n{'='*80}")
    print(f"   {'model':<38} {'norm':<10} {'n':>4} {'median K':>10} {'|K-0.25|':>10}")
    verdict = True
    for m in models:
        sub = [r for r in ok if r["model"] == m]
        if not sub:
            continue
        k = med([r["K"] for r in sub])
        norm = "RMSNorm" if "SmolLM" in m else "LayerNorm"
        print(f"   {m:<38} {norm:<10} {len(sub):>4} {k:>10.4f} {abs(k-0.25):>10.4f}")
        verdict &= abs(k - 0.25) < 0.05
    print(f"\n   -> {'YES' if verdict else 'NO'}: "
          + ("all architectures sit within 0.05 of the ambient simplex value, "
             "including both LayerNorm models with their SECOND null direction."
             if verdict else
             "at least one architecture departs from +1/4 by more than 0.05 -- "
             "which would be a finding, not a bug, but check null_projector first."))

    (OUT / "profiles.json").write_text(json.dumps(dict(
        k=K_DIM, top_k=TOP_K, n_planes=N_PLANES, rows=rows), indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'profiles.json'}")


if __name__ == "__main__":
    main()
