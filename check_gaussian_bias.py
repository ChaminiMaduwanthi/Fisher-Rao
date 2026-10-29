"""
Does the final LayerNorm's bias drive GPT-2's Gaussian-U result?

run_gaussian_matched.py replaces the unembedding only.  On a LayerNorm model it
keeps the final norm's learned bias b, so every state's Gaussian logits carry
the same random offset s U_g b.  RMSNorm has no bias, and GPT-2 is the one model
whose Gaussian K falls below the trained K (14-revision-log.md S13).  This
recomputes the control on the lowest and highest entropy bands with b removed.

The first four states are also recomputed with b kept; they must reproduce the
saved K_gauss exactly, which shows the harness is the control's own.

Usage:  python check_gaussian_bias.py gpt2
        python check_gaussian_bias.py EleutherAI/pythia-160m
Output: results/scramble/gaussian_nobias_<model>_draw.json
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
from fisherrao.metrics import fisher_metric_projected, topk_indices
from fisherrao.curvature import null_projector
from fisherrao.stats import median
from run_scramble_consistent import K_DIM, TOP_K, depths_for, frame_for, median_K, sign_test
from run_gaussian_matched import matched_scale

OUT = pathlib.Path("results/scramble")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    tag = f"{model.split('/')[-1]}_draw"
    saved = json.loads((OUT / f"gaussian_{tag}.json").read_text(encoding="utf-8"))["rows"]
    pick = [r for r in saved if r["entropy"] < 0.5 or r["entropy"] >= 4.0]
    lm = LM(model)
    print(lm.summary())
    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in depths_for(lm.n_layers):
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))
    U_real, bias_real, b = lm.U, lm.bias, lm.norm.bias
    U_g = torch.randn(U_real.shape, generator=torch.Generator().manual_seed(0),
                      dtype=torch.float64) / U_real.shape[1] ** 0.5

    def k_gauss(h0, target):
        s = matched_scale(U_g @ lm.apply_norm(h0), target)
        lm.U, lm.bias = s * U_g, None
        try:
            idx = topk_indices(lm, h0, TOP_K)
            F = frame_for(lm, h0, idx, null_projector(lm, h0), K_DIM)
            return median_K(lambda x: fisher_metric_projected(lm, h0 + F @ x, F, idx), K_DIM)
        finally:
            lm.U, lm.bias = U_real, bias_real

    rows, t0 = [], time.time()
    for j, r in enumerate(pick):
        layer, h0 = states[r["_i"]]
        if layer != r["layer"]:
            raise SystemExit(f"state {r['_i']}: layer {layer} does not match the saved {r['layer']}")
        rec = dict(_i=r["_i"], layer=layer, entropy=r["entropy"], K_real=r["K_real"], K_gauss=r["K_gauss"],
                   bias_share=float(b.detach().double().norm() / lm.apply_norm(h0).norm()))
        if j < 4:
            rec["K_gauss_rerun"] = k_gauss(h0, r["entropy"])
        lm.norm.bias = None
        try:
            rec["K_gauss_nobias"] = k_gauss(h0, r["entropy"])
        finally:
            lm.norm.bias = b
        rows.append(rec)
        print(f"   {j+1:>2}/{len(pick)}  L{layer:<2} H={r['entropy']:5.2f}  real {r['K_real']:+.4f}  "
              f"gauss {r['K_gauss']:+.4f}  without bias {rec['K_gauss_nobias']:+.4f}  "
              f"|b|/|n(h)| {rec['bias_share']:.2f}   ({time.time()-t0:.0f}s)", flush=True)

    summary = dict(model=model, n=len(rows),
                   rerun_max_abs_diff=max(abs(r["K_gauss_rerun"] - r["K_gauss"]) for r in rows if "K_gauss_rerun" in r),
                   bands={})
    for name, sub in (("[0, 0.5)", [r for r in rows if r["entropy"] < 0.5]),
                      ("[4, inf)", [r for r in rows if r["entropy"] >= 4.0])):
        summary["bands"][name] = dict(
            n=len(sub), K_real=median([r["K_real"] for r in sub]), K_gauss=median([r["K_gauss"] for r in sub]),
            K_gauss_nobias=median([r["K_gauss_nobias"] for r in sub]),
            real_gt_gauss=sign_test([r["K_real"] - r["K_gauss"] for r in sub]),
            real_gt_nobias=sign_test([r["K_real"] - r["K_gauss_nobias"] for r in sub]),
            median_abs_change=median([abs(r["K_gauss_nobias"] - r["K_gauss"]) for r in sub]),
            bias_share=median([r["bias_share"] for r in sub]))
    print(f"\n{model}: rerun with the bias kept, largest |diff| {summary['rerun_max_abs_diff']:.1e}")
    for name, v in summary["bands"].items():
        print(f"   {name:<9} n {v['n']}  real {v['K_real']:+.4f}  gauss {v['K_gauss']:+.4f}  "
              f"without bias {v['K_gauss_nobias']:+.4f}  real > without bias {v['real_gt_nobias']['pos']}/"
              f"{v['real_gt_nobias']['n']}  |b|/|n(h)| {v['bias_share']:.2f}")
    out = OUT / f"gaussian_nobias_{tag}.json"
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
