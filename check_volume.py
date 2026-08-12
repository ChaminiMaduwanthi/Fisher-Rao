"""Is the volume element actually k-stable?  (Stage 3 assumed it; verify it.)

The claim in 06-stage3-log.md S4.4 is that log-volume is well behaved where the
Riemann-derived quantities are k-fragile.  That was an eyeball judgement on one
column of numbers.  Test it the same way dR/R_ref was tested: vary k, check
whether the LAYER RANKING is preserved (Spearman rho), which is what any
layer-wise claim actually depends on.

Compare per-dimension log-volume (log geometric mean of retained eigenvalues),
which is the k-comparable form.
"""
import sys, torch
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from fisherrao import LM
from fisherrao.curvature import volume_element, select_k

lm = LM()
H, _ = lm.residual_stream("The bank was steep and muddy after the heavy rain, "
                          "so the hikers climbed down to the river")
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]
KS = [3, 4, 5, 6, 7, 10]

def ranks(d):
    s = sorted(LAYERS, key=lambda l: d[l]); return {l: i for i, l in enumerate(s)}
def rho(a, b):
    ra, rb = ranks(a), ranks(b)
    x = torch.tensor([float(ra[l]) for l in LAYERS]); y = torch.tensor([float(rb[l]) for l in LAYERS])
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])

for per_dim in (False, True):
    lbl = "log vol / k  (per-dim)" if per_dim else "log vol      (raw sum)"
    print(f"\n=== {lbl} ===")
    print(f"{'k':>3} " + "".join(f"{'L'+str(l):>9}" for l in LAYERS))
    tab = {}
    for k in KS:
        tab[k] = {l: volume_element(lm, H[l, -1], k, per_dim=per_dim) for l in LAYERS}
        print(f"{k:>3} " + "".join(f"{tab[k][l]:>9.3f}" for l in LAYERS))
    print("  Spearman rho of layer ordering:", end="")
    for a, b in zip(KS, KS[1:]):
        print(f"  k{a}/k{b}={rho(tab[a],tab[b]):+.2f}", end="")
    print(f"   |  k{KS[0]}/k{KS[-1]}={rho(tab[KS[0]],tab[KS[-1]]):+.2f}")

print("\n=== cond_eff-based k selection (cond_max = 1e2) ===")
print(f"{'layer':>6} {'k sel':>6} {'cond_eff':>10} {'trace frac':>11} {'k_eff(99%)':>11}")
for l in LAYERS:
    s = select_k(lm, H[l, -1], cond_max=1e2)
    print(f"{l:>6} {s['k']:>6} {s['cond_eff']:>10.1f} {s['trace_frac']:>11.4f} {s['k_eff_99']:>11}")
