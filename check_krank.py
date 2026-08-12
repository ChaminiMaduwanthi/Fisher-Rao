"""Is the dR/R_ref layer ranking stable across the retained rank k?

If the MAGNITUDE of the deviation is k-dependent (it is) but the RANKING of
layers is preserved, then relative statements across layers survive and only
absolute magnitudes must be quoted with their k.  If the ranking scrambles, the
deviation is not a usable signal at all and that must be said plainly.
"""
import sys, torch
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
from fisherrao import LM
from fisherrao.curvature import curvature_at

lm = LM()
H,_ = lm.residual_stream("The bank was steep and muddy after the heavy rain, so the hikers climbed down to the river")
LAYERS = [5, 10, 20, 25, 29]
KS = [4, 5, 6, 7]

tab = {}
print(f"{'k':>3} " + "".join(f"{'L'+str(l):>11}" for l in LAYERS) + "   ranking (most anomalous first)")
for k in KS:
    row = {}
    for l in LAYERS:
        c = curvature_at(lm, H[l,-1], k, top_k=512, n_planes=24)
        row[l] = c["dR_rel"]
    tab[k] = row
    order = sorted(LAYERS, key=lambda l: -row[l])
    print(f"{k:>3} " + "".join(f"{row[l]:>+10.1%} " for l in LAYERS)
          + "  " + " > ".join(f"L{l}" for l in order))

# Spearman rank correlation between consecutive k
def ranks(d):
    s = sorted(LAYERS, key=lambda l: d[l])
    return {l: i for i, l in enumerate(s)}
print("\nSpearman rank correlation of the layer ordering between k values:")
for a, b in zip(KS, KS[1:]):
    ra, rb = ranks(tab[a]), ranks(tab[b])
    x = torch.tensor([float(ra[l]) for l in LAYERS])
    y = torch.tensor([float(rb[l]) for l in LAYERS])
    r = float(torch.corrcoef(torch.stack([x, y]))[0,1])
    print(f"   k={a} vs k={b}:  rho = {r:+.3f}")
ra, rb = ranks(tab[KS[0]]), ranks(tab[KS[-1]])
x = torch.tensor([float(ra[l]) for l in LAYERS]); y = torch.tensor([float(rb[l]) for l in LAYERS])
print(f"   k={KS[0]} vs k={KS[-1]}:  rho = {float(torch.corrcoef(torch.stack([x,y]))[0,1]):+.3f}")
