"""Is the k-instability explained by k exceeding k_eff?

k_eff = directions holding 99% of the trace.  Beyond it, retained directions
have near-zero eigenvalues, G is badly conditioned there, and second derivatives
should be numerically worthless.  If the instability is confined to k > k_eff,
the rule "k <= k_eff" rescues the measurement.  If not, the deviation is simply
not identifiable and must be reported as such.
"""
import sys, torch
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
from fisherrao import LM, fisher_metric, spectrum
from fisherrao.curvature import curvature_at, effective_frame, metric_in_frame

lm = LM()
H,_ = lm.residual_stream("The bank was steep and muddy after the heavy rain, so the hikers climbed down to the river")

print("Per layer: k_eff, and dR/R_ref at k below / at / above k_eff.")
print("Also cond_eff of the RETAINED subspace at each k -- the diagnostic that")
print("should explain any blow-up.\n")
print(f"{'layer':>5} {'k_eff':>6} | " + "".join(f"{'k='+str(k):>22}" for k in (3,4,5,6,7)))
print(f"{'':>5} {'':>6} | " + "".join(f"{'dR/R':>11}{'cond':>11}" for _ in range(5)))
for l in (5, 10, 20, 25):
    ke = spectrum(fisher_metric(lm, H[l,-1], top_k=512))["k_eff"]
    cells = ""
    for k in (3,4,5,6,7):
        c = curvature_at(lm, H[l,-1], k, top_k=512, n_planes=16)
        G = metric_in_frame(lm, H[l,-1], effective_frame(lm, H[l,-1], k, top_k=512),
                            top_k=512)(torch.zeros(k, dtype=torch.float64))
        ev = torch.linalg.eigvalsh(G).clamp_min(0)
        cond = float(ev.max()/ev.min()) if float(ev.min())>0 else float('inf')
        mark = "*" if k > ke else " "
        cells += f"{c['dR_rel']:>+10.1%}{mark}{cond:>11.1e}"
    print(f"{l:>5} {ke:>6} | {cells}")
print("\n  * marks k > k_eff (retaining directions outside the effective subspace)")
