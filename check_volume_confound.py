"""Is the layer-wise log-volume profile just residual-norm growth?

G(h) = A^T (...) A with A proportional to 1/r, r = sqrt(mean(h^2)).  So G scales
as 1/r^2, hence lambda_i ~ 1/r^2 and

    log_vol_per_dim = 0.5 * mean(log lambda)  carries a  -log(r)  term
                                              BY CONSTRUCTION  (not -2 log r:
    the 0.5 and the 2 from lambda ~ r^-2 cancel).  The
residual stream norm grows enormously with depth (salience -> 3.7e4 by layer 29),
so the reported "volume contracts through the network" could be nothing but norm
growth -- exactly the artifact that invalidated the first RQ3b claim.

Equivalently: the semantic manifold is the sphere of DIRECTIONS (Stage 0), and
the metric on the unit sphere is r^2 * G_ambient.  Correcting by +log(r) per
dimension is therefore not a cosmetic adjustment -- it is what puts the volume
on the correct manifold.

Test: regress log_vol/k on log(r).  A slope near -1 with high R^2 means the raw
profile is mostly the trivial scale term.
"""
import sys, torch
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
from fisherrao import LM
from fisherrao import corpus
from fisherrao.curvature import spectral_diagnostics

lm = LM(); K_DIM, TOP_K = 5, 512
LAYERS=[1,5,10,15,20,25,28,29,30]
sents = corpus.sentences()[:12]
streams=[lm.residual_stream(s)[0] for s in sents]

rows=[]
for l in LAYERS:
    lv, lr = [], []
    for H in streams:
        for t in range(H.shape[1]):
            h=H[l,t]
            sd=spectral_diagnostics(lm,h,K_DIM,top_k=TOP_K)
            lv.append(sd["log_vol_per_dim"])
            lr.append(float(torch.log(torch.sqrt(h.pow(2).mean()+lm.eps))))
    rows.append((l,torch.tensor(lv,dtype=torch.float64),torch.tensor(lr,dtype=torch.float64)))

print(f"{'layer':>6} {'logvol/k':>10} {'log r':>9} {'logvol+logr':>13}   <- scale-corrected")
allv=[];allr=[]
for l,v,r in rows:
    corrected = v + r
    print(f"{l:>6} {float(v.mean()):>10.4f} {float(r.mean()):>9.4f} {float(corrected.mean()):>13.4f}")
    allv.append(v); allr.append(r)
V=torch.cat(allv); R=torch.cat(allr)

# regression logvol/k = a*log r + b
A=torch.stack([R,torch.ones_like(R)],1)
coef=torch.linalg.lstsq(A,V.unsqueeze(1)).solution.squeeze()
pred=A@coef; ss_res=((V-pred)**2).sum(); ss_tot=((V-V.mean())**2).sum()
print(f"\nregression  logvol/k = a*log(r) + b   over all {len(V)} points")
print(f"   a = {float(coef[0]):+.4f}   (construction predicts -1 exactly)")
print(f"   R^2 = {float(1-ss_res/ss_tot):.4f}")
if abs(float(coef[0])+1)<0.25 and float(1-ss_res/ss_tot)>0.85:
    print("   -> PROFILE IS DOMINATED BY THE TRIVIAL 1/r^2 SCALE TERM.")
    print("      The uncorrected layer profile must not be reported as a finding.")
else:
    print("   -> the profile is NOT explained by norm growth alone.")

corr=[(l,float((v+r).mean())) for l,v,r in rows]
print("\nSCALE-CORRECTED profile (logvol/k + log r) = volume on the unit sphere -- the non-trivial part:")
for l,c in corr: print(f"   layer {l:>2}: {c:+.4f}")
lo=min(corr,key=lambda x:x[1]); hi=max(corr,key=lambda x:x[1])
print(f"   range {hi[1]-lo[1]:.3f}   max at layer {hi[0]}, min at layer {lo[0]}")
