"""Two controls on the RQ3b '19-layer disagreement' result.

CONTROL A -- is the flat metrics' LATE separation just residual-norm growth?
    Stage 0 measured salience reaching 3.7e4 by layer 29.  If Euclidean/U^T U
    distances rise late only because the vectors get bigger, then dividing by the
    residual norm should move their half-depth earlier.  If it does, the honest
    claim is narrower: "the flat metrics fail to normalise", not "they localise
    the computation elsewhere".

CONTROL B -- is d_FR's EARLY rise an arccos compression artifact?
    d_FR = 2 arccos(BC) expands differences near BC=1 (similar distributions,
    early layers) and compresses near BC=0 (dissimilar, late layers).  That alone
    could manufacture "fast early rise, flat late".  Compare against measures
    without that shape: total variation (bounded, ~linear) and 1-BC.
"""
import sys, torch
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
from fisherrao import LM, manson_metric
from fisherrao.corpus import POLYSEMY
from run_polysemy import find_token

lm = LM(); G = manson_metric(lm.U); L = lm.n_layers
rows=[]
for word, sa, sb in POLYSEMY:
    Ha,ta = lm.residual_stream(sa); Hb,tb = lm.residual_stream(sb)
    ia,ib = find_token(lm,ta,word), find_token(lm,tb,word)
    if ia is None or ib is None: continue
    per=[]
    for l in range(L+1):
        ha,hb = Ha[l,ia], Hb[l,ib]
        pa,pb = lm.next_token_probs(ha), lm.next_token_probs(hb)
        bc = (pa.sqrt()*pb.sqrt()).sum().clamp(-1,1)
        diff = ha-hb
        de = float(torch.linalg.vector_norm(diff))
        du = float(torch.sqrt((diff@G@diff).clamp_min(0)))
        nrm = float((torch.linalg.vector_norm(ha)+torch.linalg.vector_norm(hb))/2)
        nrmG = float((torch.sqrt((ha@G@ha).clamp_min(0))+torch.sqrt((hb@G@hb).clamp_min(0)))/2)
        per.append(dict(
            d_fr=float(2*torch.arccos(bc)),
            tv=float(0.5*(pa-pb).abs().sum()),          # total variation
            one_bc=float(1-bc),                          # Hellinger^2
            d_euc=de, d_uu=du,
            d_euc_n=de/nrm if nrm>0 else float('nan'),   # norm-normalised
            d_uu_n=du/nrmG if nrmG>0 else float('nan'),
        ))
    rows.append(per)

keys=["d_fr","tv","one_bc","d_euc","d_euc_n","d_uu","d_uu_n"]
def halfdepth(key):
    out=[]
    for per in rows:
        v=torch.tensor([p[key] for p in per],dtype=torch.float64)
        fin=float(v[-1])
        if abs(fin)<1e-12: continue
        vn=v/fin
        idx=(vn>=0.5).nonzero()
        out.append(int(idx[0]) if len(idx) else L)
    t=torch.tensor(out,dtype=torch.float64)
    return float(t.mean()), float(t.median())

print("HALF-SEPARATION DEPTH (first layer reaching 50% of final), 30 layers\n")
print(f"{'measure':<12} {'mean':>7} {'median':>8}   note")
notes={"d_fr":"Fisher-Rao (the claim)","tv":"CONTROL B: total variation",
       "one_bc":"CONTROL B: 1-BC","d_euc":"Euclidean (raw)",
       "d_euc_n":"CONTROL A: Euclidean / ||h||","d_uu":"U^T U (raw)",
       "d_uu_n":"CONTROL A: U^T U / ||h||_G"}
for k in keys:
    m,md=halfdepth(k)
    print(f"{k:<12} {m:>7.2f} {md:>8.1f}   {notes[k]}")
