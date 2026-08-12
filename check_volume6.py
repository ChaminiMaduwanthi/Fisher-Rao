import warnings; warnings.filterwarnings("ignore")
import json, pathlib, torch
from fisherrao import LM, corpus
from fisherrao.metrics import fisher_metric, topk_indices
from fisherrao.curvature import null_projector

K = 5
OUT = pathlib.Path("results/volume6"); OUT.mkdir(parents=True, exist_ok=True)
CK = OUT / "vol6.json"
MODELS = ["gpt2", "EleutherAI/gpt-neo-125m", "HuggingFaceTB/SmolLM2-135M-Instruct",
          "JackFram/llama-160m", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m"]
TIED = {"gpt2":1, "gpt-neo-125m":1, "SmolLM2-135M-Instruct":1,
        "llama-160m":0, "pythia-70m":0, "pythia-160m":0}
done = json.loads(CK.read_text()) if CK.exists() else {}

for mid in MODELS:
    nm = mid.split("/")[-1]
    if nm in done: continue
    lm = LM(mid); gen = torch.Generator().manual_seed(0)
    pts = []
    for s in corpus.GENERAL:
        H, _ = lm.residual_stream(s)
        for l in range(1, lm.n_layers + 1):
            for t in range(H.shape[1]): pts.append(H[l, t])
    sel = torch.randperm(len(pts), generator=gen)[:150].tolist()
    v, e = [], []
    for i in sel:
        h = pts[i]
        try:
            lp = torch.log_softmax(lm.logits(h), -1)
            idx = topk_indices(lm, h, 512); P = null_projector(lm, h)
            ev = torch.linalg.eigvalsh(P @ fisher_metric(lm, h, idx=idx) @ P).clamp_min(0).flip(0)[:K]
            ev = ev[ev > 0]
            v.append(0.5*float(torch.log(ev).mean()) + float(torch.log(torch.sqrt(h.pow(2).mean()+lm.eps))))
            e.append(float(-(lp.exp()*lp).sum()))
        except Exception: pass
    m = lambda x: float(torch.tensor(x, dtype=torch.float64).median())
    un = lm.U.norm(dim=1)
    done[nm] = dict(tied=TIED[nm], V=lm.N, d=lm.d, logvol=m(v), H=m(e),
                    logU=float(torch.log(un.median())),
                    Uspread=float(torch.log(un).std()),
                    logg=float(torch.log(lm.g.abs()).mean()), n=len(v))
    CK.write_text(json.dumps(done, indent=2))
    print(f"  {nm} done", flush=True)

if len(done) == len(MODELS):
    def sp(a, b):
        x = torch.argsort(torch.argsort(torch.tensor(a, dtype=torch.float64))).double()
        y = torch.argsort(torch.argsort(torch.tensor(b, dtype=torch.float64))).double()
        return float(torch.corrcoef(torch.stack([x, y]))[0, 1])
    order = [m.split("/")[-1] for m in MODELS]
    print(f"\n{'model':<22}{'tied':>5}{'V':>7}{'logvol':>9}{'H':>7}{'log|U|':>8}{'Uspread':>9}{'log|g|':>8}")
    for nm in order:
        r = done[nm]
        print(f"{nm:<22}{r['tied']:>5}{r['V']:>7}{r['logvol']:>9.3f}{r['H']:>7.2f}"
              f"{r['logU']:>8.3f}{r['Uspread']:>9.4f}{r['logg']:>8.3f}")
    lv = [done[nm]["logvol"] for nm in order]
    print("\nSpearman of log-vol against each candidate, n=6 models:")
    for key, name in (("tied","tied (1/0)"), ("V","vocab size"), ("d","width d"),
                      ("H","entropy"), ("logU","log|U| median"),
                      ("Uspread","log|U| spread"), ("logg","log|g| mean")):
        print(f"   {name:<16} rho = {sp([done[nm][key] for nm in order], lv):+.3f}")
