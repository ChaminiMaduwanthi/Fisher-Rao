import torch, time, traceback
from fisherrao import LM, corpus
from fisherrao.metrics import fisher_metric, fisher_metric_scrambled, topk_indices
from fisherrao.curvature import riemann, scalar_curvature
lm=LM(); H,_=lm.residual_stream(corpus.sentences()[0]); h=H[20,-1]
K=5; TOP=512
idx=topk_indices(lm,h,TOP)
perm=torch.randperm(lm.U.shape[0],generator=torch.Generator().manual_seed(0))
print("setup done", flush=True)
for tag, mf in [("real", lambda x: fisher_metric(lm,x,top_k=TOP,idx=idx)),
                ("scram", lambda x: fisher_metric_scrambled(lm,x,perm,TOP,idx=idx))]:
    t=time.time(); G0=mf(h); print(f"{tag}: metric {time.time()-t:.2f}s", flush=True)
    hhat=h/torch.linalg.vector_norm(h)
    P=torch.eye(h.shape[-1],dtype=h.dtype)-torch.outer(hhat,hhat)
    ev,evec=torch.linalg.eigh(P@G0@P)
    F=evec[:,torch.argsort(ev,descending=True)[:K]].contiguous()
    g=lambda x: F.T @ mf(h+F@x) @ F
    x0=torch.zeros(K,dtype=h.dtype)
    t=time.time()
    try:
        R=riemann(g,x0)
        print(f"{tag}: riemann {time.time()-t:.1f}s  scalarR={scalar_curvature(g,x0,R=R):.4f}", flush=True)
    except Exception as e:
        print(f"{tag}: FAILED {type(e).__name__}: {e}", flush=True); traceback.print_exc()
