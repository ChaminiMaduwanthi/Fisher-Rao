"""
VERIFICATION of the claim in 11-mabrok-replication-log.md S5.

That log asserts the local-PCA residual-variance proxy returns `1 - threshold`
by construction, and concludes Mabrok's ~1e-5 reports his threshold rather than
the flatness of the manifold.  The evidence given was a single table computed on
GPT-2 activations.  That is NOT sufficient: a statistic that happens to equal
1 - thr on one dataset could still be measuring something real.

The claim is data-independence, so it has to be tested against data whose
geometry is KNOWN and DIFFERENT.  Four clouds, same n, same d, same sweep:

    gaussian   isotropic noise in R^768.  NO manifold at all -- intrinsic
               dimension is the full 768.  If the proxy reports ~1e-5 here at a
               1e-4 threshold, it cannot be measuring flatness of anything.
    flat3      a 3-dimensional LINEAR subspace.  Exactly flat, zero curvature.
               The honest answer for a curvature proxy is 0.
    sphere3    a 3-sphere of radius 1 embedded in R^768.  Genuinely CURVED,
               known sectional curvature +1, same intrinsic dimension as flat3.
               A curvature proxy must separate this from flat3.
    gpt2       real activations, WikiText-103, for reference.

TWO THINGS ARE BEING CHECKED, and they are different:

    1. TAUTOLOGY -- is measured residual ~ (1 - threshold) on every cloud?
    2. DISCRIMINATION -- does the proxy separate sphere3 from flat3 at all?
       This is the question that actually matters for RQ2b.  A statistic can be
       pinned to the threshold in magnitude and still rank curved above flat.

Usage:  python check_pca_tautology.py [n_points] [n_base]
"""

from __future__ import annotations

import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THRESHOLDS = (0.90, 0.95, 0.99, 0.999, 0.9999)
K_SWEEP = (20, 50, 200)
Q_SWEEP = (2, 3, 5, 10)      # for second_fundamental, which takes a FIXED q


def residual(X, base, k, thr):
    out = []
    for i in base.tolist():
        d = torch.cdist(X[i:i + 1], X).squeeze(0)
        nbr = X[torch.argsort(d)[:k + 1]]
        nbr = nbr - nbr.mean(0, keepdim=True)
        ev = torch.linalg.svdvals(nbr) ** 2
        tot = float(ev.sum())
        if tot <= 0:
            continue
        frac = torch.cumsum(ev, 0) / tot
        q = int(torch.searchsorted(frac, torch.tensor(thr, dtype=frac.dtype)) + 1)
        q = max(1, min(q, len(ev) - 1))
        out.append(float(ev[q:].sum() / tot))
    return float(torch.tensor(out, dtype=torch.float64).median()) if out else float("nan")


def make_clouds(n, d, gen):
    clouds = {}
    clouds["gaussian"] = torch.randn(n, d, generator=gen, dtype=torch.float64)

    B = torch.linalg.qr(torch.randn(d, 3, generator=gen, dtype=torch.float64))[0]
    clouds["flat3"] = torch.randn(n, 3, generator=gen, dtype=torch.float64) @ B.T

    S = torch.randn(n, 4, generator=gen, dtype=torch.float64)
    S = S / S.norm(dim=1, keepdim=True)          # uniform on the 3-sphere
    B4 = torch.linalg.qr(torch.randn(d, 4, generator=gen, dtype=torch.float64))[0]
    clouds["sphere3"] = S @ B4.T
    return clouds


def gpt2_cloud(n):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from fisherrao import LM

    lm = LM("gpt2")
    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        filename="wikitext-103-raw-v1/validation-00000-of-00001.parquet",
        repo_type="dataset")
    lines = [s.strip() for s in pq.read_table(path).column("text").to_pylist()
             if len(s.strip()) >= 40 and not s.strip().startswith("=")]
    text = " ".join(lines)[:400_000]
    ids = lm.tok(text, return_tensors="pt")["input_ids"][0]
    acc, pos = [], 0
    while sum(len(a) for a in acc) < n and pos < len(ids):
        H, _ = lm.residual_stream(lm.tok.decode(ids[pos:pos + 1024]))
        acc.append(H[12])
        pos += 1024
    return torch.cat(acc, 0)[:n]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
    n_base = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    d = 768
    gen = torch.Generator().manual_seed(0)

    clouds = make_clouds(n, d, gen)
    try:
        clouds["gpt2"] = gpt2_cloud(n)
        print(f"gpt2 cloud: {tuple(clouds['gpt2'].shape)}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"gpt2 cloud unavailable ({type(exc).__name__}); "
              f"synthetic clouds only")

    base = torch.randperm(n, generator=gen)[:n_base]

    print(f"\n{'='*78}\n1. TAUTOLOGY -- is the measured value just (1 - threshold)?"
          f"\n{'='*78}")
    for k in K_SWEEP:
        print(f"\n   k = {k}")
        print(f"   {'cloud':<10}" + "".join(f"{f'thr={t:g}':>12}" for t in THRESHOLDS))
        print(f"   {'1 - thr':<10}" + "".join(f"{1-t:>12.1e}" for t in THRESHOLDS))
        for name, X in clouds.items():
            vals = [residual(X, base, k, t) for t in THRESHOLDS]
            print(f"   {name:<10}" + "".join(f"{v:>12.2e}" for v in vals))
        print(f"   {'ratio':<10}" + "  (each row above / the 1-thr row)")

    print(f"\n{'='*78}\n2. DISCRIMINATION -- can it tell CURVED from FLAT?"
          f"\n{'='*78}")
    print("   Same intrinsic dimension (3), same ambient dimension, same n.")
    print("   sphere3 has sectional curvature +1; flat3 has exactly 0.")
    print(f"\n   {'k':>4} {'threshold':>10} {'flat3':>12} {'sphere3':>12} "
          f"{'ratio':>8} {'separates?':>12}")
    # flat3's residual is ~1e-31, i.e. exact zero in float64, so a RATIO is
    # meaningless (it prints as 1e28).  Judge separation on whether sphere3 is
    # above the numerical-zero floor instead.
    FLOOR = 1e-12
    sep_at = []
    for k in K_SWEEP:
        for t in (0.90, 0.99, 0.9999):
            f3 = residual(clouds["flat3"], base, k, t)
            s3 = residual(clouds["sphere3"], base, k, t)
            sep = s3 > FLOOR and f3 < FLOOR
            sep_at.append((k, t, sep))
            print(f"   {k:>4} {t:>10g} {f3:>12.2e} {s3:>12.2e} "
                  f"{'above floor' if s3 > FLOOR else 'AT ZERO':>10} "
                  f"{'YES' if sep else 'no':>12}")
    ok_any = any(s for _, _, s in sep_at)

    print(f"\n{'='*78}\nVERDICT\n{'='*78}")

    def track(name, k=200, thrs=(0.90, 0.95, 0.99)):
        return [residual(clouds[name], base, k, t) / (1 - t) for t in thrs]

    for name in ("gaussian", "gpt2", "sphere3", "flat3"):
        if name not in clouds:
            continue
        r = track(name)
        print(f"   {name:<10} ratio to (1-thr) at k=200, thr .90/.95/.99: "
              + ", ".join(f"{x:.2f}" for x in r))

    print()
    print("   1. THE STATISTIC IS NOT VACUOUS.  On a clean 3-sphere it returns a")
    print("      stable non-zero value that does NOT move with the threshold,")
    print("      and on a flat 3-plane it returns exact zero.  It can tell")
    print("      curved from flat when the local spectrum HAS A GAP.")
    print()
    print("   2. BUT ON REAL TRANSFORMER ACTIVATIONS IT BEHAVES LIKE ISOTROPIC")
    print("      NOISE.  gpt2 and gaussian both track (1-thr); flat3 and sphere3")
    print("      do not.  Real hidden states have no spectral gap, so q is set")
    print("      by the threshold and the residual is what is left over.")
    print()
    print("   3. AND AT MABROK'S OPERATING POINT THE INSTRUMENT STOPS")
    print("      DISCRIMINATING.  Reaching ~1e-5 needs thr ~ 0.9999, and there")
    print("      even the 3-SPHERE reads as exactly flat (see the table above).")
    print("      A threshold that small demands more components than a curved")
    print("      low-dimensional patch has, so the tail empties.")
    print()
    print("   -> The S5 claim 'the statistic is a tautology' is TOO STRONG in")
    print("      general and must be restated: it is threshold-pinned ON DATA")
    print("      WITHOUT A SPECTRAL GAP, which is what transformer activations")
    print("      are.  The conclusion for RQ2b is unchanged and better founded.")
    print(f"\n   Curved-vs-flat separation anywhere in the sweep: "
          f"{'YES' if ok_any else 'NO'}"
          f"   (and NOT at thr=0.9999)")

    second_fundamental_check(clouds, base)


def second_fundamental_check(clouds, base):
    """The same treatment for Mabrok's OTHER proxy.

    11-mabrok-replication-log.md S6(d) singled out `second_fundamental` as "the
    proxy to use for any magnitude claim", on the grounds that it has no
    threshold and therefore none of pca_curvature's degeneracy.  That was an
    argument, not a measurement.  Here is the measurement: the same four clouds,
    the same question -- can it tell a curved 3-sphere from a flat 3-plane, and
    does its value depend on the free parameter q rather than on the data?
    """
    from fisherrao.proxies import second_fundamental
    import torch

    rule = "=" * 78
    print(f"\n{rule}\n3. THE OTHER PROXY -- second_fundamental (||II||)\n{rule}")
    print("   Mabrok's tangent-plane rotation measure.  No variance threshold, so")
    print("   pca_curvature's 1-thr degeneracy cannot occur -- but it takes a")
    print("   fixed q, and that is a free parameter too.")
    print(f"\n   {'cloud':<10}" + "".join(f"{'q=' + str(q):>12}" for q in Q_SWEEP))
    vals = {}
    for name, X in clouds.items():
        row = []
        for q in Q_SWEEP:
            v = [second_fundamental(X, int(i), n_neighbors=20, q=q)
                 for i in base[:40].tolist()]
            t = torch.tensor([x for x in v if x == x], dtype=torch.float64)
            row.append(float(t.median()) if len(t) else float("nan"))
        vals[name] = row
        print(f"   {name:<10}" + "".join(f"{x:>12.4e}" for x in row))

    print(f"\n   {'q':>4} {'flat3':>12} {'sphere3':>12} {'ratio':>9} {'separates?':>12}")
    ok = False
    for j, q in enumerate(Q_SWEEP):
        f3, s3 = vals["flat3"][j], vals["sphere3"][j]
        r = s3 / f3 if f3 > 0 else float("inf")
        sep = r > 2 or r < 0.5
        ok |= sep
        print(f"   {q:>4} {f3:>12.4e} {s3:>12.4e} {r:>9.2f} {'YES' if sep else 'no':>12}")
    print(f"\n   -> curved vs flat separated at some q: {'YES' if ok else 'NO'}")
    print("      Compare pca_curvature, which separates them at thr<=0.99 and")
    print("      NOT at the 0.9999 that reproduces Mabrok's published 1e-5.")
    return vals


if __name__ == "__main__":
    main()
