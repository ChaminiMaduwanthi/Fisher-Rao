"""
Duplicated states in the saved results, and what they did to the paper's numbers.

Checkpointed scripts append one row per computed state and resume by skipping
states already on file.  Several checkpoints nevertheless hold the same state
twice -- rows that recur across resumes with identical values.  The analysis
scripts that build the summary .json files key their rows by state and so drop
the copies; but some paper figures were read straight from the raw .jsonl
checkpoints, where every copy counts as an independent observation.

For each source behind a number in the submitted paper this script recomputes
the statistic twice -- on the rows as stored (which must reproduce the paper)
and on distinct states -- so the size of the effect is on record rather than
assumed.  It also scans every results file for duplicated rows.

Usage:  python audit_duplicates.py
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao.stats import median, mannwhitney

OUT = pathlib.Path("results/audit")
EDGES = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def distinct(rows, key):
    seen = {}
    for r in rows:
        seen.setdefault(key(r), r)
    return list(seen.values())


def sign(d):
    d = [x for x in d if x == x]
    pos = sum(1 for x in d if x > 0)
    return pos, len(d), (pos - len(d) / 2) / (0.5 * len(d) ** 0.5) if d else float("nan")


def ebin(e):
    for lo, hi in EDGES:
        if lo <= e < hi:
            return f"[{lo},{hi})" if hi < 1e9 else f"[{lo},inf)"


def scan():
    """Every results file: rows, identical copies."""
    out = []
    for f in sorted(glob.glob("results/**/*.jsonl", recursive=True)):
        lines = [l for l in open(f, encoding="utf-8").read().splitlines() if l.strip()]
        out.append(dict(file=f.replace("\\", "/"), rows=len(lines),
                        identical_copies=len(lines) - len(set(lines))))
    return out


def scramble(label, rows):
    """Section VI.B, Tables III, V and VI, and the k-sweep -- from within_points.jsonl."""
    res = {}
    pos, n, z = sign([r["K_real"] - r["K_within"] for r in rows])
    res["overall"] = dict(n=n, K_real=median([r["K_real"] for r in rows]),
                          K_within=median([r["K_within"] for r in rows]), pos=pos, z=z)
    sw = [r for r in rows if "K_real_k4" in r]
    for k in (4, 6):
        pos, n, z = sign([r[f"K_real_k{k}"] - r[f"K_within_k{k}"] for r in sw])
        res[f"k{k}"] = dict(n=n, K_real=median([r[f"K_real_k{k}"] for r in sw]),
                            K_within=median([r[f"K_within_k{k}"] for r in sw]), pos=pos, z=z)
    res["by_layer"] = {}
    for l in sorted({r["layer"] for r in rows}):
        sub = [r for r in rows if r["layer"] == l]
        kr, kw = median([r["K_real"] for r in sub]), median([r["K_within"] for r in sub])
        pos, n, z = sign([r["K_real"] - r["K_within"] for r in sub])
        res["by_layer"][str(l)] = dict(n=n, retained=kw / kr, pos=pos)
    res["by_entropy"] = {}
    for lo, hi in EDGES:
        sub = [r for r in rows if ebin(r["entropy"]) == ebin(lo)]
        pos, n, z = sign([r["K_real"] - r["K_within"] for r in sub])
        res["by_entropy"][ebin(lo)] = dict(n=n, K_real=median([r["K_real"] for r in sub]),
                                           K_within=median([r["K_within"] for r in sub]), pos=pos)
    return res


def profiles(rows):
    """Section VI.A and Fig. 1 (per-model medians), and the negative-plane test of Section VII."""
    res = {"n": len(rows), "per_model": {}}
    for m in sorted({r["model"] for r in rows}):
        sub = [r["K"] for r in rows if r["model"] == m]
        res["per_model"][m] = dict(n=len(sub), median_K=median(sub))
    meds = [v["median_K"] for v in res["per_model"].values()]
    res["spread"] = max(meds) - min(meds)
    neg = [r for r in rows if r.get("n_planes_neg", 0) > 0]
    pos = [r for r in rows if r.get("n_planes_neg", 0) == 0]
    _, z_k, auc = mannwhitney([r["k_eff_99"] for r in neg], [r["k_eff_99"] for r in pos])
    _, z_h, _ = mannwhitney([r["entropy"] for r in neg], [r["entropy"] for r in pos])
    _, z_c, _ = mannwhitney([r["cond_eff"] for r in neg], [r["cond_eff"] for r in pos])
    res["negative_planes"] = dict(
        n_neg=len(neg), n=len(rows),
        k_eff_neg=median([r["k_eff_99"] for r in neg]), k_eff_pos=median([r["k_eff_99"] for r in pos]),
        auc=auc, z=z_k, entropy_neg=median([r["entropy"] for r in neg]),
        entropy_pos=median([r["entropy"] for r in pos]), z_entropy=z_h, z_cond=z_c)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(scan=scan())

    w = jl("results/scramble/within_points.jsonl")
    report["scramble"] = dict(rows=scramble("rows", w),
                              distinct=scramble("distinct", distinct(w, lambda r: r["_i"])))
    p = [r for r in jl("results/profiles/profile_points.jsonl")
         if isinstance(r.get("K"), (int, float)) and r["K"] == r["K"]]
    report["profiles"] = dict(rows=profiles(p),
                              distinct=profiles(distinct(p, lambda r: (r["model"], r["layer"], r["_i"]))))

    print("FILES WITH IDENTICAL COPIES")
    for s in report["scan"]:
        if s["identical_copies"]:
            print(f"   {s['file']:<52} rows {s['rows']:>4}   identical copies {s['identical_copies']:>3}")

    for name in ("rows", "distinct"):
        s = report["scramble"][name]
        o = s["overall"]
        print(f"\nSCRAMBLE ({name}): n {o['n']}  K {o['K_real']:.4f} -> {o['K_within']:.4f}  "
              f"{o['pos']}/{o['n']}  z {o['z']:+.2f}")
        for k in (4, 6):
            q = s[f"k{k}"]
            print(f"   k={k}: n {q['n']}  {q['K_real']:.4f} -> {q['K_within']:.4f}  {q['pos']}/{q['n']}  z {q['z']:+.2f}")
        print("   by layer (retained, sign):  " + "  ".join(
            f"L{l} {v['retained']*100:+.1f}% {v['pos']}/{v['n']}" for l, v in s["by_layer"].items()))
        print("   by entropy (n, real, within, sign):  " + "  ".join(
            f"{b} {v['n']} {v['K_real']:.4f} {v['K_within']:+.4f} {v['pos']}/{v['n']}"
            for b, v in s["by_entropy"].items()))

    for name in ("rows", "distinct"):
        s = report["profiles"][name]
        print(f"\nPROFILES ({name}): n {s['n']}, spread {s['spread']:.4f}")
        for m, v in s["per_model"].items():
            print(f"   {m:<38} n {v['n']:>3}  median K {v['median_K']:.4f}")
        q = s["negative_planes"]
        print(f"   negative planes: {q['n_neg']}/{q['n']}; k_eff {q['k_eff_neg']:.0f} vs {q['k_eff_pos']:.0f}; "
              f"AUC {q['auc']:.3f}, z {q['z']:+.2f}; entropy {q['entropy_neg']:.2f} vs "
              f"{q['entropy_pos']:.2f} (z {q['z_entropy']:+.2f}); conditioning z {q['z_cond']:+.2f}")

    (OUT / "duplicates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'duplicates.json'}")


if __name__ == "__main__":
    main()
