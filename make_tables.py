"""
Every table the revised paper needs, rebuilt from the saved results.

Rule (12-audit-log.md S7.5): a number is re-measured into its table from a
results file, never copied from notes.  Each table is recomputed here from
per-state rows, on distinct states, with the project's true median and midrank
Spearman (fisherrao.stats).  Where a stored summary exists the recomputed value
is compared with it and any disagreement is printed.

Tables
  1  controls       K of the real model and three controls on the same states
  2  architectures  K across four models, distinct states
  3  agreement      rank correlation between curvature instruments (E2)
  4  corpus         each instrument's correlation with entropy on two corpora
  5  behaviour      sense change against a matched same-sense control

Usage:  python make_tables.py      -> results/tables/tables.md and tables.json
"""

from __future__ import annotations

import json
import re
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao.stats import median, spearman

OUT = pathlib.Path("results/tables")
R = pathlib.Path("results")


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def sign(d):
    d = [x for x in d if x == x]
    pos = sum(1 for x in d if x > 0)
    return dict(pos=pos, n=len(d), z=(pos - len(d) / 2) / (0.5 * len(d) ** 0.5) if d else float("nan"))


def quartiles(xs):
    v = sorted(x for x in xs if x == x)
    lo, hi = v[: len(v) // 2], v[(len(v) + 1) // 2:]
    return median(lo), median(hi)


CHECKS = []


def check(label, recomputed, stored, tol=1e-6):
    ok = abs(recomputed - stored) <= tol
    CHECKS.append(dict(label=label, recomputed=recomputed, stored=stored, ok=ok))


# ---------------------------------------------------------------- 1 controls
BANDS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, float("inf")))


def band_label(lo, hi):
    return f"[{lo:g}, {hi:g})" if hi != float("inf") else f"[{lo:g}, ∞)"


def controls():
    out = {}
    for name, tag in (("SmolLM2-135M", "SmolLM2-135M-Instruct_paper"), ("LLaMA-160M", "llama-160m_draw"),
                      ("GPT-2", "gpt2_draw"), ("Pythia-160M", "pythia-160m_draw")):
        path = R / "scramble" / f"consistent_{tag}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        rows = [r for r in d["rows"] if "error" not in r]
        gpath = R / "scramble" / f"gaussian_{tag}_points.jsonl"
        gauss = {r["_i"]: r["K_gauss"] for r in jl(gpath) if "K_gauss" in r} if gpath.exists() else {}
        t = dict(n=len(rows), excluded=d["summary"].get("n_errors", 0), real=median([r["K_real"] for r in rows]))
        for c in ("within", "consistent"):
            t[c] = median([r[f"K_{c}"] for r in rows])
            t[f"real_gt_{c}"] = sign([r["K_real"] - r[f"K_{c}"] for r in rows])
        stored = d["summary"]["k5"]
        for c in ("real", "within", "consistent"):
            check(f"controls, {name}: median K, {c}", t[c], stored[c])
        check(f"controls, {name}: real > consistent", t["real_gt_consistent"]["pos"], stored["real_gt_consistent"]["pos"])
        paired = [r for r in rows if r["_i"] in gauss]
        if paired:
            t["gauss_n"] = len(paired)
            t["gauss"] = median([gauss[r["_i"]] for r in paired])
            t["real_on_gauss_states"] = median([r["K_real"] for r in paired])
            t["real_gt_gauss"] = sign([r["K_real"] - gauss[r["_i"]] for r in paired])
            gsum = R / "scramble" / f"gaussian_{tag}.json"
            if gsum.exists():
                gs = json.loads(gsum.read_text(encoding="utf-8"))["summary"]
                check(f"controls, {name}: median K, Gaussian U", t["gauss"], gs["K_gauss"])
                check(f"controls, {name}: real > Gaussian U", t["real_gt_gauss"]["pos"], gs["real_gt_gauss"]["pos"])
        t["by_entropy"] = {}
        for lo, hi in BANDS:
            sub = [r for r in rows if lo <= r["entropy"] < hi]
            if not sub:
                continue
            b = dict(n=len(sub), real=median([r["K_real"] for r in sub]),
                     consistent=median([r["K_consistent"] for r in sub]),
                     real_gt_consistent=sign([r["K_real"] - r["K_consistent"] for r in sub]))
            gsub = [r for r in sub if r["_i"] in gauss]
            if gsub:
                b.update(gauss=median([gauss[r["_i"]] for r in gsub]),
                         real_gt_gauss=sign([r["K_real"] - gauss[r["_i"]] for r in gsub]))
            t["by_entropy"][band_label(lo, hi)] = b
        # "Learning moves K towards +1/4" (14-revision-log.md S5-S6), tested state by state:
        # a state counts when the real K is the closer of the pair to 1/4.
        q = lambda x: abs(x - 0.25)
        t["quarter"] = dict(real=median([q(r["K_real"]) for r in rows]),
                            consistent=median([q(r["K_consistent"]) for r in rows]),
                            real_closer_than_consistent=sign([q(r["K_consistent"]) - q(r["K_real"]) for r in rows]))
        if paired:
            t["quarter"].update(gauss=median([q(gauss[r["_i"]]) for r in paired]),
                                real_closer_than_gauss=sign([q(gauss[r["_i"]]) - q(r["K_real"]) for r in paired]))
        sw = [r for r in rows if "K4_real" in r]
        for k in (4, 6):
            t[f"k{k}"] = dict(n=len(sw), real=median([r[f"K{k}_real"] for r in sw]),
                              within=median([r[f"K{k}_within"] for r in sw]),
                              consistent=median([r[f"K{k}_consistent"] for r in sw]),
                              real_gt_consistent=sign([r[f"K{k}_real"] - r[f"K{k}_consistent"] for r in sw]))
        out[name] = t
    return out


# ---------------------------------------------------------------- 2 architectures
def architectures():
    seen = {}
    for r in jl(R / "profiles" / "profile_points.jsonl"):
        if isinstance(r.get("K"), (int, float)) and r["K"] == r["K"]:
            seen.setdefault((r["model"], r["layer"], r["_i"]), r)
    out = {}
    for m in sorted({k[0] for k in seen}):
        ks = [r["K"] for (mm, _, _), r in seen.items() if mm == m]
        q1, q3 = quartiles(ks)
        out[m] = dict(n=len(ks), median=median(ks), q1=q1, q3=q3)
    return out


# ---------------------------------------------------------------- 3 agreement
INSTR = [("intrinsic_K", "intrinsic K"), ("intrinsic_R", "intrinsic R"), ("pca", "local-PCA (Mabrok)"),
         ("manson", "Frenet, UᵀU (Manson)"), ("king", "Euclidean angle (King)")]
STORED_E2 = {("intrinsic_K", "pca"): "1. intrinsic Fisher-Rao K (this work)|2. local-PCA residual (Mabrok)",
             ("intrinsic_K", "manson"): "1. intrinsic Fisher-Rao K (this work)|3. Frenet under U^T U (Manson)",
             ("intrinsic_K", "king"): "1. intrinsic Fisher-Rao K (this work)|4. Euclidean angle (King et al.)",
             ("pca", "manson"): "2. local-PCA residual (Mabrok)|3. Frenet under U^T U (Manson)",
             ("pca", "king"): "2. local-PCA residual (Mabrok)|4. Euclidean angle (King et al.)",
             ("manson", "king"): "3. Frenet under U^T U (Manson)|4. Euclidean angle (King et al.)"}


def finite(*vals):
    return all(isinstance(v, (int, float)) and v == v for v in vals)


def agreement():
    s4 = json.loads((R / "stage4" / "stage4.json").read_text(encoding="utf-8"))
    rows = s4["e2_sample"]
    out = {"n": len(rows), "pairs": {}}
    keys = [k for k, _ in INSTR]
    for a_i, a in enumerate(keys):
        for b in keys[a_i + 1:]:
            pc = [r for r in rows if finite(r.get(a), r.get(b))]
            pooled = spearman([r[a] for r in pc], [r[b] for r in pc])
            per = []
            for l in sorted({r["layer"] for r in pc}):
                g = [r for r in pc if r["layer"] == l]
                if len(g) >= 8:
                    v = spearman([r[a] for r in g], [r[b] for r in g])
                    if v == v:
                        per.append(v)
            within = sum(per) / len(per) if per else float("nan")
            out["pairs"][f"{a}|{b}"] = dict(n=len(pc), pooled=pooled, within=within)
            # stored values predate the midrank Spearman of 12-audit-log.md S1; the
            # PCA proxy has exact ties (values that underflow to 0), so they move by
            # up to ~1e-3 -- agreement to that tolerance is agreement
            if (a, b) in STORED_E2:
                check(f"E2 pooled {a}|{b}", pooled, s4["e2_correlations"][STORED_E2[(a, b)]], 2e-3)
            if f"{a}|{b}" in s4["within_layer_correlations"]:
                check(f"E2 within-layer {a}|{b}", within, s4["within_layer_correlations"][f"{a}|{b}"], 2e-3)
    check("E2 positive control K|R", out["pairs"]["intrinsic_K|intrinsic_R"]["pooled"],
          s4["positive_control"]["rho_K_vs_R"], 2e-3)
    return out


# ---------------------------------------------------------------- 4 corpus
CORPUS = [("K", "intrinsic K"), ("R", "intrinsic R"), ("pca", "local-PCA (Mabrok)"),
          ("manson", "Frenet U^T U (Manson)"), ("king", "Euclid angle (King)")]


def corpus():
    c = json.loads((R / "corpus_compare" / "corpus_compare.json").read_text(encoding="utf-8"))
    rows = c["rows"]
    out = {"n": {s: sum(1 for r in rows if r["source"] == s) for s in ("handwritten", "wikitext")},
           "rho_with_entropy": {}, "K_vs_R": {}}
    for key, label in CORPUS:
        d = {}
        for s in ("handwritten", "wikitext"):
            pc = [r for r in rows if r["source"] == s and finite(r.get(key), r.get("entropy"))]
            d[s] = spearman([r[key] for r in pc], [r["entropy"] for r in pc])
            check(f"corpus rho({key}, H) {s}", d[s], c["rq3a"][label][s], 2e-3)
        d["shift"] = abs(d["wikitext"] - d["handwritten"])
        out["rho_with_entropy"][key] = d
    for s in ("handwritten", "wikitext"):
        pc = [r for r in rows if r["source"] == s and finite(r.get("K"), r.get("R"))]
        out["K_vs_R"][s] = spearman([r["K"] for r in pc], [r["R"] for r in pc])
    return out


# ---------------------------------------------------------------- 5 behaviour
def behaviour():
    ss = json.loads((R / "samesense" / "samesense_SmolLM2-135M-Instruct.json").read_text(encoding="utf-8"))
    out = {"n_pairs": len(ss["rows"]), "final_layer": {}}
    for key, label in (("d_fr", "Fisher–Rao geodesic"), ("d_euc", "Euclidean"), ("d_uu", "UᵀU")):
        same = [r["same"][-1][key] for r in ss["rows"]]
        diff = [r["diff"][-1][key] for r in ss["rows"]]
        t = dict(label=label, same=median(same), diff=median(diff))
        t["ratio"] = t["diff"] / t["same"]
        t.update(sign([b - a for a, b in zip(same, diff)]))
        out["final_layer"][key] = t
        # the stored summary (and the submitted paper's Table VII) used torch's
        # lower-middle element for these 32-pair medians; confirm that is the
        # whole difference, then report the true median
        st = ss["summary"][key]["final"]
        lower = lambda xs: sorted(xs)[(len(xs) - 1) // 2]
        t["same_lower"], t["diff_lower"] = lower(same), lower(diff)
        check(f"samesense {key} same (stored = lower median)", t["same_lower"], st["same"], 1e-9)
        check(f"samesense {key} diff (stored = lower median)", t["diff_lower"], st["diff"]["median"], 1e-9)
        check(f"samesense {key} sign", t["pos"], st["diff"]["pos"], 0)
    po = json.loads((R / "polysemy" / "polysemy.json").read_text(encoding="utf-8"))["summary"]
    wi = json.loads((R / "wic" / "wic_SmolLM2-135M-Instruct.json").read_text(encoding="utf-8"))["summary"]
    out["half_depth_mean"] = dict(
        handwritten_64={k: po[k]["half_depth_mean"] for k in ("d_fr", "d_euc", "d_uu")},
        gap_vs_uu=dict(mean=po["gap"]["mean"], ci=po["gap"]["ci"], fr_earlier=po["gap"]["n_fr_earlier"],
                       n=po["gap"]["n"]),
        wic_different_sense={k: wi[k]["0"] for k in ("d_fr", "d_euc", "d_uu")})
    return out


# ---------------------------------------------------------------- 6 spectral
SPECTRAL = (("SmolLM2-135M", "volume_scramble.json"), ("LLaMA-160M", "volume_scramble_llama-160m.json"),
            ("GPT-2", "volume_scramble_gpt2.json"), ("Pythia-160M", "volume_scramble_pythia-160m.json"))
PAPER = []          # (what, recomputed, value in the submitted paper)


def spectral():
    out = {}
    for name, fn in SPECTRAL:
        path = R / "volume_scramble" / fn
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        rows = d["rows"]
        layers = sorted({r["layer"] for r in rows})
        t = dict(n=len(rows), excluded=d.get("n_errors", 0), layers=layers)
        for q, label in (("lv", "log_volume"), ("ke", "k_eff")):
            ok = [r for r in rows if finite(r[f"{q}_real"], r[f"{q}_within"])]
            t[f"{label}_rho"] = spearman([r[f"{q}_real"] for r in ok], [r[f"{q}_within"] for r in ok])
            pr = [median([r[f"{q}_real"] for r in rows if r["layer"] == l]) for l in layers]
            pw = [median([r[f"{q}_within"] for r in rows if r["layer"] == l]) for l in layers]
            t[f"{label}_profile"] = dict(real=pr, within=pw)
            t[f"{label}_min_layer"] = dict(real=layers[pr.index(min(pr))], within=layers[pw.index(min(pw))])
            t[f"{label}_range_retained"] = (max(pw) - min(pw)) / (max(pr) - min(pr))
            t[f"{label}_shape_rho"] = spearman(pr, pw)
            t[f"{label}_rho_entropy_real"] = spearman([r["entropy"] for r in ok], [r[f"{q}_real"] for r in ok])
        if "model" in d:        # written by the current run_volume_scramble.py, which uses true medians
            check(f"spectral, {name}: log-volume per-state rho", t["log_volume_rho"], d["point_rho"]["lv_within"])
            check(f"spectral, {name}: k_eff per-state rho", t["k_eff_rho"], d["point_rho"]["ke_within"])
            check(f"spectral, {name}: log-volume profile shape", t["log_volume_shape_rho"], d["shape_rho"]["within"])
            check(f"spectral, {name}: k_eff profile shape", t["k_eff_shape_rho"], d["k_eff_shape_rho"]["within"])
            check(f"spectral, {name}: rho(log-volume, entropy)", t["log_volume_rho_entropy_real"], d["rho_real_entropy"])
        out[name] = t
    if "SmolLM2-135M" in out:
        PAPER.append(("spectral: log-volume, per-state rho (SmolLM2)", out["SmolLM2-135M"]["log_volume_rho"], 0.957))
        PAPER.append(("spectral: k_eff, per-state rho (SmolLM2)", out["SmolLM2-135M"]["k_eff_rho"], 0.991))
        PAPER.append(("spectral: k_eff range retained (SmolLM2), Section VI.B text",
                      out["SmolLM2-135M"]["k_eff_range_retained"], 0.995))
    return out


# ---------------------------------------------------------------- 7 validation
def validation():
    V = R / "validation"
    out = {"ladder_autodiff": json.loads((R / "ladder" / "ladder_precision.json").read_text(encoding="utf-8"))}
    s0 = json.loads((R / "stage0" / "stage0.json").read_text(encoding="utf-8"))
    mdl = s0.get("model")
    out["gate_a_real"] = dict(model=mdl.get("model_id") if isinstance(mdl, dict) else mdl,
                              eps=s0["gate_a"]["best_eps"], rel_err=s0["gate_a"]["best_rel_err"])
    g = re.search(r"plateau at eps = (\S+), relative error (\S+)", (V / "gate_a_kl_test.txt").read_text(encoding="utf-8"))
    out["gate_a_synthetic"] = dict(eps=float(g.group(1)), rel_err=float(g.group(2)))
    PAPER.append(("Gate A on a real model (KL-Hessian relative error)", out["gate_a_real"]["rel_err"], 8.7e-6))

    pb = (V / "validate_pullback.txt").read_text(encoding="utf-8")
    sec = {int(m.group(1)): m.start() for m in re.finditer(r"^(\d)\. ", pb, re.M)}
    part = lambda k: pb[sec[k]: sec.get(k + 1, len(pb))]
    lastnum = lambda txt: [float(m.group(1)) for m in re.finditer(r"(\S+)\s+PASS\s*$", txt, re.M)]
    out["rung7"] = dict(sectional_worst=max(lastnum(part(1))), reparam_worst=max(lastnum(part(2))),
                        scalar_worst=max(lastnum(part(3))),
                        broken_assembly=float(re.search(r"term dropped: (\S+)\s+CAUGHT", pb).group(1)))

    gs = (V / "validate_geomstats.txt").read_text(encoding="utf-8")
    fam = {}
    for m in re.finditer(r"^(?:rung|RUNG) (\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s.*?(DISAGREE|AGREE)", gs, re.M):
        f_ = fam.setdefault(m.group(2), dict(rung=int(m.group(1)), n=0, agree=0, max_diff=0.0))
        f_["n"] += 1
        f_["agree"] += m.group(6) == "AGREE"
        f_["max_diff"] = max(f_["max_diff"], float(m.group(5)))
    for m in re.finditer(r"^\s*(\w+)\s*: metrics identical\?\s+(True|False)", gs, re.M):
        fam.setdefault(m.group(1), {})["metrics_identical"] = m.group(2) == "True"
    out["geomstats"] = fam

    out["null"] = [dict(model=r["short"], norm=r["norm"], expect=r["expect"], found=r["found"],
                        kl_null=r["kl_null"], kl_rand=r["kl_rand"], ok=r["pass"])
                   for r in json.loads((R / "nulltest" / "nulltest.json").read_text(encoding="utf-8"))]
    out["trace_fraction"] = json.loads((R / "tracefrac" / "tracefrac.json").read_text(encoding="utf-8"))["frac"]
    return out


# ---------------------------------------------------------------- 8 near-flat calibration
def calibration():
    import make_figures                           # Fig. 3(b) is computed there, seeded
    m = json.loads((R / "mabrok" / "mabrok.json").read_text(encoding="utf-8"))
    th = ["var0.9", "var0.95", "var0.99", "var0.999", "var0.9999"]
    layer12 = {k: [m["results"][f"L12_k{k}"].get(t) for t in th]
               for k in m["k_sweep"] if f"L12_k{k}" in m["results"]}
    ys, yp = make_figures.sphere_calibration()
    return dict(model=m["model"], thresholds=[0.9, 0.95, 0.99, 0.999, 0.9999], layer12=layer12,
                sphere=ys, plane=yp)


# ---------------------------------------------------------------- 9 LayerNorm projection check
def layernorm_check():
    out = {}
    for path in sorted((R / "layernorm").glob("projection_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        s, rows = d["summary"], d["rows"]
        out[s["model"]] = dict(n=s["n"], r_over_s=s["median_r_over_s"], metric_err=s["proj_vs_full_x1"],
                               median_abs_dK=s["median_abs_dK"],
                               max_abs_dK=max(abs(r["K_projected"] - r["K_exact"]) for r in rows),
                               package_vs_full=s.get("package_vs_full_x1"))
    return out


def f(x, nd=4):
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def markdown(T):
    L = ["# Tables for the revised paper", "",
         "Generated by `make_tables.py` from the saved results; distinct states, true medians, "
         "midrank Spearman. Do not edit by hand.", ""]
    L += ["## 1. Controls — slice sectional curvature `K` (k = 5) on the same states", "",
          "| model | n | real | inconsistent scramble (submitted paper) | consistent scramble | Gaussian `U`, entropy-matched |",
          "|---|---|---|---|---|---|"]
    for m, t in T["controls"].items():
        g = (f"{t['gauss']:.4f} (real > control {t['real_gt_gauss']['pos']}/{t['real_gt_gauss']['n']}, "
             f"z = {t['real_gt_gauss']['z']:+.2f})") if "gauss" in t else "—"
        L.append(f"| {m} | {t['n']} | {t['real']:.4f} | {t['within']:.4f} (real > control "
                 f"{t['real_gt_within']['pos']}/{t['real_gt_within']['n']}, z = {t['real_gt_within']['z']:+.2f}) | "
                 f"{t['consistent']:.4f} (real > control {t['real_gt_consistent']['pos']}/"
                 f"{t['real_gt_consistent']['n']}, z = {t['real_gt_consistent']['z']:+.2f}) | {g} |")
    L += ["", "`k`-sweep subset (same states at k = 4, 6):", "",
          "| model | k | n | real | inconsistent | consistent | real > consistent |", "|---|---|---|---|---|---|---|"]
    for m, t in T["controls"].items():
        for k in (4, 6):
            s = t[f"k{k}"]
            L.append(f"| {m} | {k} | {s['n']} | {s['real']:.4f} | {s['within']:.4f} | {s['consistent']:.4f} | "
                     f"{s['real_gt_consistent']['pos']}/{s['real_gt_consistent']['n']}, z = {s['real_gt_consistent']['z']:+.2f} |")
    L += ["", "By entropy band, `k` = 5 (median `K`; sign test of real > control):", "",
          "| model | entropy (nats) | n | real | consistent | real > consistent | Gaussian `U` | real > Gaussian `U` |",
          "|---|---|---|---|---|---|---|---|"]
    for m, t in T["controls"].items():
        for bl, b in t["by_entropy"].items():
            rc = b["real_gt_consistent"]
            gg = (f"{b['gauss']:.4f} | {b['real_gt_gauss']['pos']}/{b['real_gt_gauss']['n']}, "
                  f"z = {b['real_gt_gauss']['z']:+.2f}") if "gauss" in b else "— | —"
            L.append(f"| {m} | {bl} | {b['n']} | {b['real']:.4f} | {b['consistent']:.4f} | "
                     f"{rc['pos']}/{rc['n']}, z = {rc['z']:+.2f} | {gg} |")
    L += ["", "Distance from +1/4, `k` = 5: median \\|`K` − 1/4\\|, and the states where the real `K` is the "
          "closer of the pair to 1/4:", "",
          "| model | real | consistent | real closer | Gaussian `U` | real closer |", "|---|---|---|---|---|---|"]
    for m, t in T["controls"].items():
        qd, rc = t["quarter"], t["quarter"]["real_closer_than_consistent"]
        gg = (f"{qd['gauss']:.4f} | {qd['real_closer_than_gauss']['pos']}/{qd['real_closer_than_gauss']['n']}, "
              f"z = {qd['real_closer_than_gauss']['z']:+.2f}") if "gauss" in qd else "— | —"
        L.append(f"| {m} | {qd['real']:.4f} | {qd['consistent']:.4f} | {rc['pos']}/{rc['n']}, z = {rc['z']:+.2f} | {gg} |")
    # 13-revision-plan.md S5, E1: real and consistent "agree" when |z| < 2 and the medians differ by < 0.01.
    agree = lambda d, z: abs(z) < 2 and abs(d) < 0.01
    for c, label in (("consistent", "consistent scramble"), ("gauss", "Gaussian `U`")):
        cs = [(m, t["real"] - t[c], t[f"real_gt_{c}"]["z"]) for m, t in T["controls"].items() if c in t]
        L += ["", f"Decision rule E1 (13-revision-plan.md §5: agreement means \\|z\\| < 2 "
              f"and a median difference below 0.01), real against {label}: agreement on "
              f"{sum(agree(d, z) for _, d, z in cs)} of {len(cs)} models (" +
              "; ".join(f"{m} {d:+.4f}, z = {z:+.2f}" for m, d, z in cs) + ")."]
    ex = [f"{m}, {t['excluded']}" for m, t in T["controls"].items() if t.get("excluded")]
    if ex:
        L += ["", "States excluded because the metric is undefined there (a hidden state that is exactly zero; "
              "14-revision-log.md §12): " + "; ".join(ex) + "."]
    L += ["", "## 2. Slice sectional curvature across architectures (distinct states)", "",
          "| model | n | median `K` | quartiles |", "|---|---|---|---|"]
    for m, t in T["architectures"].items():
        L.append(f"| {m} | {t['n']} | {t['median']:.4f} | {t['q1']:.4f} – {t['q3']:.4f} |")
    L += ["", f"## 3. Agreement between instruments (E2, n = {T['agreement']['n']} states; pairwise complete)", "",
          "| pair | n | pooled ρ | within-layer mean ρ |", "|---|---|---|---|"]
    names = dict(INSTR)
    for p, t in T["agreement"]["pairs"].items():
        a, b = p.split("|")
        L.append(f"| {names[a]} – {names[b]} | {t['n']} | {t['pooled']:+.3f} | {t['within']:+.3f} |")
    L += ["", f"## 4. Correlation with predictive entropy on two corpora "
          f"(hand-written n = {T['corpus']['n']['handwritten']}, WikiText n = {T['corpus']['n']['wikitext']})", "",
          "| instrument | hand-written ρ | WikiText ρ | shift |", "|---|---|---|---|"]
    for key, label in CORPUS:
        t = T["corpus"]["rho_with_entropy"][key]
        L.append(f"| {label} | {t['handwritten']:+.3f} | {t['wikitext']:+.3f} | {t['shift']:.3f} |")
    L.append(f"\nIntrinsic `K` against `R`: ρ = {T['corpus']['K_vs_R']['handwritten']:+.3f} (hand-written), "
             f"{T['corpus']['K_vs_R']['wikitext']:+.3f} (WikiText).")
    b = T["behaviour"]
    L += ["", f"## 5. Sense change against a matched same-sense control ({b['n_pairs']} words, final layer)", "",
          "| metric | same sense (median) | sense change (median) | ratio | sign | z |", "|---|---|---|---|---|---|"]
    for key, t in b["final_layer"].items():
        L.append(f"| {t['label']} | {t['same']:.4g} | {t['diff']:.4g} | {t['ratio']:.2f}× | {t['pos']}/{t['n']} | {t['z']:+.2f} |")
    h = b["half_depth_mean"]
    L += ["", "Mean half-separation depth (layer): hand-written 64 pairs — Fisher–Rao "
          f"{h['handwritten_64']['d_fr']:.2f}, Euclidean {h['handwritten_64']['d_euc']:.2f}, UᵀU "
          f"{h['handwritten_64']['d_uu']:.2f}; paired gap against UᵀU {h['gap_vs_uu']['mean']:.2f} layers, "
          f"95 % CI [{h['gap_vs_uu']['ci'][0]:.2f}, {h['gap_vs_uu']['ci'][1]:.2f}], Fisher–Rao earlier in "
          f"{h['gap_vs_uu']['fr_earlier']}/{h['gap_vs_uu']['n']}. WiC different-sense pairs — Fisher–Rao "
          f"{h['wic_different_sense']['d_fr']:.2f}, Euclidean {h['wic_different_sense']['d_euc']:.2f}, UᵀU "
          f"{h['wic_different_sense']['d_uu']:.2f}. (Means, as computed by run_polysemy.py and run_wic.py.)"]
    S = T.get("spectral", {})
    if S:
        L += ["", "## 6. Spectral control — log-volume and effective dimension under the paired scramble "
              "(at h₀ the paper's scramble and the consistent one coincide, so this control is valid)", "",
              "Real against scrambled: ρ across states, ρ between the two layer profiles (medians by layer), the "
              "layer where each profile is lowest, and the scrambled profile's range as a share of the real one's. "
              "Last column: the real quantity's ρ with predictive entropy.", "",
              "| model | n | quantity | ρ, per state | ρ, layer profile | min layer real / scrambled | range kept "
              "| ρ with entropy |",
              "|---|---|---|---|---|---|---|---|"]
        for m, t in S.items():
            for label, name in (("log_volume", "log-volume"), ("k_eff", "`k_eff`")):
                L.append(f"| {m} | {t['n']} | {name} | {t[f'{label}_rho']:+.3f} | {t[f'{label}_shape_rho']:+.3f} | "
                         f"{t[f'{label}_min_layer']['real']} / {t[f'{label}_min_layer']['within']} | "
                         f"{t[f'{label}_range_retained']*100:.0f} % | {t[f'{label}_rho_entropy_real']:+.3f} |")
        ex = [f"{m}, {t['excluded']} of {t['n'] + t['excluded']}" for m, t in S.items() if t.get("excluded")]
        if ex:
            L += ["", "States excluded because the metric is undefined there (a hidden state that is exactly zero; "
                  "14-revision-log.md §12): " + "; ".join(ex) + "."]
        for m, t in S.items():
            L += ["", f"Effective dimension by layer, {m} (true medians):", "",
                  "| layer | " + " | ".join(str(l) for l in t["layers"]) + " |",
                  "|---|" + "---|" * len(t["layers"]),
                  "| real | " + " | ".join(f"{v:g}" for v in t["k_eff_profile"]["real"]) + " |",
                  "| scrambled | " + " | ".join(f"{v:g}" for v in t["k_eff_profile"]["within"]) + " |"]
    V = T["validation"]
    L += ["", "## 7. Validation", "", "Ladder, autodiff implementation, worst over 12 base points and 12 planes:", "",
          "| rung | manifold | expected `K` | worst error |", "|---|---|---|---|"]
    for r in V["ladder_autodiff"]:
        L.append(f"| {r['rung']} | {r['manifold']} | {r['expect']:+g} | {r['worst']:.1e} |")
    r7 = V["rung7"]
    L += ["", f"Rung 7 (synthetic pullback, `K` = 1/4 exactly): worst sectional error {r7['sectional_worst']:.1e}; "
          f"reparameterisation {r7['reparam_worst']:.1e}; scalar curvature {r7['scalar_worst']:.1e}; the broken "
          f"assembly (outer-product term dropped) fails, at {r7['broken_assembly']:.3f}.",
          "", "Independent implementation (geomstats 2.8.0):", "",
          "| family | rung | comparisons | agree | largest difference | metrics identical |", "|---|---|---|---|---|---|"]
    for fam, v in V["geomstats"].items():
        if "n" in v:
            L.append(f"| {fam} | {v['rung']} | {v['n']} | {v['agree']} | {v['max_diff']:.1e} | "
                     f"{v.get('metrics_identical', '—')} |")
    ga, gsy = V["gate_a_real"], V["gate_a_synthetic"]
    L += ["", f"KL–Hessian identity (Gate A): real model ({ga['model']}), relative error {ga['rel_err']:.1e} at "
          f"ε = {ga['eps']:g}; synthetic softmax model, {gsy['rel_err']:.1e} at ε = {gsy['eps']:g}.",
          "", "Null directions — a step of ‖h‖ along the null space against a random step of the same size:", "",
          "| model | norm | null directions expected / found | KL along null | KL along random |",
          "|---|---|---|---|---|"]
    for r in V["null"]:
        L.append(f"| {r['model']} | {r['norm']} | {r['expect']} / {r['found']} | {r['kl_null']:.1e} | {r['kl_rand']:.2f} |")
    tf = V["trace_fraction"]
    L += ["", f"Share of the quotiented metric's trace held by the slice ({tf['5']['n']} states): " +
          "; ".join(f"`k` = {k}: median {tf[k]['median']*100:.1f} %, mean {tf[k]['mean']*100:.1f} %"
                    for k in ("4", "5", "6")) + "."]
    C = T["calibration"]
    L += ["", "## 8. The near-flat reading — local-PCA residual by retained-variance threshold", "",
          "| | " + " | ".join(str(t) for t in C["thresholds"]) + " |", "|---|" + "---|" * len(C["thresholds"])]
    for k, vals in C["layer12"].items():
        L.append(f"| {C['model']} layer 12, k = {k} | " + " | ".join(f"{v:.1e}" for v in vals) + " |")
    L.append("| unit 3-sphere (`K` = +1) | " + " | ".join(f"{v:.1e}" for v in C["sphere"]) + " |")
    L.append("| flat 3-plane (`K` = 0) | " + " | ".join(f"{v:.1e}" for v in C["plane"]) + " |")
    LN = T.get("layernorm", {})
    if LN:
        L += ["", "## 9. The RMSNorm-form Jacobian on LayerNorm models, as it was before the 29 October fix", "",
              "| model | states | median r/s | metric error, median | median \\|ΔK\\| | largest \\|ΔK\\| | package after the fix |",
              "|---|---|---|---|---|---|---|"]
        for m, v in LN.items():
            pk = f"{v['package_vs_full']:.1e}" if v.get("package_vs_full") is not None else "—"
            L.append(f"| {m} | {v['n']} | {v['r_over_s']:.4f} | {v['metric_err']:.1e} | {v['median_abs_dK']:.1e} | "
                     f"{v['max_abs_dK']:.1e} | {pk} |")
    if PAPER:
        L += ["", "## Recomputed values against the submitted paper", "",
              "| quantity | recomputed | in the submitted paper |", "|---|---|---|"]
        for what, a, b in PAPER:
            L.append(f"| {what} | {a:.4g} | {b:.4g} |")
    bad = [c for c in CHECKS if not c["ok"]]
    L += ["", f"## Verification against stored summaries: {len(CHECKS) - len(bad)}/{len(CHECKS)} agree", ""]
    for c in bad:
        L.append(f"- **differs:** {c['label']}: recomputed {c['recomputed']:.6g}, stored {c['stored']:.6g}")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = dict(controls=controls(), architectures=architectures(), agreement=agreement(),
             corpus=corpus(), behaviour=behaviour(), spectral=spectral(), validation=validation(),
             calibration=calibration(), layernorm=layernorm_check())
    a = T["architectures"]
    meds = [v["median"] for v in a.values()]
    PAPER.extend([
        ("states in the cross-architecture sweep", sum(v["n"] for v in a.values()), 456),
        ("Pythia-160M median K", a["EleutherAI/pythia-160m"]["median"], 0.2738),
        ("GPT-2 median K", a["gpt2"]["median"], 0.2599),
        ("spread of median K across the four models", max(meds) - min(meds), 0.019),
        ("paper's scramble, SmolLM2, median K", T["controls"]["SmolLM2-135M"]["within"], 0.0109),
        ("paper's scramble, SmolLM2, states", T["controls"]["SmolLM2-135M"]["n"], 221),
        ("sense change / same sense, Fisher-Rao ratio", T["behaviour"]["final_layer"]["d_fr"]["ratio"], 2.03),
        ("largest cross-family |rho| between instruments",
         max(abs(v["pooled"]) for p, v in T["agreement"]["pairs"].items()
             if not p.startswith("intrinsic_K|intrinsic_R")), 0.248)])
    T["checks"] = CHECKS
    T["paper"] = [dict(what=w, recomputed=x, submitted=y) for w, x, y in PAPER]
    (OUT / "tables.json").write_text(json.dumps(T, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(T)
    (OUT / "tables.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
