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
def controls():
    out = {}
    for name, tag in (("SmolLM2-135M", "SmolLM2-135M-Instruct_paper"), ("LLaMA-160M", "llama-160m_draw")):
        rows = [r for r in json.loads((R / "scramble" / f"consistent_{tag}.json").read_text(encoding="utf-8"))["rows"]
                if "error" not in r]
        gpath = R / "scramble" / f"gaussian_{tag}_points.jsonl"
        gauss = {r["_i"]: r["K_gauss"] for r in jl(gpath) if "K_gauss" in r} if gpath.exists() else {}
        t = dict(n=len(rows), real=median([r["K_real"] for r in rows]))
        for c in ("within", "consistent"):
            t[c] = median([r[f"K_{c}"] for r in rows])
            t[f"real_gt_{c}"] = sign([r["K_real"] - r[f"K_{c}"] for r in rows])
        paired = [r for r in rows if r["_i"] in gauss]
        if paired:
            t["gauss_n"] = len(paired)
            t["gauss"] = median([gauss[r["_i"]] for r in paired])
            t["real_on_gauss_states"] = median([r["K_real"] for r in paired])
            t["real_gt_gauss"] = sign([r["K_real"] - gauss[r["_i"]] for r in paired])
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
    bad = [c for c in CHECKS if not c["ok"]]
    L += ["", f"## Verification against stored summaries: {len(CHECKS) - len(bad)}/{len(CHECKS)} agree", ""]
    for c in bad:
        L.append(f"- **differs:** {c['label']}: recomputed {c['recomputed']:.6g}, stored {c['stored']:.6g}")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = dict(controls=controls(), architectures=architectures(), agreement=agreement(),
             corpus=corpus(), behaviour=behaviour())
    T["checks"] = CHECKS
    (OUT / "tables.json").write_text(json.dumps(T, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(T)
    (OUT / "tables.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
