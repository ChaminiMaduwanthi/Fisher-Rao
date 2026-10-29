"""
Check every quantitative statement in the paper source against the saved results.

Rule (12-audit-log.md S7.5): a number in the paper is re-measured from a results
file, never copied from notes.  This script recomputes each number the paper
quotes, formats it the way the paper prints it, and checks that exact text is in
paper/fisher-rao-conference-paper.md.  Run make_tables.py first.

Usage:  python check_paper_numbers.py        (exit status 1 if any check fails)
"""

from __future__ import annotations

import json
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

R = pathlib.Path("results")
PAPER = pathlib.Path("paper/fisher-rao-conference-paper.md").read_text(encoding="utf-8")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
FAILS, N = [], [0]


def load(*parts):
    return json.loads(R.joinpath(*parts).read_text(encoding="utf-8"))


def num(x, nd):
    """Fixed decimals with a typographic minus, as the paper prints them."""
    return f"{x:.{nd}f}".replace("-", "−")


def signed(x, nd):
    return ("+" if x >= 0 else "") + num(x, nd)


def sci(x, nd=1):
    """a.b × 10⁻ⁿ"""
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m} × 10{str(int(e)).translate(SUP)}"


def has(label, text):
    N[0] += 1
    if text not in PAPER:
        FAILS.append(f"{label}: expected the text '{text}'")


T = load("tables", "tables.json")
C, S, A, V = T["controls"], T["spectral"], T["architectures"], T["validation"]
MODELS = ("SmolLM2-135M", "LLaMA-160M", "GPT-2", "Pythia-160M")

# ---- profile sample (Section V-B, abstract)
arch = [A["HuggingFaceTB/SmolLM2-135M-Instruct"], A["JackFram/llama-160m"], A["gpt2"], A["EleutherAI/pythia-160m"]]
has("profile states", f"{sum(a['n'] for a in arch)} hidden states")
for a in arch:
    has("profile median and quartiles", f"{num(a['median'], 3)} for")
    has("profile quartiles", f"{num(a['q1'], 3)}–{num(a['q3'], 3)}")
meds = [a["median"] for a in arch]
has("profile range, abstract", f"between {num(min(meds), 3)} and {num(max(meds), 3)}")
has("profile spread", f"a spread of {num(max(meds) - min(meds), 3)}")

# ---- controls (Table III, Section V-B, abstract, conclusion)
for m in MODELS:
    t = C[m]
    has(f"{m} trained", f"| {num(t['real'], 4)} ")
    has(f"{m} C0", f"{num(t['within'], 4)} ({signed(t['real_gt_within']['z'], 1)})")
    has(f"{m} C1", f"{num(t['consistent'], 4)} ({signed(t['real_gt_consistent']['z'], 1)})")
    has(f"{m} C2", f"{num(t['gauss'], 4)} ({signed(t['real_gt_gauss']['z'], 1)})")
    has(f"{m} states", f"| {t['n']} ")
ctrl = [C[m][c] for m in MODELS for c in ("consistent", "gauss")]
has("genuine-control range", f"{num(min(ctrl), 3)} to {num(max(ctrl), 3)}")
assert max(abs(x - 0.25) for x in ctrl) <= 0.035
has("within 0.035 of 1/4", "within 0.035 of 1/4")
off = [C[m]["consistent"] - C[m]["real"] for m in MODELS]
assert all(o > 0 for o in off)
has("offset against C1", f"{num(min(off), 3)} to {num(max(off), 3)}")
zc = [C[m]["real_gt_consistent"]["z"] for m in MODELS]
has("C1 z range", f"z from {num(max(zc), 1)} to {num(min(zc), 1)}")
z0 = [C[m]["real_gt_within"]["z"] for m in MODELS]
has("C0 z range", f"z from {signed(min(z0), 1)} to {signed(max(z0), 1)}")
sweep = [C[m][f"k{k}"]["real_gt_consistent"]["z"] for m in MODELS for k in (4, 6)]
assert all(z < 0 for z in sweep)
has("k-sweep", f"in all eight cases (|z| ≥ 2 in {['zero','one','two','three','four','five','six','seven','eight'][sum(abs(z) >= 2 for z in sweep)]})")
assert all(b["real_gt_consistent"]["pos"] < b["real_gt_consistent"]["n"] / 2
           for m in MODELS for b in C[m]["by_entropy"].values())
has("every entropy band", "in every entropy band fewer than half of the states have the trained value higher")
quarter = [C[m]["quarter"][f"real_closer_than_{c}"]["z"] for m in MODELS for c in ("consistent", "gauss")]
has("closer to 1/4", f"in only {['zero','one','two','three','four','five','six','seven','eight'][sum(z >= 2 for z in quarter)]} of the eight")
assert min(quarter) > -2
mid = [C[m]["by_entropy"][b]["real"] for m in MODELS for b in ("[0.5, 1)", "[1, 2)", "[2, 4)")]
midc = [C[m]["by_entropy"][b][c] for m in MODELS for b in ("[0.5, 1)", "[1, 2)", "[2, 4)") for c in ("consistent", "gauss")]
assert max(abs(x - 0.25) for x in mid) <= 0.02 and max(abs(x - 0.25) for x in midc) <= 0.06
has("mid-band closeness", "trained medians stay within 0.02 of 1/4 and the genuine controls within 0.06")
g = C["GPT-2"]
has("GPT-2 against C2", f"({num(g['gauss'], 3)} against {num(g['real'], 3)})")
nb = load("scramble", "gaussian_nobias_gpt2_draw.json")["summary"]["bands"]["[4, inf)"]
has("bias check", f"({num(nb['K_gauss_nobias'], 3)} against {num(nb['K_gauss'], 3)} above 4 nats, where the trained model gives {num(nb['K_real'], 3)})")
excl = load("scramble", "consistent_llama-160m_draw.json")["summary"]["n_errors"] + \
    load("volume_scramble", "volume_scramble_llama-160m.json")["n_errors"] + \
    (108 - A["JackFram/llama-160m"]["n"])
has("excluded LLaMA states", f"these {excl} states are excluded")
mism = max(load("scramble", f"gaussian_{t}.json")["summary"]["max_entropy_mismatch"]
           for t in ("SmolLM2-135M-Instruct_paper", "llama-160m_draw", "gpt2_draw", "pythia-160m_draw"))
assert mism < 1e-13
has("entropy matching", "to within 10⁻¹³ nats")
for tag, name in (("gpt2_draw", "GPT-2"), ("pythia-160m_draw", "Pythia-160M")):
    chk = load("scramble", f"consistent_{tag}.json")["summary"]["checks"][0]["identity_vs_real"]
    has(f"identity permutation, {name}", f"{sci(chk)} on {name}")

# ---- control validity (Table II, Section IV-D)
cv = load("scramble", "control_validity.json")["summary"]
gm = cv["gate_median_abs_dev"]
has("Gate A off h0, trained and C1", f"{sci(gm['x1_real'])}, {sci(gm['x1_cons'])}")
has("Gate A off h0, C0", f"{num(gm['x1_within_vs_cons'], 2)}–{num(gm['x1_within_vs_real'], 2)}, fails")
has("conditioning", f"{num(cv['cond_real'], 1)} against {num(cv['cond_scr'], 1)}, z = {num(cv['cond_scr_gt_real']['z'], 1)}, on {cv['n']} SmolLM2 states")

# ---- validation (Table II, Section V-A)
lad = {r["rung"]: r["worst"] for r in V["ladder_autodiff"]}
for rung in (1, 2, 3, 6):
    has(f"ladder rung {rung}", sci(lad[rung]))
has("worst analytic error", f"to {sci(max([lad[r] for r in (1, 2, 3, 6)] + [V['rung7']['sectional_worst']]))} or better")
has("rung 7", sci(V["rung7"]["sectional_worst"]))
has("reparameterisation", sci(V["rung7"]["reparam_worst"]))
has("broken assembly", f"fails, {num(V['rung7']['broken_assembly'], 3)}")
gs = V["geomstats"]
has("geomstats gamma/beta", sci(max(gs["gamma"]["max_diff"], gs["beta"]["max_diff"])))
assert gs["normal"]["agree"] == gs["normal"]["n"] and not gs["Poincare"]["agree"] and not gs["categorical"]["agree"]
has("Gate A, real model", sci(V["gate_a_real"]["rel_err"]))
nl = V["null"]
has("null directions", f"≤ {sci(max(r['kl_null'] for r in nl))} nats")
has("random step", f"{num(min(r['kl_rand'] for r in nl), 2)} to {num(max(r['kl_rand'] for r in nl), 1)} nats")
assert all(r["ok"] for r in nl) and len(nl) == 6
has("trace share", f"a median {num(V['trace_fraction']['5']['median'] * 100, 1)} % of the quotiented trace ({V['trace_fraction']['5']['n']} states)")
tr = load("stage0", "stage0.json")["truncation_error"]["500"]
has("truncation", f"truncation to 500 tokens changes the metric by {sci(tr)}")

sl = load("slice", "slice_gauss.json")["summary"]
mid_b = ("N512_k5_[0.5,1.0)", "N512_k5_[1.0,2.0)", "N512_k5_[2.0,4.0)")
eig = [sl[b]["eigen_abs_ext"] for b in mid_b]
rnd = [sl[b]["random_abs_ext"] for b in mid_b]
has("Gauss check, eigen", f"{num(min(eig), 3)} to {num(max(eig), 3)} between 0.5 and 4 nats")
has("Gauss check, Table II", f"{num(min(eig), 3)}–{num(max(eig), 3)}")
has("Gauss check, random", f"against {num(min(rnd), 2)} to {num(max(rnd), 2)} for random slices")
has("Gauss check, extremes", f"by {num(sl['N512_k5_[0.0,0.5)']['eigen_abs_ext'], 2)} and {num(sl['N512_k5_[4.0,inf)']['eigen_abs_ext'], 2)} below 0.5 and above 4 nats")
ek = [sl[b]["eigen_K"] for b in mid_b]
has("synthetic eigen K", f"give {num(min(ek), 2)} to {num(max(ek), 2)} between 0.5 and 4 nats")

ln = [load("layernorm", f)["summary"] for f in ("projection_gpt2.json", "projection_pythia-160m.json")]
ln_rows = [r for f in ("projection_gpt2.json", "projection_pythia-160m.json") for r in load("layernorm", f)["rows"]]
has("LayerNorm fix, largest", f"at most {sci(max(abs(r['K_projected'] - r['K_exact']) for r in ln_rows))}")
has("LayerNorm fix, median", f"(median {sci(max(s['median_abs_dK'] for s in ln), 0)})")

# ---- spectral (Table IV, Section V-C)
for m, short in zip(MODELS, ("SmolLM2", "LLaMA", "GPT-2", "Pythia")):
    t = S[m]
    has(f"{m} spectral row", f"| {short} ({t['n']}) | {num(t['log_volume_rho'], 3)} | {num(t['k_eff_rho'], 3)} | "
        f"{t['k_eff_min_layer']['real']} / {t['k_eff_min_layer']['within']} | "
        f"{num(t['k_eff_rho_entropy_real'], 2)} / {num(t['log_volume_rho_entropy_real'], 2)} |")
    assert t["log_volume_min_layer"]["real"] == t["log_volume_min_layer"]["within"]
ke = [S[m]["k_eff_rho"] for m in MODELS]
lv = [S[m]["log_volume_rho"] for m in MODELS]
rk = [S[m]["k_eff_range_retained"] for m in MODELS]
ent = [S[m]["k_eff_rho_entropy_real"] for m in MODELS]
has("k_eff rho range", f"{num(min(ke), 2)} to {num(max(ke), 2)} and {round(min(rk) * 100)} % to {round(max(rk) * 100)} %")
has("log-volume rho range", f"less tightly ({num(min(lv), 2)} to {num(max(lv), 2)})")
has("k_eff entropy range", f"(ρ = {num(min(ent), 2)} to {num(max(ent), 2)})")

# ---- instruments (Table V, Section V-D)
pairs, cor = T["agreement"]["pairs"], T["corpus"]
for key, label in (("intrinsic_K|intrinsic_R", "Intrinsic R"), ("intrinsic_K|pca", "Local-PCA residual"),
                   ("intrinsic_K|manson", "Frenet curvature"), ("intrinsic_K|king", "Euclidean angle")):
    p = pairs[key]
    has(f"Table V {label}", f"{signed(p['pooled'], 3)} / {signed(p['within'], 3)}")
for key in ("K", "R", "pca", "manson", "king"):
    r = cor["rho_with_entropy"][key]
    has(f"Table V entropy {key}", f"{signed(r['handwritten'], 3)} → {signed(r['wikitext'], 3)}")
three = [pairs[f"intrinsic_K|{k}"] for k in ("pca", "manson", "king")]
has("instrument pooled bound", f"|ρ| ≤ {num(max(abs(p['pooled']) for p in three), 3)}")
has("instrument within bound", f"≤ {num(max(abs(p['within']) for p in three), 3)} within layers")
has("K–R agreement", f"(ρ = {num(pairs['intrinsic_K|intrinsic_R']['pooled'], 2)})")
sh = cor["rho_with_entropy"]
has("corpus shifts, intrinsic", f"shifts by {num(sh['K']['shift'], 3)} and {num(sh['R']['shift'], 3)}")
prox = [sh[k]["shift"] for k in ("pca", "manson", "king")]
has("corpus shifts, instruments", f"against {num(min(prox), 3)} to {num(max(prox), 3)}")
has("corpus sizes", f"{cor['n']['handwritten']} hand-written and {cor['n']['wikitext']} WikiText-103 states")
has("E2 sample", f"on {T['agreement']['n']} SmolLM2 states")

cal = T["calibration"]
mab = load("mabrok", "mabrok.json")["results"]["L12_k200"]["var0.9999"]
has("near-flat value", f"reaches {sci(mab)} at a retained-variance threshold of 0.9999 with 200 neighbours")
has("sphere at 0.999", f"as {sci(cal['sphere'][3])}")

# ---- reproducibility facts
has("table checks", f"checks {len(T['checks'])} stored summaries")
assert all(c["ok"] for c in T["checks"])

print(f"{N[0]} checks against the saved results; {len(FAILS)} failed")
for f in FAILS:
    print("  FAIL", f)
sys.exit(1 if FAILS else 0)
