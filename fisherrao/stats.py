"""Rank statistics, with the tie handling done correctly.

WHY THIS MODULE EXISTS
----------------------
Every script in this project had its own copy of

    rx = torch.argsort(torch.argsort(x))

as a "rank".  That is a *permutation*, not a rank: tied values get distinct
consecutive integers, assigned in whatever order `argsort` happens to visit
them, which for a stable sort is **file order**.

For continuous quantities (`K`, entropy, log-volume) there are no ties in
float64 and the two agree exactly.  For DISCRETE quantities the difference is
not small, and it is not random either -- it imports whatever the row ordering
happens to encode.  In this project the point files are grouped by model and
then by layer, and `k_eff` varies strongly with layer, so the tie-breaking
leaked the layer ordering straight into the correlation:

    rho(n_planes_neg, k_eff_99), n = 456, 407 of 456 values tied at zero
        argsort-argsort : +0.419      <- reported as "+0.42"
        tie-corrected   : +0.118
        rho(file index, k_eff_99) = +0.510   <- the channel it leaked through

`spearman` here uses midranks, which is the standard definition, and
`median` returns the true median rather than `torch.median`'s lower-middle
element (which for even n is a different statistic and moved a reported
profile correlation from +0.867 to +0.967).

Both functions are NaN-safe by pairwise deletion, which is what the scripts
that they replace did.
"""

from __future__ import annotations

import math

__all__ = ["midranks", "spearman", "median", "mannwhitney"]


def midranks(v: list[float]) -> list[float]:
    """Ranks 1..n, with tied values sharing the average of the ranks they span."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r


def spearman(a, b, min_n: int = 3) -> float:
    """Spearman rank correlation with midrank tie correction.

    Pairs where either value is NaN are dropped (pairwise deletion).
    Returns NaN if fewer than `min_n` usable pairs remain, or if either
    variable is constant.
    """
    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x == x and y == y]
    if len(pairs) < min_n:
        return float("nan")
    ra = midranks([p[0] for p in pairs])
    rb = midranks([p[1] for p in pairs])
    n = len(pairs)
    ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va <= 0 or vb <= 0:                    # a constant variable has no ranking
        return float("nan")
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / math.sqrt(va * vb)


def median(xs) -> float:
    """True median: the average of the two middle values when n is even.

    `torch.median` returns the LOWER middle element, which is a different
    statistic and is not what any of the write-ups meant by "median".
    """
    v = sorted(float(x) for x in xs if x == x)
    n = len(v)
    if n == 0:
        return float("nan")
    m = n // 2
    return v[m] if n % 2 else 0.5 * (v[m - 1] + v[m])


def mannwhitney(a, b) -> tuple[float, float, float]:
    """Mann-Whitney U, tie-corrected normal z, and AUC (common-language effect).

    The right test when one variable is a binary grouping and the other is
    continuous -- which is the situation Spearman was being misused for on
    `n_planes_neg`, a variable that is zero at 407 of 456 points.
    """
    a = [float(x) for x in a if x == x]
    b = [float(x) for x in b if x == x]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    comb = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    r = midranks([v for v, _ in comb])
    R1 = sum(r[i] for i in range(len(comb)) if comb[i][1] == 0)
    U = R1 - n1 * (n1 + 1) / 2.0
    N = n1 + n2
    counts: dict[float, int] = {}
    for v, _ in comb:
        counts[v] = counts.get(v, 0) + 1
    tie = sum(c ** 3 - c for c in counts.values())
    sd = math.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1.0))))
    z = (U - n1 * n2 / 2.0) / sd if sd > 0 else float("nan")
    return U, z, U / (n1 * n2)
