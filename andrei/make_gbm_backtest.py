"""
Builds gbm_backtest.py from your ensemble_blend.py.

Keeps everything in ensemble_blend.py up to the blend grid (data loading, cost
model, age_hazard, schedule, merge_clusters, ...) and replaces the rest with a
backtest of the new pipe-year GBM, judged by the same walk-forward and rule.

Usage:  python make_gbm_backtest.py      then      python gbm_backtest.py
"""
SRC, OUT, MARKER = "ensemble_blend.py", "gbm_backtest.py", "\nGRID = ["

BODY = r'''
# ===================================================================
# Pipe-year GBM backtest (built by make_gbm_backtest.py)
# ===================================================================
from hazard_gbm import gbm_probability
from depletion import depletion_multiplier

# Fixed in advance. Every candidate uses the same leak-cost model, so they differ
# only in the probability of a leak within 2 years.
CANDIDATES = ["age", "age x depl", "gbm", "gbm + age x depl"]
BASE = "age x depl"                                  # = the $76.44M submission


def _norm(v):
    pos = v[v > 0]
    return v / pos.mean() if pos.size else v


_MEMBERS = {}


def members(T, start):
    """Each candidate's predicted leak $ per pipe, using leaks up to end of T only."""
    if (T, start) not in _MEMBERS:
        cost = fit_cost(lk[lk["year"] <= T])(pipes)
        a = age_hazard(last_year=T)([start, start + 1])
        age_term = (1 - (1 - a[:, 0]) * (1 - a[:, 1])) * cost
        age_depl = age_term * depletion_multiplier(before_year=start)
        gbm = gbm_probability(before_year=start, horizon=2) * cost
        assert len(gbm) == len(age_term), "pipes.csv row order doesn't match the pipeline's"
        _MEMBERS[(T, start)] = {"age": age_term, "age x depl": age_depl, "gbm": gbm,
                                "gbm + age x depl": 0.5 * _norm(gbm) + 0.5 * _norm(age_depl)}
    return _MEMBERS[(T, start)]


# ---------------------------------------------------------------
# 1. How differently do they rank? (2025, fit on leaks to 2024)
# ---------------------------------------------------------------
print("\n=== Step 1: rank correlation of the candidates (2025) ===")
m25 = members(2024, 2025)
r25 = at_risk(2025)
print(pd.DataFrame({k: m25[k][r25] for k in CANDIDATES}).rank().corr().round(2).to_string())

actual = np.where(leak_year == 2025, pipe_dollars, 0.0)[r25]
print("\nLeak $ caught by each candidate's riskiest 10% (2025):")
for k in CANDIDATES:
    top = np.argsort(-m25[k][r25])[: r25.sum() // 10]
    print(f"  {k:18s} {100 * actual[top].sum() / actual.sum():.1f}%")


# ---------------------------------------------------------------
# 2. Walk-forward on real leaks (same as ensemble_blend.py)
# ---------------------------------------------------------------
print("\n=== Step 2: walk-forward on real leaks ===")
res = []
for start in [2023, 2024, 2025]:
    T, end = start - 1, 2026
    alive = at_risk(start)
    circles = []
    for i in np.flatnonzero(must_cover & alive):
        cx, cy = (P1[i] + P2[i]) / 2
        r = length[i] / 2 + MARGIN
        circles.append({"r": r, "idx": inside(cx, cy, r), "own": np.array([i])})
    m = members(T, start)
    for k in CANDIDATES:
        _, dig, _ = schedule(circles, alive, None, static_value=np.where(alive, m[k], 0.0),
                             max_years=end - start + 1)
        got, total = leak_cost_real(dig, start, end)
        res.append({"candidate": k, "window": f"{start}-{end}", "pct_prevented": 100 * (1 - got / total)})
        print(f"  {start}-{end}  {k:18s} prevented {res[-1]['pct_prevented']:.1f}% of leak $")

table = pd.DataFrame(res).pivot(index="candidate", columns="window", values="pct_prevented")
windows = list(table.columns)
table["mean"] = table[windows].mean(axis=1)
for w in windows + ["mean"]:
    table[f"vs_best_{w}"] = table[w] - table.loc[BASE, w]
table = table.reindex(CANDIDATES)
table.to_csv("gbm_backtest.csv")
print("\n" + table.round(2).to_string())


# ---------------------------------------------------------------
# 3. Same rule as always, then a submission file for each candidate that passes
# ---------------------------------------------------------------
KEY = "2025-2026"
passing = [k for k in CANDIDATES if k not in (BASE, "age")       # "age" is reference only
           and table.loc[k, f"vs_best_{KEY}"] > 0 and table.loc[k, "vs_best_mean"] > 0]
passing.sort(key=lambda k: -table.loc[k, "mean"])
if not passing:
    print("\nNothing beat age x depl in 2025-2026 and on average: keep the $76.44M plan.")
else:
    print("\nPassed the rule (best first): " + "; ".join(
        f"{k} (+{table.loc[k, 'vs_best_mean']:.2f} pts mean)" for k in passing))
    mem, circ, merges = merge_clusters(45, 50)
    plan_circles = [{"x": c[0], "y": c[1], "r": c[2], "idx": c[3], "own": np.array(mem[k])}
                    for k, c in circ.items()]
    m_now = members(2026, 2027)
    for k in passing:
        order, dig, n_left = schedule(plan_circles, live, None,
                                      static_value=np.where(live, m_now[k], 0.0))
        covered = np.zeros(N, bool)
        for c in order:
            covered[plan_circles[c]["idx"]] = True
        assert not seed[~covered[seed]].size, f"{k}: iron pipes not covered"
        path = "submission_" + k.replace(" + ", "_plus_").replace(" x ", "_x_").replace(" ", "_") + ".csv"
        with open(path, "w") as f:
            for c in order:
                pc = plan_circles[c]
                f.write(f"\"({pc['x']:.6f}, {pc['y']:.6f})\",{pc['r']:.6f}\n")
        print(f"  Saved {path}  ({len(order)} circles, coverage check passed)")
'''

src = open(SRC).read()
if MARKER not in src:
    raise SystemExit(f"Couldn't find '{MARKER.strip()}' in {SRC}. Is this the blend script?")
open(OUT, "w").write(src[: src.index(MARKER)] + BODY)
print(f"Wrote {OUT}. Next: python hazard_gbm.py (quick check), then python {OUT}")
