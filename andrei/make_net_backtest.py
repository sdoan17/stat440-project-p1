"""
Builds net_backtest.py from your ensemble_blend.py.

What the grader showed us: every pipe is either dug OR repaired after a leak, never
both (REPLACE + REPAIR = 26,449 in every submission). So a leak costs its repair
but saves that pipe's digging (~$200/m). The value of digging a pipe early is
    P(leak before we'd dig it) x (leak cost - 200 x length)
not P(leak) x leak cost, which is what the $76.44M plan ranks by.

Candidates (fixed in advance):
  age x depl      the $76.44M plan: 2-year leak probability x depletion x leak cost
  net             same, but x (leak cost - 200 x length): long pipes with cheap leaks go later
  4yr             4-year horizon (2027-2030), to get ahead of the 2030 cohort burst
  4yr net         both

Scoring: the walk-forward with same-year credit (a dig in year k prevents leaks
from year k on), measured in TOTAL cost = leak $ - dig $ those leaks save, because
that's what the grader charges. Lower is better. Rule as always: beat the base in
2025-2026 AND on the mean of the three windows. Note: 2-year windows can't see a
4-year horizon's payoff; 2023-2026 (burst in year 4) is its real test.

Usage:  python make_net_backtest.py      then      python net_backtest.py
"""
SRC, OUT, MARKER = "ensemble_blend.py", "net_backtest.py", "\nGRID = ["

BODY = r'''
# ===================================================================
# Net-value backtest (built by make_net_backtest.py)
# ===================================================================
from depletion import depletion_multiplier

CANDIDATES = ["age x depl", "net", "4yr", "4yr net"]
BASE = "age x depl"                                  # = the $76.44M submission
SAVE = 200 * length * must_cover                     # dig $ a leak saves: that pipe's metres
DOLLARS = pipes["Pipe ID"].map(lk.groupby("Pipe ID")["Cost"].sum()).fillna(0).values


def window_costs(dig_year, start, end):
    """Same-year credit: a dig in year k (0 = start) prevents leaks from year k on."""
    window = (~np.isnan(leak_year)) & (leak_year >= start) & (leak_year <= end)
    hit = window & ((leak_year - start) < dig_year)          # leaked before we dug it
    leak_dollars = DOLLARS[hit].sum()
    saved_digging = SAVE[hit].sum()
    return leak_dollars, DOLLARS[window].sum(), leak_dollars - saved_digging


_MEMBERS = {}


def members(T, start):
    """Each candidate's priority per pipe, using leaks up to end of T only."""
    if (T, start) not in _MEMBERS:
        cost = fit_cost(lk[lk["year"] <= T])(pipes)
        a = age_hazard(last_year=T)([start + k for k in range(4)])
        p2 = 1 - (1 - a[:, 0]) * (1 - a[:, 1])
        p4 = 1 - np.prod(1 - a, axis=1)
        depl = depletion_multiplier(before_year=start)
        assert len(depl) == len(cost), "pipes.csv row order doesn't match the pipeline's"
        _MEMBERS[(T, start)] = {
            "age x depl": p2 * depl * cost,
            "net": p2 * depl * (cost - SAVE),
            "4yr": p4 * depl * cost,
            "4yr net": p4 * depl * (cost - SAVE),
        }
    return _MEMBERS[(T, start)]


# ---------------------------------------------------------------
# 0. How much does netting change things?
# ---------------------------------------------------------------
m = members(2024, 2025)
r = at_risk(2025) & must_cover
c = fit_cost(lk[lk["year"] <= 2024])(pipes)
neg = r & (c < SAVE)
print("\n=== Step 0: pipes where a leak is cheaper than digging (2025) ===")
print(f"  {int(neg.sum()):,} of {int(r.sum()):,} pipes to replace ({100 * neg.mean() / max(r.mean(), 1e-9):.1f}%) "
      f"have expected leak cost below their $200/m dig saving; 'net' puts them last.")
print(pd.DataFrame({k: m[k][r] for k in CANDIDATES}).rank().corr().round(2).to_string())


# ---------------------------------------------------------------
# 1. Walk-forward
# ---------------------------------------------------------------
print("\n=== Step 1: walk-forward, total cost = leak $ - digging saved by those leaks ===")
res = []
for start in [2023, 2024, 2025]:
    T, end = start - 1, 2026
    alive = at_risk(start)
    circles = []
    for i in np.flatnonzero(must_cover & alive):
        cx, cy = (P1[i] + P2[i]) / 2
        rr = length[i] / 2 + MARGIN
        circles.append({"r": rr, "idx": inside(cx, cy, rr), "own": np.array([i])})
    mm = members(T, start)
    for k in CANDIDATES:
        _, dig, _ = schedule(circles, alive, None, static_value=np.where(alive, mm[k], 0.0),
                             max_years=end - start + 1)
        got, total, net = window_costs(dig, start, end)
        res.append({"candidate": k, "window": f"{start}-{end}", "net $M": net / 1e6,
                    "leak % prevented": 100 * (1 - got / total)})
        print(f"  {start}-{end}  {k:12s} total cost ${net / 1e6:7.2f}M   "
              f"(leak $ prevented {res[-1]['leak % prevented']:.1f}%)")

df = pd.DataFrame(res)
net = df.pivot(index="candidate", columns="window", values="net $M").reindex(CANDIDATES)
windows = list(net.columns)
net["mean"] = net[windows].mean(axis=1)
gain = net.loc[BASE] - net                    # $M saved vs the base (positive = better)
gain.columns = [f"saves_vs_base_{w}" for w in gain.columns]
out = pd.concat([net, gain], axis=1)
out.to_csv("net_backtest.csv")
print("\nTotal cost by window ($M, lower is better) and $M saved vs the $76.44M method:")
print(out.round(2).to_string())
pct = df.pivot(index="candidate", columns="window", values="leak % prevented").reindex(CANDIDATES)
print("\n(For reference, leak $ prevented %:)")
print(pct.round(1).to_string())


# ---------------------------------------------------------------
# 2. Rule, then a submission file for each candidate that passes
# ---------------------------------------------------------------
KEY = "2025-2026"
passing = [k for k in CANDIDATES if k != BASE
           and out.loc[k, f"saves_vs_base_{KEY}"] > 0 and out.loc[k, "saves_vs_base_mean"] > 0]
passing.sort(key=lambda k: -out.loc[k, "saves_vs_base_mean"])
if not passing:
    print("\nNothing beat the $76.44M method in 2025-2026 and on average: keep it.")
else:
    print("\nPassed the rule (best first): " + "; ".join(
        f"{k} (saves ${out.loc[k, 'saves_vs_base_mean']:.2f}M a window on average)" for k in passing))
    mem, circ, merges = merge_clusters(45, 50)
    plan_circles = [{"x": cc[0], "y": cc[1], "r": cc[2], "idx": cc[3], "own": np.array(mem[kk])}
                    for kk, cc in circ.items()]
    m_now = members(2026, 2027)
    for k in passing:
        order, dig, n_left = schedule(plan_circles, live, None,
                                      static_value=np.where(live, m_now[k], 0.0))
        covered = np.zeros(N, bool)
        for cc in order:
            covered[plan_circles[cc]["idx"]] = True
        assert not seed[~covered[seed]].size, f"{k}: iron pipes not covered"
        path = "submission_" + k.replace(" ", "_") + ".csv"
        with open(path, "w") as f:
            for cc in order:
                pc = plan_circles[cc]
                f.write(f"\"({pc['x']:.6f}, {pc['y']:.6f})\",{pc['r']:.6f}\n")
        print(f"  Saved {path}  ({len(order)} circles, coverage check passed)")
'''

src = open(SRC).read()
if MARKER not in src:
    raise SystemExit(f"Couldn't find '{MARKER.strip()}' in {SRC}. Is this the blend script?")
open(OUT, "w").write(src[: src.index(MARKER)] + BODY)
print(f"Wrote {OUT}. Next: python {OUT}")
