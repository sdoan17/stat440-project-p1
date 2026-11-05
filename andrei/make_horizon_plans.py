"""
Builds horizon_plans.py from your ensemble_blend.py: writes a submission for each
horizon in HORIZONS, using exactly the same pipeline as submission_4yr.csv
(age curve x depletion x leak cost, merged circles, $10M a year).

Fractional horizons give the last year partial weight:
  4.5 years = P(leak in 2027-2030) plus half-weight exposure in 2031.

Check: it also rebuilds the 4-year plan and compares it with your submission_4yr.csv,
so you know the new files differ from your best ONLY in the horizon.

Usage:  python make_horizon_plans.py      then      python horizon_plans.py
"""
SRC, OUT, MARKER = "ensemble_blend.py", "horizon_plans.py", "\nGRID = ["

BODY = r'''
# ===================================================================
# Horizon plans (built by make_horizon_plans.py)
# ===================================================================
import os
from depletion import depletion_multiplier

HORIZONS = [4.5, 6]
CHECK = 4                       # rebuilt and compared with submission_4yr.csv

T, START = 2026, 2027
cost = fit_cost(lk[lk["year"] <= T])(pipes)
A = age_hazard(last_year=T)([START + k for k in range(7)])
risk_cost = depletion_multiplier(before_year=START) * cost

print("Building merged circles (takes a few minutes, no output until done)...")
mem, circ, merges = merge_clusters(45, 50)
plan_circles = [{"x": c[0], "y": c[1], "r": c[2], "idx": c[3], "own": np.array(mem[k])}
                for k, c in circ.items()]


def horizon_prob(H):
    full, frac = int(H), H - int(H)
    surv = np.prod(1 - A[:, :full], axis=1)
    if frac:
        surv = surv * (1 - A[:, full]) ** frac       # partial exposure in the last year
    return 1 - surv


def plan_lines(H):
    order, dig, n_left = schedule(plan_circles, live, None,
                                  static_value=np.where(live, horizon_prob(H) * risk_cost, 0.0))
    covered = np.zeros(N, bool)
    for c in order:
        covered[plan_circles[c]["idx"]] = True
    assert not seed[~covered[seed]].size, f"{H}-year plan: iron pipes not covered"
    return [f"\"({plan_circles[c]['x']:.6f}, {plan_circles[c]['y']:.6f})\",{plan_circles[c]['r']:.6f}\n"
            for c in order]


if os.path.exists("submission_4yr.csv"):
    mine = plan_lines(CHECK)
    theirs = open("submission_4yr.csv").readlines()
    same = sum(a == b for a, b in zip(mine, theirs))
    print(f"Check: rebuilt 4-year plan matches submission_4yr.csv on {same:,} of {len(theirs):,} lines"
          + ("  (identical)" if same == len(theirs) == len(mine) else "  (NOT identical: tell Claude)"))

for H in HORIZONS:
    lines = plan_lines(H)
    path = f"submission_{str(H).replace('.', 'p')}yr.csv"
    open(path, "w").writelines(lines)
    print(f"Saved {path}  ({len(lines):,} circles, coverage check passed)")
'''

src = open(SRC).read()
if MARKER not in src:
    raise SystemExit(f"Couldn't find '{MARKER.strip()}' in {SRC}. Is this the blend script?")
open(OUT, "w").write(src[: src.index(MARKER)] + BODY)
print(f"Wrote {OUT}. Next: python {OUT}")
