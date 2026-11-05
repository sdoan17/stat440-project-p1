"""
Builds hybrid_plans.py from your ensemble_blend.py: "lottery ticket" plans that keep
the $77.67M plan's exact dig order for the first K years, then re-rank the remaining
circles every year (rolling, 4-year window) for the rest of the plan.

Why: full rolling plans cut leak costs by up to $2.8M (mostly 2030, 2031 and 2036),
but construction rose by $3.7-6.1M for reasons we can't model. Keeping the early
order unchanged leaves the start of the plan exactly as in the best submission.

Tickets
  hybrid_roll_y5      4.5-year order for years 1-4, then rolling (original age curve)
  hybrid_cohroll_y5   4.5-year order for years 1-4, then rolling (cohort-shifted curve)
  hybrid_cohroll_y3   4.5-year order for years 1-2, then rolling (cohort-shifted curve)

No backtest: our windows are at most 4 years long, so they can't see anything after
the switch. These are leaderboard tickets; only your best private score counts.

Usage:  python make_hybrid_plans.py      then      python hybrid_plans.py
"""
SRC, OUT, MARKER = "ensemble_blend.py", "hybrid_plans.py", "\nGRID = ["

BODY = r'''
# ===================================================================
# Hybrid plans (built by make_hybrid_plans.py)
# ===================================================================
import os
from depletion import depletion_multiplier

TICKETS = [("hybrid_roll_y5", 4, "H0"), ("hybrid_cohroll_y5", 4, "H1"), ("hybrid_cohroll_y3", 2, "H1")]
IRON_NAMES = ["cast iron", "gray iron", "wrought iron"]
SHIFTS = np.arange(-6, 7)
MIN_LEAKS = 30
FIRST_HIST = 2019
W = 4
YEARS = 25

_mat = pipes["Material"].fillna("").str.strip().str.lower()
IRON_ALL = _mat.isin(IRON_NAMES).to_numpy()
LAY = pd.to_datetime(pipes["Lay date"], errors="coerce").dt.year.to_numpy(float)
COHORT = np.where(np.isnan(LAY), -1, (np.nan_to_num(LAY) // 5) * 5).astype(int)
LY = np.nan_to_num(np.asarray(leak_year, float), nan=9999)


def hazard_block(T, y0, y1):
    A = np.asarray(age_hazard(last_year=T)(list(range(y0, y1 + 1))), float)
    assert A.shape == (N, y1 - y0 + 1)
    return A


def fit_shifts(T):
    lo = FIRST_HIST + SHIFTS.min()
    A = hazard_block(T, lo, T + SHIFTS.max())
    size = COHORT.max() + 1 if COHORT.max() > 0 else 1
    O = np.zeros(size)
    score = np.zeros((len(SHIFTS), size))
    for si, d in enumerate(SHIFTS):
        E = np.zeros(size); LL = np.zeros(size)
        for y in range(FIRST_HIST, T + 1):
            m = IRON_ALL & (COHORT >= 0) & (LAY <= y) & (LY >= y)
            h = np.clip(A[:, y + d - lo], 1e-6, 0.999)
            E += np.bincount(COHORT[m], h[m], minlength=size)
            leak = m & (LY == y)
            LL += np.bincount(COHORT[leak], np.log(h[leak]), minlength=size)
            if si == 0:
                O += np.bincount(COHORT[leak], minlength=size)
        score[si] = LL - O * np.log(np.maximum(E, 1e-12))
    best = np.where(O >= MIN_LEAKS, SHIFTS[np.argmax(score, axis=0)], 0)
    return np.where(COHORT >= 0, best[np.maximum(COHORT, 0)], 0).astype(int)


def plan_hazard(T, start, n, delta):
    lo = start + SHIFTS.min()
    A = hazard_block(T, lo, start + n - 1 + SHIFTS.max())
    cols = (start - lo) + np.arange(n)[None, :] + delta[:, None]
    return np.take_along_axis(A, cols, axis=1)


T, START = 2026, 2027
cost = fit_cost(lk[lk["year"] <= T])(pipes)
rc = depletion_multiplier(before_year=START) * cost
H = {"H0": plan_hazard(T, START, YEARS + W, np.zeros(N, int)),
     "H1": plan_hazard(T, START, YEARS + W, fit_shifts(T))}
p45 = 1 - np.prod(1 - H["H0"][:, :4], axis=1) * (1 - H["H0"][:, 4]) ** 0.5
base_value = np.where(live, p45 * rc, 0.0)

print("Building merged circles (a few minutes, silent)...")
mem, circ, merges = merge_clusters(45, 50)
plan_circles = [{"x": c[0], "y": c[1], "r": c[2], "idx": c[3], "own": np.array(mem[k])}
                for k, c in circ.items()]


def line(c):
    pc = plan_circles[c]
    return f"\"({pc['x']:.6f}, {pc['y']:.6f})\",{pc['r']:.6f}\n"


best_lines = open("submission_4p5yr.csv").readlines() if os.path.exists("submission_4p5yr.csv") else None

for name, K, hz in TICKETS:
    # 1. the best plan's first K years, unchanged
    res, dig, n_left = schedule(plan_circles, live, None, static_value=base_value, max_years=K)
    prefix = list(res[: len(res) - n_left])
    if best_lines is not None:
        same = sum(a == b for a, b in zip([line(c) for c in prefix], best_lines))
        print(f"{name}: first {len(prefix):,} circles (years 1-{K}) match submission_4p5yr.csv on {same:,}"
              + ("  (identical)" if same == len(prefix) else "  (NOT identical: tell Claude)"))
    # 2. re-rank the rest every year, starting from the state after year K
    in_prefix = set(prefix)
    rest = [c for c in range(len(plan_circles)) if c not in in_prefix]
    alive_rest = live & (dig >= 10 ** 6)
    sub = [plan_circles[c] for c in rest]
    order2, _, _ = schedule(sub, alive_rest, np.where(alive_rest, rc, 0.0), H=H[hz][:, K:], window=W,
                            max_years=YEARS - K)
    order = prefix + [rest[j] for j in order2]
    covered = np.zeros(N, bool)
    for c in order:
        covered[plan_circles[c]["idx"]] = True
    assert not seed[~covered[seed]].size, f"{name}: iron pipes not covered"
    path = f"submission_{name}.csv"
    open(path, "w").writelines(line(c) for c in order)
    print(f"  Saved {path}  ({len(order):,} circles, coverage check passed)")
'''

src = open(SRC).read()
if MARKER not in src:
    raise SystemExit(f"Couldn't find '{MARKER.strip()}' in {SRC}. Is this the blend script?")
open(OUT, "w").write(src[: src.index(MARKER)] + BODY)
print(f"Wrote {OUT}. Next: python {OUT}")
