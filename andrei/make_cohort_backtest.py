"""
Builds cohort_backtest.py from your ensemble_blend.py.

Two fixes, tested together and apart (fixed in advance):

  COHORT TIMING. The pooled age curve runs late for newer cohorts (2030 leaks kept
  falling as the horizon reached into 2031-32). Each 5-year laying cohort gets an
  age shift (-6..+6 years) chosen by likelihood to best match WHEN its leaks
  happened, using only leaks before the plan starts. A pipe's risk is then read at
  its effective age (age + shift). Only timing is shifted, not the risk level:
  a cohort that just burst would otherwise look "fragile" while its weak pipes are
  already used up (depletion).

  ROLLING SCHEDULE. Instead of ranking every circle once by 2027-2031 risk, re-rank
  the remaining circles each year by risk over the NEXT 4 years from that year
  (your schedule() already supports this with H= and window=).

Candidates
  4.5yr               the $77.67M plan (base)
  cohort 4yr          cohort-shifted hazard, 4-year horizon, ranked once
  rolling 4yr         original hazard, re-ranked every year over the next 4 years
  cohort rolling 4yr  both

Rule: beat the base in 2023-2026 (the burst rehearsal: plan made in 2023, burst in
year 4, like the real plan and 2030) AND on the mean of the three windows.
The script writes a submission for every candidate anyway (circle building is slow),
and says which ones passed.

Usage:  python make_cohort_backtest.py      then      python cohort_backtest.py
"""
SRC, OUT, MARKER = "ensemble_blend.py", "cohort_backtest.py", "\nGRID = ["

BODY = r'''
# ===================================================================
# Cohort-timed hazard + rolling schedule (built by make_cohort_backtest.py)
# ===================================================================
import os
from depletion import depletion_multiplier

IRON_NAMES = ["cast iron", "gray iron", "wrought iron"]
SHIFTS = np.arange(-6, 7)          # years a cohort may fail earlier (+) or later (-) than the pooled curve
MIN_LEAKS = 30                     # cohorts with fewer leaks keep shift 0
FIRST_HIST = 2019
W = 4                              # horizon for the new candidates
CANDIDATES = ["4.5yr", "cohort 4yr", "rolling 4yr", "cohort rolling 4yr"]
BASE, KEY = "4.5yr", "2023-2026"

_mat = pipes["Material"].fillna("").str.strip().str.lower()
IRON_ALL = _mat.isin(IRON_NAMES).to_numpy()
LAY = pd.to_datetime(pipes["Lay date"], errors="coerce").dt.year.to_numpy(float)
COHORT = np.where(np.isnan(LAY), -1, (np.nan_to_num(LAY) // 5) * 5).astype(int)
LY = np.nan_to_num(np.asarray(leak_year, float), nan=9999)
DOLLARS = pipes["Pipe ID"].map(lk.groupby("Pipe ID")["Cost"].sum()).fillna(0).values


def hazard_block(T, y0, y1):
    """Age-curve hazard fitted on leaks up to T, for calendar years y0..y1 (pipes x years)."""
    A = np.asarray(age_hazard(last_year=T)(list(range(y0, y1 + 1))), float)
    assert A.shape == (N, y1 - y0 + 1)
    return A


def fit_shifts(T):
    """Per-cohort age shift by profile likelihood on leaks 2019..T (risk level left free)."""
    lo = FIRST_HIST + SHIFTS.min()
    A = hazard_block(T, lo, T + SHIFTS.max())
    size = COHORT.max() + 1 if COHORT.max() > 0 else 1
    O = np.zeros(size)
    score = np.zeros((len(SHIFTS), size))
    for si, d in enumerate(SHIFTS):
        E = np.zeros(size)
        LL = np.zeros(size)
        for y in range(FIRST_HIST, T + 1):
            m = IRON_ALL & (COHORT >= 0) & (LAY <= y) & (LY >= y)
            h = np.clip(A[:, y + d - lo], 1e-6, 0.999)
            E += np.bincount(COHORT[m], h[m], minlength=size)
            leak = m & (LY == y)
            LL += np.bincount(COHORT[leak], np.log(h[leak]), minlength=size)
            if si == 0:
                O += np.bincount(COHORT[leak], minlength=size)
        score[si] = LL - O * np.log(np.maximum(E, 1e-12))
    best = SHIFTS[np.argmax(score, axis=0)]
    best = np.where(O >= MIN_LEAKS, best, 0)
    zero = int(np.flatnonzero(SHIFTS == 0)[0])
    rows = []
    for c in np.unique(COHORT[COHORT >= 0]):
        rows.append({"cohort": f"{c}-{c + 4}", "iron pipes": int(((COHORT == c) & IRON_ALL).sum()),
                     f"leaks {FIRST_HIST}-{T}": int(O[c]), "shift (yrs)": int(best[c]),
                     "fit gain vs 0": round(float(score[np.argmax(score[:, c]), c] - score[zero, c]), 1)})
    delta = np.where(COHORT >= 0, best[np.maximum(COHORT, 0)], 0).astype(int)
    tab = pd.DataFrame(rows).set_index("cohort")
    edge = tab[(tab["shift (yrs)"].abs() == SHIFTS.max()) & (tab[f"leaks {FIRST_HIST}-{T}"] >= MIN_LEAKS)]
    if len(edge):
        print(f"  WARNING: {len(edge)} cohort(s) hit the +/-{SHIFTS.max()}-year limit: " + ", ".join(edge.index)
              + ". Treat the cohort candidates with suspicion.")
    return delta, tab


def plan_hazard(T, start, n, delta):
    """Hazard for plan years start..start+n-1, read at each pipe's effective age."""
    lo = start + SHIFTS.min()
    A = hazard_block(T, lo, start + n - 1 + SHIFTS.max())
    cols = (start - lo) + np.arange(n)[None, :] + delta[:, None]
    return np.take_along_axis(A, cols, axis=1)


_CACHE = {}


def build(T, start, n_years):
    if (T, start) not in _CACHE:
        cost = fit_cost(lk[lk["year"] <= T])(pipes)
        rc = depletion_multiplier(before_year=start) * cost
        delta, table = fit_shifts(T)
        n = n_years + W
        H0 = plan_hazard(T, start, n, np.zeros(N, int))
        H1 = plan_hazard(T, start, n, delta)
        p45 = 1 - np.prod(1 - H0[:, :4], axis=1) * (1 - H0[:, 4]) ** 0.5
        p4c = 1 - np.prod(1 - H1[:, :W], axis=1)
        _CACHE[(T, start)] = {"rc": rc, "H0": H0, "H1": H1, "table": table,
                              "static": {"4.5yr": p45 * rc, "cohort 4yr": p4c * rc}}
    return _CACHE[(T, start)]


def run(name, circles, alive, b, max_years):
    if name in b["static"]:
        return schedule(circles, alive, None, static_value=np.where(alive, b["static"][name], 0.0),
                        max_years=max_years)
    H = b["H1"] if name.startswith("cohort") else b["H0"]
    return schedule(circles, alive, np.where(alive, b["rc"], 0.0), H=H, window=W, max_years=max_years)


# ---------------------------------------------------------------
# 1. The shifts themselves
# ---------------------------------------------------------------
print("\n=== Step 1: cohort age shifts (+ = fails earlier than the pooled age curve) ===")
for T in [2022, 2026]:
    _, tab = fit_shifts(T)
    print(f"\nFitted on leaks {FIRST_HIST}-{T}:")
    print(tab.to_string())

# ---------------------------------------------------------------
# 2. Walk-forward (same-year credit)
# ---------------------------------------------------------------
print("\n=== Step 2: walk-forward, % of real leak $ prevented ===")
res, per_year = [], {}
for start in [2023, 2024, 2025]:
    T, end = start - 1, 2026
    ny = end - start + 1
    alive = at_risk(start)
    circles = []
    for i in np.flatnonzero(must_cover & alive):
        cx, cy = (P1[i] + P2[i]) / 2
        rr = length[i] / 2 + MARGIN
        circles.append({"r": rr, "idx": inside(cx, cy, rr), "own": np.array([i])})
    b = build(T, start, ny)
    window = (LY >= start) & (LY <= end)
    for k in CANDIDATES:
        _, dig, _ = run(k, circles, alive, b, ny)
        hit = window & ((LY - start) < dig)                      # leaked before we dug it
        res.append({"candidate": k, "window": f"{start}-{end}",
                    "prevented %": 100 * (1 - DOLLARS[hit].sum() / DOLLARS[window].sum())})
        if start == 2023:
            per_year[k] = [DOLLARS[hit & (LY == y)].sum() / 1e6 for y in range(start, end + 1)]
        print(f"  {start}-{end}  {k:20s} prevented {res[-1]['prevented %']:.1f}%")

t = pd.DataFrame(res).pivot(index="candidate", columns="window", values="prevented %").reindex(CANDIDATES)
wins = list(t.columns)
t["mean"] = t[wins].mean(axis=1)
for w in wins + ["mean"]:
    t[f"vs_base_{w}"] = t[w] - t.loc[BASE, w]
t.to_csv("cohort_backtest.csv")
print("\n" + t.round(2).to_string())
print("\nLeak $M NOT prevented by year, 2023-2026 window (2026 = the burst, year 4):")
print(pd.DataFrame(per_year, index=[str(y) for y in range(2023, 2027)]).T.reindex(CANDIDATES).round(2).to_string())

passing = [k for k in CANDIDATES if k != BASE
           and t.loc[k, f"vs_base_{KEY}"] > 0 and t.loc[k, "vs_base_mean"] > 0]
print("\nPassed the rule: " + (", ".join(f"{k} (+{t.loc[k, 'vs_base_mean']:.2f} pts mean, "
                                       f"+{t.loc[k, f'vs_base_{KEY}']:.2f} in {KEY})" for k in passing)
                               if passing else "none"))

# ---------------------------------------------------------------
# 3. Real plans for 2027 (all candidates; submit the ones that passed first)
# ---------------------------------------------------------------
print("\nBuilding merged circles (a few minutes, silent)...")
mem, circ, merges = merge_clusters(45, 50)
plan_circles = [{"x": c[0], "y": c[1], "r": c[2], "idx": c[3], "own": np.array(mem[k])}
                for k, c in circ.items()]
b = build(2026, 2027, 25)
for k in CANDIDATES:
    order, dig, n_left = run(k, plan_circles, live, b, 25)
    covered = np.zeros(N, bool)
    for c in order:
        covered[plan_circles[c]["idx"]] = True
    assert not seed[~covered[seed]].size, f"{k}: iron pipes not covered"
    lines = [f"\"({plan_circles[c]['x']:.6f}, {plan_circles[c]['y']:.6f})\",{plan_circles[c]['r']:.6f}\n"
             for c in order]
    if k == BASE:
        if os.path.exists("submission_4p5yr.csv"):
            theirs = open("submission_4p5yr.csv").readlines()
            same = sum(a == b_ for a, b_ in zip(lines, theirs))
            print(f"Check: rebuilt 4.5-year plan matches submission_4p5yr.csv on {same:,} of {len(theirs):,} lines"
                  + ("  (identical)" if same == len(theirs) == len(lines) else "  (NOT identical: tell Claude)"))
        continue
    path = "submission_" + k.replace(" ", "_") + ".csv"
    open(path, "w").writelines(lines)
    print(f"Saved {path}  ({len(lines):,} circles, coverage check passed)"
          + ("  <- PASSED" if k in passing else "  (did not pass)"))
'''

src = open(SRC).read()
if MARKER not in src:
    raise SystemExit(f"Couldn't find '{MARKER.strip()}' in {SRC}. Is this the blend script?")
open(OUT, "w").write(src[: src.index(MARKER)] + BODY)
print(f"Wrote {OUT}. Next: python {OUT}")
