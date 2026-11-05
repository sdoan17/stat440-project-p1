"""
Do pipes fail by construction batch? And how much is left to gain?

A "batch" = iron pipes that touch end to end (within 0.5 m) and were laid within
a few days of each other, i.e. probably laid by the same crew, same material lot.

CHECK 1 - does the batch signal exist?
  At the start of each test window, take every live iron pipe and compare how many
  actually leaked in the window (Observed) with how many the age-only hazard expected
  (Expected). O/E = 1 means age explains everything. If pipes whose batch-mates
  already leaked have O/E well above 1 (or well below, if batches get used up),
  the batch carries information that age doesn't.

CHECK 2 - headroom, in a simple simulator
  Same budget ($10M a year, carryover), one tight circle per pipe, dig the best
  value-per-dollar first, and count the share of real window leak $ prevented by:
    age          age hazard x average leak cost for the surface   (~ your age term)
    age x batch  the same, times a shrunk batch multiplier (O+k)/(E+k) from history
    oracle       the pipes that ACTUALLY leaked, ranked by their actual cost
  oracle minus age = the most any better model could add.
  The simulator is cruder than your walk-forward (no merged circles), so compare
  rows with each other, not with your 41%.

Windows: plan at the start of 2023 / 2024 / 2025, score on real leaks up to 2026,
using only leaks before the window to fit anything (no peeking).

Usage: python batch_diagnostics.py      (in the folder with pipes.csv and train.csv)
"""
import math

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

RATE = {"grassland": 3, "farmland": 7, "swamp": 12, "road": 17, "structure": 50, "water": 32}
IRON = ["cast iron", "gray iron", "wrought iron"]
FIRST, LAST = 2019, 2026
WINDOWS = [2023, 2024, 2025]
BUDGET = 10e6
TOUCH = 0.5          # metres: endpoints this close count as connected
DAY_GAPS = [3, 14, 60]
PRIMARY_GAP = 14     # decided in advance: the one to trust; the others only check robustness
K_PRIOR = 2.0        # shrinkage: a batch needs ~2 expected leaks of evidence to move far from 1
MAX_AGE = 130


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load():
    p = pd.read_csv("pipes.csv")
    l = pd.read_csv("train.csv")
    date_col = "Date" if "Date" in l else next(c for c in l if "date" in c.lower())
    cost_col = "Cost" if "Cost" in l else next(c for c in l if "cost" in c.lower())
    l["leak_year"] = pd.to_datetime(l[date_col]).dt.year
    l = l.sort_values("leak_year").drop_duplicates("Pipe ID")
    l = l[["Pipe ID", "leak_year", cost_col]].rename(columns={cost_col: "leak_cost"})
    p = p.merge(l, on="Pipe ID", how="left")
    p["material"] = p["Material"].fillna("unknown").str.strip().str.lower()
    p["lay"] = pd.to_datetime(p["Lay date"], errors="coerce")
    n_nodate = int((p["material"].isin(IRON) & p["lay"].isna()).sum())
    p = p[p["material"].isin(IRON) & p["lay"].notna()].reset_index(drop=True)
    p["surface"] = p["Surface"].str.strip().str.lower()
    p["lay_year"] = p["lay"].dt.year
    p["leak_year"] = p["leak_year"].fillna(9999).astype(int)
    p["leak_cost"] = p["leak_cost"].fillna(0.0)
    P1 = p[["GPS x1", "GPS y1"]].to_numpy(float)
    P2 = p[["GPS x2", "GPS y2"]].to_numpy(float)
    L = np.hypot(*(P2 - P1).T)
    rate = p["surface"].map(RATE).to_numpy(float)
    p["dig_cost"] = 100 + (rate * math.pi * (L / 2) ** 2) ** 0.85 + 200 * L   # own tight circle
    print(f"{len(p):,} iron pipes with a lay date ({n_nodate} without one are left out), "
          f"{int((p.leak_year <= LAST).sum()):,} leaked in {FIRST}-{LAST}")
    return p, P1, P2


def batches(p, P1, P2, days):
    n = len(p)
    ends = np.vstack([P1, P2])
    owner = np.r_[np.arange(n), np.arange(n)]
    pairs = cKDTree(ends).query_pairs(TOUCH, output_type="ndarray")
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    lay = p["lay"].to_numpy()
    keep = (a != b) & (np.abs((lay[a] - lay[b]) / np.timedelta64(1, "D")) <= days)
    g = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(n, n))
    return connected_components(g, directed=False)[1]


# ---------------------------------------------------------------
# Age-only hazard, fitted on years before y0
# ---------------------------------------------------------------
def fit_hazard(p, y0):
    risk = np.zeros(MAX_AGE + 1)
    ev = np.zeros(MAX_AGE + 1)
    lay, leak = p["lay_year"].to_numpy(), p["leak_year"].to_numpy()
    for t in range(FIRST, y0):
        at = (lay <= t) & (leak >= t)
        age = np.clip(t - lay[at], 0, MAX_AGE)
        np.add.at(risk, age, 1)
        np.add.at(ev, age, (leak[at] == t).astype(float))
    kernel = np.ones(5)                                 # pool ages +/- 2 years
    r = np.convolve(risk, kernel, "same")
    e = np.convolve(ev, kernel, "same")
    overall = ev.sum() / max(risk.sum(), 1)
    return np.where(r >= 30, e / np.maximum(r, 1), overall)


def expected_history(p, h, y0):
    """Expected leaks per pipe over FIRST..y0-1 while it was at risk (for the batch multiplier)."""
    lay, leak = p["lay_year"].to_numpy(), p["leak_year"].to_numpy()
    E = np.zeros(len(p))
    for t in range(FIRST, y0):
        at = (lay <= t) & (leak >= t)
        E[at] += h[np.clip(t - lay[at], 0, MAX_AGE)]
    O = (leak < y0).astype(float)
    return O, E


def window_prob(p, h, y0):
    lay = p["lay_year"].to_numpy()
    surv = np.ones(len(p))
    for t in range(y0, LAST + 1):
        surv *= 1 - h[np.clip(t - lay, 0, MAX_AGE)]
    return 1 - surv


# ---------------------------------------------------------------
# Simple simulator
# ---------------------------------------------------------------
def prevented_share(value, cost, leak_year, leak_cost, y0):
    order = np.argsort(-(value / cost), kind="stable")
    cum = np.cumsum(cost[order])
    dig_year = np.full(len(cost), 9999)
    dig_year[order] = y0 + ((cum - 1e-6) // BUDGET).astype(int)   # $10M/yr with carryover
    in_window = leak_year <= LAST
    saved = in_window & (dig_year <= leak_year)                   # dug at or before the leak year
    return 100 * leak_cost[saved].sum() / leak_cost[in_window].sum()


def run_window(p, comp, y0, detail):
    h = fit_hazard(p, y0)
    O_hist, E_hist = expected_history(p, h, y0)
    live = (p["leak_year"] >= y0).to_numpy()
    q = p[live]
    c = comp[live]
    Pwin = window_prob(q, h, y0)
    outcome = (q["leak_year"] <= LAST).to_numpy()

    # batch evidence from history, counting other members only
    bsize = np.bincount(comp)
    bO = np.bincount(comp, O_hist)
    bE = np.bincount(comp, E_hist)
    mult = (bO[c] + K_PRIOR) / (bE[c] - E_hist[live] + K_PRIOR)
    # depletion: each pipe fails once, so risk scales with the share of the batch NOT yet leaked
    remaining = (bsize[c] - bO[c] + 1) / (bsize[c] + 1)

    if detail:
        share = bO[c] / bsize[c]
        group = np.select(
            [bsize[c] == 1, bO[c] == 0, share <= 0.25, share <= 0.5],
            ["single pipe (no batch)", "batch: no mate leaked yet", "batch: <=25% leaked",
             "batch: 25-50% leaked"], "batch: >50% leaked")
        t = pd.DataFrame({"group": group, "O": outcome, "E": Pwin}).groupby("group").agg(
            pipes=("O", "size"), observed=("O", "sum"), expected=("E", "sum"))
        t["O/E"] = t["observed"] / t["expected"]
        print(f"\nCHECK 1, window {y0}-{LAST}: actual leaks vs age-only expectation, by batch history")
        print(t.round(2).to_string())

        # timing: where is each pipe relative to its batch's observed failure age?
        lay, leak = p["lay_year"].to_numpy(), p["leak_year"].to_numpy()
        prior = leak < y0
        ages = pd.Series(leak[prior] - lay[prior]).groupby(comp[prior]).median()
        peak = ages.reindex(c).to_numpy()
        gap = (y0 - q["lay_year"].to_numpy()) - peak
        has = ~np.isnan(peak)
        tg = np.select([gap < -3, gap <= 3], ["before batch's failure age (>3 yrs)",
                                              "at batch's failure age (+/-3 yrs)"],
                       "past batch's failure age (>3 yrs)")
        t2 = pd.DataFrame({"timing": tg[has], "O": outcome[has], "E": Pwin[has]}).groupby("timing").agg(
            pipes=("O", "size"), observed=("O", "sum"), expected=("E", "sum"))
        t2["O/E"] = t2["observed"] / t2["expected"]
        print("Pipes whose batch-mates already leaked, by age relative to the batch's median failure age:")
        print(t2.round(2).to_string())

    surf_cost = p.loc[p.leak_year < y0].groupby("surface")["leak_cost"].mean()
    cost_if_leak = q["surface"].map(surf_cost).fillna(p.loc[p.leak_year < y0, "leak_cost"].mean()).to_numpy()
    args = (q["dig_cost"].to_numpy(), q["leak_year"].to_numpy(), q["leak_cost"].to_numpy(), y0)
    return {"age": prevented_share(Pwin * cost_if_leak, *args),
            "age x batch": prevented_share(Pwin * mult * cost_if_leak, *args),
            "age x depletion": prevented_share(Pwin * remaining * cost_if_leak, *args),
            "oracle": prevented_share(np.where(outcome, q["leak_cost"].to_numpy(), 0.0) + 1e-9, *args)}


def main():
    p, P1, P2 = load()
    rows = []
    for days in DAY_GAPS:
        comp = batches(p, P1, P2, days)
        size = np.bincount(comp)
        in_batch = (size[comp] > 1).mean()
        print(f"\n######## batch = touching + laid within {days} days ########")
        print(f"{(size > 1).sum():,} batches of 2+ pipes; {100 * in_batch:.1f}% of pipes are in one; "
              f"median batch size {int(np.median(size[size > 1])) if (size > 1).any() else 0}, "
              f"largest {size.max()}")
        for y0 in WINDOWS:
            r = run_window(p, comp, y0, detail=(days == PRIMARY_GAP))
            rows.append({"batch days": days, "window": f"{y0}-{LAST}", **r})

    t = pd.DataFrame(rows).set_index(["batch days", "window"])
    t["batch gain"] = t["age x batch"] - t["age"]
    t["depletion gain"] = t["age x depletion"] - t["age"]
    t["headroom"] = t["oracle"] - t["age"]
    print("\nCHECK 2: % of window leak $ prevented (simple simulator; compare rows, not with your 41%)")
    print(t.round(2).to_string())

    prim = t.loc[PRIMARY_GAP]
    print("\n--- Reading it ---")
    print(f"Headroom (oracle - age), {PRIMARY_GAP}-day batches: "
          + ", ".join(f"{w}: {v:.1f} pts" for w, v in prim["headroom"].items()))
    print("  Roughly $0.3-0.5M per point on your real plan. Small headroom = stop; large = keep going.")
    wins = (prim["depletion gain"] > 0).all()
    mean_gain = prim["depletion gain"].mean()
    if wins and mean_gain >= 1:
        print(f"Depletion term beats age in every window (+{mean_gain:.1f} pts on average). "
              "Worth building into the real model and testing with your usual walk-forward rule.")
    elif wins:
        print(f"Depletion term beats age in every window, but only by {mean_gain:.1f} pts: "
              "real but small, likely inside leaderboard noise.")
    else:
        print("Depletion term does not beat age in every window: the batch idea doesn't pay off "
              "in this form. Look at CHECK 1 to see whether the signal is absent or just mis-modelled.")


if __name__ == "__main__":
    main()
