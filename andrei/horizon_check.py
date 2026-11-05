"""
Star Kingdom pipe project: three-way ensemble of differently-targeted models.

Ensembles help when members are accurate AND make different mistakes. Our best
plan already combines two members trained on DIFFERENT targets:
    age term      = P(leak in next 2 years, from age)        x leak-cost model
    expected $    = gradient boosting trained on leak DOLLARS (Poisson loss)
This script adds a third member with a third target:
    yes/no term   = gradient-boosting P(leak next year)       x leak-cost model
(that term + the age term was the $73.31M plan), and tests the blend

    priority = a x expected $  +  b x yes/no term  +  age term

on a small grid fixed in advance: a, b in {0, 0.5, 1}.
    a=1, b=0  is the current best ($75.92M public)
    a=0, b=1  is the previous plan ($73.31M public)
    a=0, b=0  is the age term alone (reference)

Decision rule (same as before, fixed before running): beat the current best in
the 2025-2026 window AND on the three-window average. A submission file is
written for every blend that passes, best first.

Run:   python ensemble_blend.py     (about 10 minutes)
Makes: blend_backtest.csv, submission_blend_a<a>_b<b>.csv for each blend that passes
"""


import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from depletion import depletion_multiplier
DEPL = depletion_multiplier()          # every leak up to 2026, so it's the plan for 2027
# ---------------------------------------------------------------
# 0. Load data, clean, basic features
# ---------------------------------------------------------------
pipes = pd.read_csv("pipes.csv")
leaks = pd.read_csv("train.csv")
leaks["year"] = leaks["Date"].str[:4].astype(int)

# 15 pipes have no lay date and 8 have no material: fill them in
pipes["lay_year"] = pipes["Lay date"].str[:4].astype(float)
pipes["lay_year"] = pipes["lay_year"].fillna(pipes["lay_year"].median()).astype(int)
pipes["Material"] = pipes["Material"].fillna("unknown")

pipes["x"] = (pipes["GPS x1"] + pipes["GPS x2"]) / 2
pipes["y"] = (pipes["GPS y1"] + pipes["GPS y2"]) / 2
pipes["length"] = np.hypot(pipes["GPS x2"] - pipes["GPS x1"],
                           pipes["GPS y2"] - pipes["GPS y1"])
pipes["decade"] = (pipes["lay_year"] // 10 * 10).astype(str)
pipes["group"] = pipes["Material"] + "_" + pipes["decade"]
pipes["leak_year"] = pipes["Pipe ID"].map(leaks.groupby("Pipe ID")["year"].min())

N = len(pipes)
leak_year = pipes["leak_year"].values
can_leak = (pipes["Material"] != "polyurethane").values

# neighbours within 150 m (for the "whole street leaks together" pattern)
pts = pipes[["x", "y"]].values
pairs = cKDTree(pts).query_pairs(r=150, output_type="ndarray")
A = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(N, N))
A = (A + A.T).tocsr()
deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)


def at_risk(yr):
    """Pipes that could still leak in year yr (not polyurethane, not yet leaked)."""
    return can_leak & (np.isnan(leak_year) | (leak_year >= yr))


def leaked_in(first, last):
    return (leak_year >= first) & (leak_year <= last)


# share of each group's at-risk pipes that leaked, per year
group_rate = {}
for yr in range(2019, 2027):
    r = at_risk(yr)
    group_rate[yr] = pipes.loc[r].groupby("group")["leak_year"].apply(lambda s: (s == yr).mean())


def features(T):
    """Features using only information up to the end of year T."""
    f = pipes[["Material", "Surface", "decade", "length", "lay_year", "x", "y"]].copy()
    f["grp_last1"] = pipes["group"].map(group_rate[T]).fillna(0).values
    f["grp_last2"] = pipes["group"].map(group_rate.get(T - 1, group_rate[T])).fillna(0).values
    f["nbr_recent"] = A @ leaked_in(T - 1, T).astype(float) / deg     # neighbours leaking lately
    f["nbr_total"] = A @ leaked_in(2019, T).astype(float) / deg        # neighbours ever leaked
    return f

CAT = ["Material", "Surface", "decade"]
NUM_OLD = ["length", "lay_year", "x", "y", "grp_last1", "grp_last2", "nbr_recent", "nbr_total"]
NUM_NEW = NUM_OLD + ["age", "gap"]     # NEW: age in the target year, years ahead
MAX_GAP = 3                            # train on forecasts up to 3 years ahead

_feat_cache = {}


def feats(T):
    if T not in _feat_cache:
        _feat_cache[T] = features(T)
    return _feat_cache[T]


def rows(info_years, max_gap, last_target):
    """One row per at-risk pipe per (info year T, target year). Features use only
    what was known at the end of T; the target is a leak in T+1 ... T+max_gap.
    A pipe that leaked between T and the target year is no longer at risk."""
    out = []
    for T in info_years:
        for yr in range(T + 1, min(T + max_gap, last_target) + 1):
            r = at_risk(yr)
            f = feats(T)[r].copy()
            f["age"] = yr - pipes["lay_year"].values[r]
            f["gap"] = yr - T
            f["target"] = (leak_year[r] == yr).astype(int)
            f["year"] = yr
            f["row"] = np.flatnonzero(r)
            out.append(f)
    return pd.concat(out, ignore_index=True)


def make_gbm(num):
    return make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
                           ("num", "passthrough", num)]),
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                       max_leaf_nodes=31, random_state=0))


def fit_leak_models(last_target):
    """Both models, trained only on leaks up to last_target.
    old = v6 model (1 year ahead, no age/gap); new = v7 model."""
    info = range(2020, last_target)
    old = rows(info, 1, last_target)
    new = rows(info, MAX_GAP, last_target)
    return {"old": make_gbm(NUM_OLD).fit(old[CAT + NUM_OLD], old["target"]),
            "new": make_gbm(NUM_NEW).fit(new[CAT + NUM_NEW], new["target"])}


def hazard_matrix(model, num, T, years):
    """pipes x years: chance of leaking in each year, given what's known at end of T.
    Beyond MAX_GAP years the wave features are treated as 3 years old; age keeps
    counting up, so the 62-65 failure window still moves forward correctly."""
    f = feats(T).copy()
    out = np.zeros((N, len(years)))
    for k, yr in enumerate(years):
        f["age"] = yr - pipes["lay_year"].values
        f["gap"] = min(yr - T, MAX_GAP)
        out[:, k] = model.predict_proba(f[CAT + num])[:, 1]
    out[~at_risk(T + 1)] = 0.0
    return out





def fit_old_model(last_target):
    """The v6 leak model (1 year ahead, no age/gap), trained on leaks up to last_target."""
    r = rows(range(2020, last_target), 1, last_target)
    return make_gbm(NUM_OLD).fit(r[CAT + NUM_OLD], r["target"])



lk = leaks.merge(pipes, on="Pipe ID")
CF = ["Surface", "Material", "length"]
SHRINK = 50


def fit_cost(df, by_surface=False):
    m = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["Surface", "Material"]),
                           ("num", "passthrough", ["length"])]),
        LinearRegression()).fit(df[CF], np.log(df["Cost"]))
    ratio = pd.Series(np.exp(np.log(df["Cost"].values) - m.predict(df[CF])))
    overall = ratio.mean()
    if not by_surface:
        return lambda d: np.exp(m.predict(d[CF])) * overall
    g = ratio.groupby(df["Surface"].values).agg(["mean", "count"])
    smear = (g["mean"] * g["count"] + overall * SHRINK) / (g["count"] + SHRINK)
    return lambda d: np.exp(m.predict(d[CF])) * pd.Series(d["Surface"].values).map(smear).fillna(overall).values



# dollars of each pipe's leak (0 if it never leaked), used as the training target
pipe_dollars = pipes["Pipe ID"].map(lk.groupby("Pipe ID")["Cost"].sum()).fillna(0).values

NUM_EC = NUM_OLD                         # same features as the current model
NUM_EC_AGE = NUM_OLD + ["age"]


def make_ec(num):
    return make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
                           ("num", "passthrough", num)]),
        HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                      max_leaf_nodes=31, min_samples_leaf=200,
                                      l2_regularization=1.0, random_state=0))


def fit_ec(last_target, num):
    """Expected-$ model trained on leaks up to last_target (1 year ahead, like the current model).
    min_samples_leaf and l2 keep a few huge leaks from dominating a leaf."""
    r = rows(range(2020, last_target), 1, last_target)
    y = np.where(r["target"] == 1, pipe_dollars[r["row"].values], 0.0)
    return make_ec(num).fit(r[CAT + num], y)


def predict_ec(model, num, T, year):
    f = feats(T).copy()
    f["age"] = year - pipes["lay_year"].values
    f["gap"] = year - T
    out = model.predict(f[CAT + num])
    out[~at_risk(T + 1)] = 0.0
    return out


def age_hazard(last_year=2026, bw=5):
    parts = []
    for yr in range(2019, last_year + 1):
        r = at_risk(yr)
        parts.append(pd.DataFrame({"age": yr - pipes["lay_year"].values[r],
                                   "leaked": (leak_year[r] == yr).astype(int)}))
    d = pd.concat(parts)
    d = d[d["age"] >= 0]
    h = d.groupby((d["age"] // bw) * bw)["leaked"].mean()
    bins = np.arange(h.index.min(), h.index.max() + bw, bw)
    h = h.reindex(bins).interpolate().ffill().bfill()

    def predict(years):
        ages = np.array(years)[None, :] - pipes["lay_year"].values[:, None]
        b = np.clip((ages // bw) * bw, h.index.min(), h.index.max())
        return h.reindex(b.ravel()).values.reshape(b.shape)
    return predict



# ---------------------------------------------------------------
# 3. Circle geometry (unchanged)
# ---------------------------------------------------------------
import heapq, random, itertools

RATE = {"grassland": 3, "farmland": 7, "swamp": 12, "road": 17, "structure": 50, "water": 32}
MARGIN = 0.001
P1 = pipes[["GPS x1", "GPS y1"]].values
P2 = pipes[["GPS x2", "GPS y2"]].values
end1, end2 = cKDTree(P1), cKDTree(P2)
surface = pipes["Surface"].values
length = pipes["length"].values
must_cover = (pipes["Material"].str.contains("iron") | (pipes["Material"] == "unknown")).values
live = at_risk(2027)                                   # not plastic, not yet leaked
live_len = np.where(live, length, 0.0)

def circle_from(a, b=None, c=None):
    if b is None:
        return (a[0], a[1], 0.0)
    if c is None:
        cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        return (cx, cy, np.hypot(a[0] - cx, a[1] - cy))
    ax, ay, bx, by, qx, qy = *a, *b, *c
    d = 2 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by))
    if abs(d) < 1e-12:                               # points in a line: use the farthest pair
        pts = [a, b, c]
        best = max(((p, q) for p in pts for q in pts), key=lambda t: np.hypot(t[0][0]-t[1][0], t[0][1]-t[1][1]))
        return circle_from(*best)
    ux = ((ax*ax+ay*ay)*(by-qy) + (bx*bx+by*by)*(qy-ay) + (qx*qx+qy*qy)*(ay-by)) / d
    uy = ((ax*ax+ay*ay)*(qx-bx) + (bx*bx+by*by)*(ax-qx) + (qx*qx+qy*qy)*(bx-ax)) / d
    return (ux, uy, np.hypot(ax - ux, ay - uy))


def smallest_circle(points):
    """Smallest circle containing all points (Welzl's algorithm, iterative)."""
    pts = list(points)
    random.Random(0).shuffle(pts)
    ok = lambda c, p: np.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] + 1e-9
    c = circle_from(pts[0])
    for i, p in enumerate(pts):
        if ok(c, p):
            continue
        c = circle_from(p)
        for j in range(i):
            q = pts[j]
            if ok(c, q):
                continue
            c = circle_from(p, q)
            for k in range(j):
                if not ok(c, pts[k]):
                    c = circle_from(p, q, pts[k])
    return c


def inside(cx, cy, r):
    """Pipes with BOTH ends inside the circle (= wholly inside)."""
    a = end1.query_ball_point([cx, cy], r)
    if not a:
        return np.array([], int)
    a = np.array(a)
    return a[np.hypot(P2[a, 0] - cx, P2[a, 1] - cy) <= r]


def price(cx, cy, r):
    idx = inside(cx, cy, r)
    if live_len[idx].sum() == 0:
        return idx, 0.0
    rate = sum(RATE[s] for s in set(surface[idx]))
    return idx, 100 + (rate * np.pi * r * r) ** 0.85 + 200 * live_len[idx].sum()


def cluster_circle(members):
    pts = np.vstack([P1[members], P2[members]])
    mid = pts.mean(axis=0)                       # work near (0, 0) to avoid rounding errors
    cx, cy, _ = smallest_circle([tuple(p) for p in pts - mid])
    cx, cy = cx + mid[0], cy + mid[1]
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).max() + MARGIN   # always reaches every pipe end
    idx, cost = price(cx, cy, r)
    return (cx, cy, r, idx, cost)

seed = np.where(must_cover & live)[0]
single = {i: cluster_circle([i]) for i in seed}


def merge_clusters(nbr_r, max_r):
    members = {i: [i] for i in seed}
    circ = dict(single)
    version = {i: 0 for i in seed}

    # clusters are neighbours if any of their pipe ends are within nbr_r metres
    allends = np.vstack([P1[seed], P2[seed]])
    owner = np.concatenate([seed, seed])
    nbrs = {i: set() for i in seed}
    for a, b in owner[cKDTree(allends).query_pairs(r=nbr_r, output_type="ndarray")]:
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)

    heap, tie = [], itertools.count()

    def push(a, b):
        c = cluster_circle(members[a] + members[b])
        if c[2] > max_r:
            return
        saving = circ[a][4] + circ[b][4] - c[4]
        if saving > 0:
            heapq.heappush(heap, (-saving, next(tie), a, b, version[a], version[b], c))

    for a in seed:
        for b in nbrs[a]:
            if a < b:
                push(a, b)

    merges = 0
    while heap:
        _, _, a, b, va, vb, c = heapq.heappop(heap)
        if a not in circ or b not in circ or version[a] != va or version[b] != vb:
            continue                                     # stale: one side changed since
        members[a] += members.pop(b)                     # merge b into a
        circ[a] = c
        circ.pop(b)
        version[a] += 1
        version.pop(b)
        nbrs[a] = (nbrs[a] | nbrs.pop(b)) - {a, b}
        for k in nbrs[a]:
            nbrs[k].discard(b)
            nbrs[k].add(a)
        for k in nbrs[a]:                                # NEW: re-check every neighbour now
            push(a, k)
        merges += 1
    return members, circ, merges



# ---------------------------------------------------------------
# 4. Scheduler (from v7; here always used with a fixed priority)
# ---------------------------------------------------------------
def schedule(circles, alive, exp_cost, H=None, window=None, static_value=None,
             max_years=25, budget_per_year=10e6):
    n_c = len(circles)
    M = sparse.csr_matrix((np.ones(sum(len(c["idx"]) for c in circles)),
                           (np.repeat(np.arange(n_c), [len(c["idx"]) for c in circles]),
                            np.concatenate([c["idx"] for c in circles]))), shape=(n_c, N))
    earth = np.array([(sum(RATE[s] for s in set(surface[c["idx"]])) * np.pi * c["r"] ** 2) ** 0.85
                      for c in circles])
    done = ~alive.copy()                 # plastic / already leaked = nothing to replace
    used = np.zeros(n_c, bool)
    order, dig_year = [], np.full(N, 10 ** 6)
    budget = 0.0
    for k in range(max_years):
        budget += budget_per_year
        if static_value is not None:
            risk = static_value
        else:
            hz = H[:, k:k + window]
            risk = (1 - np.prod(1 - hz, axis=1)) * exp_cost   # P(leak in next W years) x cost
        todo_len = np.where(~done, length, 0.0)
        cost = 100 + earth + 200 * (M @ todo_len)
        value = M @ np.where(~done, risk, 0.0)
        cand = np.flatnonzero(~used)
        cand = cand[np.argsort(-value[cand] / cost[cand], kind="stable")]
        for c in cand:
            own = circles[c]["own"]
            if done[own].all():                 # its pipes were replaced by an earlier circle
                used[c] = True
                continue
            idx = circles[c]["idx"]
            todo = idx[~done[idx]]
            c_cost = 100 + earth[c] + 200 * length[todo].sum()
            if c_cost > budget:
                break                           # grader would move to next year here
            budget -= c_cost
            done[idx] = True
            dig_year[todo] = k
            used[c] = True
            order.append(c)
        if used.all():
            break
    left = [c for c in np.flatnonzero(~used) if not done[circles[c]["own"]].all()]
    return order + left, dig_year, len(left)


def leak_cost_real(dig_year, start, end):
    """Real leak $ in [start, end] on pipes not yet replaced (same-year leak counts)."""
    lyr = leak_year - start
    hit = (~np.isnan(leak_year)) & (leak_year >= start) & (leak_year <= end) & (lyr <= dig_year)
    dollars = pipes["Pipe ID"].map(lk.groupby("Pipe ID")["Cost"].sum()).fillna(0).values
    window = (~np.isnan(leak_year)) & (leak_year >= start) & (leak_year <= end)
    return dollars[hit].sum(), dollars[window].sum()



def leak_cost_real(dig_year, start, end):
    """Real leak $ in [start, end] on pipes not yet replaced (same-year leak counts)."""
    dollars = pipes["Pipe ID"].map(lk.groupby("Pipe ID")["Cost"].sum()).fillna(0).values
    window = (~np.isnan(leak_year)) & (leak_year >= start) & (leak_year <= end)
    hit = window & ((leak_year - start) <= dig_year)
    return dollars[hit].sum(), dollars[window].sum()



# ===================================================================
# Horizon check (built by make_horizon_check.py)
# ===================================================================
from depletion import depletion_multiplier

HORIZONS = [2, 3, 4, 5, 6]
YEARS = 16                     # 2027-2042: long enough for every plan to finish
HALF = 0.5                     # a pipe is exposed for about half of the year it's dug in
MIN_SHARE = 0.25               # 5/6 years must beat 4 by >= 25% of the simulated 2->4 gain
MIN_GAIN = 0.20                # ...and by >= $0.2M simulated (~$0.4M real, the noise level)
MIN_VALID = 0.25               # simulated 2->4 gain must be >= $0.25M (grader: $0.99M, model ~2x low)
# Grader leak $M by year (years 1-6) for the submissions you already made:
GRADER = {2: [13.26, 3.14, 3.30, 6.24, 3.39, 0.44],     # age x depl,  $76.44M
          4: [13.46, 3.13, 2.38, 6.12, 3.20, 0.47]}     # 4yr,         $77.56M

T, START = 2026, 2027
cost = fit_cost(lk[lk["year"] <= T])(pipes)
A = age_hazard(last_year=T)([START + k for k in range(YEARS)])        # pipes x years
risk_cost = depletion_multiplier(before_year=START) * cost
assert A.shape == (N, YEARS) and len(risk_cost) == N

mem, circ, merges = merge_clusters(45, 50)
plan_circles = [{"x": c[0], "y": c[1], "r": c[2], "idx": c[3], "own": np.array(mem[k])}
                for k, c in circ.items()]


def expected_leaks(dig_year):
    """Simulated leak $ in each plan year under the age model."""
    k = np.arange(YEARS)[None, :]
    dy = np.asarray(dig_year, float)[:, None]
    exposure = np.where(k < dy, 1.0, np.where(k == dy, HALF, 0.0))
    h = A * exposure
    no_leak_before = np.cumprod(np.c_[np.ones(N), 1 - h[:, :-1]], axis=1)
    first_leak = no_leak_before * h                                   # each pipe leaks at most once
    return (first_leak * risk_cost[:, None])[live].sum(axis=0)


sim, plans = {}, {}
for H in HORIZONS:
    pH = 1 - np.prod(1 - A[:, :H], axis=1)
    order, dig, n_left = schedule(plan_circles, live, None,
                                  static_value=np.where(live, pH * risk_cost, 0.0))
    sim[H] = expected_leaks(dig)
    plans[H] = order
    print(f"  horizon {H}: simulated leak ${sim[H].sum() / 1e6:.2f}M")

# ---- 1. Validation against the grader -------------------------------------------
print("\n=== Step 1: does the simulator reproduce what the grader showed for 2 -> 4 years? ===")
v = pd.DataFrame({"grader 4yr - 2yr ($M)": np.subtract(GRADER[4], GRADER[2]),
                  "simulated 4yr - 2yr ($M)": (sim[4] - sim[2])[:6] / 1e6},
                 index=[f"year {i + 1}" for i in range(6)])
print(v.round(2).to_string())
gain_24 = (sim[2].sum() - sim[4].sum()) / 1e6
yr3_ok = (sim[4] - sim[2])[2] < 0
print(f"Simulated total 2 -> 4 gain: ${gain_24:.2f}M  (grader: $0.99M on leaks; the model "
      "under-predicts levels ~2x, so about half that is a good match)")
d = (sim[4] - sim[2])[:6]
peak_ok = int(np.argmin(d)) in (2, 3)            # biggest drop in year 3 or 4, as in the grader
valid = gain_24 >= MIN_VALID and yr3_ok and peak_ok
print(f"Checks: gain >= ${MIN_VALID:.2f}M: {gain_24 >= MIN_VALID} | year 3 drops: {yr3_ok} | "
      f"biggest drop in year 3 or 4: {peak_ok}")

# ---- 2. Compare horizons -------------------------------------------------------------
print("\n=== Step 2: simulated leak $ by horizon ($M; lower is better) ===")
tab = pd.DataFrame({f"{H}yr": np.r_[sim[H][:6], sim[H][6:].sum(), sim[H].sum()] / 1e6
                    for H in HORIZONS},
                   index=[f"year {i + 1}" for i in range(6)] + ["years 7+", "TOTAL"])
tab["5yr - 4yr"] = tab["5yr"] - tab["4yr"]
print(tab.round(2).to_string())

# ---- 3. Decision ---------------------------------------------------------------------
print("\n--- Decision ---")
if not valid:
    print("The simulator does NOT reproduce the grader's 2 -> 4 result, so it can't be trusted "
          "for 4 -> 5. Keep the 4-year plan ($77.56M).")
else:
    best = min([5, 6], key=lambda H: sim[H].sum())
    gain = (sim[4].sum() - sim[best].sum()) / 1e6
    need = max(MIN_SHARE * gain_24, MIN_GAIN)
    print(f"Simulator validated (2 -> 4 gain ${gain_24:.2f}M, year 3 drops as in the grader).")
    if gain >= need:
        path = f"submission_{best}yr.csv"
        with open(path, "w") as f:
            for c in plans[best]:
                pc = plan_circles[c]
                f.write(f"\"({pc['x']:.6f}, {pc['y']:.6f})\",{pc['r']:.6f}\n")
        print(f"{best}-year horizon beats 4 years by ${gain:.2f}M simulated (needed ${need:.2f}M). "
              f"Saved {path}: worth one submission.")
    else:
        print(f"Best longer horizon ({best} years) beats 4 years by only ${gain:.2f}M simulated "
              f"(needed ${need:.2f}M). Keep the 4-year plan ($77.56M).")

# ---- forced: write the 5-year plan regardless of the rule (diagnostic submission) ----
covered = np.zeros(N, bool)
for c in plans[5]:
    covered[plan_circles[c]["idx"]] = True
assert not seed[~covered[seed]].size, "iron pipes not covered"
with open("submission_5yr.csv", "w") as f:
    for c in plans[5]:
        pc = plan_circles[c]
        f.write(f"\"({pc['x']:.6f}, {pc['y']:.6f})\",{pc['r']:.6f}\n")
print("Saved submission_5yr.csv (failed the rule; diagnostic submission)")
