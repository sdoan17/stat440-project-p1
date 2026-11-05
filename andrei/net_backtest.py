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
