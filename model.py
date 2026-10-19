"""
Star Kingdom pipe project: v4 plan (leak model + cheapest circles + best order).

  1. Chance of leaking next year: gradient boosting (beat logistic regression).
     Leaks come in WAVES, so it uses recent trends for each pipe's group
     (material + decade) and for its neighbours within 150 m.
  2. Leak cost: linear regression on log(cost) from surface, material, length.
  3. Priority = (gradient boosting chance + chance of leaking in 2027-2028 from
     PIPE AGE) x expected leak cost. The age part is Andrei's finding: pipes
     fail at about 62-65 years old.
  4. Circles: one tight circle per pipe, then neighbouring pipes are merged
     into one circle whenever that is cheaper. Circles are dug in order of
     priority per dollar; redundant circles are dropped.

Run:   python model.py      (about 1 minute)
Needs: pandas numpy scikit-learn scipy
Makes: pipe_risk.csv, submission.csv
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
NUM = ["length", "lay_year", "x", "y", "grp_last1", "grp_last2", "nbr_recent", "nbr_total"]


def rows(target_years, info_year=None):
    """One row per at-risk pipe per target year. Features come from the year
    before (or from info_year if given, to mimic forecasting further ahead)."""
    out = []
    for yr in target_years:
        T = info_year if info_year else yr - 1
        f = features(T)[at_risk(yr)].copy()
        f["target"] = (leak_year[at_risk(yr)] == yr).astype(int)
        f["Pipe ID"] = pipes.loc[at_risk(yr), "Pipe ID"].values
        out.append(f)
    return pd.concat(out, ignore_index=True)


def make_models():
    logit = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                           ("num", StandardScaler(), NUM)]),
        LogisticRegression(max_iter=3000))
    gbm = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=False), CAT),
                           ("num", "passthrough", NUM)]),
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                       max_leaf_nodes=31, random_state=0))
    return {"Logistic regression": logit, "Gradient boosting": gbm}


# ---------------------------------------------------------------
# 1 + 4. Chance of leaking: train on 2019-2024, test on 2025-2026
#        (test features only use info up to 2024, like a real forecast)
# ---------------------------------------------------------------
print("\n=== Step 1: chance of leaking (train 2019-2024, test 2025-2026) ===")
train = rows(range(2021, 2025))           # needs 2 years of history, so starts 2021
test = rows([2025, 2026], info_year=2024)

aucs = {}
for name, m in make_models().items():
    m.fit(train[CAT + NUM], train["target"])
    p = m.predict_proba(test[CAT + NUM])[:, 1]
    aucs[name] = roc_auc_score(test["target"], p)
    top = np.argsort(-p)[: len(p) // 10]
    print(f"{name:20s} AUC = {aucs[name]:.3f} | top 10% riskiest pipes caught "
          f"{test['target'].values[top].sum() / test['target'].sum():.0%} of leaks")

# ---------------------------------------------------------------
# 2. Leak cost: log(cost) ~ surface + material + length
# ---------------------------------------------------------------
print("\n=== Step 2: leak cost (train 2019-2024, test 2025-2026) ===")
lk = leaks.merge(pipes, on="Pipe ID")
CF = ["Surface", "Material", "length"]


def fit_cost(df):
    m = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["Surface", "Material"]),
                           ("num", "passthrough", ["length"])]),
        LinearRegression()).fit(df[CF], np.log(df["Cost"]))
    # convert log back to AVERAGE dollars, since rare huge leaks drive the score
    smear = np.mean(np.exp(np.log(df["Cost"]) - m.predict(df[CF])))
    return lambda d: np.exp(m.predict(d[CF])) * smear


cost_fn = fit_cost(lk[lk["year"] <= 2024])
te = lk[lk["year"] >= 2025]
pred = cost_fn(te)
print(f"Predicted total ${pred.sum()/1e6:.1f}M vs actual ${te['Cost'].sum()/1e6:.1f}M | "
      f"correlation (log scale) {np.corrcoef(np.log(pred), np.log(te['Cost']))[0, 1]:.2f}")

# ---------------------------------------------------------------
# Refit on all years, predict for 2027 using info up to 2026
# ---------------------------------------------------------------
best = max(aucs, key=aucs.get)
print(f"\nUsing {best} for the plan (best test AUC).")
everything = rows(range(2021, 2027))
leak_model = make_models()[best].fit(everything[CAT + NUM], everything["target"])
cost_fn = fit_cost(lk)

now = features(2026)
pipes["p_leak"] = leak_model.predict_proba(now[CAT + NUM])[:, 1]
pipes["exp_cost"] = cost_fn(pipes)
pipes["current"] = np.where(at_risk(2027), pipes["Material"], "polyurethane")
pipes.loc[~at_risk(2027), "p_leak"] = 0.0      # already leaked = now plastic, no risk

# ---------------------------------------------------------------
# 3. Risk = chance x cost  (expected leak $ per year)
# ---------------------------------------------------------------
# Backtests showed leak COST matters as much as leak chance (a few huge road
# leaks dominate), so every ranking below multiplies chance by expected cost.
# ---------------------------------------------------------------
# 3b. Age forecast (from Andrei's notebook): pipes fail sharply at about
#     62-65 years old, so a pipe's AGE in 2027-2028 predicts the next leak
#     wave. Yearly leak rate per 5-year age bin, fitted on 2019-2026.
# ---------------------------------------------------------------
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


H = age_hazard()([2027, 2028])
pipes["p_leak_2y"] = 1 - (1 - H[:, 0]) * (1 - H[:, 1])        # chance of leaking in 2027-2028

# Priority = (leak model chance + age-based 2-year chance) x expected leak cost.
# We compared three orders on two tests:
#                         real 2022-26 leaks (avg / worst)   200 simulated futures
#   leak model only (v5)      56% / 40% prevented             $66.7M typical score
#   age only                  66% / 60%                       $60.3M
#   both together (this)      61% / 52%                       $67.9M
# Only the combination beat v5 on BOTH tests, so it is the safest choice.
pipes["risk"] = pipes["p_leak"] * pipes["exp_cost"]                       # expected leak $/year
pipes["priority"] = np.where(at_risk(2027),
                             (pipes["p_leak"] + pipes["p_leak_2y"]) * pipes["exp_cost"], 0)
pipes[["Pipe ID", "Material", "Surface", "p_leak", "exp_cost", "risk"]] \
    .sort_values("risk", ascending=False).to_csv("pipe_risk.csv", index=False)
print("Saved pipe_risk.csv (riskiest pipes first)")


# ---------------------------------------------------------------
# 4. Build circles as cheaply as possible
# ---------------------------------------------------------------
# Idea: the cheapest circle around ONE pipe has its centre at the pipe's
# middle and radius = half its length. Digging cost grows with radius^1.7,
# so one small circle per pipe is usually cheapest. But each circle also
# pays a $100 fee, so where short pipes meet (junctions, short chains),
# ONE circle around several pipes can be cheaper. We merge neighbours
# only when the merged circle costs less than the separate ones.
import heapq, random

RATE = {"grassland": 3, "farmland": 7, "swamp": 12, "road": 17, "structure": 50, "water": 32}
MARGIN = 0.001            # metres added to every radius so rounding never cuts a pipe
P1 = pipes[["GPS x1", "GPS y1"]].values
P2 = pipes[["GPS x2", "GPS y2"]].values
end1, end2 = cKDTree(P1), cKDTree(P2)
surface = pipes["Surface"].values
length = pipes["length"].values
live = (pipes["current"] != "polyurethane").values      # still needs replacing in 2027
live_len = np.where(live, length, 0.0)
prio = pipes["priority"].values
must_cover = (pipes["Material"].str.contains("iron") | (pipes["Material"] == "unknown")).values


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


# start: one cluster per live pipe that must be replaced
seed = np.where(must_cover & live)[0]
parent = {i: i for i in seed}
members = {i: [i] for i in seed}
circ = {i: cluster_circle([i]) for i in seed}

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

# candidate neighbours: pipes whose ends are within 30 m of each other
allends = np.vstack([P1[seed], P2[seed]])
owner = np.concatenate([seed, seed])
pairs = cKDTree(allends).query_pairs(r=30.0, output_type="ndarray")
cand = {(min(owner[a], owner[b]), max(owner[a], owner[b])) for a, b in pairs if owner[a] != owner[b]}

MAX_R = 25.0
heap = []
def push(a, b):
    ma = members[a] + members[b]
    c = cluster_circle(ma)
    if c[2] > MAX_R:
        return
    saving = circ[a][4] + circ[b][4] - c[4]
    if saving > 0:
        heapq.heappush(heap, (-saving, a, b, len(ma)))

for a, b in cand:
    push(a, b)

merges = 0
while heap:
    s, a, b, n = heapq.heappop(heap)
    ra, rb = find(a), find(b)
    if ra == rb or ra != a or rb != b or len(members[a]) + len(members[b]) != n:
        if ra != rb:
            push(min(ra, rb), max(ra, rb))      # cluster changed since: re-evaluate
        continue
    # merge b into a
    parent[b] = a
    members[a] = members[a] + members.pop(b)
    circ[a] = cluster_circle(members[a])
    circ.pop(b)
    merges += 1
    # new candidates: neighbours of any member
    # (lazy: neighbours are re-pushed when their stale entries pop)
print(f"Merged {merges} times: {len(seed)} pipes -> {len(circ)} circles")

# ---------------------------------------------------------------
# 5. Put the circles in order and remove redundant ones
# ---------------------------------------------------------------
C = pd.DataFrame([{"x": c[0], "y": c[1], "r": c[2], "idx": c[3], "cost": c[4],
                   "own": members[k], "prio": prio[c[3]][live[c[3]]].sum()}
                  for k, c in circ.items()])
C["value"] = C["prio"] / C["cost"].clip(lower=1)

# Pipes that already leaked in 2019-2026 are plastic now, so they need no circle.
# (The grader charges $100 + digging for EVERY circle, even one with nothing left
# to replace, so circles around already-fixed pipes are pure waste. Plans
# without them have been accepted by the grader.)
C = C.sort_values("value", ascending=False).reset_index(drop=True)


def dedupe_and_price(C):
    """Walk the list in order like the grader: drop circles whose own pipes are
    already covered, and price each circle given what earlier ones replaced."""
    done = np.zeros(N, bool)          # replaced (or covered) by an earlier circle
    keep, costs = [], []
    for k, c in enumerate(C.itertuples()):
        if done[c.own].all():
            continue
        idx = c.idx
        todo = idx[live[idx] & ~done[idx]]
        rate = sum(RATE[s] for s in set(surface[idx]))          # every circle is charged,
        cost = 100 + (rate * np.pi * c.r ** 2) ** 0.85 + 200 * length[todo].sum()  # even if empty
        done[idx] = True
        keep.append(k)
        costs.append(cost)
    return C.iloc[keep].reset_index(drop=True), np.array(costs)


final, costs = dedupe_and_price(C)
final["cost_now"] = costs
yr = np.floor(np.cumsum(costs) / 10e6)          # rough year each circle is dug
print(f"{len(final)} circles, digging cost ${costs.sum()/1e6:.1f}M, "
      f"done in about {int(yr.max()) + 1} years")

with open("submission.csv", "w") as f:
    for c in final.itertuples():
        f.write(f"\"({c.x:.6f}, {c.y:.6f})\",{c.r:.6f}\n")
print("Saved submission.csv")
