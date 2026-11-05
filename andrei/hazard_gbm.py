"""
Pipe-year hazard model with gradient boosting.

One row per pipe per year it was at risk (2021 onward), target = "leaked that year".
Every feature is computed from leaks BEFORE that year only, so the model learns
how risk evolves, and the same features can be built for 2027 without peeking.

Features
  static     age, lay year (cohort), material, surface, length, x, y
  batch      batch size; share of the batch already leaked (depletion);
             batch leaks in the last 2 years; years since the batch's first leak
             (batch = iron pipes touching end to end and laid within 14 days)
  nearby     share of pipes within 200 m (outside the pipe's own batch) that have
             leaked, all-time and in the last 2 years

gbm_probability(before_year=S, horizon=2) returns, for every row of pipes.csv,
P(leak in years S .. S+horizon-1), using only leaks before S. Non-iron pipes are
included (material is a feature); polyurethane and already-leaked pipes get 0.

Quick check on its own:  python hazard_gbm.py
  Compares the full model with a "no history" version (static features only) on
  2023, 2024 and 2025: AUC, and leak $ caught by the riskiest 10%.
"""
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

IRON = ["cast iron", "gray iron", "wrought iron"]
MATERIALS = IRON + ["copper", "brass", "unknown"]
SURFACES = ["grassland", "farmland", "swamp", "road", "structure", "water"]
TRAIN_FROM = 2021        # need at least 2 years of leak history behind every training row
TOUCH, DAYS = 0.5, 14    # batch definition (same as depletion.py)
NBR_R = 200.0            # metres for the "nearby" features
RECENT = 2               # "recent" = the last 2 years
PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200,
              l2_regularization=1.0, random_state=0)   # fixed in advance, not tuned

STATIC = ["age", "lay_year", "material", "surface", "length", "x", "y"]
HISTORY = ["batch_size", "batch_share_leaked", "batch_recent", "yrs_since_batch_first",
           "nbr_share_leaked", "nbr_recent"]
FEATURES = STATIC + HISTORY
CATEGORICAL = ["material", "surface"]

_DATA, _MODELS = {}, {}


# ---------------------------------------------------------------
# Static pieces (built once)
# ---------------------------------------------------------------
def _data(pipes_path="pipes.csv", leaks_path="train.csv"):
    key = (pipes_path, leaks_path)
    if key in _DATA:
        return _DATA[key]
    p = pd.read_csv(pipes_path)
    l = pd.read_csv(leaks_path)
    date_col = "Date" if "Date" in l else next(c for c in l if "date" in c.lower())
    cost_col = "Cost" if "Cost" in l else next(c for c in l if "cost" in c.lower())
    l["year"] = pd.to_datetime(l[date_col]).dt.year
    first = l.sort_values("year").drop_duplicates("Pipe ID").set_index("Pipe ID")
    N = len(p)

    material = p["Material"].fillna("unknown").str.strip().str.lower()
    surface = p["Surface"].str.strip().str.lower()
    lay = pd.to_datetime(p["Lay date"], errors="coerce")
    P1 = p[["GPS x1", "GPS y1"]].to_numpy(float)
    P2 = p[["GPS x2", "GPS y2"]].to_numpy(float)
    mid = (P1 + P2) / 2

    d = {
        "N": N,
        "leak_year": p["Pipe ID"].map(first["year"]).fillna(9999).to_numpy(int),
        "leak_cost": p["Pipe ID"].map(first[cost_col]).fillna(0.0).to_numpy(float),
        "eligible": (material != "polyurethane").to_numpy(),
        "lay_year": (lay.dt.year + lay.dt.dayofyear / 365.25).to_numpy(float),   # NaN if missing
        "material": material.map({m: i for i, m in enumerate(MATERIALS)}).fillna(len(MATERIALS)).to_numpy(int),
        "surface": surface.map({s: i for i, s in enumerate(SURFACES)}).fillna(len(SURFACES)).to_numpy(int),
        "surface_name": surface.to_numpy(),
        "length": np.hypot(*(P2 - P1).T),
        "x": mid[:, 0], "y": mid[:, 1],
    }

    # batches: iron pipes with a lay date, touching end to end, laid within DAYS
    use = (material.isin(IRON) & lay.notna()).to_numpy()
    rows = np.flatnonzero(use)
    n = rows.size
    owner = np.r_[np.arange(n), np.arange(n)]
    pairs = cKDTree(np.vstack([P1[use], P2[use]])).query_pairs(TOUCH, output_type="ndarray")
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    lay_u = lay[use].to_numpy()
    keep = (a != b) & (np.abs((lay_u[a] - lay_u[b]) / np.timedelta64(1, "D")) <= DAYS)
    comp_u = connected_components(sparse.coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])),
                                                    shape=(n, n)), directed=False)[1]
    comp = np.arange(N) + comp_u.max() + 1          # everyone else: a batch of one
    comp[rows] = comp_u
    _, comp = np.unique(comp, return_inverse=True)
    d["comp"] = comp
    d["batch_size"] = np.bincount(comp)[comp].astype(float)

    # nearby pipes (eligible, within NBR_R, not in the same batch)
    el = np.flatnonzero(d["eligible"])
    pr = cKDTree(mid[el]).query_pairs(NBR_R, output_type="ndarray")
    i, j = el[pr[:, 0]], el[pr[:, 1]]
    keep = comp[i] != comp[j]
    i, j = i[keep], j[keep]
    A = sparse.coo_matrix((np.ones(2 * i.size), (np.r_[i, j], np.r_[j, i])), shape=(N, N)).tocsr()
    d["A"] = A
    d["nbr_n"] = np.asarray(A.sum(1)).ravel()
    print(f"[hazard_gbm] {N:,} pipes, {int((np.bincount(comp) > 1).sum()):,} batches, "
          f"{np.median(d['nbr_n'][el]):.0f} median neighbours within {NBR_R:.0f} m")
    _DATA[key] = d
    return d


def _features(d, hist_end, t):
    """Features for target year t, using leaks with year < hist_end."""
    ly = d["leak_year"]
    comp = d["comp"]
    before = (ly < hist_end).astype(float)
    recent = ((ly < hist_end) & (ly >= hist_end - RECENT)).astype(float)
    b_leaked = np.bincount(comp, before)[comp]
    b_recent = np.bincount(comp, recent)[comp]
    first = np.full(comp.max() + 1, np.inf)
    m = ly < hist_end
    np.minimum.at(first, comp[m], ly[m].astype(float))   # each batch's earliest leak year
    first[np.isinf(first)] = np.nan
    nb = np.maximum(d["nbr_n"], 1)
    return pd.DataFrame({
        "age": t - d["lay_year"],
        "lay_year": d["lay_year"],
        "material": d["material"],
        "surface": d["surface"],
        "length": d["length"],
        "x": d["x"], "y": d["y"],
        "batch_size": d["batch_size"],
        "batch_share_leaked": b_leaked / d["batch_size"],
        "batch_recent": b_recent,
        "yrs_since_batch_first": t - first[comp],
        "nbr_share_leaked": (d["A"] @ before) / nb,
        "nbr_recent": (d["A"] @ recent) / nb,
    })


def _at_risk(d, t):
    return d["eligible"] & (d["leak_year"] >= t)


# ---------------------------------------------------------------
# Fit and predict
# ---------------------------------------------------------------
def fit(before_year, features=FEATURES, pipes_path="pipes.csv", leaks_path="train.csv"):
    key = (before_year, tuple(features), pipes_path, leaks_path)
    if key in _MODELS:
        return _MODELS[key]
    d = _data(pipes_path, leaks_path)
    X, y = [], []
    for t in range(TRAIN_FROM, before_year):
        r = _at_risk(d, t)
        X.append(_features(d, t, t)[r])
        y.append(d["leak_year"][r] == t)
    X = pd.concat(X, ignore_index=True)[list(features)]
    y = np.concatenate(y)
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        categorical_features=[f in CATEGORICAL for f in features], **PARAMS).fit(X, y)
    print(f"[hazard_gbm] fit on {TRAIN_FROM}-{before_year - 1}: {len(y):,} pipe-years, "
          f"{int(y.sum()):,} leaks, {time.time() - t0:.0f}s")
    _MODELS[key] = model
    return model


def gbm_probability(before_year, horizon=2, features=FEATURES,
                    pipes_path="pipes.csv", leaks_path="train.csv"):
    """P(leak in years before_year .. before_year+horizon-1) for every row of pipes.csv."""
    d = _data(pipes_path, leaks_path)
    model = fit(before_year, features, pipes_path, leaks_path)
    r = _at_risk(d, before_year)
    surv = np.ones(r.sum())
    for h in range(horizon):            # history stays fixed at before_year; age moves on
        X = _features(d, before_year, before_year + h)[r][list(features)]
        surv *= 1 - model.predict_proba(X)[:, 1]
    out = np.zeros(d["N"])
    out[r] = 1 - surv
    return out


# ---------------------------------------------------------------
# Quick check
# ---------------------------------------------------------------
def _auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.sum()
    return (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def main():
    d = _data()
    rows = []
    for S in [2023, 2024, 2025]:
        r = _at_risk(d, S)
        y = d["leak_year"][r] == S
        actual = np.where(y, d["leak_cost"][r], 0.0)
        hist = d["leak_year"] < S
        surf_cost = pd.Series(d["leak_cost"][hist]).groupby(d["surface_name"][hist]).mean()
        cost = pd.Series(d["surface_name"][r]).map(surf_cost).fillna(d["leak_cost"][hist].mean()).to_numpy()
        for label, feats in [("no history (static only)", STATIC), ("full model", FEATURES)]:
            p = gbm_probability(S, horizon=1, features=feats)[r]
            top = np.argsort(-(p * cost))[: r.sum() // 10]
            rows.append({"year": S, "model": label, "AUC": _auc(p, y),
                         "leak $ in riskiest 10%": 100 * actual[top].sum() / actual.sum()})
    t = pd.DataFrame(rows).pivot(index="model", columns="year")
    print("\nOne-year-ahead check (fit on leaks before each year, scored on that year):")
    print(t.round(3).to_string())

    from sklearn.inspection import permutation_importance
    S = 2025
    r = _at_risk(d, S)
    X = _features(d, S, S)[r][FEATURES]
    y = d["leak_year"][r] == S
    pi = permutation_importance(fit(S), X, y, scoring="roc_auc", n_repeats=3, random_state=0)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    print("\nWhat the full model leans on (AUC lost when each feature is shuffled, 2025):")
    print(imp.round(4).to_string())


if __name__ == "__main__":
    main()
