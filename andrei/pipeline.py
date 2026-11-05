"""
Star Kingdom Pipe Project -- end-to-end scaffold.

    1. risk model  -> expected leak cost for every pipe, every year 2027-2051
    2. circles     -> candidate dig circles (baseline: one tight circle per iron pipe)
    3. scheduler   -> order circles by expected leak cost avoided per dollar
    4. simulator   -> replay the list under the grader's rules, report a score
    5. export      -> submission CSV

Usage:
    python pipeline.py --pipes pipes.csv --train train.csv --out submission.csv

Every stage is a plain function so teammates can swap in better versions
(gradient boosting, merged circles, smarter ordering) without touching the rest.
"""
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eda import ANNUAL_BUDGET, build_panel, load

START, END = 2027, 2051
PLAN_YEARS = np.arange(START, END + 1)
EPS = 0.05  # metres of slack so rounding never pushes a pipe end outside its circle

CAT = ["material", "surface"]
NUM = ["age", "age_sq", "log_length"]


# --------------------------------------------------------------------------
# 1. Risk model: P(leak in year) x E[cost | leak]
# --------------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    df["age_sq"] = df.age ** 2
    df["log_length"] = np.log1p(df.length_m)
    return df


def preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ])


def fit_leak_model(panel):
    """Logistic regression: does this pipe leak this year?"""
    return Pipeline([("pre", preprocessor()),
                     ("lr", LogisticRegression(max_iter=2000))]).fit(panel[CAT + NUM], panel.leaked)


def fit_cost_model(panel):
    """Linear regression on log(cost per leak), with Duan smearing to get back the mean."""
    leaks = panel[panel.leaked == 1].copy()
    y = np.log(leaks.leak_cost / leaks.n_leaks)
    model = Pipeline([("pre", preprocessor()), ("ols", LinearRegression())]).fit(leaks[CAT + NUM], y)
    smear = np.mean(np.exp(y - model.predict(leaks[CAT + NUM])))
    return model, smear


def predict_expected_cost(df, leak_model, cost_model, smear):
    p = leak_model.predict_proba(df[CAT + NUM])[:, 1]
    cost = np.exp(cost_model.predict(df[CAT + NUM])) * smear
    return p, cost


def validate(panel, valid_from=2025):
    """Train on years < valid_from and score on the rest: time-based split, no leakage."""
    tr, va = panel[panel.year < valid_from], panel[panel.year >= valid_from]
    lm = fit_leak_model(tr)
    cm, smear = fit_cost_model(tr)
    p, c = predict_expected_cost(va, lm, cm, smear)
    print(f"validation {valid_from}+: AUC {roc_auc_score(va.leaked, p):.3f}, "
          f"log loss {log_loss(va.leaked, p):.4f}")
    check = pd.DataFrame({"year": va.year, "pred_leaks": p, "actual_leaks": va.n_leaks,
                          "pred_cost": p * c, "actual_cost": va.leak_cost}).groupby("year").sum()
    print(check.round(0))
    print("If pred_cost is far below actual_cost, the year-on-year cost trend is not "
          "captured by age alone. Consider a trend term or a tail model.")


def expected_cost_matrix(pipes, leak_model, cost_model, smear):
    """Array (n_pipes, 25): expected leak cost of each pipe in each plan year."""
    grid = pipes.loc[pipes.index.repeat(len(PLAN_YEARS))].copy()
    grid["year"] = np.tile(PLAN_YEARS, len(pipes))
    grid["age"] = grid.year - grid.lay_year
    grid = add_features(grid)
    p, c = predict_expected_cost(grid, leak_model, cost_model, smear)
    return (p * c).reshape(len(pipes), len(PLAN_YEARS))


# --------------------------------------------------------------------------
# 2. Circles
# --------------------------------------------------------------------------
@dataclass
class Circle:
    cx: float
    cy: float
    r: float


class PipeIndex:
    """Find which pipes sit fully inside a circle.

    A pipe is a straight segment and a circle is convex, so the pipe is inside
    exactly when both endpoints are. Its midpoint is then inside too, so the
    KD-tree query on midpoints never misses a candidate.
    """

    def __init__(self, pipes):
        self.p = pipes
        self.tree = cKDTree(pipes[["mid_x", "mid_y"]].to_numpy())
        self.x1, self.y1 = pipes.gps_x1.to_numpy(), pipes.gps_y1.to_numpy()
        self.x2, self.y2 = pipes.gps_x2.to_numpy(), pipes.gps_y2.to_numpy()
        self.length = pipes.length_m.to_numpy()
        self.rate = pipes.surface_rate.to_numpy()

    def inside(self, c):
        cand = np.asarray(self.tree.query_ball_point([c.cx, c.cy], c.r), dtype=int)
        if cand.size == 0:
            return cand
        ok = ((np.hypot(self.x1[cand] - c.cx, self.y1[cand] - c.cy) <= c.r) &
              (np.hypot(self.x2[cand] - c.cx, self.y2[cand] - c.cy) <= c.r))
        return cand[ok]

    def cost(self, c, members):
        """Price a circle given the (still unreplaced) pipes inside it."""
        if members.size == 0:
            return 0.0
        rate = np.unique(self.rate[members]).sum()  # each surface type counted once
        area = np.pi * c.r ** 2
        return 100 + (rate * area) ** 0.85 + 200 * self.length[members].sum()


def baseline_circles(pipes):
    """One tight circle per iron pipe. Always feasible; a benchmark to beat."""
    iron = pipes[pipes.is_iron]
    return [Circle(x, y, l / 2 + EPS) for x, y, l in zip(iron.mid_x, iron.mid_y, iron.length_m)]

# TODO (circle team): merge nearby same-surface, similar-risk pipes into larger
# circles (e.g. DBSCAN on midpoints, then a minimum enclosing circle per cluster).
# Keep a merge only if the simulator score improves.


# --------------------------------------------------------------------------
# 3. Scheduler
# --------------------------------------------------------------------------
def order_by_value(circles, idx, exp_cost):
    """Riskiest per dollar first: expected 2027 leak cost avoided / circle cost."""
    scores = []
    for c in circles:
        m = idx.inside(c)
        cost = idx.cost(c, m)
        scores.append(exp_cost[m, 0].sum() / cost if cost > 0 else 0)
    return [circles[i] for i in np.argsort(scores)[::-1]]


# --------------------------------------------------------------------------
# 4. Simulator
# --------------------------------------------------------------------------
def simulate(circles, pipes, idx, exp_cost, verbose=True):
    """Replay the grader's rules and estimate the score.

    Assumes a circle is dug at the start of its year, so the pipes in it cause
    no leak cost from that year on. Expected-value leak accounting: ignores that
    a pipe which leaks is replaced for free (TODO once the rule is confirmed).
    """
    n = len(pipes)
    replaced = np.zeros(n, bool)
    dig_year = np.full(n, END + 1)
    year, balance, spent = START, ANNUAL_BUDGET, 0.0
    spend_by_year = {}

    for c in circles:
        members = idx.inside(c)
        members = members[~replaced[members]]
        cost = idx.cost(c, members)
        if cost == 0:
            continue  # empty circle: free, skipped
        while cost > balance:
            year += 1
            balance += ANNUAL_BUDGET
            if year > END:
                raise RuntimeError("Plan runs past 2051: submission would be rejected")
        balance -= cost
        spent += cost
        spend_by_year[year] = spend_by_year.get(year, 0) + cost
        replaced[members] = True
        dig_year[members] = year

    missing = pipes.is_iron.to_numpy() & ~replaced
    if missing.any():
        raise RuntimeError(f"{missing.sum()} iron pipes not covered: submission would be rejected")

    still_in_ground = PLAN_YEARS[None, :] < dig_year[:, None]
    leak_cost = (exp_cost * still_in_ground).sum()
    leftover = ANNUAL_BUDGET * len(PLAN_YEARS) - spent
    score = leftover - leak_cost
    if verbose:
        print(f"digging done by {max(spend_by_year)}; spent ${spent / 1e6:,.1f}M, "
              f"leftover ${leftover / 1e6:,.1f}M, expected leaks ${leak_cost / 1e6:,.1f}M, "
              f"score ${score / 1e6:,.1f}M")
    return score, spend_by_year


# --------------------------------------------------------------------------
# 5. Export
# --------------------------------------------------------------------------
def export(circles, path):
    with open(path, "w") as f:
        for c in circles:
            f.write(f'"({c.cx:.3f}, {c.cy:.3f})",{c.r:.3f}\n')
    print(f"wrote {len(circles)} circles to {path}")


def main(pipes_path, train_path, out_path):
    pipes, train = load(pipes_path, train_path)
    pipes = pipes.dropna(subset=["lay_year", "surface_rate"]).reset_index(drop=True)
    panel = add_features(build_panel(pipes, train))

    validate(panel)
    leak_model = fit_leak_model(panel)
    cost_model, smear = fit_cost_model(panel)
    exp_cost = expected_cost_matrix(pipes, leak_model, cost_model, smear)

    idx = PipeIndex(pipes)
    circles = baseline_circles(pipes)

    print("\nUnordered baseline (file order):")
    simulate(circles, pipes, idx, exp_cost)
    print("Ordered by risk per dollar:")
    ordered = order_by_value(circles, idx, exp_cost)
    simulate(ordered, pipes, idx, exp_cost)
    export(ordered, out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipes", default="pipes.csv")
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    main(a.pipes, a.train, a.out)
