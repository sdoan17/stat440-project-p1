"""
Star Kingdom Pipe Project -- exploratory data analysis.

Usage:
    python eda.py --pipes pipes.csv --train train.csv --out eda_out

Prints a text report, writes figures and summary tables to --out, and writes
panel.csv (one row per pipe per year, 2019-2026) for the modelling step.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

TRAIN_YEARS = range(2019, 2027)
ANNUAL_BUDGET = 10_000_000
N_YEARS = 25

# Substring of the lower-cased surface label -> $/m^2
# ("grassland" matches "grass", "structure" matches "struct", ...)
SURFACE_RATES = {"struct": 50, "water": 32, "road": 17,
                 "swamp": 12, "farm": 7, "grass": 3}

PIPE_COLS = ["pipe_id", "lay_date", "material", "surface",
             "gps_x1", "gps_y1", "gps_x2", "gps_y2"]
TRAIN_COLS = ["pipe_id", "date", "time", "cost"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def snake(name):
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def surface_rate(label):
    label = str(label).lower()
    for key, rate in SURFACE_RATES.items():
        if key in label:
            return rate
    return np.nan


def load(pipes_path, train_path):
    pipes = pd.read_csv(pipes_path)
    train = pd.read_csv(train_path)
    pipes.columns = [snake(c) for c in pipes.columns]
    train.columns = [snake(c) for c in train.columns]

    for df, cols, name in [(pipes, PIPE_COLS, "pipes"), (train, TRAIN_COLS, "train")]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns {missing}; "
                             f"found {list(df.columns)}. Rename them in load().")

    pipes["pipe_id"] = pipes["pipe_id"].astype(str).str.strip()
    pipes["lay_date"] = pd.to_datetime(pipes["lay_date"], errors="coerce")
    pipes["lay_year"] = pipes["lay_date"].dt.year
    pipes["material"] = pipes["material"].astype(str).str.strip().str.lower()
    pipes["surface"] = pipes["surface"].astype(str).str.strip().str.lower()
    pipes["length_m"] = np.hypot(pipes.gps_x2 - pipes.gps_x1, pipes.gps_y2 - pipes.gps_y1)
    pipes["mid_x"] = (pipes.gps_x1 + pipes.gps_x2) / 2
    pipes["mid_y"] = (pipes.gps_y1 + pipes.gps_y2) / 2
    pipes["is_iron"] = pipes["material"].str.contains("iron")
    pipes["surface_rate"] = pipes["surface"].map(surface_rate)

    train["pipe_id"] = train["pipe_id"].astype(str).str.strip()
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    train["year"] = train["date"].dt.year
    train["cost"] = pd.to_numeric(train["cost"], errors="coerce")
    train["time"] = train["time"].astype(str).str.strip()
    return pipes, train


# --------------------------------------------------------------------------
# Pipe-year panel: the modelling table
# --------------------------------------------------------------------------
def build_panel(pipes, train):
    """One row per (pipe, year) for every year 2019-2026 the pipe existed."""
    leaks = (train.groupby(["pipe_id", "year"])
                  .agg(n_leaks=("cost", "size"), leak_cost=("cost", "sum"))
                  .reset_index())
    frames = []
    for y in TRAIN_YEARS:
        alive = pipes.loc[pipes.lay_year <= y, ["pipe_id"]].copy()
        alive["year"] = y
        frames.append(alive)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(leaks, on=["pipe_id", "year"], how="left")
    panel[["n_leaks", "leak_cost"]] = panel[["n_leaks", "leak_cost"]].fillna(0)
    keep = ["pipe_id", "material", "surface", "surface_rate", "is_iron",
            "length_m", "lay_year", "mid_x", "mid_y"]
    panel = panel.merge(pipes[keep], on="pipe_id")
    panel["age"] = panel.year - panel.lay_year
    panel["leaked"] = (panel.n_leaks > 0).astype(int)
    panel = panel.sort_values(["pipe_id", "year"]).reset_index(drop=True)
    panel["prev_leaks"] = panel.groupby("pipe_id").n_leaks.cumsum() - panel.n_leaks
    return panel


def rate_table(panel, by):
    """Leak frequency and severity per group, normalised by exposure."""
    g = panel.groupby(by, observed=True)
    out = pd.DataFrame({
        "pipe_years": g.size(),
        "km_years": g.length_m.sum() / 1000,
        "leaks": g.n_leaks.sum(),
        "leak_cost": g.leak_cost.sum(),
    })
    out["leaks_per_1000_pipe_yr"] = 1000 * out.leaks / out.pipe_years
    out["leaks_per_km_yr"] = out.leaks / out.km_years
    out["mean_cost_per_leak"] = out.leak_cost / out.leaks.replace(0, np.nan)
    out["exp_cost_per_pipe_yr"] = out.leak_cost / out.pipe_years
    return out.sort_values("exp_cost_per_pipe_yr", ascending=False)


def min_circle_cost(pipes):
    """Cost of the smallest circle around each pipe on its own."""
    r = pipes.length_m / 2
    return 100 + (pipes.surface_rate * np.pi * r**2) ** 0.85 + 200 * pipes.length_m


def network_degree(pipes, tol=0.01):
    """How many pipe ends meet at each point (1 = dead end, 3+ = junction)."""
    ends = np.vstack([pipes[["gps_x1", "gps_y1"]].to_numpy(),
                      pipes[["gps_x2", "gps_y2"]].to_numpy()])
    keys = pd.Series(list(map(tuple, np.round(ends / tol).astype(np.int64))))
    return keys.value_counts()


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def main(pipes_path, train_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160, "display.max_columns", 20,
                  "display.float_format", "{:,.2f}".format)
    pipes, train = load(pipes_path, train_path)

    section("1. Overview and data quality")
    print(f"pipes: {pipes.shape}, train: {train.shape}")
    print("missing values in pipes:\n", pipes[PIPE_COLS].isna().sum()[lambda s: s > 0])
    print("missing values in train:\n", train[TRAIN_COLS].isna().sum()[lambda s: s > 0])
    print("duplicate pipe IDs in pipes:", pipes.pipe_id.duplicated().sum())
    print("zero-length pipes:", (pipes.length_m == 0).sum())
    unmapped = pipes.loc[pipes.surface_rate.isna(), "surface"].unique()
    print("surfaces with no rate (fix SURFACE_RATES):", list(unmapped))
    orphans = ~train.pipe_id.isin(pipes.pipe_id)
    print("leaks whose pipe is not in pipes.csv:", orphans.sum())
    lt = train.merge(pipes, on="pipe_id", how="inner")
    print("leaks dated before the pipe was laid:", (lt.date < lt.lay_date).sum())
    print(f"x range {pipes[['gps_x1', 'gps_x2']].min().min():,.0f}-"
          f"{pipes[['gps_x1', 'gps_x2']].max().max():,.0f}, "
          f"y range {pipes[['gps_y1', 'gps_y2']].min().min():,.0f}-"
          f"{pipes[['gps_y1', 'gps_y2']].max().max():,.0f}")

    section("2. Pipes: materials, surfaces, age, length")
    mat = pipes.groupby(["material", "is_iron"]).agg(
        n=("pipe_id", "size"), km=("length_m", lambda s: s.sum() / 1000),
        median_len=("length_m", "median"), median_lay_year=("lay_year", "median"))
    print(mat.sort_values("n", ascending=False))
    iron = pipes[pipes.is_iron]
    print(f"\nIron pipes: {len(iron):,}, total {iron.length_m.sum() / 1000:,.1f} km "
          f"-> ${200 * iron.length_m.sum() / 1e6:,.1f}M in pipe cost alone")
    print("\nIron pipes by surface:")
    print(iron.groupby("surface").agg(n=("pipe_id", "size"),
                                      km=("length_m", lambda s: s.sum() / 1000)))
    print("\nLay year range:", pipes.lay_year.min(), "-", pipes.lay_year.max())
    deg = network_degree(pipes)
    print("\nPipe-end connectivity (pipe ends meeting at a point -> number of points):")
    print(deg.value_counts().sort_index().head(8))

    section("3. Leaks over time and cost distribution")
    by_year = train.groupby("year").agg(
        leaks=("cost", "size"), total=("cost", "sum"),
        mean=("cost", "mean"), median=("cost", "median"), max=("cost", "max"))
    print(by_year)
    print("\nGrowth 2019->2026: leaks x{:.2f}, total cost x{:.2f}, median cost x{:.2f}".format(
        *(by_year.iloc[-1][["leaks", "total", "median"]] / by_year.iloc[0][["leaks", "total", "median"]])))
    q = train.cost.quantile([.5, .9, .99, .999])
    print("\nCost quantiles:\n", q)
    srt = train.cost.sort_values(ascending=False)
    for frac in (0.001, 0.01):
        k = max(1, int(frac * len(srt)))
        print(f"top {frac:.1%} of leaks ({k}) = {srt.iloc[:k].sum() / srt.sum():.1%} of all leak cost")
    print("\nTen largest leaks:")
    print(lt.nlargest(10, "cost")[["pipe_id", "date", "cost", "material", "surface", "length_m"]])
    print(f"\nshare of leaks with time '00:00' (possible missing time): "
          f"{(train.time == '00:00').mean():.1%} (uniform would be ~0.07%)")
    repeats = train.pipe_id.value_counts()
    print("\nleaks per leaking pipe:\n", repeats.value_counts().sort_index())
    print("If many pipes leak more than once, 'leaks fix themselves' may not mean "
          "the pipe becomes plastic. Check the assignment page.")

    section("4. Leak rates by risk factor (exposure-adjusted)")
    panel = build_panel(pipes, train)
    print(f"panel: {len(panel):,} pipe-years, "
          f"{pipes.lay_year.isna().sum()} pipes dropped for missing lay date")
    panel["age_bucket"] = pd.cut(panel.age, [-1, 10, 20, 30, 40, 50, 60, 70, 80, 200])
    tables = {}
    for by in ["material", "surface", "age_bucket", ["is_iron", "age_bucket"]]:
        name = by if isinstance(by, str) else "_".join(by)
        tables[name] = rate_table(panel, by)
        print(f"\n--- by {name} ---")
        print(tables[name])
        tables[name].to_csv(out / f"rates_by_{name}.csv")

    section("5. Feasibility: one circle per iron pipe")
    c = min_circle_cost(iron)
    print(f"total ${c.sum() / 1e6:,.1f}M vs ${ANNUAL_BUDGET * N_YEARS / 1e6:,.0f}M budget "
          f"-> about {c.sum() / ANNUAL_BUDGET:.1f} years of budget")
    print("per-pipe circle cost quantiles:\n", c.quantile([.1, .5, .9, .99]))
    print("cost share of pipe metres:", f"{(200 * iron.length_m).sum() / c.sum():.1%}")

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    by_year.leaks.plot.bar(ax=ax[0], title="Leaks per year")
    (by_year.total / 1e6).plot.bar(ax=ax[1], title="Leak cost per year ($M)")
    fig.tight_layout(); fig.savefig(out / "leaks_by_year.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(train.cost.clip(lower=1)), bins=60)
    ax.set(xlabel="log10(cost $)", ylabel="leaks", title="Leak cost distribution")
    fig.tight_layout(); fig.savefig(out / "cost_hist.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    tm = rate_table(panel, ["material", "age_bucket"]).reset_index()
    tm = tm[tm.pipe_years >= 200]
    for m, d in tm.groupby("material"):
        d = d.sort_values("age_bucket")
        ax.plot(d.age_bucket.astype(str), d.leaks_per_1000_pipe_yr, marker="o", label=m)
    ax.set(xlabel="age (years)", ylabel="leaks per 1,000 pipe-years",
           title="Leak rate by age and material")
    ax.legend(fontsize=8); plt.xticks(rotation=45)
    fig.tight_layout(); fig.savefig(out / "rate_by_age_material.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    segs = pipes[["gps_x1", "gps_y1", "gps_x2", "gps_y2"]].to_numpy().reshape(-1, 2, 2)
    colors = np.where(pipes.is_iron, "#444444", "#4a90d9")
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=0.4))
    ax.scatter(lt.mid_x, lt.mid_y, s=np.clip(lt.cost / 2000, 1, 200),
               c="crimson", alpha=0.3, linewidths=0)
    ax.autoscale(); ax.set_aspect("equal")
    ax.set_title("Pipes (grey = iron, blue = other) and leaks (red, sized by cost)")
    fig.tight_layout(); fig.savefig(out / "map.png", dpi=200); plt.close(fig)

    panel.drop(columns="age_bucket").to_csv(out / "panel.csv", index=False)
    print(f"\nWrote figures, rate tables and panel.csv to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipes", default="pipes.csv")
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--out", default="eda_out")
    a = ap.parse_args()
    main(a.pipes, a.train, a.out)
