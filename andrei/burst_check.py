"""
Do the leak bursts (2022, 2026, and year 4 = 2030 in the grader) hit the same pipes?

For each way of grouping pipes (place, cohort, age, surface, material, length), compare
each group's leak rate in a burst year with its rate in normal years (2021, 2023-2025).
"relative burst ratio" = how much harder the burst hit this group than the network
as a whole (1.0 = same as everyone).

The test that matters is OUT OF SAMPLE: rank groups by how hard 2022 hit them, then
look at 2026. If the groups 2022 hit hardest were also hit hardest in 2026, bursts
are predictable and the plan can get ahead of 2030. If not, a burst is like bad
weather: it hits everyone, and no dig order can dodge it.

Usage: python burst_check.py      (in the folder with pipes.csv and train.csv)
"""
import numpy as np
import pandas as pd

BURST = [2022, 2026]
NORMAL = [2021, 2023, 2024, 2025]
CELL = 1000.0            # metres, for the "place" grouping
MIN_PIPE_YEARS = 300     # skip groups too small to say anything

p = pd.read_csv("pipes.csv")
l = pd.read_csv("train.csv")
date_col = "Date" if "Date" in l else next(c for c in l if "date" in c.lower())
cost_col = "Cost" if "Cost" in l else next(c for c in l if "cost" in c.lower())
l["date"] = pd.to_datetime(l[date_col])
l["year"], l["month"] = l["date"].dt.year, l["date"].dt.month

# --- 1. Is a burst a short event or a whole bad year? --------------------------------
print("=== Leaks by month (all materials) ===")
bymonth = l.pivot_table(index="year", columns="month", values=cost_col, aggfunc="size", fill_value=0)
print(bymonth.to_string())
yr = l.groupby("year")[cost_col].agg(leaks="size", total_M="sum", mean_cost="mean")
yr["total_M"] /= 1e6
print("\n=== Leaks per year ===")
print(yr.round(2).to_string())

# --- 2. Pipe-year panel ----------------------------------------------------------------
first = l.sort_values("date").drop_duplicates("Pipe ID").set_index("Pipe ID")
ly = p["Pipe ID"].map(first["year"]).fillna(9999).astype(int).to_numpy()
mat = p["Material"].fillna("unknown").str.strip().str.lower()
elig = (mat != "polyurethane").to_numpy()
lay = pd.to_datetime(p["Lay date"], errors="coerce").dt.year.to_numpy(float)
x = (p["GPS x1"] + p["GPS x2"]).to_numpy() / 2
y = (p["GPS y1"] + p["GPS y2"]).to_numpy() / 2
length = np.hypot(p["GPS x2"] - p["GPS x1"], p["GPS y2"] - p["GPS y1"]).to_numpy()
static = pd.DataFrame({
    "place (1 km cell)": (x // CELL).astype(int).astype(str) + "_" + (y // CELL).astype(int).astype(str),
    "lay cohort (5 yrs)": pd.Series((lay // 5) * 5).map(lambda v: "none" if np.isnan(v) else f"{int(v)}"),
    "surface": p["Surface"].str.strip().str.lower(),
    "material": mat,
    "length quartile": pd.qcut(length, 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"]).astype(str),
})

rows = []
for t in BURST + NORMAL:
    r = elig & (ly >= t)
    d = static[r].copy()
    d["age (5 yrs)"] = pd.Series(((t - lay[r]) // 5) * 5).map(
        lambda v: "none" if np.isnan(v) else f"{int(v)}-{int(v) + 4}").to_numpy()
    d["year"] = t
    d["leak"] = ly[r] == t
    rows.append(d)
panel = pd.concat(rows, ignore_index=True)


def overall_ratio(b):
    n = panel[panel.year.isin(NORMAL)]
    return panel.loc[panel.year == b, "leak"].mean() / n["leak"].mean()


R = {b: overall_ratio(b) for b in BURST}
print(f"\nWhole network: 2022 leak rate = {R[2022]:.2f}x normal years, 2026 = {R[2026]:.2f}x")


def table(col):
    g = panel.groupby([col, "year"])["leak"].agg(["sum", "count"]).unstack("year")
    normal_leaks = g["sum"][NORMAL].sum(axis=1)
    normal_n = g["count"][NORMAL].sum(axis=1)
    out = pd.DataFrame({"pipe-years": g["count"].sum(axis=1), "normal rate %": 100 * normal_leaks / normal_n})
    for b in BURST:
        rate_b = g["sum"][b] / g["count"][b]
        out[f"rel {b}"] = (rate_b / (normal_leaks / normal_n)) / R[b]
        out[f"extra leaks {b}"] = g["sum"][b] - (normal_leaks / normal_n) * g["count"][b]
    big = (g["count"][BURST].min(axis=1) >= MIN_PIPE_YEARS / 3) & (normal_n >= MIN_PIPE_YEARS)
    return out[big].replace([np.inf, -np.inf], np.nan).dropna()


print("\n=== Does 2022 predict 2026? (out-of-sample test, per grouping) ===")
print("corr = do the same groups get hit harder in both bursts (by group, weighted by size)")
print("top-fifth = 2026 relative ratio of the fifth of pipes in groups 2022 hit hardest\n")
summary = []
for col in ["place (1 km cell)", "lay cohort (5 yrs)", "age (5 yrs)", "surface", "material", "length quartile"]:
    t = table(col)
    if len(t) < 3:
        continue
    w = t["pipe-years"]
    a, b = t["rel 2022"], t["rel 2026"]
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    corr = np.average((a - ma) * (b - mb), weights=w) / np.sqrt(
        np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w))
    s = t.sort_values("rel 2022", ascending=False)
    cum = s["pipe-years"].cumsum() / s["pipe-years"].sum()
    top = s[cum <= 0.2] if (cum <= 0.2).any() else s.iloc[:1]
    top26 = np.average(top["rel 2026"], weights=top["pipe-years"])
    rest = s.drop(top.index)
    rest26 = np.average(rest["rel 2026"], weights=rest["pipe-years"]) if len(rest) else np.nan
    spread = np.sqrt(np.average((b - mb) ** 2, weights=w))
    summary.append({"grouping": col, "groups": len(t), "corr 2022 vs 2026": corr,
                    "top-fifth in 2026": top26, "rest in 2026": rest26, "spread 2026": spread})
    if col != "place (1 km cell)":
        print(f"--- {col} ---")
        print(t.round(2).to_string())
        print()
    else:
        print(f"--- {col}: {len(t)} cells; 8 hit hardest in 2022 and how they fared in 2026 ---")
        print(t.sort_values("rel 2022", ascending=False).head(8).round(2).to_string())
        print()

sm = pd.DataFrame(summary).set_index("grouping")
print("=== Summary ===")
print(sm.round(2).to_string())

print("\n--- Reading it ---")
if "age (5 yrs)" in sm.index:
    print(f"Age: corr {sm.loc['age (5 yrs)', 'corr 2022 vs 2026']:.2f}. Your age model already covers "
          "whatever age explains, so it doesn't count as a new lever.")
cand = sm.drop(index="age (5 yrs)", errors="ignore")
best = cand.loc[cand["corr 2022 vs 2026"].idxmax()]
gap = best["top-fifth in 2026"] - best["rest in 2026"]
if best["corr 2022 vs 2026"] >= 0.5 and gap >= 0.3:
    print(f"Repeatable: grouping by {best.name}, the groups 2022 hit hardest were hit "
          f"{best['top-fifth in 2026']:.2f}x as hard in 2026 vs {best['rest in 2026']:.2f}x for the rest. "
          "A burst-aware weight for 2030 is worth testing on the 2023-2026 window.")
elif best["corr 2022 vs 2026"] >= 0.3:
    print(f"Weak repeat (best: {best.name}, corr {best['corr 2022 vs 2026']:.2f}). Maybe worth one "
          "pre-committed test, but expect a small gain.")
else:
    print("Bursts don't hit the same groups twice: they behave like network-wide bad years. "
          "No dig order can get ahead of 2030; keep the $76.44M plan.")
