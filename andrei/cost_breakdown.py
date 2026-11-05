"""
Where does our construction money go?

Replays a submission year by year ($10M a year, unspent money carries over,
circles with nothing left to replace are skipped) and splits the construction
cost into the four parts of the formula on slide 7:

    cost = 100 + (surface rate * area)^0.85 + 200 * (metres of non-polyurethane pipe)

  groundbreaking   $100 per circle
  area term        (rate * area)^0.85, rate = sum of the DISTINCT surfaces in the circle
  iron metres      $200/m of iron pipe   -> the same for every team, can't be reduced
  other metres     $200/m of copper/brass/unknown pipe we didn't need to dig

Everything except "iron metres" is money a better circle layout can save.

Usage:  python cost_breakdown.py submission_blend_a0p0_b0p0.csv
Run it in the folder that holds pipes.csv and train.csv.
"""
import math
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RATE = {"grassland": 3, "farmland": 7, "swamp": 12, "road": 17, "structure": 50, "water": 32}
SURFACES = list(RATE)
RATE_ARR = np.array([RATE[s] for s in SURFACES], float)
IRON = ["cast iron", "gray iron", "wrought iron"]
YEAR_BUDGET = 10e6
CENTER = re.compile(r'"?\(\s*([^,\s]+)\s*,\s*([^)\s]+)\s*\)"?')


class Network:
    """The pipe network as it stands at the start of 2027."""

    def __init__(self, folder="."):
        pipes = pd.read_csv(f"{folder}/pipes.csv")
        leaks = pd.read_csv(f"{folder}/train.csv", usecols=["Pipe ID"])
        self.N = len(pipes)
        self.material = pipes["Material"].fillna("unknown").str.strip().str.lower().to_numpy()
        surf = pipes["Surface"].str.strip().str.lower()
        if not surf.isin(SURFACES).all():
            raise ValueError(f"Unknown surfaces: {sorted(set(surf) - set(SURFACES))}")
        self.code = surf.map({s: i for i, s in enumerate(SURFACES)}).to_numpy()
        self.P1 = pipes[["GPS x1", "GPS y1"]].to_numpy(float)
        self.P2 = pipes[["GPS x2", "GPS y2"]].to_numpy(float)
        self.length = np.hypot(*(self.P2 - self.P1).T)
        self.tree = cKDTree((self.P1 + self.P2) / 2)
        self.max_half = self.length.max() / 2 + 1e-6
        leaked = pipes["Pipe ID"].isin(leaks["Pipe ID"]).to_numpy()
        self.live = (self.material != "polyurethane") & ~leaked      # still needs digging if inside a circle
        self.iron = np.isin(self.material, IRON)
        self.unknown = self.material == "unknown"

    def query(self, x, y, r):
        """(pipes fully inside, pipes with at least one end inside) for one circle."""
        c = np.asarray(self.tree.query_ball_point((x, y), r + self.max_half), dtype=int)
        if c.size == 0:
            return c, c
        d1 = np.hypot(self.P1[c, 0] - x, self.P1[c, 1] - y)
        d2 = np.hypot(self.P2[c, 0] - x, self.P2[c, 1] - y)
        return c[(d1 <= r) & (d2 <= r)], c[(d1 <= r) | (d2 <= r)]

    def rate(self, idx):
        """Sum of the distinct surface rates among these pipes."""
        return RATE_ARR[np.unique(self.code[idx])].sum() if len(idx) else 0.0


def read_circles(path):
    circles = []
    for line in open(path):
        line = line.strip()
        if not line or "," not in line:
            continue
        c, r = line.rsplit(",", 1)
        m = CENTER.fullmatch(c.strip())
        if not m:
            continue                                        # header or junk line
        circles.append((float(m[1]), float(m[2]), float(r)))
    return circles


def write_circles(path, circles):
    with open(path, "w") as f:
        for x, y, r in circles:
            f.write(f"\"({x:.6f}, {y:.6f})\",{r:.6f}\n")


def replay(net, circles, rule="inside"):
    """One row per circle that gets dug. rule: which pipes' surfaces set the rate
    ('inside' = both ends inside, our scorer; 'ends' = at least one end inside,
    the closest match in grader_pricing_check.py)."""
    done = ~net.live.copy()
    year, budget, rows = 1, YEAR_BUDGET, []
    for k, (x, y, r) in enumerate(circles):
        ins, ends = net.query(x, y, r)
        todo = ins[~done[ins]]
        if todo.size == 0:
            continue                                        # nothing left to replace: skipped, free
        src = ins if rule == "inside" else ends
        codes = np.unique(net.code[src])
        rate = RATE_ARR[codes].sum()
        area_term = (rate * math.pi * r * r) ** 0.85
        area_if_own = (net.rate(todo) * math.pi * r * r) ** 0.85
        iron_m = net.length[todo][net.iron[todo]].sum()
        other_m = net.length[todo].sum() - iron_m
        cost = 100 + area_term + 200 * (iron_m + other_m)
        while cost > budget:
            year += 1
            budget += YEAR_BUDGET
        budget -= cost
        done[ins] = True
        rows.append({"circle": k, "year": year, "cost": cost, "ground": 100.0,
                     "area": area_term, "area_if_own": area_if_own,
                     "iron_m": 200 * iron_m, "other_m": 200 * other_m,
                     "n_surf": len(codes), "rate": rate, "r": r,
                     "iron_pipes": int(net.iron[todo].sum()),
                     "struct_or_water": bool(np.isin(codes, [SURFACES.index("structure"),
                                                             SURFACES.index("water")]).any())})
    return pd.DataFrame(rows)


def m(v):
    return f"${v / 1e6:,.2f}M"


def breakdown(net, d, title):
    total = d["cost"].sum()
    print(f"\n=== {title} ===")
    print(f"Circles dug: {len(d):,}   construction: {m(total)}   last year of digging: {d['year'].max()}")
    parts = [("groundbreaking ($100/circle)", d["ground"].sum()),
             ("area term (rate*area)^0.85", d["area"].sum()),
             ("iron metres ($200/m)  [fixed]", d["iron_m"].sum()),
             ("other metres ($200/m copper/brass/unknown)", d["other_m"].sum())]
    for name, v in parts:
        print(f"  {name:<45} {m(v):>10}  {100 * v / total:5.1f}%")
    avoidable = total - d["iron_m"].sum()
    print(f"  {'-> money a better layout competes for':<45} {m(avoidable):>10}  {100 * avoidable / total:5.1f}%")
    return avoidable


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "submission_blend_a0p0_b0p0.csv"
    net = Network()
    circles = read_circles(path)
    print(f"Read {len(circles):,} circles from {path}")
    d = replay(net, circles)
    avoidable = breakdown(net, d, "Construction cost, our scorer's pricing")

    floor = 200 * net.length[net.live & net.iron].sum()
    print(f"\nIron floor (200 x every live iron metre): {m(floor)}  "
          f"(iron component above should match; a gap means uncovered or leak-freed pipes)")

    print("\n--- Where the area term comes from ---")
    g = d.assign(surfaces=d["n_surf"].clip(upper=4).map({1: "1", 2: "2", 3: "3", 4: "4+"}))
    t = g.groupby("surfaces").agg(circles=("cost", "size"), area_M=("area", "sum"),
                                  avg_rate=("rate", "mean"), avg_radius=("r", "mean"))
    t["area_M"] /= 1e6
    t["share"] = (100 * t["area_M"] / t["area_M"].sum()).round(1)
    print("By number of distinct surfaces in the circle (rates add up across surfaces):")
    print(t.round(2).to_string())
    sw = d.loc[d["struct_or_water"], "area"].sum()
    print(f"Area term in circles touching structure ($50) or water ($32): {m(sw)} "
          f"of {m(d['area'].sum())}")
    bystander = d["area"].sum() - d["area_if_own"].sum()
    print(f"Paid for surfaces of pipes we are NOT replacing in that circle: {m(bystander)}")

    print("\n--- By year ---")
    y = d.groupby("year").agg(circles=("cost", "size"), construction_M=("cost", "sum"),
                              area_M=("area", "sum"), iron_pipes=("iron_pipes", "sum"))
    y[["construction_M", "area_M"]] /= 1e6
    print(y.round(2).to_string())

    d2 = replay(net, circles, rule="ends")
    print(f"\nSame file under the other plausible grader rule (surfaces of pipes with an end inside): "
          f"{m(d2['cost'].sum())}")

    print("\n--- Verdict ---")
    if avoidable >= 15e6:
        print(f"{m(avoidable)} is reducible in principle. Run circle_optimizer.py next.")
    elif avoidable >= 6e6:
        print(f"{m(avoidable)} is reducible. A few $M is a stretch; circle_optimizer.py will tell you.")
    else:
        print(f"Only {m(avoidable)} is reducible. Layout gains will be small; probably not worth it.")


if __name__ == "__main__":
    main()
