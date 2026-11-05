"""
Why does the SAME set of circles cost $139.0M-$145.6M in the grader, depending only
on dig order, when our scorer says $138.76M for all of them?

Replays every submission file you've scored under several candidate pricing rules,
and compares each rule's construction total with the grader's. The right rule should
track the grader closely: high correlation, and a near-constant gap (the remaining gap
is the grader's future leaks, which we can't simulate).

Candidate rules for the surface rate of a circle (pipe metres are charged the same way
in all of them: $200/m for pipes still to replace):
  inside            surfaces of all pipes fully inside        (our scorer; order-free)
  inside, not done  surfaces of pipes fully inside not yet replaced (order matters)
  ends              surfaces of all pipes with an end inside  (order-free)
  ends, not done    surfaces of pipes with an end inside not yet replaced (order matters)

Usage: python pricing_check.py      (folder with pipes.csv, train.csv, cost_breakdown.py
                                     and the submission files)
"""
import math
import os

import numpy as np
import pandas as pd

from cost_breakdown import Network, read_circles

# Grader construction totals ($M) from the breakdowns you've pasted
GRADER = {
    "submission_blend_a0p7_b0p2.csv": 139.03,
    "submission_blend_a0p0_b0p0_depl.csv": 139.14,
    "submission_gbm.csv": 143.47,
    "submission_4yr.csv": 139.01,
    "submission_4yr_net.csv": 139.86,
    "submission_4yr_opt.csv": 139.57,
    "submission_5yr.csv": 140.12,
    "submission_4p5yr.csv": 139.54,
    "submission_3p5yr.csv": 139.04,
    "submission_5p5yr.csv": 140.74,
    "submission_cohort_4yr.csv": 140.23,
    "submission_cohort_rolling_4yr.csv": 143.22,
    "submission_rolling_4yr.csv": 145.61,
}
RULES = ["inside", "inside, not done", "ends", "ends, not done"]


def replay_all_rules(net, circles):
    done = ~net.live.copy()
    total = dict.fromkeys(RULES, 0.0)
    for x, y, r in circles:
        ins, ends = net.query(x, y, r)
        todo = ins[~done[ins]]
        if todo.size == 0:
            continue                                   # nothing left to replace: skipped
        area = math.pi * r * r
        metres = 200 * net.length[todo].sum()
        srcs = {"inside": ins, "inside, not done": todo,
                "ends": ends, "ends, not done": ends[~done[ends]]}
        for k, s in srcs.items():
            total[k] += 100 + (net.rate(s) * area) ** 0.85 + metres
        done[ins] = True
    return {k: v / 1e6 for k, v in total.items()}


def main():
    net = Network()
    rows = []
    for f, g in GRADER.items():
        if not os.path.exists(f):
            print(f"  (skipping {f}: not in this folder)")
            continue
        t = replay_all_rules(net, read_circles(f))
        rows.append({"file": f.replace("submission_", "").replace(".csv", ""), "grader": g, **t})
        print(f"  replayed {f}")
    d = pd.DataFrame(rows).set_index("file").sort_values("grader")
    print("\nConstruction $M: grader vs each candidate rule")
    print(d.round(2).to_string())

    print("\nHow well each rule tracks the grader across files:")
    summ = []
    for k in RULES:
        gap = d["grader"] - d[k]
        summ.append({"rule": k, "correlation": d["grader"].corr(d[k]),
                     "spread of rule ($M)": d[k].max() - d[k].min(),
                     "gap mean ($M)": gap.mean(), "gap std ($M)": gap.std()})
    s = pd.DataFrame(summ).set_index("rule")
    print(s.round(3).to_string())
    print(f"\nGrader spread across these files: ${d['grader'].max() - d['grader'].min():.2f}M")

    best = s["gap std ($M)"].idxmin()
    if s.loc[best, "correlation"] >= 0.8 and s.loc[best, "spread of rule ($M)"] >= 1.0:
        print(f"\nBest match: '{best}' (corr {s.loc[best, 'correlation']:.2f}, gap std "
              f"${s.loc[best, 'gap std ($M)']:.2f}M). This rule explains the order effect: we can now "
              "build plans that keep the rolling plans' leak savings without the construction penalty.")
    else:
        print("\nNo candidate rule explains the grader's order effect (none tracks it closely). "
              "The extra construction must come from something we don't model, such as the "
              "grader's future leaks.")


if __name__ == "__main__":
    main()
