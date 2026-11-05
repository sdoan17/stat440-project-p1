"""
Builds horizon_check.py from your ensemble_blend.py.

Can't backtest a 5-year horizon (the data ends in 2026), so this uses the age model
to SIMULATE each horizon's expected leak cost over the real plan (2027 onward):
  1. For each horizon (2-6 years), build the real plan the same way as the 4-year
     submission (age x depletion x leak cost, merged circles, $10M a year).
  2. Expected leak $ per year = every pipe's chance of leaking before its circle is
     dug (half exposure in the year it's dug), x its depletion-scaled leak cost.
  3. VALIDATION FIRST: the grader already showed 2 -> 4 years cut leaks by $0.99M,
     mostly in year 3. The simulator must reproduce that, or it isn't trusted.
  4. Rule (fixed in advance): the simulator must show a 2 -> 4 gain of at least $0.25M
     with the biggest drop in year 3 or 4; then switch from 4 years only if 5 or 6
     beats 4 by at least a quarter of that gain AND at least $0.2M simulated.

The age model under-predicts leak LEVELS by about 2x (the O/E of ~2 we saw), so
compare differences and patterns, not absolute dollars.

Usage:  python make_horizon_check.py      then      python horizon_check.py
"""
SRC, OUT, MARKER = "ensemble_blend.py", "horizon_check.py", "\nGRID = ["

BODY = r'''
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
'''

src = open(SRC).read()
if MARKER not in src:
    raise SystemExit(f"Couldn't find '{MARKER.strip()}' in {SRC}. Is this the blend script?")
open(OUT, "w").write(src[: src.index(MARKER)] + BODY)
print(f"Wrote {OUT}. Next: python {OUT}")
