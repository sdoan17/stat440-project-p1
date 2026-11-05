"""
Depletion multiplier for every row of pipes.csv (same order as the file).

Batch = iron pipes touching end to end (within 0.5 m) and laid within 14 days.
Each pipe fails at most once, so a batch that has mostly leaked has little risk left:
    multiplier = (pipes in batch not yet leaked + 1) / (batch size + 1)
Pipes that aren't iron or have no lay date get 1.0 (no change).

Usage in ensemble_blend.py:
    from depletion import depletion_multiplier
    DEPL = depletion_multiplier()            # uses every leak in train.csv (plan for 2027)
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

IRON = ["cast iron", "gray iron", "wrought iron"]
TOUCH = 0.5
DAYS = 14


def depletion_multiplier(pipes_path="pipes.csv", leaks_path="train.csv", before_year=2027):
    p = pd.read_csv(pipes_path)
    l = pd.read_csv(leaks_path)
    date_col = "Date" if "Date" in l else next(c for c in l if "date" in c.lower())
    l = l[pd.to_datetime(l[date_col]).dt.year < before_year]
    leaked_all = p["Pipe ID"].isin(l["Pipe ID"]).to_numpy()

    material = p["Material"].fillna("unknown").str.strip().str.lower()
    lay_all = pd.to_datetime(p["Lay date"], errors="coerce")
    use = (material.isin(IRON) & lay_all.notna()).to_numpy()
    rows = np.flatnonzero(use)
    n = rows.size

    P1 = p.loc[use, ["GPS x1", "GPS y1"]].to_numpy(float)
    P2 = p.loc[use, ["GPS x2", "GPS y2"]].to_numpy(float)
    owner = np.r_[np.arange(n), np.arange(n)]
    pairs = cKDTree(np.vstack([P1, P2])).query_pairs(TOUCH, output_type="ndarray")
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    lay = lay_all[use].to_numpy()
    keep = (a != b) & (np.abs((lay[a] - lay[b]) / np.timedelta64(1, "D")) <= DAYS)
    g = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(n, n))
    comp = connected_components(g, directed=False)[1]

    size = np.bincount(comp)
    gone = np.bincount(comp, leaked_all[rows].astype(float))
    mult = np.ones(len(p))
    mult[rows] = (size[comp] - gone[comp] + 1) / (size[comp] + 1)
    return mult


if __name__ == "__main__":
    m = depletion_multiplier()
    print(f"{len(m):,} rows; multiplier min {m.min():.2f}, median {np.median(m):.2f}, "
          f"share below 0.5: {(m < 0.5).mean():.1%}")
