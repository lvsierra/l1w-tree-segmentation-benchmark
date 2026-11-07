"""Which reference trees do the methods find hard?

Inputs (from verify_boundary_fp.py): verify_out/trees_<result>.csv for the nine
canonical results, and verify_out/reference_tree_centroids.csv (same order as
the sorted tree ids). Output: verify_out/hard_trees.csv + a printed summary.
"""
import numpy as np
import pandas as pd

D = "verify_out"
RESULTS = {  # canonical files only (the ones that reproduce Table 3.2)
    "treeX": "treeX_preset",
    "TreeLearn pre": "TreeLearn_pretrained",
    "TreeLearn ft": "TreeLearn_finetuned_ep20",
    "SAT pre": "SAT_pretrained",
    "SAT ft": "SAT_finetuned_candB_SATftA",
    "TLS2Trees": "TLS2Trees_pretrained",
    "ForAINet": "ForAINet_pretrained",
    "FF3D pre": "FF3D_pretrained",
    "FF3D ft": "FF3D_finetuned_candB_ft3",
}

# 1. one table: a row per reference tree, a column per result (matched IoU, 0 if missed)
iou = {}
size = None
for name, f in RESULTS.items():
    t = pd.read_csv(f"{D}/trees_{f}.csv").set_index("gt_id").sort_index()
    iou[name] = t["matched_iou"].fillna(0.0)
    if name == "treeX":                       # native GT sizes
        size = t["gt_points"]
iou = pd.DataFrame(iou)

# 2. per-tree difficulty
tab = pd.DataFrame(index=iou.index)
tab["points"] = size
tab["missed_by"] = (iou == 0).sum(axis=1)                 # results that did not find it
tab["weak_by"] = ((iou > 0) & (iou < 0.8)).sum(axis=1)     # found but poorly cut
tab["mean_score"] = iou.mean(axis=1)                       # 0 = always missed, 1 = perfect
tab["worst_iou"] = iou.min(axis=1)

# 3. context from the centroids: crowding and position in the plot
c = pd.read_csv(f"{D}/reference_tree_centroids.csv").to_numpy()
c = pd.DataFrame(c, index=iou.index, columns=["x", "y"])
xy = c[["x", "y"]].to_numpy()
d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
np.fill_diagonal(d, np.inf)
tab["nn_dist_m"] = d.min(axis=1)                           # distance to nearest reference tree
tab["n_within_5m"] = (d < 5).sum(axis=1)

def hull(p):
    p = sorted(map(tuple, p))
    cr = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], q) <= 0: lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up) >= 2 and cr(up[-2], up[-1], q) <= 0: up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])
h = hull(xy); a, b = h, np.roll(h, -1, 0)
edge = np.full(len(xy), np.inf)
for (ax_, ay), (bx, by) in zip(a, b):
    ex, ey = bx-ax_, by-ay; px, py = xy[:, 0]-ax_, xy[:, 1]-ay
    t = np.clip((px*ex+py*ey)/(ex*ex+ey*ey), 0, 1)
    edge = np.minimum(edge, np.hypot(px-t*ex, py-t*ey))
tab["dist_to_edge_m"] = edge                               # distance to edge of evaluated area

tab = tab.join(iou.add_prefix("iou_"))
tab.sort_values("mean_score").to_csv(f"{D}/hard_trees.csv", float_format="%.4f")

# 4. summary
print("Trees missed by k of the 9 results:")
print(tab["missed_by"].value_counts().sort_index().to_string())
print("\nTrees found by all 9 but cut poorly (IoU<0.8) by >=3:",
      int(((tab.missed_by == 0) & (tab.weak_by >= 3)).sum()))

hard = tab["mean_score"] <= tab["mean_score"].quantile(0.10)
print(f"\nHardest 10% (n={hard.sum()}, mean score <= {tab.mean_score.quantile(.10):.3f}) vs rest - medians:")
cols = ["points", "nn_dist_m", "n_within_5m", "dist_to_edge_m"]
print(pd.DataFrame({"hardest": tab.loc[hard, cols].median(),
                    "rest": tab.loc[~hard, cols].median()}).round(2).to_string())

rk = lambda s: s.rank().to_numpy()
print("\nSpearman correlation with mean_score (all 156 trees):")
for col in cols:
    r = np.corrcoef(rk(tab[col]), rk(tab["mean_score"]))[0, 1]
    print(f"  {col:15s} {r:+.2f}")

print("\nPer result: trees missed, and overlap with the hardest 10%:")
for name in iou:
    m = iou[name] == 0
    print(f"  {name:14s} missed {int(m.sum()):3d}   of which in hardest 10%: {int((m & hard).sum())}")

print("\nHardest 12 trees:")
print(tab.sort_values("mean_score").head(12)[
    ["points", "missed_by", "weak_by", "mean_score", "nn_dist_m", "n_within_5m", "dist_to_edge_m"]
].round(2).to_string())

# correlation of per-tree IoU across results
print("\nMean pairwise Spearman correlation of per-tree IoU between results:",
      round(np.nanmean(np.where(np.eye(9, dtype=bool), np.nan, iou.rank().corr().to_numpy())), 2))
