"""At what distance does per-tree difficulty stop being correlated?

Reads verify_out/hard_trees.csv (per-tree difficulty = mean matched IoU across
the nine results, plus each tree's position via gt_centroids_native.csv) and
estimates spatial autocorrelation in distance bands: Moran's I with binary
band weights, with a permutation test (the tree scores are shuffled over the
fixed positions, so the null keeps the plot's geometry).

The point is to choose the bootstrap block size from the data
"""
import csv
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "verify_out"
RNG = np.random.default_rng(0)
NPERM = 2000
EDGES = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40]


def read(p):
    with open(os.path.join(OUT, p)) as f:
        return list(csv.DictReader(f))


hard = {int(r["gt_id"]): r for r in read("hard_trees.csv")}
cen = {int(r["gt_id"]): (float(r["x"]), float(r["y"])) for r in read("gt_centroids_native.csv")}
ids = sorted(set(hard) & set(cen))
xy = np.array([cen[i] for i in ids])
z = np.array([float(hard[i]["mean_score"]) for i in ids])
nn = np.array([float(hard[i]["nn_dist_m"]) for i in ids])
z = z - z.mean()
n = len(ids)
print(f"{n} reference trees with a difficulty score\n")

print("Spacing of the reference trees (distance to nearest neighbour):")
for q in (10, 25, 50, 75, 90):
    print(f"  p{q:<3d} {np.percentile(nn, q):6.2f} m")
print(f"  mean {nn.mean():.2f} m\n")

d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).T).T
np.fill_diagonal(d, np.inf)

print(f"Moran's I of per-tree difficulty by distance band ({NPERM} permutations):")
print(f"  {'band':>12}  {'pairs':>6}  {'I':>7}  {'p':>6}")
ranges = []
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    w = ((d >= lo) & (d < hi)).astype(float)
    npairs = int(w.sum() / 2)
    if npairs < 30:
        continue
    def moran(v):
        return (n / w.sum()) * (v @ w @ v) / (v @ v)
    obs = moran(z)
    null = np.array([moran(RNG.permutation(z)) for _ in range(NPERM)])
    p = (np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()
    ranges.append((lo, hi, obs, p))
    print(f"  {lo:4.0f}-{hi:<4.0f} m  {npairs:6d}  {obs:+7.3f}  {p:6.3f}")

sig = [r for r in ranges if r[3] < 0.05 and r[2] > 0]
if sig:
    print(f"\nPositive autocorrelation is detectable out to "
          f"{max(r[1] for r in sig):.0f} m; beyond that the bands are "
          f"indistinguishable from the permuted null.")
