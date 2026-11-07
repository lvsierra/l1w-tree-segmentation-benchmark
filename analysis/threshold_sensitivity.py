"""Sensitivity of the boundary-discounting rule to the 2 m buffer.

Reads only the stored verification output (fp_*.csv, centroids_*.csv,
gt_centroids_*.csv). Nothing is re-run and no prediction array is touched.
Pretrained SAT is scored on its own cloud, so its false-positive centroids are
brought into the native frame by the constant translation estimated from the
156 reference-tree centroids, exactly as in Table 3.3.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "barkla_verify"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "verify_out"

# TP / FN per canonical result (Table 3.2)
CANON = [
    ("treeX",     "preset",     "treeX_preset",                 151, 5),
    ("TreeLearn", "pretrained", "TreeLearn_pretrained",         156, 0),
    ("TreeLearn", "fine-tuned", "TreeLearn_finetuned_ep20",     155, 1),
    ("SAT",       "pretrained", "SAT_pretrained",               141, 15),
    ("SAT",       "fine-tuned", "SAT_finetuned_candB_SATftA",   149, 7),
    ("TLS2Trees", "pretrained", "TLS2Trees_pretrained",         153, 3),
    ("ForAINet",  "pretrained", "ForAINet_pretrained",          126, 30),
    ("FF3D",      "pretrained", "FF3D_pretrained",              149, 7),
    ("FF3D",      "fine-tuned", "FF3D_finetuned_candB_ft3",     151, 5),
]
THRESH = [0, 1, 2, 3, 5]


def convex_hull(pts):
    pts = sorted(map(tuple, pts))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


def signed_dist_to_hull(xy, hull):
    a, b = hull, np.roll(hull, -1, axis=0)
    d = np.full(len(xy), np.inf)
    inside = np.ones(len(xy), bool)
    for (ax, ay), (bx, by) in zip(a, b):
        ex, ey = bx - ax, by - ay
        px, py = xy[:, 0] - ax, xy[:, 1] - ay
        t = np.clip((px * ex + py * ey) / (ex * ex + ey * ey), 0, 1)
        d = np.minimum(d, np.hypot(px - t * ex, py - t * ey))
        inside &= (ex * py - ey * px) >= 0
    return np.where(inside, -d, d)


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


gt_n = read(os.path.join(OUT, "gt_centroids_native.csv"))
ids_n = np.array([int(r["gt_id"]) for r in gt_n])
xy_n = np.array([[float(r["x"]), float(r["y"])] for r in gt_n])
hull = convex_hull(xy_n)

# translation between the SAT cloud and the native one, from the shared trees
gt_s = read(os.path.join(OUT, "gt_centroids_sat.csv"))
ids_s = np.array([int(r["gt_id"]) for r in gt_s])
xy_s = np.array([[float(r["x"]), float(r["y"])] for r in gt_s])
common = np.intersect1d(ids_n, ids_s)
off = xy_n[np.searchsorted(ids_n, common)] - xy_s[np.searchsorted(ids_s, common)]
t = off.mean(0)
resid = np.hypot(*(off - t).T)
print(f"SAT -> native translation ({t[0]:.3f}, {t[1]:.3f}) m, "
      f"residual median {np.median(resid):.3f} max {resid.max():.3f} "
      f"over {len(common)} trees\n")

rows = []
for method, regime, key, tp, fn in CANON:
    fp_rows = read(os.path.join(OUT, f"fp_{key}.csv"))
    d = np.array([r["signed_dist_to_tree_hull_m"] for r in fp_rows])
    if (d == "").all():                       # SAT pretrained: no distances stored
        cen = {int(r["pred_id"]): (float(r["x"]), float(r["y"]))
               for r in read(os.path.join(OUT, f"centroids_{key}.csv"))}
        pts = np.array([cen[int(r["pred_id"])] for r in fp_rows]) + t
        dist = signed_dist_to_hull(pts, hull)
    else:
        dist = d.astype(float)
    n_fp = len(fp_rows)
    kept = [n_fp - int((dist > k).sum()) for k in THRESH]
    rows.append((method, regime, n_fp, kept,
                 [f1(tp, k, fn) for k in kept], f1(tp, n_fp, fn)))

hdr = "  ".join(f"{k:>2} m" for k in THRESH)
print(f"{'result':24s} {'FP':>4}   discounted F1 by buffer:  {hdr}")
for method, regime, n_fp, kept, f1s, strict in rows:
    name = f"{method} {regime}"
    print(f"{name:24s} {n_fp:4d}   strict {strict:.3f}   " +
          "  ".join(f"{v:.3f}" for v in f1s))
print()
print(f"{'result':24s} " + "  ".join(f"{k:>3} m" for k in THRESH) + "   (FP kept)")
for method, regime, n_fp, kept, f1s, strict in rows:
    print(f"{method + ' ' + regime:24s} " + "  ".join(f"{k:5d}" for k in kept))

f1_by_t = np.array([r[4] for r in rows])
print("\nrank correlation of the nine results between buffers:")
for i, k in enumerate(THRESH):
    order = np.argsort(-f1_by_t[:, i])
    print(f"  {k} m: " + " > ".join(f"{rows[j][0]} {rows[j][1]}" for j in order))
