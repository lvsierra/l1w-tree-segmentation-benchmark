"""
export_centroids.py - plan-view centroid of every predicted instance, and of
the 156 reference trees, for each canonical result.

Needed for the spatial block bootstrap: false positives have no reference tree
to attach them to, so they are placed in the plot by their own centroid and
resampled with the block they fall in.

Coordinates:
  * the eight results scored on the native cloud use --xyz (native vert .npy);
  * pretrained SAT is scored on its own 86.9-million-point cloud, so its
    coordinates come from --sat-laz (its PredInstance field must equal the
    prediction array, which the script checks).

Output, in --out:
  centroids_<result>.csv        pred_id, n_points, x, y
  gt_centroids_native.csv       gt_id, n_points, x, y      (from the native cloud)
  gt_centroids_sat.csv          gt_id, n_points, x, y      (from the SAT cloud)
  extent.json                   coordinate ranges of both clouds

Nothing else is recomputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

BG = (0, -1)


def centroids(labels, xy):
    """(ids, counts, x, y) for every non-background label."""
    labels = np.asarray(labels).astype(np.int64, copy=False).ravel()
    keep = ~np.isin(labels, BG)
    ids, inv, cnt = np.unique(labels[keep], return_inverse=True, return_counts=True)
    x = np.bincount(inv, weights=xy[keep, 0], minlength=len(ids)) / cnt
    y = np.bincount(inv, weights=xy[keep, 1], minlength=len(ids)) / cnt
    return ids, cnt, x, y


def write(path, ids, cnt, x, y, idname):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([idname, "n_points", "x", "y"])
        for i, c, a, b in zip(ids, cnt, x, y):
            w.writerow([int(i), int(c), round(float(a), 3), round(float(b), 3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True, help="same methods.tsv as verify_boundary_fp.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xyz", required=True, help="native coordinates .npy, shape (85289646, 3+)")
    ap.add_argument("--gt-native", required=True)
    ap.add_argument("--sat-laz", default=None, help="SAT output .laz with PredInstance (86.9M points)")
    ap.add_argument("--gt-sat", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.methods) as f:
        rows = [r for r in csv.DictReader(
            (l for l in f if l.strip() and not l.startswith("#")), delimiter="\t")]

    xy = np.asarray(np.load(args.xyz, mmap_mode="r")[:, :2], dtype=np.float64)
    print(f"[native] {len(xy):,} points  x {xy[:,0].min():.2f}..{xy[:,0].max():.2f}"
          f"  y {xy[:,1].min():.2f}..{xy[:,1].max():.2f}")
    extent = {"native": [float(xy[:, 0].min()), float(xy[:, 0].max()),
                         float(xy[:, 1].min()), float(xy[:, 1].max())]}

    gt = np.asarray(np.load(args.gt_native))
    ids, cnt, gx, gy = centroids(gt, xy)
    write(os.path.join(args.out, "gt_centroids_native.csv"), ids, cnt, gx, gy, "gt_id")
    print(f"[native] {len(ids)} reference trees")
    del gt

    sat_xy = None
    if args.sat_laz:
        import laspy
        las = laspy.read(args.sat_laz)
        sat_xy = np.c_[np.asarray(las.x), np.asarray(las.y)]
        sat_pred_field = np.asarray(las["PredInstance"]).astype(np.int64)
        del las
        print(f"[sat]    {len(sat_xy):,} points  x {sat_xy[:,0].min():.2f}..{sat_xy[:,0].max():.2f}"
              f"  y {sat_xy[:,1].min():.2f}..{sat_xy[:,1].max():.2f}")
        extent["sat"] = [float(sat_xy[:, 0].min()), float(sat_xy[:, 0].max()),
                         float(sat_xy[:, 1].min()), float(sat_xy[:, 1].max())]
        if args.gt_sat:
            gs = np.asarray(np.load(args.gt_sat))
            ids_s, cnt_s, sx, sy = centroids(gs, sat_xy)
            write(os.path.join(args.out, "gt_centroids_sat.csv"), ids_s, cnt_s, sx, sy, "gt_id")
            # the two clouds must place the same trees in the same place
            common = np.intersect1d(ids, ids_s)
            d = np.hypot(gx[np.searchsorted(ids, common)] - sx[np.searchsorted(ids_s, common)],
                         gy[np.searchsorted(ids, common)] - sy[np.searchsorted(ids_s, common)])
            print(f"[check]  {len(common)} trees in both clouds; centroid offset "
                  f"median {np.median(d):.3f} m, max {d.max():.3f} m "
                  f"-> {'SAME coordinate frame' if d.max() < 1.0 else 'DIFFERENT frames, do not mix'}")
            del gs

    for r in rows:
        name = f"{r['method']}_{r['regime']}"
        pred = np.asarray(np.load(r["pred"], mmap_mode="r"))
        if len(pred) == len(xy):
            coords, src = xy, "native"
        elif sat_xy is not None and len(pred) == len(sat_xy):
            same = float((sat_pred_field == pred).mean())
            if same < 0.999:
                print(f"  !! {name}: SAT cloud does not match this prediction "
                      f"({same:.4f}), skipped")
                continue
            coords, src = sat_xy, "sat"
        else:
            print(f"  !! {name}: no coordinates for {len(pred):,} points, skipped")
            continue
        ids_p, cnt_p, px, py = centroids(pred, coords)
        write(os.path.join(args.out, f"centroids_{name}.csv"), ids_p, cnt_p, px, py, "pred_id")
        print(f"  {name:28s} {len(ids_p):5d} predicted instances  ({src} coordinates)")
        del pred

    with open(os.path.join(args.out, "extent.json"), "w") as f:
        json.dump(extent, f, indent=2)
    print(f"\nDone. Files in {args.out}/")


if __name__ == "__main__":
    main()
