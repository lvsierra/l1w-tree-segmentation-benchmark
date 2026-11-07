"""
verify_boundary_fp.py - are the L1W "false positives" real mistakes, or real
trees that simply have no label in the 156-tree ground truth?

For every method listed in a TSV file it:
  1. re-scores the prediction with exactly the rule of instance_iou_eval.py
     (greedy one-to-one IoU matching, tau = 0.5, background ids 0 and -1)
     -> must reproduce Table 3.2 (TP/FP/FN). If it does not, the paths are wrong.
  2. describes every false positive: size, and which fraction of its points
     falls on GT label -1, on GT label 0, and on reference trees;
  3. re-scores under two "ignore" rules applied to UNMATCHED predictions only:
       rule A: ignore a prediction if > 50 % of its points have GT label -1
       rule B: ignore a prediction if > 50 % of its points are GT background
               (label 0 or -1)  - the eligibility rule of the FOR-instance arm
     Ignored predictions count as neither TP nor FP. Matched predictions are
     never ignored (a match needs IoU >= 0.5 with a real tree).
  4. optionally (--laz), uses the point coordinates to measure where each FP
     lies relative to the convex hull of the 156 reference trees;
  5. saves per-tree outcomes (best matched IoU per reference tree) for the
     later bootstrap / hardest-tree analyses.

Nothing is overwritten: all output goes to --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

BG = (0, -1)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_labels(path: str, field: str | None = None) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    if path.endswith(".npz"):
        z = np.load(path)
        if field is None:
            raise SystemExit(f"{path} is .npz, keys={list(z.keys())}: add ':<key>' to the path")
        return z[field]
    raise SystemExit(f"unsupported file type: {path}")


def parse_path(p: str):
    # allow "file.npz:key"
    if ".npz:" in p:
        f, k = p.rsplit(":", 1)
        return f, k
    return p, None


# --------------------------------------------------------------------------
# scoring - same rule as instance_iou_eval.py, vectorised for 85M points
# --------------------------------------------------------------------------
def score(gt: np.ndarray, pred: np.ndarray, tau: float):
    gt = np.asarray(gt).astype(np.int64, copy=False).ravel()
    pred = np.asarray(pred).astype(np.int64, copy=False).ravel()
    if gt.shape != pred.shape:
        raise ValueError(f"length mismatch: gt {gt.shape} vs pred {pred.shape}")

    gt_fg = ~np.isin(gt, BG)
    pr_fg = ~np.isin(pred, BG)

    g_ids, g_sizes = np.unique(gt[gt_fg], return_counts=True)

    # predicted instances + background composition of each one
    p_ids, p_inv, p_sizes = np.unique(pred[pr_fg], return_inverse=True, return_counts=True)
    g_under_p = gt[pr_fg]
    n_m1 = np.bincount(p_inv, weights=(g_under_p == -1), minlength=len(p_ids)).astype(np.int64)
    n_0 = np.bincount(p_inv, weights=(g_under_p == 0), minlength=len(p_ids)).astype(np.int64)
    n_tree = p_sizes - n_m1 - n_0
    del g_under_p

    # intersections over points that are foreground in both
    both = gt_fg & pr_fg
    gi = np.searchsorted(g_ids, gt[both])
    pi = np.searchsorted(p_ids, pred[both])
    key = gi.astype(np.int64) * len(p_ids) + pi
    del gi, pi, both
    ukey, inter = np.unique(key, return_counts=True)
    del key
    pair_g = ukey // len(p_ids)
    pair_p = ukey % len(p_ids)
    iou = inter / (g_sizes[pair_g] + p_sizes[pair_p] - inter)

    # best IoU of every prediction with any reference tree (for diagnostics)
    best_iou_p = np.zeros(len(p_ids))
    np.maximum.at(best_iou_p, pair_p, iou)

    # greedy one-to-one, highest IoU first, stop below tau
    order = np.argsort(-iou, kind="stable")
    used_g = np.zeros(len(g_ids), bool)
    used_p = np.zeros(len(p_ids), bool)
    match_iou_g = np.full(len(g_ids), np.nan)
    for k in order:
        if iou[k] < tau:
            break
        g, p = pair_g[k], pair_p[k]
        if used_g[g] or used_p[p]:
            continue
        used_g[g] = used_p[p] = True
        match_iou_g[g] = iou[k]

    return dict(g_ids=g_ids, g_sizes=g_sizes, p_ids=p_ids, p_sizes=p_sizes,
                n_m1=n_m1, n_0=n_0, n_tree=n_tree, best_iou_p=best_iou_p,
                used_g=used_g, used_p=used_p, match_iou_g=match_iou_g,
                n_gt_points=int(gt_fg.sum()))


def metrics(tp: int, fp: int, fn: int, match_iou):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    m = float(np.nanmean(match_iou)) if tp else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 4), recall=round(r, 4),
                f1=round(f1, 4), mean_iou=round(m, 4))


# optional geometry
def convex_hull(pts: np.ndarray) -> np.ndarray:
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


def signed_dist_to_hull(xy: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """> 0 outside the hull, < 0 inside (metres)."""
    a = hull
    b = np.roll(hull, -1, axis=0)
    d = np.full(len(xy), np.inf)
    inside = np.ones(len(xy), bool)
    for (ax, ay), (bx, by) in zip(a, b):
        ex, ey = bx - ax, by - ay
        px, py = xy[:, 0] - ax, xy[:, 1] - ay
        t = np.clip((px * ex + py * ey) / (ex * ex + ey * ey), 0, 1)
        d = np.minimum(d, np.hypot(px - t * ex, py - t * ey))
        inside &= (ex * py - ey * px) >= 0          # counter-clockwise hull
    return np.where(inside, -d, d)


def centroids(labels: np.ndarray, ids: np.ndarray, xy: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int64, copy=False).ravel()
    m = np.isin(labels, ids)
    idx = np.searchsorted(ids, labels[m])
    cnt = np.bincount(idx, minlength=len(ids))
    cx = np.bincount(idx, weights=xy[m, 0], minlength=len(ids)) / np.maximum(cnt, 1)
    cy = np.bincount(idx, weights=xy[m, 1], minlength=len(ids)) / np.maximum(cnt, 1)
    return np.c_[cx, cy]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--laz", default=None,
                    help="native L1W .laz/.las (85,289,646 points, same order as the native GT)")
    ap.add_argument("--xyz", default=None,
                    help="alternative to --laz: .npy of shape (N, 3+) with the native point coordinates")
    ap.add_argument("--xyz-labels", default=None,
                    help="instance labels stored alongside --xyz; used only to check that the "
                         "coordinates are in the same point order as the native GT")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.methods) as f:
        rows = list(csv.DictReader((l for l in f if l.strip() and not l.startswith("#")),
                                   delimiter="\t"))

    gt_cache: dict[str, np.ndarray] = {}
    census_done: set[str] = set()
    xy = hull = None
    if args.laz:
        import laspy
        las = laspy.read(args.laz)
        xy = np.c_[np.asarray(las.x), np.asarray(las.y)]
        print(f"[laz] {len(xy):,} points")
    elif args.xyz:
        v = np.load(args.xyz, mmap_mode="r")
        xy = np.asarray(v[:, :2], dtype=np.float64)
        print(f"[xyz] {len(xy):,} points from {os.path.basename(args.xyz)}")
    order_checked = not (args.xyz and args.xyz_labels)

    summary = []
    for r in rows:
        name = f"{r['method']}_{r['regime']}"
        t0 = time.time()
        gpath, gkey = parse_path(r["gt"])
        if gpath not in gt_cache:
            gt_cache[gpath] = np.asarray(load_labels(gpath, gkey)).astype(np.int64)
        gt = gt_cache[gpath]
        ppath, pkey = parse_path(r["pred"])
        pred = np.asarray(load_labels(ppath, pkey))

        if gpath not in census_done:
            vals, cnts = np.unique(gt[gt < 1], return_counts=True)
            n_inst = len(np.unique(gt[gt >= 1]))
            print(f"\n[GT census] {os.path.basename(gpath)}: {len(gt):,} points, "
                  f"{n_inst} instances (ids >= 1)")
            for v, c in zip(vals, cnts):
                print(f"    label {v:>3}: {c:>12,} points ({100 * c / len(gt):.1f} %)")
            census_done.add(gpath)

        if not order_checked and xy is not None and len(gt) == len(xy):
            chk = score(gt, np.load(args.xyz_labels, mmap_mode="r"), args.tau)
            n_ok = int(chk["used_g"].sum())
            miou = float(np.nanmean(chk["match_iou_g"])) if n_ok else 0.0
            same = n_ok >= 0.95 * len(chk["g_ids"]) and miou > 0.95
            print(f"[order check] labels next to the coordinates match {n_ok}/{len(chk['g_ids'])} "
                  f"reference trees, mean IoU {miou:.3f} -> "
                  f"{'SAME point order, coordinates used' if same else 'DIFFERENT order, coordinates IGNORED'}")
            if not same:
                xy = None
            order_checked = True
            del chk

        s = score(gt, pred, args.tau)
        tp = int(s["used_g"].sum())
        fn = len(s["g_ids"]) - tp
        unm = ~s["used_p"]
        fp = int(unm.sum())
        base = metrics(tp, fp, fn, s["match_iou_g"])

        frac_m1 = s["n_m1"] / s["p_sizes"]
        frac_bg = (s["n_m1"] + s["n_0"]) / s["p_sizes"]
        ign_a = unm & (frac_m1 > 0.5)
        ign_b = unm & (frac_bg > 0.5)
        rule_a = metrics(tp, fp - int(ign_a.sum()), fn, s["match_iou_g"])
        rule_b = metrics(tp, fp - int(ign_b.sum()), fn, s["match_iou_g"])

        # classify every FP by where its points fall
        cat = np.full(len(s["p_ids"]), "", object)
        cat[unm & (frac_m1 > 0.5)] = "mostly_unlabelled(-1)"
        cat[unm & (frac_m1 <= 0.5) & (frac_bg > 0.5)] = "mostly_ground(0)"
        cat[unm & (frac_bg <= 0.5)] = "mostly_on_reference_trees"
        counts = {c: int((cat[unm] == c).sum()) for c in
                  ("mostly_unlabelled(-1)", "mostly_ground(0)", "mostly_on_reference_trees")}

        exp = {k: r.get(f"expected_{k}", "").strip() for k in ("tp", "fp", "fn")}
        ok = all(not exp[k] or int(exp[k]) == base[k] for k in exp)

        geo = {}
        dist = np.full(len(s["p_ids"]), np.nan)
        if xy is not None and len(gt) == len(xy):
            if hull is None:
                gcent = centroids(gt, s["g_ids"], xy)
                hull = convex_hull(gcent)
                np.savetxt(os.path.join(args.out, "reference_tree_centroids.csv"), gcent,
                           delimiter=",", header="x,y", comments="", fmt="%.3f")
            pc = centroids(pred, s["p_ids"][unm], xy)
            dist[unm] = signed_dist_to_hull(pc, hull)
            geo = dict(fp_outside_hull=int((dist[unm] > 0).sum()),
                       fp_outside_hull_by_2m=int((dist[unm] > 2).sum()),
                       fp_inside_hull=int((dist[unm] <= 0).sum()))
        elif xy is not None:
            geo = dict(note="coordinates skipped: point count differs from the .laz (SAT-aligned GT)")

        # per-FP table
        with open(os.path.join(args.out, f"fp_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pred_id", "n_points", "frac_gt_minus1", "frac_gt_0", "frac_on_trees",
                        "best_iou_any_tree", "category", "signed_dist_to_tree_hull_m"])
            for i in np.flatnonzero(unm):
                w.writerow([int(s["p_ids"][i]), int(s["p_sizes"][i]), round(frac_m1[i], 4),
                            round(s["n_0"][i] / s["p_sizes"][i], 4),
                            round(s["n_tree"][i] / s["p_sizes"][i], 4),
                            round(float(s["best_iou_p"][i]), 4), cat[i],
                            "" if np.isnan(dist[i]) else round(float(dist[i]), 2)])

        # per-tree outcomes (for bootstrap / hardest trees later)
        with open(os.path.join(args.out, f"trees_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gt_id", "gt_points", "matched", "matched_iou"])
            for gid, gs, u, mi in zip(s["g_ids"], s["g_sizes"], s["used_g"], s["match_iou_g"]):
                w.writerow([int(gid), int(gs), int(u), "" if np.isnan(mi) else round(float(mi), 4)])

        fp_sizes = s["p_sizes"][unm]
        rec = dict(method=r["method"], regime=r["regime"], reproduces_table=ok,
                   baseline=base, rule_A_ignore_unlabelled=rule_a,
                   rule_B_ignore_background=rule_b, fp_categories=counts,
                   fp_median_points=int(np.median(fp_sizes)) if fp else 0,
                   geometry=geo, seconds=round(time.time() - t0, 1))
        summary.append(rec)

        flag = "OK " if ok else "!! "
        print(f"\n{flag}{name}: TP {tp} FP {fp} FN {fn} F1 {base['f1']}"
              f"{'' if ok else '   <-- does NOT match Table 3.2, check the paths'}")
        print(f"    FP by location: {counts}   median FP size {rec['fp_median_points']:,} pts")
        print(f"    rule A (ignore >50% label -1):      FP {rule_a['fp']:>4}  F1 {rule_a['f1']}")
        print(f"    rule B (ignore >50% background):    FP {rule_b['fp']:>4}  F1 {rule_b['f1']}")
        if geo:
            print(f"    geometry: {geo}")
        del pred, s

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "regime", "reproduces_table",
                    "TP", "FP", "FN", "F1",
                    "FP_ruleA", "F1_ruleA", "FP_ruleB", "F1_ruleB",
                    "FP_unlabelled", "FP_ground", "FP_on_trees", "FP_median_points",
                    "FP_outside_hull"])
        for s in summary:
            b, a, bb, c = s["baseline"], s["rule_A_ignore_unlabelled"], s["rule_B_ignore_background"], s["fp_categories"]
            w.writerow([s["method"], s["regime"], s["reproduces_table"],
                        b["tp"], b["fp"], b["fn"], b["f1"],
                        a["fp"], a["f1"], bb["fp"], bb["f1"],
                        c["mostly_unlabelled(-1)"], c["mostly_ground(0)"],
                        c["mostly_on_reference_trees"], s["fp_median_points"],
                        s["geometry"].get("fp_outside_hull", "")])
    print(f"\nDone. Results in {args.out}/ (summary.csv, summary.json, fp_*.csv, trees_*.csv)")
    if not all(s["reproduces_table"] for s in summary):
        print("NOTE: rows marked '!!' did not reproduce Table 3.2. Where two candidate files were "
              "listed for one result, this is expected for the wrong candidate; otherwise check the path.")


if __name__ == "__main__":
    main()
