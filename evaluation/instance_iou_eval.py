"""
Shared instance-segmentation evaluator (IoU family, F1 @ IoU>=tau).

One evaluator for BOTH methods (SAT and treeX) so their numbers are directly
comparable. This is the IoU/F1 family.

Inputs are two per-point integer instance-label arrays of equal length:
    gt_labels[i]   : ground-truth instance id of point i
    pred_labels[i] : predicted instance id of point i
Background / non-tree points carry an id listed in `background_ids` and are
excluded from instance construction (but still count against a prediction's IoU
if the prediction leaks onto them -- see note below).

Matching: greedy, highest-IoU-first, one-to-one, threshold tau (default 0.5).
Reported: precision, recall, F1, mean IoU over matched pairs, plus raw TP/FP/FN.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def _instance_point_counts(labels: np.ndarray, background_ids) -> dict[int, int]:
    mask = ~np.isin(labels, list(background_ids))
    ids, counts = np.unique(labels[mask], return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist()))


def iou_match(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    tau: float = 0.5,
    background_ids=(-1, 0),
):
    """Greedy one-to-one IoU matching between GT and predicted instances."""
    gt_labels = np.asarray(gt_labels).ravel()
    pred_labels = np.asarray(pred_labels).ravel()
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            f"length mismatch: gt {gt_labels.shape} vs pred {pred_labels.shape} "
            " the two arrays must be point-aligned (same order, same N)."
        )

    gt_sizes = _instance_point_counts(gt_labels, background_ids)
    pred_sizes = _instance_point_counts(pred_labels, background_ids)
    n_gt, n_pred = len(gt_sizes), len(pred_sizes)

    # Intersection counts over points where BOTH labels are foreground.
    fg = (~np.isin(gt_labels, list(background_ids))) & (
        ~np.isin(pred_labels, list(background_ids))
    )
    if fg.sum() == 0 or n_gt == 0:
        return _summarise([], gt_sizes, pred_sizes, tau)

    df = pd.DataFrame({"g": gt_labels[fg], "p": pred_labels[fg]})
    inter = (
        df.groupby(["g", "p"], sort=False)
        .size()
        .reset_index(name="i")
    )
    inter["iou"] = inter.apply(
        lambda r: r["i"]
        / (gt_sizes[r["g"]] + pred_sizes[r["p"]] - r["i"]),
        axis=1,
    )

    # Greedy: take highest IoU pairs first, one GT and one pred used at most once.
    inter = inter.sort_values("iou", ascending=False)
    used_g, used_p, matches = set(), set(), []
    for _, r in inter.iterrows():
        if r["iou"] < tau:
            break
        g, p = int(r["g"]), int(r["p"])
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        matches.append((g, p, float(r["iou"])))

    return _summarise(matches, gt_sizes, pred_sizes, tau)


def _summarise(matches, gt_sizes, pred_sizes, tau):
    tp = len(matches)
    fn = len(gt_sizes) - tp
    fp = len(pred_sizes) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0
    return {
        "tau": tau,
        "n_gt": len(gt_sizes),
        "n_pred": len(pred_sizes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou_matched": round(mean_iou, 4),
        "matches": matches,
    }


def _load_labels(path: str, field: str | None):
    """Load a per-point label array from .npy / .npz / .las|.laz.
    For .npz/.las you must pass --field (e.g. 'instance', 'treeID', 'PredInstance')."""
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        z = np.load(path)
        if field is None:
            raise ValueError(f"{path} is .npz; keys={list(z.keys())} -- pass --*-field")
        return z[field]
    if path.endswith((".las", ".laz")):
        import laspy
        las = laspy.read(path)
        if field is None:
            raise ValueError(
                f"{path} is LAS; dims={list(las.point_format.dimension_names)} -- pass --*-field"
            )
        return np.asarray(las[field])
    raise ValueError(f"unsupported file type: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT labels file (.npy/.npz/.las)")
    ap.add_argument("--pred", required=True, help="Pred labels file (.npy/.npz/.las)")
    ap.add_argument("--gt-field", default=None)
    ap.add_argument("--pred-field", default=None)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--background", default="-1,0",
                    help="comma-separated background/non-tree ids")
    args = ap.parse_args()
    bg = tuple(int(x) for x in args.background.split(","))

    gt = _load_labels(args.gt, args.gt_field)
    pred = _load_labels(args.pred, args.pred_field)
    res = iou_match(gt, pred, tau=args.tau, background_ids=bg)
    print(f"tau={res['tau']}  n_gt={res['n_gt']}  n_pred={res['n_pred']}")
    print(f"TP={res['tp']}  FP={res['fp']}  FN={res['fn']}")
    print(f"precision={res['precision']}  recall={res['recall']}  "
          f"F1={res['f1']}  mean_IoU(matched)={res['mean_iou_matched']}")


if __name__ == "__main__":
    main()
