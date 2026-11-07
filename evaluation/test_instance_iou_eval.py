"""
Unit tests for the shared evaluator (instance_iou_eval.py).

Each case is a tiny synthetic point cloud (10 points) whose correct answer can be
worked out by hand, so the evaluator's output can be checked exactly. Labels:
0 = background, 1, 2, ... = tree instances. tau = 0.5 throughout.

Run:  python test_instance_iou_eval.py        (plain asserts, no pytest needed)
      python -m pytest test_instance_iou_eval.py   (also works)
"""
import numpy as np

from instance_iou_eval import iou_match

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def check(res, tp, fp, fn, f1, miou):
    assert (res["tp"], res["fp"], res["fn"]) == (tp, fp, fn), res
    assert abs(res["f1"] - f1) < 1e-4, res
    assert abs(res["mean_iou_matched"] - miou) < 1e-4, res


@case
def test_perfect_prediction():
    """Two trees predicted exactly: F1 = 1, IoU = 1."""
    gt = [1, 1, 1, 1, 2, 2, 2, 2, 0, 0]
    res = iou_match(gt, gt)
    check(res, tp=2, fp=0, fn=0, f1=1.0, miou=1.0)
    return "Perfect prediction", res


@case
def test_label_permutation():
    """Predicted IDs are arbitrary: relabelling 1->7, 2->3 changes nothing."""
    gt = [1, 1, 1, 1, 2, 2, 2, 2, 0, 0]
    pred = [7, 7, 7, 7, 3, 3, 3, 3, 0, 0]
    res = iou_match(gt, pred)
    check(res, tp=2, fp=0, fn=0, f1=1.0, miou=1.0)
    return "Instance IDs permuted", res


@case
def test_split_tree():
    """Tree 1 (6 pts) split 4+2: larger part matches (IoU 4/6), smaller is an FP.
    Over-segmentation costs precision, not recall (cf. Figure 2.2)."""
    gt = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    pred = [1, 1, 1, 1, 5, 5, 2, 2, 2, 2]
    res = iou_match(gt, pred)
    # P = 2/3, R = 1, F1 = 0.8; mean IoU = (4/6 + 1) / 2
    check(res, tp=2, fp=1, fn=0, f1=0.8, miou=(4 / 6 + 1) / 2)
    return "One tree split in two", res


@case
def test_merged_trees_at_threshold():
    """Two 4-point trees predicted as one 8-point instance: each pair has IoU
    exactly 0.5 (>= tau), so one-to-one matching gives 1 TP, 1 FN."""
    gt = [1, 1, 1, 1, 2, 2, 2, 2, 0, 0]
    pred = [9, 9, 9, 9, 9, 9, 9, 9, 0, 0]
    res = iou_match(gt, pred)
    # P = 1, R = 0.5, F1 = 2/3
    check(res, tp=1, fp=0, fn=1, f1=2 / 3, miou=0.5)
    return "Two trees merged (IoU = tau)", res


@case
def test_small_background_leak():
    """Prediction covers the 4-point tree plus 2 background points:
    IoU = 4/6, still a match."""
    gt = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    pred = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    res = iou_match(gt, pred)
    check(res, tp=1, fp=0, fn=0, f1=1.0, miou=4 / 6)
    return "Prediction leaks onto 2 background pts", res


@case
def test_large_background_leak():
    """Prediction covers the 4-point tree plus 5 background points:
    IoU = 4/9 < tau, so the tree is missed AND the prediction is an FP.
    Absorbing background cannot inflate the score."""
    gt = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    res = iou_match(gt, pred)
    check(res, tp=0, fp=1, fn=1, f1=0.0, miou=0.0)
    return "Prediction leaks onto 5 background pts", res


@case
def test_empty_prediction():
    """Nothing predicted: every tree is a false negative."""
    gt = [1, 1, 1, 1, 2, 2, 2, 2, 0, 0]
    pred = [0] * 10
    res = iou_match(gt, pred)
    check(res, tp=0, fp=0, fn=2, f1=0.0, miou=0.0)
    return "Empty prediction", res


@case
def test_length_mismatch_raises():
    """Misaligned arrays must fail loudly rather than be truncated."""
    gt = [1, 1, 1, 1, 2, 2, 2, 2, 0, 0]
    pred = [1, 1, 1, 1, 2, 2, 2, 2, 0]
    try:
        iou_match(gt, pred)
    except ValueError:
        return "Arrays of different length", None
    raise AssertionError("length mismatch was not rejected")


if __name__ == "__main__":
    print(f"{'Case':42s} {'TP':>3s} {'FP':>3s} {'FN':>3s} {'F1':>6s} {'mIoU':>6s}  result")
    for fn in CASES:
        name, res = fn()
        if res is None:
            print(f"{name:42s} {'':>3s} {'':>3s} {'':>3s} {'':>6s} {'':>6s}  PASS (ValueError)")
        else:
            print(f"{name:42s} {res['tp']:3d} {res['fp']:3d} {res['fn']:3d} "
                  f"{res['f1']:6.3f} {res['mean_iou_matched']:6.3f}  PASS")
    print(f"\nAll {len(CASES)} tests passed.")
