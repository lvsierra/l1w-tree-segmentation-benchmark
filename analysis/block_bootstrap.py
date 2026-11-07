"""Spatial block bootstrap for the L1W benchmark.

Units: the 156 reference trees and every unmatched prediction (false positive),
each placed in the plot by its centroid. The plot is divided into square blocks;
a replicate resamples whole blocks with replacement, so a block brings its own
TP, FN and FP with it. The same block sample is applied to all nine results
(paired), which makes the confidence interval of a difference directly computable.

SAT's pretrained output lives on its own cloud, whose coordinates differ from the
native ones by a constant translation; the translation is estimated from the 156
reference-tree centroids and removed, and the residual is reported.
"""
import numpy as np, pandas as pd

D = "verify_out"
R = 10000
RNG = np.random.default_rng(42)
RESULTS = {"treeX preset": "treeX_preset",
           "TreeLearn pretrained": "TreeLearn_pretrained",
           "TreeLearn fine-tuned": "TreeLearn_finetuned_ep20",
           "SAT pretrained": "SAT_pretrained",
           "SAT fine-tuned": "SAT_finetuned_candB_SATftA",
           "TLS2Trees pretrained": "TLS2Trees_pretrained",
           "ForAINet pretrained": "ForAINet_pretrained",
           "FF3D pretrained": "FF3D_pretrained",
           "FF3D fine-tuned": "FF3D_finetuned_candB_ft3"}

gt = pd.read_csv(f"{D}/gt_centroids_native.csv").set_index("gt_id").sort_index()
gs = pd.read_csv(f"{D}/gt_centroids_sat.csv").set_index("gt_id").sort_index()
SHIFT = np.array([(gs.x - gt.x).mean(), (gs.y - gt.y).mean()])
resid = np.hypot((gs.x - gt.x) - SHIFT[0], (gs.y - gt.y) - SHIFT[1])
print(f"SAT cloud offset {SHIFT.round(3)} m, residual max {resid.max():.3f} m")
gxy = gt[["x", "y"]].to_numpy()


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


H = hull(gxy); A, B_ = H, np.roll(H, -1, 0)


def dist_outside(xy):
    d = np.full(len(xy), np.inf); ins = np.ones(len(xy), bool)
    for (ax, ay), (bx, by) in zip(A, B_):
        ex, ey = bx-ax, by-ay; px, py = xy[:, 0]-ax, xy[:, 1]-ay
        t = np.clip((px*ex+py*ey)/(ex*ex+ey*ey), 0, 1)
        d = np.minimum(d, np.hypot(px-t*ex, py-t*ey)); ins &= (ex*py-ey*px) >= 0
    return np.where(ins, -d, d)


data = {}
for name, f in RESULTS.items():
    t = pd.read_csv(f"{D}/trees_{f}.csv").set_index("gt_id").sort_index()
    fp = pd.read_csv(f"{D}/fp_{f}.csv")
    c = pd.read_csv(f"{D}/centroids_{f}.csv").set_index("pred_id")
    xy = c.loc[fp.pred_id, ["x", "y"]].to_numpy()
    if name == "SAT pretrained":
        xy = xy - SHIFT
    data[name] = dict(matched=t.matched.to_numpy().astype(float),
                      iou=t.matched_iou.fillna(0).to_numpy(),
                      fp_xy=xy, fp_out=dist_outside(xy) > 2)


def blocks(size):
    ids = {}
    def bid(xy):
        k = np.floor(xy / size).astype(int)
        return np.array([ids.setdefault((a, b), len(ids)) for a, b in zip(k[:, 0], k[:, 1])])
    gb = bid(gxy)
    fb = {n: bid(d["fp_xy"]) for n, d in data.items()}
    return gb, fb, len(ids)


def f1(tp, fp, fn):
    return np.where(2*tp+fp+fn > 0, 2*tp/np.maximum(2*tp+fp+fn, 1e-9), 0.0)


def run(size, naive=False):
    gb, fb, nb = blocks(size)
    if naive:
        m = len(gxy)
        idx = RNG.integers(0, m, (R, m))
        out = {}
        for n, d in data.items():
            tp = d["matched"][idx].sum(1); fn = m - tp
            fp = np.full(R, d["fp_xy"].shape[0], float)
            out[n] = dict(f1=f1(tp, fp, fn))
        return out, m
    per = {}
    for n, d in data.items():
        per[n] = dict(tp=np.bincount(gb, weights=d["matched"], minlength=nb),
                      fn=np.bincount(gb, weights=1-d["matched"], minlength=nb),
                      iou=np.bincount(gb, weights=d["iou"], minlength=nb),
                      fp=np.bincount(fb[n], minlength=nb).astype(float),
                      fpc=np.bincount(fb[n][~d["fp_out"]], minlength=nb).astype(float))
    draw = RNG.integers(0, nb, (R, nb))
    W = np.stack([np.bincount(row, minlength=nb) for row in draw]).astype(float)
    out = {}
    for n, p in per.items():
        tp = W @ p["tp"]; fn = W @ p["fn"]; iou = W @ p["iou"]; fp = W @ p["fp"]
        out[n] = dict(f1=f1(tp, fp, fn), f1c=f1(tp, W @ p["fpc"], fn),
                      prec=np.where(tp+fp > 0, tp/np.maximum(tp+fp, 1e-9), 0),
                      rec=np.where(tp+fn > 0, tp/np.maximum(tp+fn, 1e-9), 0),
                      miou=np.where(tp > 0, iou/np.maximum(tp, 1), 0))
    return out, nb


ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))

print("\nBlock diagnostics")
for s in (10, 15, 20):
    gb, fb, nb = blocks(s)
    tpb = np.bincount(gb, minlength=nb)
    fpb = np.mean([np.bincount(fb[n], minlength=nb).mean() for n in data])
    print(f"  {s:2d} m: {nb:3d} blocks, {tpb[tpb>0].mean():.1f} reference trees per occupied "
          f"block, {fpb:.1f} false positives per block on average")

boot, nb = run(15)
naive, m = run(15, naive=True)
print(f"\n95% CIs, block bootstrap (15 m, {nb} blocks, {R} replicates)")
rows = []
for n in RESULTS:
    b = boot[n]
    lo, hi = ci(b["f1"]); lc, hc = ci(b["f1c"]); li, hj = ci(b["miou"])
    pl, ph = ci(b["prec"]); rl, rh = ci(b["rec"]); nl, nh = ci(naive[n]["f1"])
    rows.append(dict(result=n, f1_lo=lo, f1_hi=hi, f1c_lo=lc, f1c_hi=hc, miou_lo=li,
                     miou_hi=hj, prec_lo=pl, prec_hi=ph, rec_lo=rl, rec_hi=rh,
                     naive_lo=nl, naive_hi=nh))
    print(f"{n:22s} F1 [{lo:.3f},{hi:.3f}] w{hi-lo:.3f} | corrected [{lc:.3f},{hc:.3f}] | "
          f"IoU [{li:.3f},{hj:.3f}] | naive [{nl:.3f},{nh:.3f}] w{nh-nl:.3f}")

print("\nPaired differences in F1")
PAIRS = [("FF3D fine-tuned", "treeX preset"), ("FF3D fine-tuned", "FF3D pretrained"),
         ("TreeLearn pretrained", "TreeLearn fine-tuned"), ("SAT fine-tuned", "SAT pretrained"),
         ("FF3D pretrained", "treeX preset"), ("treeX preset", "TLS2Trees pretrained"),
         ("treeX preset", "ForAINet pretrained")]
diffs = []
for a, b in PAIRS:
    for key, lab in (("f1", "strict"), ("f1c", "corrected")):
        d = boot[a][key] - boot[b][key]
        lo, hi = ci(d)
        sig = "differs" if lo > 0 or hi < 0 else "not distinguishable"
        diffs.append(dict(a=a, b=b, protocol=lab, mean=d.mean(), lo=lo, hi=hi, verdict=sig))
        print(f"  {a:20s} - {b:20s} ({lab:9s}) {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {sig}")

print("\nCI width vs resampling scheme (mean over the nine results, strict F1)")
for s in (10, 20):
    bo, nbs = run(s)
    print(f"  block {s:2d} m ({nbs:3d} blocks): {np.mean([np.diff(ci(bo[n]['f1']))[0] for n in RESULTS]):.3f}")
print(f"  block 15 m ({nb:3d} blocks): {np.mean([np.diff(ci(boot[n]['f1']))[0] for n in RESULTS]):.3f}")
print(f"  naive per-tree            : {np.mean([np.diff(ci(naive[n]['f1']))[0] for n in RESULTS]):.3f}")

pd.DataFrame(rows).to_csv(f"{D}/bootstrap_ci.csv", index=False, float_format="%.4f")
pd.DataFrame(diffs).to_csv(f"{D}/bootstrap_diffs.csv", index=False, float_format="%.4f")

print("\nBoundary-corrected point estimates (FP centroid > 2 m outside the reference extent removed)")
for n, d in data.items():
    tp = d["matched"].sum(); fn = len(d["matched"]) - tp
    fp = len(d["fp_out"]); fpc = int((~d["fp_out"]).sum())
    print(f"  {n:22s} FP {fp:4d} -> {fpc:4d}   F1 {f1(tp,fp,fn):.3f} -> {f1(tp,fpc,fn):.3f}")
