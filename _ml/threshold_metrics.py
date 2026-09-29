# -*- coding: utf-8 -*-
"""训练报告里的指标是「不过滤」的，但界面默认有 50% 门槛。
这里把**门槛下的指标**也算出来，两个都给。"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import (MODEL_FEATS, merge_candidates, snap_to_peak,
                           snap_to_peak_by_score)
from sklearn.isotonic import IsotonicRegression

from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TOL = 20.0
SEED = 0
SIGMA = 2.0
W = 12.0
GT = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
rows = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))["rows"]
works = sorted({r["rj"] for r in rows if r["rj"] in GT})
bywork = {w: [r for r in rows if r["rj"] == w] for w in works}
for r in rows:
    g = GT.get(r["rj"], {}).get(r["track"])
    r["_d"] = min(abs(r["t"] - x) for x in g) if g else None

CACHE = ROOT / "_ml" / "_final_oof2.pkl"


def X(rs): return np.array([[r[f] for f in MODEL_FEATS] for r in rs], float)
def Y(rs): return np.array([r["TP"] for r in rs], int)
def SOFT(rs):
    return np.array([float(np.exp(-(r["_d"] / SIGMA) ** 2)) if r["_d"] is not None else 0.0
                     for r in rs], float)


if CACHE.exists():
    OOF, OOF_LAB = pickle.load(open(CACHE, "rb"))
    print("  复用缓存的样本外分数")
else:
    OOF = {}; OOF_LAB = []
    for hold in works:
        tr = [r for r in rows if r["rj"] != hold]
        te = bywork[hold]
        xr, yr = X(tr), Y(tr)
        xt = X(te)
        clf = [RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_depth=10,
                                      class_weight="balanced", random_state=SEED,
                                      n_jobs=-1).fit(xr, yr),
               HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                              max_depth=10, min_samples_leaf=5,
                                              l2_regularization=1.0,
                                              class_weight="balanced",
                                              random_state=SEED).fit(xr, yr),
               make_pipeline(StandardScaler(), LogisticRegression(
                   class_weight="balanced", max_iter=3000, C=0.3)).fit(xr, yr)]
        reg = [RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_depth=10,
                                     random_state=SEED, n_jobs=-1).fit(xr, SOFT(tr)),
               HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                             max_depth=10, min_samples_leaf=5,
                                             l2_regularization=1.0,
                                             random_state=SEED).fit(xr, SOFT(tr)),
               make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(xr, SOFT(tr))]
        sc = np.mean([m.predict_proba(xt)[:, 1] for m in clf], axis=0)
        rf = np.mean([m.predict(xt) for m in reg], axis=0)
        OOF[hold] = [(float(s), float(g), r["track"], float(r["t"]),
                      float(r.get("peak", 0.0))) for s, g, r in zip(sc, rf, te)]
        OOF_LAB += [float(r["TP"]) for r in te]
    pickle.dump((OOF, OOF_LAB), open(CACHE, "wb"))
    print("  已算出样本外分数")

# **自己拟合 iso**，与 train_model.py 完全一致 ——
# 借交付 pkl 的 iso 会差 3 个点（iso 是大台阶，台阶落点稍有不同结果就变）
_oof_raw = np.concatenate([np.array([x[0] for x in OOF[h]]) for h in works])
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(_oof_raw, np.array(OOF_LAB))
print(f"  iso 拟合于 {len(_oof_raw)} 个样本外预测（与 train_model.py 同口径）")


def run(thr):
    """thr 是校准概率门槛"""
    per_p, per_r = [], []
    errs = []
    n_all = 0; empty_tracks = 0
    for hold in works:
        by = defaultdict(list)
        for s, g, tk, t, pk in OOF[hold]:
            by[tk].append({"time": t, "score": s, "mmss": "", "peak": pk, "_ref": g})
        ftp = fn = 0; cov = set(); gtot = 0
        for tk, cand in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            gtot += len(g)
            base = merge_candidates([{k: v for k, v in c.items() if k != "_ref"}
                                     for c in cand], top=3)
            evs = snap_to_peak_by_score(base, cand,
                                        np.array([c["_ref"] for c in cand]), window=W)
            kept = []
            for e in evs:
                p = float(np.clip(iso.predict([e["score"]])[0], 0.0, 0.95))
                if thr > 0 and p < thr:
                    continue
                kept.append(e)
            if not kept and g:
                empty_tracks += 1
            for e in kept:
                fn += 1
                h = [x for x in g if abs(e["time"] - x) <= TOL]
                if h:
                    ftp += 1
                    for x in h:
                        cov.add(x)
                    errs.append(min(abs(e["time"] - x) for x in g))
        per_p.append(ftp / fn if fn else float("nan"))
        per_r.append(len(cov) / max(1, gtot))
        n_all += fn
    e = np.array(errs) if errs else np.array([0.0])
    pp = [x for x in per_p if not np.isnan(x)]
    P = float(np.mean(pp)) if pp else 0.0
    R = float(np.mean(per_r))
    return P, R, 2 * P * R / max(1e-9, P + R), n_all, (e <= 5).mean(), e.mean(), empty_tracks


nt = sum(len(GT[w]) for w in works)
print()
print("  ===== 带回归精修的新模型，在不同门槛下的指标 =====")
print("  %-14s %8s %8s %7s %9s %8s %8s" % (
    "门槛", "精确率", "召回率", "F1", "每轨候选", "≤5秒", "平均误差"))
print("  " + "-" * 72)
for thr in (0.0, 0.17, 0.25, 0.3, 0.5, 0.6, 0.66, 0.67, 0.7, 0.9):
    P, R, F, n, q, m, et = run(thr)
    lbl = "不过滤" if thr == 0 else "≥ %.0f%%" % (thr * 100)
    star = "  ← 界面默认" if abs(thr - 0.5) < 1e-9 else ""
    print("  %-14s %7.1f%% %7.1f%% %6.3f %8.2f %7.0f%% %7.1f秒%s" % (
        lbl, P * 100, R * 100, F, n / nt, q * 100, m, star))
print()
print("  对照：`train_model.py` 报告的数字就是「不过滤」那一行")
