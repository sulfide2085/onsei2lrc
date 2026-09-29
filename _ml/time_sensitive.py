# -*- coding: utf-8 -*-
"""时间敏感的模型：三种做法对比

上次只试了「按距离给正例降权」（sample_weight）—— 那等于把远处的正例往负例推，
会误伤召回。换更正确的形式重测：

  A. 基线：二分类（真值 ±20 秒内的峰全算正例）
  B. 软标签回归：y = exp(-(d/σ)²)，直接学「离真值有多近」
  C. 混合：A 的分类分 与 B 的回归分 取平均

关键指标除了精确率/召回率，还有**时间精度**（≤5 秒比例、平均误差）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import MODEL_FEATS, merge_candidates, snap_to_peak

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TOL = 20.0
SEED = 0
GT = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
rows = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))["rows"]
works = sorted({r["rj"] for r in rows if r["rj"] in GT})
bywork = {w: [r for r in rows if r["rj"] == w] for w in works}

# 每行到最近真值的距离
for r in rows:
    g = GT.get(r["rj"], {}).get(r["track"])
    r["_d"] = min(abs(r["t"] - x) for x in g) if g else None


def X(rs): return np.array([[r[f] for f in MODEL_FEATS] for r in rs], float)
def Y(rs): return np.array([r["TP"] for r in rs], int)
def SOFT(rs, sigma):
    return np.array([np.exp(-(r["_d"] / sigma) ** 2) if r["_d"] is not None else 0.0
                     for r in rs], float)


def fit_clf(rs):
    xr, yr = X(rs), Y(rs)
    return [RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_depth=10,
                                   class_weight="balanced", random_state=SEED,
                                   n_jobs=-1).fit(xr, yr),
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=10,
                                           min_samples_leaf=5, l2_regularization=1.0,
                                           class_weight="balanced",
                                           random_state=SEED).fit(xr, yr),
            make_pipeline(StandardScaler(), LogisticRegression(
                class_weight="balanced", max_iter=3000, C=0.3)).fit(xr, yr)]


def fit_reg(rs, sigma):
    xr, yr = X(rs), SOFT(rs, sigma)
    return [RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_depth=10,
                                  random_state=SEED, n_jobs=-1).fit(xr, yr),
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=10,
                                          min_samples_leaf=5, l2_regularization=1.0,
                                          random_state=SEED).fit(xr, yr),
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(xr, yr)]


def pred_clf(ms, rs):
    xt = X(rs)
    return np.mean([m.predict_proba(xt)[:, 1] for m in ms], axis=0)


def pred_reg(ms, rs):
    xt = X(rs)
    return np.mean([m.predict(xt) for m in ms], axis=0)


def norm01(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / max(1e-9, hi - lo)


def evaluate(mode, sigma=5.0, alpha=0.5):
    per_p, per_r = [], []
    errs = []
    for hold in works:
        tr = [r for r in rows if r["rj"] != hold]
        te = bywork[hold]
        if mode == "clf":
            sc = pred_clf(fit_clf(tr), te)
        elif mode == "reg":
            sc = pred_reg(fit_reg(tr, sigma), te)
        else:
            a = norm01(pred_clf(fit_clf(tr), te))
            b = norm01(pred_reg(fit_reg(tr, sigma), te))
            sc = alpha * a + (1 - alpha) * b
        by = defaultdict(list)
        for s, r in zip(sc, te):
            by[r["track"]].append({"time": float(r["t"]), "score": float(s), "mmss": "",
                                   "peak": float(r.get("peak", 0.0))})
        ftp = fn = 0; cov = set(); gtot = 0
        for tk, cand in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            gtot += len(g)
            for c in snap_to_peak(merge_candidates(cand, top=3), cand, window=12.0):
                fn += 1
                h = [x for x in g if abs(c["time"] - x) <= TOL]
                if h:
                    ftp += 1
                    for x in h:
                        cov.add(x)
                    errs.append(min(abs(c["time"] - x) for x in g))
        per_p.append(ftp / max(1, fn)); per_r.append(len(cov) / max(1, gtot))
    e = np.array(errs) if errs else np.array([0.0])
    P, R = float(np.mean(per_p)), float(np.mean(per_r))
    return P, R, 2 * P * R / max(1e-9, P + R), (e <= 5).mean(), e.mean()


print("  %-34s %7s %7s %7s %7s %7s" % (
    "做法", "精确率", "召回率", "F1", "≤5秒", "平均误差"))
print("  " + "-" * 76)
P, R, F, q, m = evaluate("clf")
print("  %-34s %6.1f%% %6.1f%% %6.3f %6.0f%% %6.1f秒" % (
    "A 基线（二分类）", P * 100, R * 100, F, q * 100, m))
print()
for sg in (2.0, 3.0, 5.0, 8.0, 12.0):
    P, R, F, q, m = evaluate("reg", sg)
    print("  %-34s %6.1f%% %6.1f%% %6.3f %6.0f%% %6.1f秒" % (
        "B 软标签回归 σ=%.0f" % sg, P * 100, R * 100, F, q * 100, m))
print()
for sg in (3.0, 5.0, 8.0):
    for al in (0.3, 0.5, 0.7):
        P, R, F, q, m = evaluate("mix", sg, al)
        print("  %-34s %6.1f%% %6.1f%% %6.3f %6.0f%% %6.1f秒" % (
            "C 混合 σ=%.0f  分类权重%.1f" % (sg, al), P * 100, R * 100, F, q * 100, m))
