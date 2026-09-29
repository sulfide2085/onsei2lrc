# -*- coding: utf-8 -*-
"""两阶段：分类器选事件，回归器定精确时刻

刚才发现软标签回归能大幅提升时间精度（≤5秒 53% → 68%），但全局排序变差
（精确率 48.9% → 31.2%）。

但如果**只用它做局部精修**呢？
  1. 分类器挑出 top-3 个事件（保住精确率）
  2. 在每个事件的 ±15 秒窗口内，用回归分挑最准的那一个（拿到时间精度）
两个模型的职责分开，可能同时拿到两边的好处。
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


def pc(ms, rs):
    xt = X(rs)
    return np.mean([m.predict_proba(xt)[:, 1] for m in ms], axis=0)


def pr(ms, rs):
    xt = X(rs)
    return np.mean([m.predict(xt) for m in ms], axis=0)


def evaluate(sigma=3.0, W=15.0, use_reg=True, use_energy=False, blend=1.0):
    per_p, per_r = [], []
    errs = []
    for hold in works:
        tr = [r for r in rows if r["rj"] != hold]
        te = bywork[hold]
        clf = pc(fit_clf(tr), te)
        reg = pr(fit_reg(tr, sigma), te) if (use_reg or use_energy) else None
        cand_all = [{"time": float(r["t"]), "score": float(s), "mmss": "",
                     "peak": float(r.get("peak", 0.0)), "_reg": float(g) if reg is not None else 0.0}
                    for s, g, r in zip(clf, reg if reg is not None else clf, te)]
        by = defaultdict(list)
        for c, r in zip(cand_all, te):
            by[r["track"]].append(c)
        ftp = fn = 0; cov = set(); gtot = 0
        for tk, cand in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            gtot += len(g)
            # 第一步：分类分选事件
            evs = merge_candidates([{k: v for k, v in c.items() if k != "_reg"}
                                    for c in cand], top=3)
            # 第二步：每个事件在 ±W 内重新定位
            for e in evs:
                near = [c for c in cand if abs(c["time"] - e["time"]) <= W]
                if near:
                    if use_energy:
                        best = max(near, key=lambda c: c["peak"])
                    elif use_reg:
                        if blend >= 1.0:
                            best = max(near, key=lambda c: c["_reg"])
                        else:
                            lo = min(c["_reg"] for c in near)
                            hi = max(c["_reg"] for c in near)
                            def key(c):
                                z = (c["_reg"] - lo) / max(1e-9, hi - lo)
                                return blend * z + (1 - blend) * (c["score"] if False else 0)
                            best = max(near, key=key)
                    else:
                        best = e
                    e = dict(e); e["time"] = best["time"]
                fn += 1
                h = [x for x in g if abs(e["time"] - x) <= TOL]
                if h:
                    ftp += 1
                    for x in h:
                        cov.add(x)
                    errs.append(min(abs(e["time"] - x) for x in g))
        per_p.append(ftp / max(1, fn)); per_r.append(len(cov) / max(1, gtot))
    e = np.array(errs) if errs else np.array([0.0])
    P, R = float(np.mean(per_p)), float(np.mean(per_r))
    return P, R, 2 * P * R / max(1e-9, P + R), (e <= 5).mean(), e.mean(), (e <= 10).mean()


print("  %-40s %7s %7s %7s %7s %7s" % (
    "做法", "精确率", "召回率", "F1", "≤5秒", "平均误差"))
print("  " + "-" * 82)
P, R, F, q, m, q10 = evaluate(use_reg=False, use_energy=False)
print("  %-40s %6.1f%% %6.1f%% %6.3f %6.0f%% %6.1f秒" % (
    "① 不精修（基线）", P * 100, R * 100, F, q * 100, m))
P, R, F, q, m, q10 = evaluate(use_reg=False, use_energy=True)
print("  %-40s %6.1f%% %6.1f%% %6.3f %6.0f%% %6.1f秒" % (
    "② 用能量最大精修（现工具）", P * 100, R * 100, F, q * 100, m))
print()
for sg in (2.0, 3.0, 5.0, 8.0):
    for W in (10.0, 15.0, 20.0):
        P, R, F, q, m, q10 = evaluate(sigma=sg, W=W, use_reg=True, use_energy=False)
        star = ""
        print("  %-40s %6.1f%% %6.1f%% %6.3f %6.0f%% %6.1f秒" % (
            "③ 回归精修 σ=%.0f 窗口±%.0f秒" % (sg, W), P * 100, R * 100, F, q * 100, m))
