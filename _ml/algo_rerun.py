# -*- coding: utf-8 -*-
"""用统一口径（合并 + 吸附）重跑各算法对比

之前那张算法对比表的精确率/召回率是在**错误口径**下算的
（直接取分数最高的 topn，没实现工具里的候选选取）。
AUC 不受影响（它是在全部候选上算的），但精确率/召回率列必须重跑。
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import MODEL_FEATS, merge_candidates, snap_to_peak

from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TOL = 20.0
SEED = 0
GT = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]

blob = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))
rows = blob["rows"]
fb = ROOT / "climax_feedback.jsonl"
if fb.exists():
    import re
    latest = {}
    for line in fb.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r.get("audio"), round(r.get("time", 0), 1))] = r
    for (a, _), r in latest.items():
        if r.get("verdict") not in (0, 1) or not r.get("feats"):
            continue
        m = re.search(r"(RJ\d+)", str(a))
        if not m:
            continue
        row = {k: float(r["feats"].get(k, 0.0)) for k in MODEL_FEATS}
        row.update({"TP": bool(r["verdict"]), "rj": m.group(1), "track": "user",
                    "t": float(r["time"]), "txt": float(r["feats"].get("txt", 0.0))})
        rows.append(row)

works = sorted({r["rj"] for r in rows if r["rj"] in GT})
bywork = {w: [r for r in rows if r["rj"] == w] for w in works}


def X(rs): return np.array([[r[f] for f in MODEL_FEATS] for r in rs], float)
def Y(rs): return np.array([r["TP"] for r in rs], int)


MODELS = {
    "随机森林 (d=10)": lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=5, max_depth=10, class_weight="balanced",
        random_state=SEED, n_jobs=-1),
    "直方图梯度提升": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=10, min_samples_leaf=5,
        l2_regularization=1.0, class_weight="balanced", random_state=SEED),
    "逻辑回归": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.3)),
    "支持向量机 RBF": lambda: make_pipeline(
        StandardScaler(), SVC(C=2.0, gamma="scale", class_weight="balanced",
                              probability=True, random_state=SEED)),
    "极端随机树": lambda: ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=5, max_depth=10, class_weight="balanced",
        random_state=SEED, n_jobs=-1),
    "只用 pre30（单特征）": None,
}

print("  用统一口径（取前4 → 合并30s → 吸附12s）重跑 10 折…")
print()
print("  %-22s %7s %8s %8s %8s %8s" % ("模型", "AUC", "精确率", "召回率", "≤5秒", "平均误差"))
print("  " + "-" * 70)

results = {}
for name, ctor in MODELS.items():
    oof = {}
    for hold in works:
        tr = [r for r in rows if r["rj"] != hold]
        te = bywork[hold]
        feats = ["pre30"] if ctor is None else MODEL_FEATS
        xr = np.array([[r[f] for f in feats] for r in tr], float)
        yr = Y(tr)
        xt = np.array([[r[f] for f in feats] for r in te], float)
        if ctor is None:
            p = (xt[:, 0] - xr[:, 0].mean()) / max(1e-9, xr[:, 0].std())
            p = 1 / (1 + np.exp(-p))
        else:
            m = ctor().fit(xr, yr)
            p = m.predict_proba(xt)[:, 1]
        oof[hold] = (p, te)
    ps = np.concatenate([oof[h][0] for h in works])
    lb = np.concatenate([Y(oof[h][1]) for h in works]).astype(bool)
    auc = ((ps[lb][:, None] > ps[~lb][None, :]).sum()
           + 0.5 * (ps[lb][:, None] == ps[~lb][None, :]).sum()) / (lb.sum() * (~lb).sum())
    cov = set(); n = tp = 0; errs = []
    for hold in works:
        p, te = oof[hold]
        by = defaultdict(list)
        for s, r in zip(p, te):
            by[r["track"]].append({"time": float(r["t"]), "score": float(s), "mmss": "",
                                   "peak": float(r["peak"])})
        for tk, cand in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            for c in snap_to_peak(merge_candidates(cand, top=3), cand, window=12.0):
                n += 1
                h = [x for x in g if abs(c["time"] - x) <= TOL]
                if h:
                    tp += 1
                    for x in h:
                        cov.add((hold, tk, x))
                    errs.append(min(abs(c["time"] - x) for x in g))
    gtot = sum(len(GT[w][t]) for w in works for t in GT[w])
    e = np.array(errs) if errs else np.array([0.0])
    results[name] = (auc, tp / max(1, n), len(cov) / gtot, e)
    print("  %-22s %6.3f %7.1f%% %7.1f%% %7.0f%% %7.1f秒" % (
        name, auc, tp / max(1, n) * 100, len(cov) / gtot * 100,
        (e <= 5).mean() * 100, e.mean()))

print()
print("  对照：旧口径（不抑制、不合并、不吸附）报出来的数字")
print("    %-20s %7s %8s %8s" % ("模型", "AUC", "精确率", "召回率"))
for k, v in (("随机森林 (d=10)", (0.923, 0.585, 0.706)),
             ("直方图梯度提升", (0.930, 0.570, 0.698)),
             ("逻辑回归", (0.917, 0.531, 0.727)),
             ("支持向量机 RBF", (0.915, 0.503, 0.710)),
             ("极端随机树", (0.892, 0.549, 0.644)),
             ("只用 pre30（单特征）", (0.862, 0.417, 0.594))):
    print("    %-20s %7.3f %7.1f%% %7.1f%%" % (k, v[0], v[1] * 100, v[2] * 100))
print()
print("  ⚠ AUC 列不受口径影响（它是在全部候选上算的），但精确率/召回率列变化很大。")
pickle.dump(results, open(ROOT / "_ml" / "_algo_rerun.pkl", "wb"))
