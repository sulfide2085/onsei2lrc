# -*- coding: utf-8 -*-
"""为什么实验脚本报 57.8%，而 train_model.py 报 48.9%？

两个可能的差异：
  A. 聚合方式：实验脚本是「全部折汇总后算」（pooled = 总命中/总候选），
     train_model.py 是「逐折算精确率再取平均」（mean of per-fold）。
     两者在候选数不均等时差别很大。
  B. 训练数据：实验脚本含旧 ✓/✗ 标注，train_model.py 现在默认不含。
把两个因素分开测。
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

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TOL = 20.0
SEED = 0
GT = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]

RF = dict(n_estimators=400, min_samples_leaf=5, max_depth=10, class_weight="balanced",
          random_state=SEED, n_jobs=-1)
HGB = dict(max_iter=300, learning_rate=0.06, max_depth=10, min_samples_leaf=5,
           l2_regularization=1.0, class_weight="balanced", random_state=SEED)
LR = dict(class_weight="balanced", max_iter=3000, C=0.3)


def X(rs): return np.array([[r[f] for f in MODEL_FEATS] for r in rs], float)
def Y(rs): return np.array([r["TP"] for r in rs], int)


def fit(rs):
    xr, yr = X(rs), Y(rs)
    return [RandomForestClassifier(**RF).fit(xr, yr),
            HistGradientBoostingClassifier(**HGB).fit(xr, yr),
            make_pipeline(StandardScaler(), LogisticRegression(**LR)).fit(xr, yr)]


def score(models, rs):
    xt = X(rs)
    return np.mean([m.predict_proba(xt)[:, 1] for m in models], axis=0)


def run(with_feedback):
    rows = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))["rows"]
    if with_feedback:
        import re
        latest = {}
        for line in (ROOT / "climax_feedback.jsonl").read_text(encoding="utf-8").splitlines():
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
                        "t": float(r["time"]), "txt": float(r["feats"].get("txt", 0.0)),
                        "peak": float(r["feats"].get("peak", 0.0))})
            rows.append(row)
    works = sorted({r["rj"] for r in rows if r["rj"] in GT})
    bywork = {w: [r for r in rows if r["rj"] == w] for w in works}

    per_fold, pooled_tp, pooled_n = [], 0, 0
    cov = set()
    for hold in works:
        tr = [r for r in rows if r["rj"] != hold]
        te = bywork[hold]
        sc = score(fit(tr), te)
        by = defaultdict(list)
        for s, r in zip(sc, te):
            by[r["track"]].append({"time": float(r["t"]), "score": float(s), "mmss": "",
                                   "peak": float(r.get("peak", 0.0))})
        ftp = fn_ = 0
        for tk, cand in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            for c in snap_to_peak(merge_candidates(cand, top=3), cand, window=12.0):
                fn_ += 1
                h = [x for x in g if abs(c["time"] - x) <= TOL]
                if h:
                    ftp += 1
                    for x in h:
                        cov.add((hold, tk, x))
        per_fold.append(ftp / max(1, fn_))
        pooled_tp += ftp
        pooled_n += fn_
    gtot = sum(len(GT[w][t]) for w in works for t in GT[w])
    return (np.mean(per_fold), pooled_tp / max(1, pooled_n),
            len(cov) / max(1, gtot), per_fold, pooled_tp, pooled_n)


print("  %-34s %14s %14s %10s" % ("配置", "逐折平均精确率", "汇总精确率", "召回率"))
print("  " + "-" * 78)
res = {}
for wf in (True, False):
    mean_p, pool_p, rec, pf, tp, n = run(wf)
    res[wf] = (mean_p, pool_p, rec, pf, tp, n)
    print("  %-34s %13.1f%% %13.1f%% %9.1f%%" % (
        "训练含旧✓✗标注" if wf else "训练只用官方标注", mean_p * 100, pool_p * 100, rec * 100))

print()
print("  ===== 逐折明细（训练含旧标注）=====")
for w, p in zip(sorted(GT), res[True][3]):
    print("    %-12s %5.1f%%" % (w, p * 100))
print("    合计命中 %d / 候选 %d" % (res[True][4], res[True][5]))
print()
print("  两个数字为什么不同：")
print("    实验脚本（kde_hybrid.py 等）用**汇总**：总命中 / 总候选")
print("    train_model.py 用**逐折平均**：每折各算一次精确率，再取算术平均")
print("    两者只在各折候选数完全相同时才相等。这里每折候选数从 12 到 30 不等，")
print("    差异就被放大了 —— 候选少的折（精确率高）在平均里被赋予了和候选多的折同等的权重。")
