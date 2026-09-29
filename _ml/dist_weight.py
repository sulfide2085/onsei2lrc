# -*- coding: utf-8 -*-
"""按「离真值的距离」给训练样本加权，让模型倾向挑最近的那个峰

现在的标注方式：真值 ±20 秒内的所有峰都是正例 → 模型只学到「这附近有高潮」，
没有动力区分「哪个峰最准」。结果就是候选池里明明有 1.3~2.4 秒的峰，
模型却挑了个 4~11 秒的。

试两种加权：
  A. 软加权  w = exp(-(d/σ)²)，远离真值的正例权重降低
  B. 硬标注  只有最近的那个峰算正例（±2 秒内），其余全部算负例
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import MODEL_FEATS, merge_candidates

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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


def dist_to_gt(r):
    g = GT.get(r["rj"], {}).get(r["track"])
    if not g:
        return None
    return min(abs(r["t"] - x) for x in g)


# 给每行预先算好「离真值的距离」
for r in rows:
    r["_d"] = dist_to_gt(r)


def X(rs): return np.array([[r[f] for f in MODEL_FEATS] for r in rs], float)
def Y(rs): return np.array([r["TP"] for r in rs], int)


def make_weights(rs, mode, sigma=5.0, hard_tol=2.0):
    if mode == "none":
        return None
    w = np.ones(len(rs))
    for i, r in enumerate(rs):
        d = r["_d"]
        if d is None or not r["TP"]:
            continue                      # 负例权重不变
        if mode == "soft":
            w[i] = float(np.exp(-(d / sigma) ** 2))
        elif mode == "hard":
            w[i] = 1.0 if d <= hard_tol else 0.0
    return w


def labels(rs, mode, hard_tol=2.0):
    y = Y(rs)
    if mode != "hard":
        return y
    out = y.copy()
    for i, r in enumerate(rs):
        d = r["_d"]
        if out[i] == 1 and (d is None or d > hard_tol):
            out[i] = 0                    # 只有最近的那个算正例
    return out


def fit(rs, mode, sigma):
    xr = X(rs)
    yr = labels(rs, mode)
    sw = make_weights(rs, mode, sigma)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_depth=10,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=10,
                                         min_samples_leaf=5, l2_regularization=1.0,
                                         class_weight="balanced", random_state=SEED)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=3000, C=0.3))
    rf.fit(xr, yr, sample_weight=sw)
    hgb.fit(xr, yr, sample_weight=sw)
    # Pipeline 不接受裸的 sample_weight，要用 stepname__参数 的形式
    lr.fit(xr, yr, logisticregression__sample_weight=sw)
    return [rf, hgb, lr]


def score(models, rs):
    xt = X(rs)
    return np.mean([m.predict_proba(xt)[:, 1] for m in models], axis=0)


def evaluate(mode, sigma=5.0):
    oof = {}
    for hold in works:
        tr = [r for r in rows if r["rj"] != hold]
        te = bywork[hold]
        oof[hold] = (score(fit(tr, mode, sigma), te), te)
    # AUC
    ps, lb = [], []
    for hold in works:
        p, te = oof[hold]
        ps.append(p); lb.append(Y(te))
    ps = np.concatenate(ps); lb = np.concatenate(lb).astype(bool)
    auc = ((ps[lb][:, None] > ps[~lb][None, :]).sum()
           + 0.5 * (ps[lb][:, None] == ps[~lb][None, :]).sum()) / (lb.sum() * (~lb).sum())
    # 精确率/召回率/时间误差
    cov = set(); n = tp = 0; errs = []
    for hold in works:
        p, te = oof[hold]
        by = defaultdict(list)
        for s, r in zip(p, te):
            by[r["track"]].append({"time": float(r["t"]), "score": float(s), "mmss": ""})
        for tk, cand in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            for c in merge_candidates(cand, top=3):
                n += 1
                h = [x for x in g if abs(c["time"] - x) <= TOL]
                if h:
                    tp += 1
                    for x in h:
                        cov.add((hold, tk, x))
                    errs.append(min(abs(c["time"] - x) for x in g))
    gtot = sum(len(GT[w][t]) for w in works for t in GT[w])
    e = np.array(errs) if errs else np.array([0.0])
    return auc, tp / max(1, n), len(cov) / gtot, e


print("  %-30s %7s %8s %8s %9s %7s %7s" % (
    "训练方式", "AUC", "精确率", "召回率", "平均误差", "≤3秒", "≤5秒"))
print("  " + "-" * 80)
for lbl, mode, sg in (
    ("现状（等权，±20s 都算正例）", "none", 0),
    ("软加权 σ=2", "soft", 2.0),
    ("软加权 σ=3", "soft", 3.0),
    ("软加权 σ=5", "soft", 5.0),
    ("软加权 σ=8", "soft", 8.0),
    ("硬标注（只最近峰算正例）", "hard", 0),
):
    a, p, r, e = evaluate(mode, sg)
    print("  %-30s %6.3f %7.1f%% %7.1f%% %8.1f秒 %6.0f%% %6.0f%%" % (
        lbl, a, p * 100, r * 100, e.mean(), (e <= 3).mean() * 100, (e <= 5).mean() * 100))
