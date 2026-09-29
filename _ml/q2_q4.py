# -*- coding: utf-8 -*-
"""回答两个问题：
  ② ASR 到底有没有用？—— 把 txt 特征整个去掉，看指标变化
  ④ 训练集内表现如何？—— 过拟合有多严重（训练集内 vs 留一作品）
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
rows = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))["rows"]
works = sorted({r["rj"] for r in rows if r["rj"] in GT})
bywork = {w: [r for r in rows if r["rj"] == w] for w in works}

CACHE = ROOT / "_ml" / "_ablate_oof.pkl"

RF = dict(n_estimators=400, min_samples_leaf=5, max_depth=10, class_weight="balanced",
          random_state=SEED, n_jobs=-1)
HGB = dict(max_iter=300, learning_rate=0.06, max_depth=10, min_samples_leaf=5,
           l2_regularization=1.0, class_weight="balanced", random_state=SEED)
LR = dict(class_weight="balanced", max_iter=3000, C=0.3)


def make(feats):
    def X(rs): return np.array([[r[f] for f in feats] for r in rs], float)
    def Y(rs): return np.array([r["TP"] for r in rs], int)

    def fit(rs):
        xr, yr = X(rs), Y(rs)
        return [RandomForestClassifier(**RF).fit(xr, yr),
                HistGradientBoostingClassifier(**HGB).fit(xr, yr),
                make_pipeline(StandardScaler(), LogisticRegression(**LR)).fit(xr, yr)]

    def score(ms, rs):
        xt = X(rs)
        return np.mean([m.predict_proba(xt)[:, 1] for m in ms], axis=0)
    return fit, score


def metrics(pred):        # pred: {hold: (scores, rows)}
    per_p, per_r = [], []
    ptp = pn = pcov = gtot_all = 0
    errs = []
    for hold, (sc, rs) in pred.items():
        by = defaultdict(list)
        for s, r in zip(sc, rs):
            by[r["track"]].append({"time": float(r["t"]), "score": float(s),
                                   "mmss": "", "peak": float(r.get("peak", 0.0))})
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
        ptp += ftp; pn += fn; pcov += len(cov); gtot_all += gtot
    e = np.array(errs) if errs else np.array([0.0])
    return (np.mean(per_p), np.mean(per_r), ptp / max(1, pn),
            pcov / max(1, gtot_all), e)


def cv(feats, tag):
    """留一作品交叉验证"""
    fit, score = make(feats)
    pred = {}
    for hold in works:
        tr = [r for r in rows if r["rj"] != hold]
        pred[hold] = (score(fit(tr), bywork[hold]), bywork[hold])
    return metrics(pred)


def in_sample(feats):
    """在全部 10 部上训练，再在同一批数据上评估"""
    fit, score = make(feats)
    allr = [r for r in rows if r["rj"] in GT]
    ms = fit(allr)
    pred = {}
    for hold in works:
        pred[hold] = (score(ms, bywork[hold]), bywork[hold])
    return metrics(pred)


print("=" * 92)
print("  ② ASR 有没有用：把 txt 特征去掉对比（txt 就是 ASR 转写带来的唯一贡献）")
print("=" * 92)
NO_TXT = [f for f in MODEL_FEATS if f != "txt"]
print("  %-30s %8s %8s %8s %8s" % ("特征集", "逐折精确", "逐折召回", "汇总精确", "汇总召回"))
print("  " + "-" * 66)
r_full = cv(MODEL_FEATS, "full")
r_notx = cv(NO_TXT, "notxt")
print("  %-30s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (
    "全部 21 个（含 txt）", r_full[0]*100, r_full[1]*100, r_full[2]*100, r_full[3]*100))
print("  %-30s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (
    "20 个（去掉 txt = 不用 ASR）", r_notx[0]*100, r_notx[1]*100, r_notx[2]*100, r_notx[3]*100))
print("  %-30s %+7.1f %+7.1f %+7.1f %+7.1f" % (
    "差异", (r_notx[0]-r_full[0])*100, (r_notx[1]-r_full[1])*100,
    (r_notx[2]-r_full[2])*100, (r_notx[3]-r_full[3])*100))

print()
print("=" * 92)
print("  ④ 过拟合有多严重：训练集内 vs 留一作品")
print("=" * 92)
ins = in_sample(MODEL_FEATS)
print("  %-30s %9s %9s %9s %9s" % ("", "逐折精确", "逐折召回", "汇总精确", "汇总召回"))
print("  " + "-" * 70)
print("  %-30s %8.1f%% %8.1f%% %8.1f%% %8.1f%%" % (
    "训练集内（10 部全用来训）", ins[0]*100, ins[1]*100, ins[2]*100, ins[3]*100))
print("  %-30s %8.1f%% %8.1f%% %8.1f%% %8.1f%%" % (
    "留一作品（9 部训 1 部测）", r_full[0]*100, r_full[1]*100, r_full[2]*100, r_full[3]*100))
print("  %-30s %+8.1f %+8.1f %+8.1f %+8.1f" % (
    "过拟合的幅度", (ins[0]-r_full[0])*100, (ins[1]-r_full[1])*100,
    (ins[2]-r_full[2])*100, (ins[3]-r_full[3])*100))
print()
print("  ===== 逐作品对照 =====")
print("  %-14s %12s %12s" % ("作品", "训练集内", "留一作品"))
print("  " + "-" * 42)
fit, score = make(MODEL_FEATS)
ms_all = fit([r for r in rows if r["rj"] in GT])
for hold in works:
    a = score(ms_all, bywork[hold])
    tr = [r for r in rows if r["rj"] != hold]
    b = score(fit(tr), bywork[hold])
    ma = metrics({hold: (a, bywork[hold])})
    mb = metrics({hold: (b, bywork[hold])})
    print("  %-14s %11.1f%% %11.1f%%" % (hold, ma[0]*100, mb[0]*100))
print()
print("  ===== 时间误差对照 =====")
print("    训练集内 平均 %.1f 秒，≤5 秒 %.0f%%" % (ins[4].mean(), (ins[4] <= 5).mean()*100))
print("    留一作品 平均 %.1f 秒，≤5 秒 %.0f%%" % (r_full[4].mean(), (r_full[4] <= 5).mean()*100))
