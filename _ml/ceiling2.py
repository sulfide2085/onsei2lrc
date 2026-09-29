# -*- coding: utf-8 -*-
"""重算精确率上限 + 概率阈值的取舍

上次算错了：我假设「一轨 1 个高潮 → 最多 1 个候选命中」，
但 ±20 秒容差下，两个相距 36 秒的候选可以**同时**命中同一个真值点。
"""
import json
import pickle
import sys
from collections import defaultdict, Counter
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

print("=" * 90)
print("  ① 训练集里，每轨候选池中「真正算命中」的峰有多少个？")
print("=" * 90)
cnt = Counter()
for rj in GT:
    for tk, g in GT[rj].items():
        peaks = [r for r in rows if r["rj"] == rj and r["track"] == tk]
        if not peaks:
            continue
        tp = [p for p in peaks if g and any(abs(p["t"] - x) <= TOL for x in g)]
        # 合并后还剩几个「不同事件」
        ev = merge_candidates([{"time": p["t"], "score": 1.0, "mmss": ""} for p in tp], top=99)
        cnt[len(ev)] += 1
print("  %-22s %6s %10s" % ("候选池中的命中事件数", "轨数", "占比"))
print("  " + "-" * 40)
for k in sorted(cnt):
    print("  %-22s %6d %9.0f%%" % (k, cnt[k], cnt[k] / sum(cnt.values()) * 100))
print("  " + "-" * 40)
print("  %-22s %6d" % ("合计", sum(cnt.values())))
print()

# 上限：完美排序器 = 命中事件全部排在误报之前，每轨取前 3
ceil_tp = sum(min(3, k) * v for k, v in cnt.items())
ceil_n = sum(cnt.values()) * 3
print("  【完美排序器】每轨把命中事件全部排在前面，取前 3：")
print("    精确率上限 = %d/%d = %.1f%%" % (ceil_tp, ceil_n, ceil_tp / ceil_n * 100))
print("    （注意：这不是 100% —— 因为 52%% 的轨只有 1 个高潮，")
print("      每轨却要输出 3 个，结构上就有 2 个位置注定是错的）")
print()

# ---- 真正的模型，跑阈值扫描 ----
CACHE = ROOT / "_ml" / "_thr_oof.pkl"
if CACHE.exists():
    OOF = pickle.load(open(CACHE, "rb"))
else:
    RF = dict(n_estimators=400, min_samples_leaf=5, max_depth=10,
              class_weight="balanced", random_state=SEED, n_jobs=-1)
    HGB = dict(max_iter=300, learning_rate=0.06, max_depth=10, min_samples_leaf=5,
               l2_regularization=1.0, class_weight="balanced", random_state=SEED)
    LR = dict(class_weight="balanced", max_iter=3000, C=0.3)

    def X(rs): return np.array([[r[f] for f in MODEL_FEATS] for r in rs], float)
    def Y(rs): return np.array([r["TP"] for r in rs], int)
    OOF = {}
    for hold in works:
        tr = [r for r in rows if r["rj"] != hold]
        te = bywork[hold]
        xr, yr = X(tr), Y(tr)
        ms = [RandomForestClassifier(**RF).fit(xr, yr),
              HistGradientBoostingClassifier(**HGB).fit(xr, yr),
              make_pipeline(StandardScaler(), LogisticRegression(**LR)).fit(xr, yr)]
        xt = X(te)
        sc = np.mean([m.predict_proba(xt)[:, 1] for m in ms], axis=0)
        OOF[hold] = [(float(s), r["track"], float(r["t"]), float(r.get("peak", 0.0)))
                     for s, r in zip(sc, te)]
    pickle.dump(OOF, open(CACHE, "wb"))
    print("  （已算出样本外分数）")
print()
print("=" * 90)
print("  ② 概率阈值扫描：少输出一些，精确率能提到多少")
print("=" * 90)
print("  %-14s %8s %8s %9s %9s %9s" % (
    "概率门槛", "逐折精确", "逐折召回", "每轨候选", "≤5秒", "平均误差"))
print("  " + "-" * 66)
for thr in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
    per_p, per_r = [], []
    n_all = 0; errs = []
    for hold in works:
        by = defaultdict(list)
        for s, tk, t, pk in OOF[hold]:
            by[tk].append({"time": t, "score": s, "mmss": "", "peak": pk,
                           "probability": s})
        ftp = fn = 0; cov = set(); gtot = 0
        for tk, cand in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            gtot += len(g)
            evs = snap_to_peak(merge_candidates(cand, top=3), cand, window=12.0)
            evs = [c for c in evs if c["score"] >= thr]
            for c in evs:
                fn += 1
                h = [x for x in g if abs(c["time"] - x) <= TOL]
                if h:
                    ftp += 1
                    for x in h:
                        cov.add(x)
                    errs.append(min(abs(c["time"] - x) for x in g))
        per_p.append(ftp / max(1, fn) if fn else 0.0)
        per_r.append(len(cov) / max(1, gtot))
        n_all += fn
    e = np.array(errs) if errs else np.array([0.0])
    print("  %-14s %7.1f%% %7.1f%% %8.2f %8.0f%% %8.1f秒" % (
        ("不过滤" if thr == 0 else "≥ %.1f" % thr),
        np.mean(per_p) * 100, np.mean(per_r) * 100,
        n_all / sum(len(GT[w]) for w in works),
        (e <= 5).mean() * 100, e.mean()))
