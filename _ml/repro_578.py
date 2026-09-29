# -*- coding: utf-8 -*-
"""逐行复现当初那段扫描代码，看 57.8% 到底从哪来"""
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
allrows = list(rows)

CACHE = ROOT / "_ml" / "_oof_repro.pkl"
if CACHE.exists():
    OOF = pickle.load(open(CACHE, "rb"))
    print("  复用样本外分数缓存")
else:
    RF = dict(n_estimators=400, min_samples_leaf=5, max_depth=10,
              class_weight="balanced", random_state=SEED, n_jobs=-1)
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

    def score(ms, rs):
        xt = X(rs)
        return np.mean([m.predict_proba(xt)[:, 1] for m in ms], axis=0)

    OOF = {}
    for hold in works:
        tr = [r for r in allrows if r["rj"] != hold]
        te = bywork[hold]
        sc = score(fit(tr), te)
        OOF[hold] = [{**r, "_s": float(s)} for s, r in zip(sc, te)]
    pickle.dump(OOF, open(CACHE, "wb"))
    print("  已算好样本外分数")


# ===== 当初那段扫描代码，逐行照抄 =====
def old_sweep(W):
    cov = set(); n = tp = 0; errs = []
    for hold in works:
        by = defaultdict(list)
        for r in OOF[hold]:
            by[r["track"]].append(r)
        for tk, grp in by.items():
            gts = GT.get(hold, {}).get(tk)
            if gts is None:      # 空列表（0 次的音轨）也要评估
                continue
            cands = [{"time": r["t"], "score": r["_s"], "mmss": ""}
                     for r in sorted(grp, key=lambda x: -x["_s"])[:4]]
            for c in merge_candidates(cands, top=3):
                t = c["time"]
                if W > 0:
                    w = [r for r in grp if abs(r["t"] - t) <= W]
                    if w:
                        t = max(w, key=lambda r: r["peak"])["t"]
                n += 1
                h = [g for g in gts if abs(t - g) <= TOL]
                if h:
                    tp += 1
                    for g in h:
                        cov.add((hold, tk, g))
                    errs.append(min(abs(t - g) for g in gts))
    gtot = sum(len(GT[w][k]) for w in works for k in GT[w])
    e = np.array(errs) if errs else np.array([0.0])
    return tp / max(1, n), len(cov) / max(1, gtot), e, n


# ===== 现在 train_model.py 的口径 =====
def now(W):
    cov = set(); n = tp = 0; errs = []
    for hold in works:
        by = defaultdict(list)
        for r in OOF[hold]:
            by[r["track"]].append(r)
        for tk, grp in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            cand = [{"time": float(r["t"]), "score": float(r["_s"]), "mmss": "",
                     "peak": float(r.get("peak", 0.0))} for r in grp]
            for c in snap_to_peak(merge_candidates(cand, top=3), cand, window=W):
                n += 1
                h = [x for x in g if abs(c["time"] - x) <= TOL]
                if h:
                    tp += 1
                    for x in h:
                        cov.add((hold, tk, x))
                    errs.append(min(abs(c["time"] - x) for x in g))
    gtot = sum(len(GT[w][k]) for w in works for k in GT[w])
    e = np.array(errs) if errs else np.array([0.0])
    return tp / max(1, n), len(cov) / max(1, gtot), e, n


print()
print("  %-32s %7s %7s %8s %7s %7s %6s" % (
    "口径", "精确率", "召回率", "平均误差", "≤3秒", "≤5秒", "候选"))
print("  " + "-" * 78)
for lbl, f in (("当初扫描 W=0", lambda: old_sweep(0)),
               ("当初扫描 W=12", lambda: old_sweep(12.0)),
               ("现在口径 W=0", lambda: now(0)),
               ("现在口径 W=12", lambda: now(12.0))):
    p, r, e, n = f()
    print("  %-32s %6.1f%% %6.1f%% %7.1f秒 %6.0f%% %6.0f%% %6d" % (
        lbl, p * 100, r * 100, e.mean(), (e <= 3).mean() * 100,
        (e <= 5).mean() * 100, n))

# 关键差异排查：手动标注的「user」轨
print()
print("  排查：track == 'user' 的行（旧 ✓/✗ 标注）")
for hold in works:
    u = [r for r in OOF[hold] if r["track"] == "user"]
    if u:
        print("    %s 有 %d 行 track='user'" % (hold, len(u)))
print("    这些行的 GT.get(hold,{}).get('user') = None → 两种情况都会被跳过")
