# -*- coding: utf-8 -*-
"""门槛扫描要用**校准概率**，跟界面显示的一致

之前那次扫描用的是原始分（ensemble 平均分），而 WebUI 的「最低概率」
比的是 isotonic 校准后的概率。两者尺度差很远：
    原始分 0.7  →  校准概率约 0.50
    原始分 0.9  →  校准概率约 0.92
所以「≥50% → 61.7%」那个说法对不上界面行为。重测。
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import MODEL_FEATS, merge_candidates, snap_to_peak, ClimaxModel

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

model = ClimaxModel()          # 只是为了拿它的 isotonic 校准器

CACHE = ROOT / "_ml" / "_prob_oof.pkl"
if CACHE.exists():
    OOF = pickle.load(open(CACHE, "rb"))
    print("  复用缓存的样本外分数")
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
    print("  已算出样本外分数")

print()
print("  原始分 → 校准概率 的对应关系：")
for v in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    print("    原始分 %.2f  →  %.0f%%" % (v, float(model.iso.predict([v])[0]) * 100))
print()


def sweep(thr):
    """thr 是**校准概率**门槛，与界面一致"""
    per_p, per_r = [], []
    n_all = 0; errs = []
    for hold in works:
        by = defaultdict(list)
        for s, tk, t, pk in OOF[hold]:
            by[tk].append({"time": t, "score": s, "mmss": "", "peak": pk})
        ftp = fn = 0; cov = set(); gtot = 0
        for tk, cand in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            gtot += len(g)
            evs = snap_to_peak(merge_candidates(cand, top=3), cand, window=12.0)
            for c in evs:
                p = float(model.predict_prob(np.array([c["score"]]))[0])
                if thr > 0 and p < thr:
                    continue
                fn += 1
                h = [x for x in g if abs(c["time"] - x) <= TOL]
                if h:
                    ftp += 1
                    for x in h:
                        cov.add(x)
                    errs.append(min(abs(c["time"] - x) for x in g))
        per_p.append(ftp / fn if fn else float("nan"))
        per_r.append(len(cov) / max(1, gtot))
        n_all += fn
    e = np.array(errs) if errs else np.array([0.0])
    pp = [x for x in per_p if not np.isnan(x)]
    return (np.mean(pp) if pp else 0.0, np.mean(per_r), n_all,
            (e <= 5).mean() * 100, sum(1 for x in per_p if np.isnan(x)))


print("  %-14s %9s %9s %9s %9s %9s" % (
    "最低概率", "逐折精确", "逐折召回", "每轨候选", "≤5秒", "整轨无候选"))
print("  " + "-" * 68)
for thr in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
    p, r, n, q, empty = sweep(thr)
    nt = sum(len(GT[w]) for w in works)
    mark = ""
    print("  %-14s %8.1f%% %8.1f%% %8.2f %8.0f%% %9d%s" % (
        ("不过滤" if thr == 0 else "≥ %.0f%%" % (thr * 100)),
        p * 100, r * 100, n / nt, q, empty, mark))
print()
print("  （「整轨无候选」= 有多少轨因为全低于门槛而一个候选都不给）")
