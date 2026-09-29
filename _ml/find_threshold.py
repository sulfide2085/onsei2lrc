# -*- coding: utf-8 -*-
"""找最优门槛：细粒度扫描所有**实际存在的**校准概率取值

为什么步长这么大？因为 isotonic 是**阶梯函数**，它把连续的原始分压成了
少数几个台阶。所以在台阶之间设门槛没有任何效果 —— 不是门槛步长大，
是**可选的台阶本身就没几个**。
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

TOL = 20.0
GT = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
rows = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))["rows"]
works = sorted({r["rj"] for r in rows if r["rj"] in GT})
OOF = pickle.load(open(ROOT / "_ml" / "_prob_oof.pkl", "rb"))
model = ClimaxModel()

# 收集全部候选：校准概率、原始分、是否命中、所在轨
recs = []
for hold in works:
    by = defaultdict(list)
    for s, tk, t, pk in OOF[hold]:
        by[tk].append({"time": t, "score": s, "mmss": "", "peak": pk})
    for tk, cand in by.items():
        g = GT.get(hold, {}).get(tk)
        if g is None:
            continue
        for c in snap_to_peak(merge_candidates(cand, top=3), cand, window=12.0):
            p = float(model.predict_prob(np.array([c["score"]]))[0])
            hit = any(abs(c["time"] - x) <= TOL for x in g)
            recs.append({"work": hold, "track": tk, "prob": p, "raw": c["score"],
                         "hit": hit, "time": c["time"],
                         "err": (min(abs(c["time"] - x) for x in g) if g else None),
                         "n_gt": len(g)})
n_pool = len(recs)
probs = np.array([r["prob"] for r in recs])
print(f"  合并后候选共 {n_pool} 个（所有折）")
print()
print("  ===== 校准概率的实际取值分布 =====")
uniq = sorted(set(np.round(probs, 4)))
print(f"    不同取值只有 {len(uniq)} 个：{['%.0f%%' % (u*100) for u in uniq]}")
print()
print("    各取值上挂了多少候选、命中率多少：")
print("    %-10s %8s %8s" % ("概率", "候选数", "实际命中率"))
print("    " + "-" * 30)
for u in uniq:
    m = np.abs(probs - u) < 1e-6
    if m.sum() >= 1:
        h = sum(1 for r, k in zip(recs, m) if k and r["hit"])
        print("    %-10s %8d %7.0f%%" % ("%.0f%%" % (u * 100), int(m.sum()),
                                          h / max(1, int(m.sum())) * 100))
print()


def evaluate(thr, key="prob"):
    """按门槛过滤后，算逐折平均精确率/召回率/F1"""
    per_p, per_r = [], []
    for w in works:
        sub = [r for r in recs if r["work"] == w and r[key] >= thr]
        n = len(sub)
        tp = sum(1 for r in sub if r["hit"])
        cov = set()
        for r in sub:
            if r["hit"]:
                cov.add(r["track"])
        gtot = sum(len(GT[w][t]) for t in GT[w])
        # 命中点数（近似：每个命中候选覆盖 1 个点）
        ptp = sum(1 for r in sub if r["hit"])
        per_p.append(tp / n if n else float("nan"))
        per_r.append(ptp / max(1, gtot))
        del cov
    pp = [x for x in per_p if not np.isnan(x)]
    P = np.mean(pp) if pp else 0.0
    R = np.mean(per_r)
    F = 2 * P * R / max(1e-9, P + R)
    n = sum(1 for r in recs if r[key] >= thr)
    return P, R, F, n / sum(len(GT[w]) for w in works)


print("  ===== 细粒度扫描（门槛取遍所有实际台阶）=====")
print("  %-10s %9s %9s %9s %9s" % ("门槛", "精确率", "召回率", "F1", "每轨候选"))
print("  " + "-" * 50)
best = None
for u in uniq:
    P, R, F, npt = evaluate(u)
    star = ""
    if best is None or F > best[2]:
        best = (u, P, R, F); star = "  ← F1 最高"
    print("  %-10s %8.1f%% %8.1f%% %8.3f %8.2f%s" % (
        "≥ %.0f%%" % (u * 100), P * 100, R * 100, F, npt, star))
P, R, F, npt = evaluate(0.0)
print("  %-10s %8.1f%% %8.1f%% %8.3f %8.2f" % ("不过滤", P * 100, R * 100, F, npt))
print()
print("  F1 最优：门槛 %.0f%%  →  精确 %.1f%%  召回 %.1f%%  F1 %.3f" % (
    best[0] * 100, best[1] * 100, best[2] * 100, best[3]))
print()
print("  ===== 若门槛设在**台阶之间**，结果完全一样（证明步长不是问题）=====")
for thr in (0.52, 0.55, 0.58, 0.60):
    P, R, F, npt = evaluate(thr)
    print("  %-10s %8.1f%% %8.1f%% %8.3f %8.2f" % ("≥ %.0f%%" % (thr * 100), P * 100, R * 100, F, npt))
