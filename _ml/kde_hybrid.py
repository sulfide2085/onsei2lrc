# -*- coding: utf-8 -*-
"""混合方案：先按模型分取前 K 个，再在这 K 个上做 KDE

纯 KDE 在全部候选上算密度会失败：低分峰扎堆的地方密度虚高。
所以先限定在模型认可的前 K 个候选上，再让密度决定精确时刻。
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import merge_candidates, snap_to_peak

exec((ROOT / "_ml" / "merge_test.py").read_text(encoding="utf-8")
     .split('print("跑 10 折')[0])
OOF = pickle.load(open(ROOT / "_ml" / "_oof_cache.pkl", "rb"))
GRID = 0.5


def hybrid(cands, top=3, K=6, h=8.0, kern="gauss", power=2.0,
           min_sep=30.0, use_snap=True):
    c = sorted(cands, key=lambda x: -x["score"])[:K]
    if not c:
        return []
    t = np.array([x["time"] for x in c], float)
    s = np.array([x["score"] for x in c], float)
    w = np.clip(s, 0, None) ** power
    if w.sum() <= 0:
        w = np.ones_like(w)
    lo, hi = t.min() - 5 * h, t.max() + 5 * h
    g = np.arange(lo, hi, GRID)
    d = np.abs(g[:, None] - t[None, :])
    if kern == "gauss":
        Kk = np.exp(-0.5 * (d / h) ** 2)
    elif kern == "tri":
        Kk = np.clip(1 - d / h, 0, None)
    else:
        Kk = np.where(d < h, 0.5 * (1 + np.cos(np.pi * d / h)), 0.0)
    dens = Kk @ w
    picked = []
    for i in np.argsort(-dens):
        tt = float(g[i])
        if all(abs(tt - x) > min_sep for x in picked):
            picked.append(tt)
        if len(picked) >= top:
            break
    picked = sorted(picked)
    if use_snap:
        evs = snap_to_peak([{"time": x, "score": 0, "mmss": ""} for x in picked],
                           cands, window=12.0)
        picked = [e["time"] for e in evs]
    return picked


def ev(fn):
    cov = set(); n = tp = 0; errs = []
    for hold in works:
        by = defaultdict(list)
        for r in OOF[hold]:
            by[r["track"]].append(r)
        for tk, grp in by.items():
            gts = GT.get(hold, {}).get(tk)
            if not gts:
                continue
            cands = [{"time": float(r["t"]), "score": float(r["_s"]), "mmss": "",
                      "peak": float(r["peak"])} for r in grp]
            for tt in fn(cands):
                n += 1
                hh = [g for g in gts if abs(tt - g) <= TOL]
                if hh:
                    tp += 1
                    for g in hh:
                        cov.add((hold, tk, g))
                    errs.append(min(abs(tt - g) for g in gts))
    gtot = sum(len(GT[w][k]) for w in works for k in GT[w])
    e = np.array(errs) if errs else np.array([0.0])
    return tp / max(1, n), len(cov) / max(1, gtot), e


def cur(c):
    return [x["time"] for x in snap_to_peak(merge_candidates(c, top=3), c, window=12.0)]


p, r, e = ev(cur)
print("  %-38s %6.1f%% %6.1f%% %7.1f秒 %5.0f%% %5.0f%%" % (
    "【当前】合并30s + 吸附12s", p * 100, r * 100, e.mean(),
    (e <= 3).mean() * 100, (e <= 5).mean() * 100))
print()
print("  %-38s %6s %6s %7s %5s %5s" % ("混合：先取前K，再KDE", "精确", "召回", "误差", "≤3秒", "≤5秒"))
print("  " + "-" * 74)
for K in (4, 6, 8, 12):
    for h in (5.0, 8.0, 12.0):
        p, r, e = ev(lambda c, K=K, h=h: hybrid(c, K=K, h=h))
        print("  %-38s %5.1f%% %5.1f%% %6.1f秒 %4.0f%% %4.0f%%" % (
            f"top{K} + KDE h={h:.0f}s + 吸附", p * 100, r * 100, e.mean(),
            (e <= 3).mean() * 100, (e <= 5).mean() * 100))
print()
for K in (6, 8):
    p, r, e = ev(lambda c, K=K: hybrid(c, K=K, h=8.0, use_snap=False))
    print("  %-38s %5.1f%% %5.1f%% %6.1f秒 %4.0f%% %4.0f%%" % (
        f"top{K} + KDE h=8s（不吸附）", p * 100, r * 100, e.mean(),
        (e <= 3).mean() * 100, (e <= 5).mean() * 100))
print()
for K in (6, 8):
    for pw in (1.0, 4.0):
        p, r, e = ev(lambda c, K=K, pw=pw: hybrid(c, K=K, h=8.0, power=pw))
        print("  %-38s %5.1f%% %5.1f%% %6.1f秒 %4.0f%% %4.0f%%" % (
            f"top{K} + KDE h=8s p^{pw:.0f} + 吸附", p * 100, r * 100, e.mean(),
            (e <= 3).mean() * 100, (e <= 5).mean() * 100))
