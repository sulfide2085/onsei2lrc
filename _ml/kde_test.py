# -*- coding: utf-8 -*-
"""用户的想法：把峰视作概率云，密度叠加后取最大密度处

形式化就是核密度估计（KDE）：
    density(t) = Σ_i  p_i · K((t − t_i) / h)
其中 p_i 是模型给该峰的分数，K 是衰减核，h 是带宽。
取 density 的局部极大值就是「高潮时刻的最大后验估计」。

为什么它能修好「一次高潮被拆成两个」：
两个相近的峰，它们之间的密度是**两个核之和**，只要靠得够近，
和的峰值就落在中间 —— 中点不是人为规定的，是算出来的。

要测的关键参数：
  · 核形状（高斯 / 三角 / 余弦）
  · 带宽 h
  · 权重是否取幂（p^1 / p^2 / p^3）—— 防止低分峰把密度摊平
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
GRID = 0.5          # 密度网格步长（秒）


def kde_select(cands, top=3, h=10.0, kernel="gauss", power=1.0, min_sep=30.0):
    """在候选时间上算密度，取前 top 个局部极大"""
    if not cands:
        return []
    t = np.array([c["time"] for c in cands], float)
    s = np.array([c["score"] for c in cands], float)
    w = np.clip(s, 0.0, None) ** power
    if w.sum() <= 0:
        w = np.ones_like(w)

    lo, hi = t.min() - 5 * h, t.max() + 5 * h
    g = np.arange(lo, hi, GRID)
    d = np.abs(g[:, None] - t[None, :])
    if kernel == "gauss":
        K = np.exp(-0.5 * (d / h) ** 2)
    elif kernel == "tri":
        K = np.clip(1.0 - d / h, 0.0, None)
    else:                                   # cos
        K = np.where(d < h, 0.5 * (1 + np.cos(np.pi * d / h)), 0.0)
    dens = K @ w

    order = np.argsort(-dens)
    picked = []
    for i in order:
        tt = float(g[i])
        if all(abs(tt - x) > min_sep for x in picked):
            picked.append(tt)
        if len(picked) >= top:
            break
    return sorted(picked)


def evaluate(fn):
    cov = set(); n = tp = 0; errs = []
    for hold in works:
        by = defaultdict(list)
        for r in OOF[hold]:
            by[r["track"]].append(r)
        for tk, grp in by.items():
            gts = GT.get(hold, {}).get(tk)
            if gts is None:      # 空列表（0 次的音轨）也要评估
                continue
            cands = [{"time": float(r["t"]), "score": float(r["_s"]), "mmss": "",
                      "peak": float(r["peak"])}
                     for r in grp]
            for tt in fn(cands):
                n += 1
                h = [g for g in gts if abs(tt - g) <= TOL]
                if h:
                    tp += 1
                    for g in h:
                        cov.add((hold, tk, g))
                    errs.append(min(abs(tt - g) for g in gts))
    gtot = sum(len(GT[w][k]) for w in works for k in GT[w])
    e = np.array(errs) if errs else np.array([0.0])
    return tp / max(1, n), len(cov) / max(1, gtot), e


def cur(cands):
    """当前实现：合并 30s + 吸附到 ±12s 内最大峰"""
    ev = snap_to_peak(merge_candidates(cands, top=3), cands, window=12.0)
    return [c["time"] for c in ev]


print("  基准：当前实现（取前4 → 合并30s → 吸附±12s）")
p, r, e = evaluate(cur)
print("    %-30s 精确 %5.1f%%  召回 %5.1f%%  平均误差 %4.1f秒  ≤5秒 %3.0f%%" % (
    "", p * 100, r * 100, e.mean(), (e <= 5).mean() * 100))
print()
print("  %-34s %7s %7s %8s %7s %7s" % ("KDE 配置", "精确率", "召回率", "平均误差", "≤3秒", "≤5秒"))
print("  " + "-" * 78)
for kern in ("gauss", "tri", "cos"):
    for h in (5.0, 10.0, 15.0, 20.0):
        p, r, e = evaluate(lambda c, k=kern, hh=h: kde_select(c, h=hh, kernel=k))
        print("  %-34s %6.1f%% %6.1f%% %7.1f秒 %6.0f%% %6.0f%%" % (
            f"{kern}  h={h:.0f}s  p^1", p * 100, r * 100, e.mean(),
            (e <= 3).mean() * 100, (e <= 5).mean() * 100))
print()
for pw in (2.0, 3.0, 5.0):
    for h in (8.0, 12.0, 20.0):
        p, r, e = evaluate(lambda c, hh=h, pp=pw: kde_select(c, h=hh, power=pp))
        print("  %-34s %6.1f%% %6.1f%% %7.1f秒 %6.0f%% %6.0f%%" % (
            f"gauss h={h:.0f}s  p^{pw:.0f}", p * 100, r * 100, e.mean(),
            (e <= 3).mean() * 100, (e <= 5).mean() * 100))
