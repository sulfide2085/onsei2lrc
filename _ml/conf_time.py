# -*- coding: utf-8 -*-
"""验证：置信度高的候选，时间误差是不是真的更小？

用户观察：红色（高置信度）候选误差 2~3 秒，橙色（中）偏 10~30 秒。
如果成立，说明「有没有高潮」和「时间准不准」是同一件事的两面 ——
那标注口径和界面呈现都要跟着改。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

exec((ROOT / "_ml" / "merge_test.py").read_text(encoding="utf-8")
     .split('print("跑 10 折')[0])

OOF = {}
for hold in works:
    tr = [r for r in allrows if r["rj"] != hold]
    te = bywork[hold]
    sc = score(fit(tr), te)
    OOF[hold] = [(float(s), r["track"], float(r["t"]), r["rj"]) for s, r in zip(sc, te)]

# 用 30 秒合并后的候选，记录它的原始分
rows = []
for hold in works:
    by = defaultdict(list)
    for s, tk, t, rj in OOF[hold]:
        by[(rj, tk)].append((s, tk, t, rj))
    for (rj, tk), grp in by.items():
        gts = GT.get(rj, {}).get(tk)
        if gts is None:      # 空列表（0 次的音轨）也要评估
            continue
        grp = sorted(grp, key=lambda x: -x[0])
        reps = []
        for s, tk_, t, rj_ in grp[:4]:
            hit = None
            for x in reps:
                if abs(t - x["time"]) <= 30.0:
                    hit = x
                    break
            if hit:
                hit["m"].append(t)
                hit["time"] = float(np.mean(hit["m"]))
                hit["score"] = max(hit["score"], s)
            else:
                reps.append({"time": t, "score": s, "m": [t]})
        reps.sort(key=lambda x: -x["score"])
        for i, c in enumerate(reps[:3]):
            g = min(gts, key=lambda x: abs(c["time"] - x))
            rows.append({"score": c["score"], "rank": i, "err": abs(c["time"] - g),
                         "hit": abs(c["time"] - g) <= TOL})

print(f"  合并后候选 {len(rows)} 个，其中命中 {sum(r['hit'] for r in rows)} 个")
print()
print("  ===== 按原始分分档看时间误差（只看命中的）=====")
print("  %-14s %6s %9s %9s %8s %8s %8s" % (
    "原始分", "命中数", "平均误差", "中位误差", "≤2秒", "≤3秒", "≤5秒"))
print("  " + "-" * 68)
hits = [r for r in rows if r["hit"]]
for lo, hi, name in ((0.90, 1.01, "[0.90, 1.00]"), (0.80, 0.90, "[0.80, 0.90)"),
                     (0.70, 0.80, "[0.70, 0.80)"), (0.60, 0.70, "[0.60, 0.70)"),
                     (0.50, 0.60, "[0.50, 0.60)"), (0.40, 0.50, "[0.40, 0.50)"),
                     (0.00, 0.40, "[0.00, 0.40)")):
    m = [r for r in hits if lo <= r["score"] < hi]
    if not m:
        continue
    e = np.array([r["err"] for r in m])
    print("  %-14s %6d %8.1f秒 %8.1f秒 %7.0f%% %7.0f%% %7.0f%%" % (
        name, len(m), e.mean(), np.median(e),
        (e <= 2).mean()*100, (e <= 3).mean()*100, (e <= 5).mean()*100))

print()
print("  ===== 按工具显示的置信度档位 =====")
print("  %-14s %6s %9s %9s %8s %8s %8s" % (
    "置信度", "命中数", "平均误差", "中位误差", "≤2秒", "≤3秒", "≤5秒"))
print("  " + "-" * 68)
for lo, hi, name in ((0.60, 1.01, "高（≥60%）"), (0.30, 0.60, "中（30~60%）"),
                     (0.00, 0.30, "低（<30%）")):
    m = [r for r in hits if lo <= r["score"] < hi]
    if not m:
        continue
    e = np.array([r["err"] for r in m])
    print("  %-14s %6d %8.1f秒 %8.1f秒 %7.0f%% %7.0f%% %7.0f%%" % (
        name, len(m), e.mean(), np.median(e),
        (e <= 2).mean()*100, (e <= 3).mean()*100, (e <= 5).mean()*100))

print()
print("  ===== 按候选排名（第1/2/3个）=====")
print("  %-14s %6s %9s %9s %8s %8s" % ("排名", "命中数", "平均误差", "中位误差", "≤3秒", "≤5秒"))
print("  " + "-" * 58)
for i in range(3):
    m = [r for r in hits if r["rank"] == i]
    if not m:
        continue
    e = np.array([r["err"] for r in m])
    print("  %-14s %6d %8.1f秒 %8.1f秒 %7.0f%% %7.0f%%" % (
        f"第 {i+1} 个", len(m), e.mean(), np.median(e),
        (e <= 3).mean()*100, (e <= 5).mean()*100))

print()
print("  ===== 相关系数 =====")
sc = np.array([r["score"] for r in hits])
er = np.array([r["err"] for r in hits])
print(f"    原始分 与 时间误差 的相关系数: {np.corrcoef(sc, er)[0,1]:+.3f}")
print(f"    （负数 = 分数越高误差越小，即用户的观察成立）")
