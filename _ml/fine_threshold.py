# -*- coding: utf-8 -*-
"""连续性门槛 vs 台阶门槛：能不能找到更好的甜点？

isotonic 把连续分压成 19 个台阶，所以在台阶之间设门槛完全无效。
换成对**原始分**设门槛（连续），就能取到任意前缀。问题是：
更细的门槛能不能换来更好的 F1？
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

src = (ROOT / "_ml" / "find_threshold.py").read_text(encoding="utf-8")
exec(src.split('print("  ===== 细粒度扫描')[0])

print()
print("  ===== 用原始分做门槛（连续，无台阶）=====")
print("  %-12s %9s %9s %9s %9s" % ("原始分门槛", "精确率", "召回率", "F1", "每轨候选"))
print("  " + "-" * 52)
rows_out = []
for i in range(20, 100, 2):
    thr = i / 100
    P, R, F, npt = evaluate(thr, key="raw")
    rows_out.append((thr, P, R, F, npt))
best = max(rows_out, key=lambda x: x[3])
for thr, P, R, F, npt in rows_out:
    if i % 1 == 0 and (round(thr * 100) % 10 == 0 or abs(thr - best[0]) < 1e-9):
        mark = "  ← F1 最高" if abs(thr - best[0]) < 1e-9 else ""
        print("  %-12s %8.1f%% %8.1f%% %8.3f %8.2f%s" % (
            "≥ %.2f" % thr, P * 100, R * 100, F, npt, mark))
print()
print("  原始分最优: %.2f → 精确 %.1f%%  召回 %.1f%%  F1 %.3f" % (
    best[0], best[1] * 100, best[2] * 100, best[3]))

print()
print("  ===== 全部前缀穷举（任意子集里最好的那个点）=====")
allr = sorted(recs, key=lambda r: -r["raw"])
best_any = None
for k in range(1, len(allr) + 1):
    sub = allr[:k]
    per_p = []; per_r = []
    for w in works:
        s = [r for r in sub if r["work"] == w]
        n = len(s); tp = sum(1 for r in s if r["hit"])
        gtot = sum(len(GT[w][t]) for t in GT[w])
        per_p.append(tp / n if n else float("nan"))
        per_r.append(tp / max(1, gtot))
    pp = [x for x in per_p if not np.isnan(x)]
    P = np.mean(pp) if pp else 0.0; R = np.mean(per_r)
    F = 2 * P * R / max(1e-9, P + R)
    if best_any is None or F > best_any[3]:
        best_any = (k, P, R, F, sub[-1]["raw"])
print("  取前 %d 个候选（原始分 ≥ %.3f）：" % (best_any[0], best_any[4]))
print("    精确 %.1f%%  召回 %.1f%%  F1 %.3f" % (
    best_any[1] * 100, best_any[2] * 100, best_any[3]))
print()
print("  三种口径对照：")
print("    校准概率 ≥50%%（当前默认）   F1 0.690")
print("    原始分连续门槛最优           F1 %.3f" % best[3])
print("    穷举全部前缀最优             F1 %.3f" % best_any[3])
