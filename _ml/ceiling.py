# -*- coding: utf-8 -*-
"""精确率的理论上限是多少？

工具每轨最多输出 3 个候选。如果一轨只有 1 个真高潮，那 3 个里必然有 2 个是错的
—— 精确率上限就是 33%。所以「48.9%」要跟上限比，才知道还剩多少空间。
"""
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

GT = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
TOL = 20.0
TOPN = 3

print("=" * 88)
print("  官方标注里每轨有几个高潮点？")
print("=" * 88)
dist = Counter()
for rj in GT:
    for tk, g in GT[rj].items():
        dist[len(g)] += 1
print("  %-12s %6s %10s %12s" % ("每轨高潮数", "轨数", "占比", "该轨精确率上限"))
print("  " + "-" * 46)
tot_tr = sum(dist.values())
ceil_sum = 0
for n in sorted(dist):
    ceiling = min(TOPN, n) / TOPN
    ceil_sum += dist[n] * ceiling
    print("  %-12s %6d %9.0f%% %11.0f%%" % (
        n, dist[n], dist[n] / tot_tr * 100, ceiling * 100))
print("  " + "-" * 46)
print("  %-12s %6d" % ("合计", tot_tr))
print("  理论精确率上限（每轨都取满 3 个）= %d/%d = %.1f%%" % (
    ceil_sum, tot_tr * TOPN, ceil_sum / (tot_tr * TOPN) * 100))
print()

# 更接近真实：候选之间有 30 秒合并，所以一轨能容纳的「不同事件」上限
print("  更精确的上限：把「同一轨内相距 <30 秒的真值点」算作同一次高潮")
merged = 0
for rj in GT:
    for tk, g in GT[rj].items():
        if not g:
            continue
        g = sorted(g)
        groups = 1
        for a, b in zip(g, g[1:]):
            if b - a > 30.0:
                groups += 1
        merged += min(TOPN, groups)
print("  理论上限 = %d/%d = %.1f%%" % (
    merged, tot_tr * TOPN, merged / (tot_tr * TOPN) * 100))
print()

# 实际能达到的：假设模型能完美挑出所有命中候选
cur = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
print("=" * 88)
print("  对照：实际做到多少")
print("=" * 88)
raw = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))["rows"]
by = defaultdict(list)
for r in raw:
    by[(r["rj"], r["track"])].append(r)
cand_per_track = [len(v) for v in by.values()]
print("  候选池：平均每轨 %.0f 个峰（中位 %d，最少 %d，最多 %d）" % (
    np.mean(cand_per_track), int(np.median(cand_per_track)),
    min(cand_per_track), max(cand_per_track)))
print()
print("  也就是说：模型要从平均 %.0f 个峰里挑出最多 3 个，" % np.mean(cand_per_track))
print("  而其中「真正该挑的」按定义为 %d 个（每轨最多 3 个，没有高潮的轨 0 个）。" % TOPN)
print()

# 一个几乎完美的排序器能做到多少？按「离真值最近的 3 个峰」挑
oracle_tp = oracle_n = 0
for rj in GT:
    for tk, g in GT[rj].items():
        peaks = by.get((rj, tk), [])
        if not peaks:
            continue
        # 完美排序：按到最近真值的距离排序（没有真值就随便取）
        if g:
            scored = sorted(peaks, key=lambda r: min(abs(r["t"] - x) for x in g))
        else:
            scored = sorted(peaks, key=lambda r: -r["t"])
        picked = []
        for r in scored:
            if all(abs(r["t"] - p["t"]) > 30.0 for p in picked):
                picked.append(r)
            if len(picked) >= TOPN:
                break
        for p in picked:
            oracle_n += 1
            if g and any(abs(p["t"] - x) <= TOL for x in g):
                oracle_tp += 1
print("  【理想排序器】按「离真值最近」挑，再走同样的合并流程：")
print("    %d/%d = %.1f%%" % (oracle_tp, oracle_n, oracle_tp / max(1, oracle_n) * 100))
print("    这是**特征能表达的上限** —— 即使模型完美，也只能到这里。")
print()
print("  实际模型：48.9%（逐折平均）")
print("  → 距离理想排序器还有 %.1f 个点的空间" % (
    oracle_tp / max(1, oracle_n) * 100 - 48.9))
