# -*- coding: utf-8 -*-
"""同一个真值附近有多个峰，能不能区分出「最近的那个」？

诊断：候选池里离真值最近的峰，中位误差只有 1.3~5.3 秒；
但模型选中的候选，中位误差 3~11 秒。差距全在「选哪个峰」。
所以问题变成：邻近的峰之间，有没有可分的特征？
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import MODEL_FEATS

exec((ROOT / "_ml" / "merge_test.py").read_text(encoding="utf-8")
     .split('print("跑 10 折')[0])

OOF = {}
for hold in works:
    tr = [r for r in allrows if r["rj"] != hold]
    te = bywork[hold]
    sc = score(fit(tr), te)
    OOF[hold] = [{**r, "_s": float(s)} for s, r in zip(sc, te)]

near_pos, near_neg = [], []
for hold in works:
    by = defaultdict(list)
    for r in OOF[hold]:
        by[r["track"]].append(r)
    for tk, grp in by.items():
        gts = GT.get(hold, {}).get(tk)
        if gts is None:      # 空列表（0 次的音轨）也要评估
            continue
        for g in gts:
            # 真值 ±15 秒内的所有峰
            w = [r for r in grp if abs(r["t"] - g) <= 15.0]
            if not w:
                continue
            closest = min(w, key=lambda r: abs(r["t"] - g))
            for r in w:
                if r is closest:
                    near_pos.append((abs(r["t"] - g), r))
                else:
                    near_neg.append((abs(r["t"] - g), r))

print(f"  真值 ±15 秒内：『最近峰』{len(near_pos)} 个，其它峰 {len(near_neg)} 个")
print()
print("  ===== 特征对比 =====")
print("  %-12s %11s %11s %10s %8s" % ("特征", "最近峰均值", "其它峰均值", "差值", "区分度"))
print("  " + "-" * 58)
rows = []
for f in MODEL_FEATS:
    a = np.array([r[f] for _, r in near_pos])
    b = np.array([r[f] for _, r in near_neg])
    sd = np.sqrt((a.var() + b.var()) / 2)
    d = (a.mean() - b.mean()) / max(1e-9, sd)      # 标准化差值
    rows.append((abs(d), f, a.mean(), b.mean(), d))
for _, f, ma, mb, d in sorted(rows, reverse=True)[:10]:
    print("  %-12s %11.2f %11.2f %10.2f %+8.2f" % (f, ma, mb, ma - mb, d))
print()
print("  （区分度 = 标准化均值差，0.2 以上才算有点用，0.5 以上算明显）")
print()

# 分数对比
print("  ===== 模型分的对比 =====")
d_pos = []
for _, r in near_pos:
    d_pos.append(r["t"])
d_neg = []
for _, r in near_neg:
    d_neg.append(r["t"])
print(f"    最近峰的模型分：{np.mean([r['_s'] for _, r in near_pos]):.4f}")
print(f"    其它峰的模型分：{np.mean([r['_s'] for _, r in near_neg]):.4f}")
sc_p = np.array([r["_s"] for _, r in near_pos])
sc_n = np.array([r["_s"] for _, r in near_neg])
print(f"    最近峰分数更高的比例：{(sc_p.mean() > sc_n.mean())*100:.0f}% 整体")
win = 0
tot = 0
byw = defaultdict(list)
for _, r in near_pos:
    byw[(r["rj"], r["track"], round(r["t"], 0))].append(r)
for _, r in near_neg:
    byw[(r["rj"], r["track"], round(r["t"], 0))].append(r)
print()
print("  逐真值点比较「最近峰是不是分最高的」：")
wins = defaultdict(lambda: [0, 0])
for hold in works:
    by = defaultdict(list)
    for r in OOF[hold]:
        by[r["track"]].append(r)
    for tk, grp in by.items():
        gts = GT.get(hold, {}).get(tk)
        if gts is None:      # 空列表（0 次的音轨）也要评估
            continue
        for g in gts:
            w = [r for r in grp if abs(r["t"] - g) <= 15.0]
            if len(w) < 2:
                continue
            closest = min(w, key=lambda r: abs(r["t"] - g))
            best = max(w, key=lambda r: r["_s"])
            wins[hold][0] += 1
            if best is closest:
                wins[hold][1] += 1
tp = sum(v[0] for v in wins.values()); tc = sum(v[1] for v in wins.values())
for rj in sorted(wins):
    print("    %-12s %2d / %2d" % (rj, wins[rj][1], wins[rj][0]))
print("    " + "-" * 26)
print("    %-12s %2d / %2d = %.0f%%" % ("合计", tc, tp, tc / max(1, tp) * 100))
print()
print("  → 如果这个比例高，说明模型分数**已经**能挑对，只是被合并/取前3弄丢了")
print("  → 如果低，说明分数挑不对，需要新特征")
