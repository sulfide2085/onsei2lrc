# -*- coding: utf-8 -*-
"""候选点的时间误差：是随机的还是有系统性偏移？

如果有系统性偏移（比如总是偏晚 3 秒），直接减掉就行，不需要新模型。
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import merge_candidates

exec((ROOT / "_ml" / "merge_test.py").read_text(encoding="utf-8")
     .split('print("跑 10 折')[0])

OOF = {}
for hold in works:
    tr = [r for r in allrows if r["rj"] != hold]
    te = bywork[hold]
    sc = score(fit(tr), te)
    OOF[hold] = [{**r, "_s": float(s)} for s, r in zip(sc, te)]

signed, absd, per_work = [], [], defaultdict(list)
hit_feats = []
for hold in works:
    by = defaultdict(list)
    for r in OOF[hold]:
        by[r["track"]].append(r)
    for tk, grp in by.items():
        gts = GT.get(hold, {}).get(tk)
        if gts is None:      # 空列表（0 次的音轨）也要评估
            continue
        grp.sort(key=lambda x: -x["_s"])
        picks = merge_candidates(
            [{"time": r["t"], "score": r["_s"], "mmss": ""} for r in grp], top=3)
        for p in picks:
            d = [(abs(p["time"] - g), p["time"] - g) for g in gts]
            e, sg = min(d, key=lambda x: x[0])
            if e <= TOL:
                signed.append(sg); absd.append(e)
                per_work[hold].append(sg)
                src = next(r for r in grp if abs(r["t"] - p["time"]) < 40)
                hit_feats.append((e, sg, src))

signed = np.array(signed); absd = np.array(absd)
print(f"  命中候选 {len(signed)} 个（±{TOL:.0f} 秒内算命中）")
print()
print("  ===== 有符号误差（候选 − 真值；正数=偏晚）=====")
print(f"    平均   {signed.mean():+6.2f} 秒")
print(f"    中位   {np.median(signed):+6.2f} 秒")
print(f"    标准差 {signed.std():6.2f} 秒")
print(f"    偏晚 {int((signed>0).sum())} 个 / 偏早 {int((signed<0).sum())} 个 / 正好 {int((signed==0).sum())} 个")
print()
print("  ===== 直方图 =====")
for lo, hi in ((-25,-15),(-15,-10),(-10,-5),(-5,0),(0,5),(5,10),(10,15),(15,25)):
    m = (signed >= lo) & (signed < hi)
    if m.sum():
        bar = "█" * int(m.sum() / max(1, len(signed)) * 100)
        print(f"    [{lo:+3d},{hi:+3d}) {int(m.sum()):>3}  {bar}")
print()
print("  ===== 绝对误差 =====")
for thr in (2, 5, 10, 15, 20):
    print(f"    ≤{thr:>2} 秒: {int((absd<=thr).sum()):>3} / {len(absd)}  = {(absd<=thr).mean()*100:5.1f}%")
print(f"    平均 {absd.mean():.2f} 秒   中位 {np.median(absd):.2f} 秒")
print()
print("  ===== 逐作品的有符号误差（看是不是某几部特别偏）=====")
print("  %-12s %5s %9s %9s" % ("作品", "命中", "平均偏移", "平均绝对"))
print("  " + "-" * 40)
for w in sorted(per_work):
    a = np.array(per_work[w])
    print("  %-12s %5d %+8.2f 秒 %8.2f 秒" % (w, len(a), a.mean(), np.abs(a).mean()))

print()
print("  ===== 误差和候选自身特征有没有关系 =====")
E = np.array([x[0] for x in hit_feats])
feats = ["pre30", "pre60", "post2", "post10", "peak", "prominence", "plateau",
         "rise", "decay", "recover", "rel_time", "contrast", "signature"]
print("  %-12s %10s %10s" % ("特征", "与误差相关", "与有符号误差"))
print("  " + "-" * 34)
rows = []
for f in feats:
    v = np.array([x[2][f] for x in hit_feats])
    sg = np.array([x[1] for x in hit_feats])
    r1 = np.corrcoef(v, E)[0, 1]
    r2 = np.corrcoef(v, sg)[0, 1]
    rows.append((abs(r2), f, r1, r2))
for _, f, r1, r2 in sorted(rows, reverse=True)[:6]:
    print("  %-12s %+10.3f %+10.3f" % (f, r1, r2))
