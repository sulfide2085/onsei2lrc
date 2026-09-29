# -*- coding: utf-8 -*-
"""观察：相距 60 秒内的成对候选，到底能不能和「真的是两次高潮」区分开

用户观察：模型有时把一次高潮拆成两个相距 ~20 秒的候选，真值在中间。
但 RJ01586001 一轨里确实有 5 个真高潮挤在 147 秒内。
所以问题不是「要不要合并」，而是「怎么判断这两个候选是同一次还是两次」。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import MODEL_FEATS

exec((ROOT / "_ml" / "merge_test.py").read_text(encoding="utf-8")
     .split('print("跑 10 折')[0])

print("  预计算 10 折样本外分数（要用样本外，否则是在看训练集）…")
OOF = {}
for hold in works:
    tr = [r for r in allrows if r["rj"] != hold]
    te = bywork[hold]
    sc = score(fit(tr), te)
    OOF[hold] = [{**r, "_s": float(s)} for s, r in zip(sc, te)]
print("  完成\n")

W = 60.0
pairs = {"两次真高潮": [], "一次被拆开": [], "含误报": []}
allc = []
for hold in works:
    by = defaultdict(list)
    for r in OOF[hold]:
        by[r["track"]].append(r)
    for tk, grp in by.items():
        gts = GT.get(hold, {}).get(tk)
        if gts is None:
            continue
        grp.sort(key=lambda x: -x["_s"])
        top = grp[:3]
        top.sort(key=lambda x: x["t"])
        for a, b in zip(top, top[1:]):
            if b["t"] - a["t"] > W:
                continue
            ha = [g for g in gts if abs(a["t"] - g) <= TOL]
            hb = [g for g in gts if abs(b["t"] - g) <= TOL]
            # 窗口里有多少个不同的真值点
            win = [g for g in gts if a["t"] - TOL <= g <= b["t"] + TOL]
            if ha and hb and len(set(ha) | set(hb)) >= 2:
                k = "两次真高潮"
            elif ha and hb:
                k = "一次被拆开"
            else:
                k = "含误报"
            pairs[k].append((a, b, len(win), hold, tk))

print(f"  相距 ≤{W:.0f} 秒的成对候选：{sum(len(v) for v in pairs.values())} 对")
for k, v in pairs.items():
    print(f"    {k:<12} {len(v):>3} 对")
print()

FEATS = ["pre30", "pre60", "pre10", "post2", "post10", "post30", "peak",
         "signature", "contrast", "rise", "decay", "recover", "prominence",
         "plateau", "rel_time", "rank_energy"]


def desc(name, groups):
    print("  " + "=" * 88)
    print(f"  【{name}】{len(groups)} 对")
    print("  " + "=" * 88)
    if not groups:
        print("    （没有样本）\n")
        return
    print("  %-10s %8s %8s %8s %8s %8s %8s" % (
        "量", "时间差", "分数差", "高分/低分", "前点分", "后点分", "谁在前"))
    print("  " + "-" * 66)
    dt = [b["t"] - a["t"] for a, b, *_ in groups]
    ds = [abs(a["_s"] - b["_s"]) for a, b, *_ in groups]
    ratio = [max(a["_s"], b["_s"]) / max(1e-6, min(a["_s"], b["_s"])) for a, b, *_ in groups]
    hi_first = sum(1 for a, b, *_ in groups if a["_s"] > b["_s"])
    print("  %-10s %8.1f %8.3f %8.2f %8.3f %8.3f %8s" % (
        "均值", np.mean(dt), np.mean(ds), np.mean(ratio),
        np.mean([x["_s"] for x, _, *_ in [(a, b) for a, b, *_ in groups]]),
        np.mean([b["_s"] for _, b, *_ in groups]),
        f"{hi_first}/{len(groups)}"))
    print("  %-10s %8.1f %8.3f %8.2f" % (
        "中位", np.median(dt), np.median(ds), np.median(ratio)))
    print()
    print("  %-12s %10s %10s %10s %9s" % ("特征", "前点均值", "后点均值", "差值", "谁更大"))
    print("  " + "-" * 58)
    for f in FEATS:
        va = np.mean([a[f] for a, b, *_ in groups])
        vb = np.mean([b[f] for a, b, *_ in groups])
        bigger = "后" if vb > va else "前"
        print("  %-12s %10.2f %10.2f %10.2f %9s" % (f, va, vb, vb - va, bigger))
    print()


desc("两次真高潮（不该合并）", pairs["两次真高潮"])
desc("一次被拆开（应该合并）", pairs["一次被拆开"])
