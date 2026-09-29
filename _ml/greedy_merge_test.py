# -*- coding: utf-8 -*-
"""决定性实验（修正版）

上一次的实现有链式合并 bug：候选 A(100s) 和 B(118s) 合并后，
C(136s) 又跟 B 合并 —— 20 秒窗口最终把 779 个候选串成一大簇。
正确做法是**只跟簇的代表点比距离**，不让簇随时间漂移。

目标：既不要「同一次高潮出现两个候选」，也不要因为丢弃而抓低分候补填坑。
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

exec((ROOT / "_ml" / "merge_test.py").read_text(encoding="utf-8")
     .split('print("跑 10 折')[0])

print("  预计算 10 折样本外分数…")
OOF = {}
for hold in works:
    tr = [r for r in allrows if r["rj"] != hold]
    te = bywork[hold]
    sc = score(fit(tr), te)
    OOF[hold] = [(float(s), r["track"], float(r["t"]), r["rj"]) for s, r in zip(sc, te)]
print("  完成\n")


def select_v2(cands, topn=3, merge=0.0, gap=15.0, mode="mean", w_by_score=False):
    """按分数降序扫描；与已有代表点相距 <= merge 的并进去，否则开新点。

    merge=0 时退化成当前的间隔抑制（gap 生效）。
    与旧实现的区别：比较对象是**簇的代表点时间**，不是簇内任意成员 —— 杜绝链式漂移。
    """
    by = defaultdict(list)
    for c in cands:
        by[(c[3], c[1])].append(c)
    out = []
    for (rj, tk), grp in by.items():
        grp.sort(key=lambda x: -x[0])
        reps = []                          # [score, time, [成员时间]]
        for sc, tk_, t, rj_ in grp:
            near = [r for r in reps if abs(t - r[1]) <= max(merge, 1e-9)]
            if near:
                r = min(near, key=lambda x: abs(t - x[1]))
                r[2].append(t)
                if mode == "best":
                    pass                                   # 时间保持不动
                elif w_by_score:
                    w = np.array([max(sc, 1e-6)] + [1.0] * (len(r[2]) - 1))
                    r[1] = float(np.average([t] + r[2][:-1], weights=w))
                else:
                    r[1] = float(np.mean(r[2]))
                r[0] = max(r[0], sc)
            elif len(reps) < topn:
                reps.append([sc, t, [t]])
            else:
                break                                      # 已经够 topn 个不同事件
            if len(reps) >= topn and merge > 0:
                # 继续扫描是为了把后续候选并进已有簇，不新增点
                pass
        reps.sort(key=lambda x: -x[0])
        out += [(r[0], tk, r[1], rj, len(r[2])) for r in reps[:topn]]
    return out


def report(label, picked):
    p, r, tp, n, cv, g = evaluate(picked)
    nt = sum(len(GT[w]) for w in works)
    dup = sum(x[4] - 1 for x in picked if len(x) > 4)
    print("  %-38s %7.1f%% %7.1f%% %6d %6.2f %7d" % (
        label, p * 100, r * 100, n, n / nt, dup))
    return p, r


def base(gap):
    return sum([select(OOF[h], merge=0, gap=gap) for h in works], [])


print("  %-38s %8s %8s %6s %6s %7s" % ("规则", "精确率", "召回率", "候选", "每轨", "并掉"))
print("  " + "-" * 80)
report("间隔15s 贪心拉满3个　【工具现状】", base(15))
report("不抑制直接取前3　【旧评估口径】", base(0.0))
print()
for m in (15, 20, 30, 45, 60, 90):
    picked = sum([select_v2(OOF[h], merge=float(m)) for h in works], [])
    report(f"合并窗口 {m}s（代表点取均值）", picked)
print()
for m in (30, 60):
    picked = sum([select_v2(OOF[h], merge=float(m), mode="best") for h in works], [])
    report(f"合并窗口 {m}s（代表点取最高分者）", picked)
for m in (30, 60):
    picked = sum([select_v2(OOF[h], merge=float(m), w_by_score=True) for h in works], [])
    report(f"合并窗口 {m}s（按分数加权均值）", picked)
