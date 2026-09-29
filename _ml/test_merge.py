# -*- coding: utf-8 -*-
"""merge_candidates 单元测试"""
import sys
from pathlib import Path

sys.path.insert(0, r"D:\pyitme\onsei2lrc")
from climax_finder import merge_candidates  # noqa: E402

ok = [0, 0]


def ck(name, cond, detail=""):
    ok[1] += 1
    if cond:
        ok[0] += 1
    print(f"  {'✓' if cond else '✗'} {name}" + (f"　{detail}" if detail else ""))


def c(t, s, p=None):
    d = {"time": float(t), "mmss": "%02d:%02d" % (t // 60, t % 60),
         "score": s, "text": str(t)}
    if p is not None:
        d["probability"] = p
    return d


print("  ① 两个相近候选 → 合并取中点")
r = merge_candidates([c(610, 0.51, 0.51), c(632, 0.94, 0.94)], top=3)
ck("合并成 1 个事件", len(r) == 1, f"{len(r)} 个")
ck("时间取中点 621s", r and r[0]["time"] == 621.0, r[0]["mmss"] if r else "")
ck("分数取最高 0.94", r and abs(r[0]["score"] - 0.94) < 1e-9, r[0]["score"] if r else "")
ck("概率取最高 0.94", r and abs(r[0].get("probability", 0) - 0.94) < 1e-9)
ck("记录了合并数 2", r and r[0]["merged"] == 2)
ck("mmss 已重算", r and r[0]["mmss"] == "10:21", r[0]["mmss"] if r else "")

print()
print("  ② 三点链：比的是「与代表点的距离」，不是与任意成员")
r = merge_candidates([c(610, 0.9), c(630, 0.8), c(650, 0.7)], top=3, pool=3)
# 610 建簇；630 距 610 = 20 <= 30 并入，代表点取均值 620；
# 650 距代表点 620 = 30 <= 30 并入，代表点变 (610+630+650)/3 = 630
ck("三个点并成一个事件", len(r) == 1, f"{len(r)} 个")
ck("时间 630s", r and r[0]["time"] == 630.0, r[0]["mmss"] if r else "")
ck("合并数 3", r and r[0]["merged"] == 3)

print()
print("  ③ 超出合并窗口 → 保持独立")
r = merge_candidates([c(600, 0.9), c(700, 0.8)], top=3)
ck("两个独立事件", len(r) == 2, f"{len(r)} 个")
ck("按分数排序", r and r[0]["time"] == 600.0)

print()
print("  ④ pool 限制")
r = merge_candidates([c(100, 0.9), c(400, 0.8), c(700, 0.7), c(710, 0.6)],
                     top=3, pool=3)
ck("pool=3 时第 4 个不参与", len(r) == 3 and all(x["time"] != 710.0 for x in r))
r = merge_candidates([c(100, 0.9), c(400, 0.8), c(700, 0.7), c(710, 0.6)],
                     top=3, pool=4)
ck("pool=4 时 700/710 合并", len(r) == 3, f"{len(r)} 个")

print()
print("  ⑤ 合并后按分数重排")
r = merge_candidates([c(700, 0.5), c(705, 0.5), c(100, 0.9)], top=3)
ck("最高分事件排第一", r and r[0]["time"] == 100.0,
   " → ".join(x["mmss"] for x in r))

print()
print("  ⑦ 绝不凑数（质量优先原则）")
print()
# 只有 1 个候选够格时，就只给 1 个 —— 不能为了凑够 top=3 而塞低分候选
r = merge_candidates([c(100, 0.95), c(500, 0.10), c(900, 0.05)], top=3, pool=3)
ck("不会造出比输入更多的候选", len(r) <= 3, f"{len(r)} 个")
# merge_candidates 本身不做门槛过滤（那是调用方上游的事），
# 所以按真实调用顺序测：先过滤，再合并。
# 契约是：**绝不为了凑够 top 个而塞低分候选**。


def pipeline(raw, thr, top=3):
    kept = [x for x in raw if x.get("probability", 0) >= thr]
    return merge_candidates(kept, top=top, pool=top + 1)


raw = [c(100, 0.95, 0.95), c(500, 0.20, 0.20), c(900, 0.10, 0.10)]
ck("只有 1 个过门槛 → 只给 1 个", len(pipeline(raw, 0.5)) == 1,
   f"{len(pipeline(raw, 0.5))} 个")
ck("门槛放低 → 才给更多", len(pipeline(raw, 0.05)) == 3,
   f"{len(pipeline(raw, 0.05))} 个")
ck("不过滤时也不会超过 top",
   len(pipeline([c(i * 100, 0.9, 0.9) for i in range(9)], 0.0, top=3)) == 3)
# 一轨只有 1 个真高潮时输出 1 个就是对的 —— 不该硬凑到 3
ck("两个高分且相距远 → 给 2 个",
   len(pipeline([c(100, 0.95, 0.95), c(180, 0.90, 0.90)], 0.5)) == 2)

print()
print("  ⑧ 边界")
ck("空输入返回空", merge_candidates([]) == [])
ck("单元素", len(merge_candidates([c(5, 0.9)])) == 1)
ck("top 限制生效", len(merge_candidates([c(i * 100, 0.9) for i in range(9)],
                                        top=3, pool=9)) == 3)

print()
print("  %d / %d 通过" % (ok[0], ok[1]))
sys.exit(0 if ok[0] == ok[1] else 1)
