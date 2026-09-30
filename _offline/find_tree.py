# -*- coding: utf-8 -*-
"""找出 RF 里走向不同的那棵树，以及分叉在哪个节点

上一版这里写成了 `while a == b:` 配 `break`，两条路径若在首轮就分叉，
比较逻辑会自旋。改成显式的「同时走一步，发现不同就停」。
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

blob = pickle.load(open(ROOT / "climax_model.pkl", "rb"))
rf = blob["models"]["RF"]
feats = blob["feats"]
js = json.loads((ROOT / "_offline" / "js_rf41.json").read_text(encoding="utf-8"))
x32 = np.array(js["x"], dtype=np.float32)

print("  Python 用同一组 float32 特征遍历 400 棵树…")
mine = []
for t in rf.estimators_:
    tt = t.tree_
    n = 0
    guard = 0
    while tt.children_left[n] != -1 and guard < 200:
        n = (tt.children_left[n] if x32[tt.feature[n]] <= tt.threshold[n]
             else tt.children_right[n])
        guard += 1
    mine.append(n)

bad = [i for i in range(len(mine)) if mine[i] != js["leaves"][i]]
print(f"    走向不同的树: {len(bad)} 棵  {bad[:8]}")
print()

for i in bad[:2]:
    t = rf.estimators_[i]
    tt = t.tree_
    print(f"  第 {i} 棵树:  Python 叶子 {mine[i]}　JS 叶子 {js['leaves'][i]}")
    a, b = 0, 0
    for step in range(50):
        if a != b:
            print(f"    → 在上一步分叉")
            break
        fa, fb = int(tt.feature[a]), int(tt.feature[b])
        thr = float(tt.threshold[a])
        xv = float(x32[fa])
        go_left_py = xv <= thr
        if fa != fb or not go_left_py:
            print(f"    分叉节点 {a}")
            print(f"      特征            {feats[fa]}")
            print(f"      x (float32)     {xv!r}")
            print(f"      x (float64)     {float(x32[fa].astype(np.float64))!r}")
            print(f"      阈值 (float64)  {thr!r}")
            print(f"      x <= 阈值 ?     {go_left_py}")
            print(f"      阈值转 float32  {float(np.float32(thr))!r}")
            print(f"      float32(阈值) > 阈值 ? {float(np.float32(thr)) > thr}")
            break
        a = int(tt.children_left[a]) if go_left_py else int(tt.children_right[a])
        b = a
    print()

# JS 侧的阈值长什么样
d = json.loads((ROOT / "_offline" / "model.json").read_text(encoding="utf-8"))
tjs = d["models"]["RF"]["trees"][bad[0]] if bad else None
if tjs is not None:
    tpy = rf.estimators_[bad[0]].tree_
    diff = [abs(tjs["t"][k] - float(tpy.threshold[k]))
            for k in range(len(tjs["t"]))]
    k = int(np.argmax(diff))
    print(f"  第 {bad[0]} 棵树导出阈值的最大偏差: {max(diff):.3e}（节点 {k}）")
    print(f"    Python float64 阈值 = {float(tpy.threshold[k])!r}")
    print(f"    导出后 JSON 的值     = {tjs['t'][k]!r}")
    print(f"    导出值 <= 原阈值 ?   {tjs['t'][k] <= float(tpy.threshold[k])}")
