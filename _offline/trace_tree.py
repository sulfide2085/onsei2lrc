# -*- coding: utf-8 -*-
"""同时走两条路径：sklearn 的树 vs 导出的 JSON，找出**第一个决策不同**的节点"""
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
D = json.loads((ROOT / "_offline" / "model.json").read_text(encoding="utf-8"))
js = json.loads((ROOT / "_offline" / "js_rf41.json").read_text(encoding="utf-8"))
x32 = np.array(js["x"], dtype=np.float32)

TI = 77
tt = rf.estimators_[TI].tree_
T = D["models"]["RF"]["trees"][TI]

print(f"  第 {TI} 棵树：节点数 sklearn={tt.node_count}  JSON={len(T['f'])}")
print()

n_py = n_js = 0
for step in range(60):
    fa_py, fa_js = int(tt.feature[n_py]), int(T["f"][n_js])
    thr_py, thr_js = float(tt.threshold[n_py]), float(T["t"][n_js])
    left_py = bool(x32[fa_py] <= thr_py)
    left_js = bool(x32[fa_js] <= thr_js)

    same = (fa_py == fa_js) and (left_py == left_js)
    mark = "" if same else "   ← 这里不同！"
    print(f"    步{step:2d}  py节点{n_py:4d} 特征{fa_py:2d}({feats[fa_py]:<11}) "
          f"x={float(x32[fa_py]):>16.8f} 阈值={thr_py:>18.10f} → {'左' if left_py else '右'}"
          f"   |  js节点{n_js:4d} 特征{fa_js:2d} 阈值={thr_js:>18.10f} "
          f"→ {'左' if left_js else '右'}{mark}")
    if not same:
        print()
        print("  === 差异详情 ===")
        print(f"    特征          {feats[fa_py]} (idx {fa_py})")
        print(f"    x (float32)   {float(x32[fa_py])!r}")
        print(f"    Python 阈值   {thr_py!r}")
        print(f"    JSON   阈值   {thr_js!r}")
        print(f"    两者差        {thr_py - thr_js:.6e}")
        print(f"    x 落在两者之间？ {thr_js < float(x32[fa_py]) <= thr_py}")
        print()
        print(f"    → Python 判定 {'左' if left_py else '右'}，JS 判定 {'左' if left_js else '右'}")
        break

    n_py = int(tt.children_left[n_py]) if left_py else int(tt.children_right[n_py])
    n_js = int(T["l"][n_js]) if left_js else int(T["r"][n_js])
    if n_py != n_js:
        print(f"    → 步{step} 后节点分叉: py={n_py} js={n_js}")
        break
else:
    print("    （60 步内未发现差异）")
