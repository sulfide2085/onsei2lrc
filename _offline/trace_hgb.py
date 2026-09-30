# -*- coding: utf-8 -*-
"""追 HGB 的分叉点

怀疑点：sklearn 预测 HGB 时会先**分箱**（bin），然后用 bin_threshold 比较，
而不是用 num_threshold。两者在边界上可能不等价。
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

blob = pickle.load(open(ROOT / "climax_model.pkl", "rb"))
h = blob["models"]["HGB"]
feats = blob["feats"]
D = json.loads((ROOT / "_offline" / "model.json").read_text(encoding="utf-8"))
js = json.loads((ROOT / "_offline" / "js_rf41.json").read_text(encoding="utf-8"))

# 换一个候选：59（HGB 误差最大的那个）
ref = json.loads((ROOT / "_offline" / "ref_out.json").read_text(encoding="utf-8"))
x32 = np.array(ref["X32"][59], dtype=np.float32)

print(f"  HGB：迭代 {len(h._predictors)} 轮，每轮 {len(h._predictors[0])} 棵树")
print(f"  _bin_mapper 存在？ {hasattr(h, '_bin_mapper') and h._bin_mapper is not None}")
print()

# sklearn 的原始分
z_sk = float(h.decision_function(x32.reshape(1, -1))[0])
print(f"  sklearn decision_function = {z_sk:.10f}")

# 手工按 num_threshold 累加
z_nt = float(np.ravel(h._baseline_prediction)[0])
for stage in h._predictors:
    for t in stage:
        nd = t.nodes
        n = 0
        while not nd["is_leaf"][n]:
            n = (nd["left"][n] if x32[nd["feature_idx"][n]] <= nd["num_threshold"][n]
                 else nd["right"][n])
        z_nt += float(nd["value"][n])
print(f"  按 num_threshold 累加      = {z_nt:.10f}   差 {z_nt - z_sk:.3e}")

# 分箱后再累加
if hasattr(h, "_bin_mapper") and h._bin_mapper is not None:
    bm = h._bin_mapper
    xb = bm.transform(x32.reshape(1, -1))
    print(f"  分箱后 dtype = {xb.dtype}  值 = {xb[0][:8]}")
    z_bin = float(np.ravel(h._baseline_prediction)[0])
    for stage in h._predictors:
        for t in stage:
            nd = t.nodes
            n = 0
            while not nd["is_leaf"][n]:
                n = (nd["left"][n] if xb[0][nd["feature_idx"][n]] <= nd["bin_threshold"][n]
                     else nd["right"][n])
            z_bin += float(nd["value"][n])
    print(f"  按 bin_threshold 累加      = {z_bin:.10f}   差 {z_bin - z_sk:.3e}")
print()
print("  → 若后者更接近，说明必须导出 bin_threshold + 分箱边界")
print()
# 看 bin_mapper 的结构
if hasattr(h, "_bin_mapper") and h._bin_mapper is not None:
    bm = h._bin_mapper
    print("  BinMapper 属性:", [a for a in dir(bm) if not a.startswith('__')][:20])
    for attr in ("bin_thresholds_", "bin_edges_", "n_bins_"):
        if hasattr(bm, attr):
            v = getattr(bm, attr)
            print(f"    {attr}: type={type(v).__name__} len={len(v) if hasattr(v,'__len__') else '-'}")
