# -*- coding: utf-8 -*-
"""直接验证**导出的 JSON**：用与 JS 即将采用的相同逻辑在 Python 里跑一遍

这里刻意用「JS 思维」写：
  · 遍历时把特征值转 float32（np.float32(x) 对应 JS 的 Math.fround）
  · 阈值已经是 float32，直接比较
  · 二叉树用数组下标递推，不碰 sklearn 对象

如果这一份与 sklearn 完全一致，那么 JS 只要照抄这个逻辑就不会错。
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

blob = pickle.load(open(ROOT / "climax_model.pkl", "rb"))
M = blob["models"]
D = json.loads((ROOT / "_offline" / "model.json").read_text(encoding="utf-8"))
rows = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))["rows"]
feats = D["feats"]
X64 = np.array([[float(r[f]) for f in feats] for r in rows])
X32 = X64.astype(np.float32)
print(f"  样本数 {len(X64)}  特征数 {len(feats)}")


# ------------------------------------------------------------------ 树遍历
def leaf_index(tree, x32):
    """返回落到的叶子下标。x32 是 float32 数组"""
    f, t, l, rr = tree["f"], tree["t"], tree["l"], tree["r"]
    n = 0
    while l[n] != -1:
        n = l[n] if x32[f[n]] <= t[n] else rr[n]
    return n


def clf_forest_score(d, x32):
    """分类森林 → 概率"""
    vals = [t["v"][leaf_index(t, x32) * 2 + 1] for t in d["trees"]]
    if d.get("is_hgb"):
        # HGB：叶子值是**已乘过学习率**的原始分，先求和再 sigmoid
        z = d["init"] + sum(vals)
        return 1.0 / (1.0 + np.exp(-z))
    return float(np.mean(vals))


def reg_forest_score(d, x32):
    vals = [t["v"][leaf_index(t, x32)] for t in d["trees"]]
    if d.get("is_hgb"):
        return d["init"] + sum(vals)
    return float(np.mean(vals))


def linear_score(d, x32):
    x = np.asarray(x32, dtype=np.float64)
    z = (x - np.array(d["mean"])) / np.array(d["scale"])
    v = float(z @ np.array(d["coef"]) + d["intercept"])
    if d["kind"] == "logreg":
        return 1.0 / (1.0 + np.exp(-v))
    return v


def score(name, x32):
    d = D["models"][name]
    if d["kind"] == "clf_forest":
        return clf_forest_score(d, x32)
    if d["kind"] == "reg_forest":
        return reg_forest_score(d, x32)
    return linear_score(d, x32)


# ------------------------------------------------------------------ 对拍
CLF = ["RF", "HGB", "LR"]
REG = ["REG_RF", "REG_HGB", "REG_RIDGE"]
N = 300                                   # 抽样对拍（全量 3981×6 太慢）
sel = np.linspace(0, len(X64) - 1, N).astype(int)

print()
print("  %-12s %-16s %16s" % ("模型", "sklearn 参考", "JSON 导出最大误差"))
print("  " + "-" * 48)
ok = True
for name in CLF + REG:
    ref = (M[name].predict_proba(X64)[:, 1] if hasattr(M[name], "predict_proba")
           else M[name].predict(X64))
    mine = np.array([score(name, X32[i]) for i in sel])
    err = float(np.max(np.abs(mine - ref[sel])))
    good = err < 1e-6
    ok &= good
    print("  %-12s %-16s %15.2e  %s" % (
        name, type(M[name]).__name__[:16], err, "OK" if good else "✗"))

print()
raw_ref = np.mean([M[k].predict_proba(X64)[:, 1] for k in CLF], axis=0)
raw_mine = np.array([np.mean([score(k, X32[i]) for k in CLF]) for i in sel])
print("  集成原始分最大误差   %.2e" % float(np.max(np.abs(raw_mine - raw_ref[sel]))))

reg_ref = np.mean([M[k].predict(X64) for k in REG], axis=0)
reg_mine = np.array([np.mean([score(k, X32[i]) for k in REG]) for i in sel])
print("  精修分最大误差       %.2e" % float(np.max(np.abs(reg_mine - reg_ref[sel]))))

# isotonic
iso = blob["iso"]
xs = np.array(D["iso"]["x"])
ys = np.array(D["iso"]["y"])


def my_iso(v):
    """numpy 的 isotonic.predict 语义：右边界，超出则夹到端点"""
    i = np.searchsorted(xs, v, side="right") - 1
    i = min(max(i, 0), len(ys) - 1)
    return float(ys[i])


probe = np.linspace(0, 1, 200)
iso_err = max(abs(my_iso(v) - float(iso.predict([v])[0])) for v in probe)
print("  isotonic 最大误差     %.2e" % iso_err)
ok &= iso_err < 1e-9

print()
print("  结论:", "导出格式完全可用 ✓" if ok else "✗ 仍有不一致")
