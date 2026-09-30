# -*- coding: utf-8 -*-
"""在 Python 里先「手工」实现一遍推理，确认与 sklearn 完全一致 ——
然后再把同一套逻辑翻译成 JS。否则 JS 那边出错根本分不清是哪一层的问题。"""
import pickle
import sys
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

blob = pickle.load(open(ROOT / "climax_model.pkl", "rb"))
M = blob["models"]


# ---------------------------------------------------------------- 树遍历
def rf_proba(tree, x):
    """sklearn 分类树：value 已归一化，直接取第 1 类"""
    tt = tree.tree_
    n = 0
    while tt.children_left[n] != -1:
        n = (tt.children_left[n] if x[tt.feature[n]] <= tt.threshold[n]
             else tt.children_right[n])
    return float(tt.value[n].ravel()[1])


def hgb_leaf(tp, x):
    """HGB 的 TreePredictor：nodes 是结构化数组，叶子 value 是原始分数"""
    nd = tp.nodes
    n = 0
    while not nd["is_leaf"][n]:
        n = (nd["left"][n] if x[nd["feature_idx"][n]] <= nd["num_threshold"][n]
             else nd["right"][n])
    return float(nd["value"][n])


def my_predict(name, X):
    m = M[name]
    cls = type(m).__name__
    out = []
    for x in X:
        if cls == "RandomForestClassifier":
            p = np.mean([rf_proba(t, x) for t in m.estimators_])
        elif cls == "HistGradientBoostingClassifier":
            z = float(np.ravel(m._baseline_prediction)[0])
            for stage in m._predictors:
                for t in stage:
                    z += m.learning_rate * hgb_leaf(t, x)
            p = 1.0 / (1.0 + np.exp(-z))
        elif cls == "RandomForestRegressor":
            p = np.mean([_reg_leaf(t, x) for t in m.estimators_])
        elif cls == "HistGradientBoostingRegressor":
            v = float(np.ravel(m._baseline_prediction)[0])
            for stage in m._predictors:
                for t in stage:
                    v += m.learning_rate * hgb_leaf(t, x)
            p = v
        elif cls == "Pipeline":
            inner = m.steps[-1][1]
            sc = m.steps[0][1]
            z = (np.asarray(x) - sc.mean_) / sc.scale_
            if "Logistic" in type(inner).__name__:
                p = float(inner.predict_proba(z.reshape(1, -1))[0, 1])
            else:
                p = float(inner.predict(z.reshape(1, -1))[0])
        else:
            raise SystemExit("未知类型 " + cls)
        out.append(float(p))
    return np.array(out)


def _reg_leaf(t, x):
    tt = t.tree_
    n = 0
    while tt.children_left[n] != -1:
        n = (tt.children_left[n] if x[tt.feature[n]] <= tt.threshold[n]
             else tt.children_right[n])
    return float(tt.value[n].ravel()[0])


# ---------------------------------------------------------------- 对拍
rng = np.random.default_rng(0)
n_feat = len(blob["feats"])
# 用真实特征分布采样更有意义：从 features_v2.json 里抽
import json
rows = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))["rows"]
idx = rng.choice(len(rows), size=40, replace=False)
X = np.array([[float(rows[i][f]) for f in blob["feats"]] for i in idx])

print("  逐模型对拍（40 个真实样本）")
print("  %-12s %-34s %14s" % ("模型", "类型", "最大绝对误差"))
print("  " + "-" * 64)
allok = True
for name, m in M.items():
    cls = type(m).__name__
    mine = my_predict(name, X)
    if hasattr(m, "predict_proba"):
        ref = m.predict_proba(X)[:, 1]
    else:
        ref = m.predict(X)
    err = float(np.max(np.abs(mine - ref)))
    ok = err < 1e-5
    allok &= ok
    print("  %-12s %-34s %13.2e  %s" % (name, cls, err, "OK" if ok else "✗ 不一致"))

print()
# 整个集成
raw = np.average([my_predict(k, X) for k in ("RF", "HGB", "LR")], axis=0)
ref_raw = np.mean([M[k].predict_proba(X)[:, 1] for k in ("RF", "HGB", "LR")], axis=0)
print("  集成原始分最大误差     %.2e" % float(np.max(np.abs(raw - ref_raw))))
refine = np.mean([my_predict(k, X) for k in ("REG_RF", "REG_HGB", "REG_RIDGE")], axis=0)
ref_ref = np.mean([M[k].predict(X) for k in ("REG_RF", "REG_HGB", "REG_RIDGE")], axis=0)
print("  精修分最大误差         %.2e" % float(np.max(np.abs(refine - ref_ref))))
print()
print("  结论:", "手工实现与 sklearn 完全一致 ✓" if allok else "有不一致，必须先修")
