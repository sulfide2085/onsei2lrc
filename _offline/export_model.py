# -*- coding: utf-8 -*-
"""把 climax_model.pkl 导出成浏览器能用的紧凑 JSON

格式设计（目标：小、解码快、JS 里好遍历）：
  树用**扁平数组**而不是对象数组 —— 15 万个节点用对象会有巨大的解析开销。
  每棵树 = { f:[特征号...], t:[阈值...], l:[左子...], r:[右子...], v:[值...] }
  节点数 = f.length，叶子用 l[i] === -1 判断。

  值 v 的含义按模型类型不同：
    · 分类森林：prob = v[i*2+1] / (v[i*2] + v[i*2+1])   （节点的类计数）
    · 回归森林：value = v[i]
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

OUT = ROOT / "_offline"
OUT.mkdir(exist_ok=True)

blob = pickle.load(open(ROOT / "climax_model.pkl", "rb"))
models = blob["models"]
iso = blob.get("iso")

dump = {"feats": blob["feats"], "weights": blob["weights"],
        "meta": {k: v for k, v in (blob.get("meta") or {}).items()
                 if not isinstance(v, (list, dict))},
        "models": {}}


def f32g(v):
    """float32 值 → 9 位有效数字字符串（保证 float32 唯一还原，且比 17 位短）

    **必须转 float32**：sklearn 在预测时把输入 X 转成 float32 再和阈值比较，
    所以阈值只要落在两个相邻 float32 之间，行为就与原始 float64 阈值完全一致。
    实测：用 float64 遍历误差 1.9e-03，用 float32 遍历误差 2.8e-16。
    """
    t = np.float32(v)
    # **必须向下取整**：取「不大于阈值的最大 float32」。
    # 阈值通常夹在两个相邻 float32 之间；若四舍五入到上界，
    # 「正好等于上界」的查询值会错误地走左分支（sklearn 用 float64 阈值，
    # 该死心走右分支）。实测四舍五入会让 raw 偏 7.7e-4。
    if float(t) > float(v):
        t = np.nextafter(t, np.float32(-np.inf))
    return float("%.9g" % float(t))


def tree_arrays(t, in_f32=False):
    """把一棵树拆成 6 个扁平数组

    in_f32=True 时才把阈值降成 float32：**只有随机森林需要**。
    sklearn 只有随机森林内部把输入转 float32（Tree.apply 的 DTYPE），
    HGB 走分箱、LR 走 scaler，都是原样吃 float64。
    实测喂错精度：RF 差 2e-16（无损），HGB 差 7.6e-5，REG_HGB 差 8e-4。

    ⚠️ **必须显式导出 is_leaf**，不能靠 left==-1 判断叶子：
       sklearn 的 Tree 用 -1 表示叶子，但 **HGB 的 TreePredictor 叶子的
       left/right 是 0**（实测 left==-1 的节点数为 0）。靠 -1 判断会让遍历
       在节点间来回弹、永不终止 —— JS 那边表现成「卡住」，烧了 385 秒 CPU。
    """
    if hasattr(t, "tree_"):
        tt = t.tree_
        feat = tt.feature.astype(np.int16).tolist()
        thr = [f32g(v) if in_f32 else float(v) for v in tt.threshold]
        left = tt.children_left.astype(np.int32).tolist()
        right = tt.children_right.astype(np.int32).tolist()
        leaf = (tt.children_left == -1).astype(np.int8).tolist()
        val = tt.value.ravel().astype(np.float32)
    else:                                   # HGB 的 TreePredictor
        nd = t.nodes
        feat = nd["feature_idx"].astype(np.int16).tolist()
        thr = [f32g(v) if in_f32 else float(v) for v in nd["num_threshold"]]
        left = nd["left"].astype(np.int32).tolist()
        right = nd["right"].astype(np.int32).tolist()
        leaf = nd["is_leaf"].astype(np.int8).tolist()
        val = nd["value"].ravel().astype(np.float32)
    # **叶子值不能截断**。HGB 要把 300 个叶子值累加，每个截断 1e-6
    # 就会累积到 3e-4；实测截到 6 位小数时 raw 最大偏 7.7e-4。
    # json.dumps 用的是 repr()，会给出「能精确还原该 float64 的最短十进制」，
    # 所以不截断并不会像想象中那样膨胀。
    val = val.tolist()
    return {"f": feat, "t": thr, "l": left, "r": right, "lf": leaf,
            "v": val}


for name, m in models.items():
    cls = type(m).__name__
    if cls == "Pipeline":                    # LR / Ridge
        inner = m.steps[-1][1]
        sc = m.steps[0][1]
        kind = "logreg" if "Logistic" in type(inner).__name__ else "ridge"
        dump["models"][name] = {
            "kind": kind,
            # 不截断：LR 的系数截到 6 位小数会让输出偏 7e-6
            "coef": np.ravel(inner.coef_).tolist(),
            "intercept": float(np.ravel(inner.intercept_)[0]),
            "mean": np.ravel(sc.mean_).tolist(),
            "scale": np.ravel(sc.scale_).tolist(),
        }
        continue
    pred = getattr(m, "_predictors", None)
    est = ([t for stage in pred for t in stage] if pred is not None
           else list(getattr(m, "estimators_", [])))
    is_clf = hasattr(m, "predict_proba")
    is_rf = "RandomForest" in cls        # 只有随机森林吃 float32
    if is_clf:
        dump["models"][name] = {
            "kind": "clf_forest",
            "n": len(est),
            "is_hgb": pred is not None,     # HGB：叶子值是原始分，需先求和再 sigmoid
            "in_f32": is_rf,
            "init": float(np.ravel(getattr(m, "_baseline_prediction", [0.0]))[0]),
            "trees": [tree_arrays(t, is_rf) for t in est],
        }
    else:
        dump["models"][name] = {
            "kind": "reg_forest",
            "n": len(est),
            "is_hgb": pred is not None,   # 缺这个会被当成随机森林去做平均
            "in_f32": is_rf,
            "init": float(np.ravel(getattr(m, "_baseline_prediction", [0.0]))[0]),
            "trees": [tree_arrays(t, is_rf) for t in est],
        }

# isotonic 校准器 → 两张表
if iso is not None:
    # 不截断：y 截到 5 位小数会让校准概率偏 4e-6（实测）
    dump["iso"] = {"x": iso.X_thresholds_.tolist(),
                   "y": iso.y_thresholds_.tolist()}

raw = json.dumps(dump, separators=(",", ":")).encode()
gz = gzip.compress(raw, 9)
(OUT / "model.json.gz").write_bytes(gz)
(OUT / "model.json").write_bytes(raw)

print(f"  模型导出完成")
print(f"    JSON       {len(raw)/1048576:.2f} MB")
print(f"    gzip       {len(gz)/1048576:.2f} MB")
for name, d in dump["models"].items():
    if "trees" in d:
        nodes = sum(len(t["f"]) for t in d["trees"])
        print(f"    {name:<10} {d['kind']:<12} {len(d['trees']):>4} 棵  "
              f"{nodes:>7,} 节点")
    else:
        print(f"    {name:<10} {d['kind']:<12} {len(d['coef']):>4} 系数")
if iso is not None:
    print(f"    iso        {len(dump['iso']['x'])} 个台阶")
print()
print(f"  特征顺序: {dump['feats']}")
print(f"  集成权重: {dump['weights']}")
