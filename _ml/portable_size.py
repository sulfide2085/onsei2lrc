# -*- coding: utf-8 -*-
"""问题①：这个模型能不能塞进单个 HTML 让别人直接用？

要回答三个子问题：
  a) 模型导出成可在浏览器里跑的形式，有多大？
  b) 音频解码 + 特征提取能不能在浏览器里做？
  c) 总共会是多少？
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np


def node_count(t):
    """兼容两种树：sklearn 的 Tree 用 tree_.node_count；HGB 的 TreePredictor 用 nodes 数组"""
    if hasattr(t, "tree_"):
        return int(t.tree_.node_count)
    n = getattr(t, "nodes", None)
    return int(len(n)) if n is not None else 0

blob = pickle.load(open(ROOT / "climax_model.pkl", "rb"))
models = blob["models"]

print("=" * 76)
print("  模型结构")
print("=" * 76)
print("  %-12s %-22s %10s %12s" % ("名字", "类型", "树/迭代数", "节点总数"))
print("  " + "-" * 62)
total_nodes = 0
for k, m in models.items():
    cls = type(m).__name__
    if cls == "Pipeline":
        inner = m.steps[-1][1]
        cls = "Pipeline(" + type(inner).__name__ + ")"
        c = getattr(inner, "coef_", None)
        n = int(c.size) if c is not None else 0
        print("  %-12s %-22s %10s %12s" % (k, cls, "-", f"{n} 系数"))
        total_nodes += n
        continue
    est = getattr(m, "estimators_", None)
    pred = getattr(m, "_predictors", None)       # HistGradientBoosting 在这
    if pred is not None:
        flat = [t for stage in pred for t in stage]
        ntree = len(flat); nodes = sum(node_count(t) for t in flat)
        est = flat
    elif est is not None:
        ntree = len(est); nodes = sum(node_count(t) for t in est)
    else:
        ntree = "-"; nodes = 0
    print("  %-12s %-22s %10s %12s" % (k, cls, ntree, format(nodes, ",")))
    total_nodes += nodes

print("  " + "-" * 62)
print("  %-12s %-22s %10s %12s" % ("合计", "", "", format(total_nodes, ",")))
print()

# 估算 JSON 导出体积：每个节点大约需要 4 个数（特征号、阈值、左、右）
print("=" * 76)
print("  JSON 导出体积估算")
print("=" * 76)
sz_compact = total_nodes * 22          # 紧凑数组格式，约 22 字节/节点
sz_verbose = total_nodes * 90          # 带键名的对象格式
print("  节点总数           %s" % format(total_nodes, ","))
print("  紧凑数组格式（推荐）  约 %.1f MB" % (sz_compact / 1048576))
print("  带键名对象格式      约 %.1f MB" % (sz_verbose / 1048576))
print()
pkl = (ROOT / "climax_model.pkl").stat().st_size
print("  对照：pickle %.1f MB" % (pkl / 1048576))
print()

# 实测：真正导出一次看看
print("=" * 76)
print("  实测导出（紧凑格式 + 量化到 4 位小数）")
print("=" * 76)
try:
    import gzip
    dump = {}
    for k, m in models.items():
        cls = type(m).__name__
        if cls == "Pipeline":
            inner = m.steps[-1][1]
            if hasattr(inner, "coef_"):
                dump[k] = {"coef": np.round(inner.coef_.ravel(), 5).tolist(),
                           "intercept": float(np.ravel(inner.intercept_)[0]),
                           "mean": np.round(m.steps[0][1].mean_, 4).tolist(),
                           "scale": np.round(m.steps[0][1].scale_, 4).tolist(),
                           "type": "linear"}
            continue
        pred = getattr(m, "_predictors", None)
        est = ([t for stage in pred for t in stage] if pred is not None
               else getattr(m, "estimators_", None))
        if est is None:
            continue
        trees = []
        for t in est:
            if hasattr(t, "tree_"):
                tt = t.tree_
                trees.append({
                    "f": tt.feature.tolist(),
                    "t": np.round(tt.threshold, 4).tolist(),
                    "l": tt.children_left.tolist(),
                    "r": tt.children_right.tolist(),
                    "v": np.round(tt.value.ravel(), 5).tolist(),
                })
            else:
                nd = t.nodes
                trees.append({
                    "f": nd["feature_idx"].tolist(),
                    "t": np.round(nd["num_threshold"], 4).tolist(),
                    "l": nd["left"].tolist(),
                    "r": nd["right"].tolist(),
                    "v": np.round(nd["value"].ravel(), 5).tolist(),
                })
        dump[k] = {"type": "forest", "trees": trees}
    raw = json.dumps(dump, separators=(",", ":")).encode()
    gz = gzip.compress(raw, 9)
    print("  JSON 原始   %.1f MB" % (len(raw) / 1048576))
    print("  gzip 压缩后  %.1f MB" % (len(gz) / 1048576))
    print("  → 浏览器原生支持 gzip 解压（DecompressionStream），所以分发用压缩版")
    print()
    print("  单个 HTML 文件的总大小估算：")
    js = 40 * 1024                     # 播放器 + 推理 + 特征提取代码
    html = 30 * 1024
    print("    模型（gzip）      %6.1f MB" % (len(gz) / 1048576))
    print("    推理 + 特征代码     %6.1f KB" % (js / 1024))
    print("    HTML/CSS          %6.1f KB" % (html / 1024))
    print("    " + "-" * 30)
    print("    合计              %6.1f MB" % ((len(gz) + js + html) / 1048576))
except Exception as e:
    print("  导出失败:", e)

print()
print("=" * 76)
print("  浏览器端能不能做音频解码 + 特征提取？")
print("=" * 76)
for item, how in (
    ("MP3/FLAC/WAV 解码", "Web Audio API 的 decodeAudioData()，原生支持"),
    ("重采样到 16 kHz 单声道", "OfflineAudioContext 指定 sampleRate=16000"),
    ("能量包络（0.05 秒帧 / 0.1 秒窗）", "纯 JS 循环算 RMS，34k 帧毫秒级"),
    ("95 分位峰 + 8 秒最小间隔", "排序 + 贪心，毫秒级"),
    ("21 个特征", "全是均值/中位数/斜率，纯 JS"),
    ("FFT（zcr/centroid/flatness）", "AnalyserNode 或自写 radix-2，1 秒片段而已"),
    ("模型推理", "遍历树，每个候选 ~1500 节点 × 58 候选 = 9 万次比较，毫秒级"),
):
    print("  %-30s %s" % (item, how))
