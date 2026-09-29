# -*- coding: utf-8 -*-
"""测试：把相近的候选点合并，对 10 折指标有什么影响

用户观察：模型有时把一个高潮拆成两个相距 ~20 秒的候选，真值在中间。
但不是所有作品都这样 —— RJ01586001 一轨里有 5 个真高潮挤在 147 秒内。
所以要测，不能拍脑袋定规则。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
from climax_finder import MODEL_FEATS

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TOL = 20.0
TOPN = 3
SEED = 0
GT = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]

blob = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))
rows = blob["rows"]
fb = ROOT / "climax_feedback.jsonl"
if fb.exists():
    import re
    latest = {}
    for line in fb.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r.get("audio"), round(r.get("time", 0), 1))] = r
    for (a, _), r in latest.items():
        if r.get("verdict") not in (0, 1) or not r.get("feats"):
            continue
        m = re.search(r"(RJ\d+)", str(a))
        if not m:
            continue
        row = {k: float(r["feats"].get(k, 0.0)) for k in MODEL_FEATS}
        row.update({"TP": bool(r["verdict"]), "rj": m.group(1), "track": "user",
                    "t": float(r["time"]), "txt": float(r["feats"].get("txt", 0.0))})
        rows.append(row)

works = sorted({r["rj"] for r in rows if r["rj"] in GT})
bywork = {w: [r for r in rows if r["rj"] == w] for w in works}
allrows = [r for r in rows]

RF = dict(n_estimators=400, min_samples_leaf=5, max_depth=10, class_weight="balanced",
          random_state=SEED, n_jobs=-1)
HGB = dict(max_iter=300, learning_rate=0.06, max_depth=10, min_samples_leaf=5,
           l2_regularization=1.0, class_weight="balanced", random_state=SEED)
LR = dict(class_weight="balanced", max_iter=3000, C=0.3)


def X(rs): return np.array([[r[f] for f in MODEL_FEATS] for r in rs], float)
def Y(rs): return np.array([r["TP"] for r in rs], int)


def fit(rs):
    xr, yr = X(rs), Y(rs)
    return [RandomForestClassifier(**RF).fit(xr, yr),
            HistGradientBoostingClassifier(**HGB).fit(xr, yr),
            make_pipeline(StandardScaler(), LogisticRegression(**LR)).fit(xr, yr)]


def score(models, rs):
    xt = X(rs)
    return np.mean([m.predict_proba(xt)[:, 1] for m in models], axis=0)


def select(cands, topn=TOPN, merge=0.0, gap=15.0, mode="mid"):
    """cands: [(score, track, time, rj)] → 选出 topn

    merge: 合并窗口（秒）。同一轨内相距 <= merge 的候选视为同一次高潮。
    mode:  mid = 取中点；best = 取分最高的那个
    """
    out = []
    by = defaultdict(list)
    for c in cands:
        by[(c[3], c[1])].append(c)
    for (rj, tk), group in by.items():
        group.sort(key=lambda x: x[0], reverse=True)
        if merge > 0:
            used = [False] * len(group)
            merged = []
            for i, c in enumerate(group):
                if used[i]:
                    continue
                cluster = [c]
                used[i] = True
                for j in range(i + 1, len(group)):
                    if used[j]:
                        continue
                    if any(abs(group[j][2] - x[2]) <= merge for x in cluster):
                        cluster.append(group[j])
                        used[j] = True
                if len(cluster) == 1:
                    merged.append(c)
                else:
                    sc = max(x[0] for x in cluster)
                    t = (np.mean([x[2] for x in cluster]) if mode == "mid"
                         else max(cluster, key=lambda x: x[0])[2])
                    merged.append((sc, tk, t, rj))
            group = merged
        group.sort(key=lambda x: -x[0])
        picked = []
        for c in group:
            if all(abs(c[2] - p[2]) > gap for p in picked):
                picked.append(c)
            if len(picked) >= topn:
                break
        out += picked
    return out


def evaluate(picked):
    tp = n = 0
    cov = set(); gtot = 0
    by = defaultdict(list)
    for c in picked:
        by[(c[3], c[1])].append(c)
    for rj in works:
        for tk, g in GT[rj].items():
            gtot += len(g)
            for c in by.get((rj, tk), []):
                n += 1
                h = [x for x in g if abs(c[2] - x) <= TOL]
                if h:
                    tp += 1
                    for x in h:
                        cov.add((rj, tk, x))
    return tp / max(1, n), len(cov) / max(1, gtot), tp, n, len(cov), gtot


print("跑 10 折，每折记录样本外分数，然后对比不同合并规则…")
oof = {}
for hold in works:
    tr = [r for r in allrows if r["rj"] != hold]
    te = bywork[hold]
    sc = score(fit(tr), te)
    oof[hold] = [(float(s), r["track"], float(r["t"]), r["rj"]) for s, r in zip(sc, te)]
print("  完成")
print()

CONFIGS = [
    ("当前（不合并，间隔 15s）",      dict(merge=0,  gap=15)),
    ("间隔改 20s",                    dict(merge=0,  gap=20)),
    ("间隔改 60s（粗暴拉开）",         dict(merge=0,  gap=60)),
    ("合并 20s → 中点",               dict(merge=20, gap=15, mode="mid")),
    ("合并 30s → 中点",               dict(merge=30, gap=15, mode="mid")),
    ("合并 45s → 中点",               dict(merge=45, gap=15, mode="mid")),
    ("合并 60s → 中点",               dict(merge=60, gap=15, mode="mid")),
    ("合并 60s → 取高分",             dict(merge=60, gap=15, mode="best")),
    ("合并 30s → 取高分",             dict(merge=30, gap=15, mode="best")),
]

print("  %-30s %7s %8s %8s %8s" % ("规则", "精确率", "召回率", "候选数", "每轨"))
print("  " + "-" * 68)
results = {}
n_tracks = sum(len(GT[w]) for w in works)
for label, kw in CONFIGS:
    # 关键：10 折的候选要**汇总后一起评估**。
    # 每折只产出一部作品的候选，却拿全部 92 个标注点当分母，召回率会差十倍。
    all_picked = []
    for hold in works:
        all_picked += select(oof[hold], **kw)
    p, r, tp, n, cv, gt_ = evaluate(all_picked)
    results[label] = (p, r, n)
    print("  %-30s %6.1f%% %7.1f%% %8d %8.2f" % (label, p*100, r*100, n, n/n_tracks))

print()
print(f"  基准核对：候选 {sum(v[2] for v in results.values())/len(results):.0f} 个"
      f"（{n_tracks} 轨 × 最多 3 个 = 最多 {n_tracks*3}）")
best = max(results.items(), key=lambda kv: (kv[1][0] + kv[1][1]))
print("  按「精确率+召回率」最高：%s（精确 %.1f%% / 召回 %.1f%%）" % (
    best[0], best[1][0]*100, best[1][1]*100))
