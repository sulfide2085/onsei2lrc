# -*- coding: utf-8 -*-
"""改 merge_candidates：按分数往下扫，直到凑够 top 个**不同事件**

现在的做法是按分数取前 top+1 个候选来合并。问题是：
同一处高潮会散成好几个峰（用户的观察），这些峰分数都很高，
**把整个池子占满了**，别处真正的高潮根本进不来。

实测例证（RJ362169 track04，官方标注 19:09 / 26:16）：
  8 个候选概率 >=0.5，但分成两团：
    26:06 / 26:25 / 26:34 / 25:58  ← 四处峰会合并成一个事件
    19:04 / 19:13 / 18:55          ← 另一团
  旧的 top=3 / pool=4 → 池子里 4 个名额全是 26:xx 那一团 → 只出 1 个候选，丢掉 19:09

新做法：一直往下扫，遇到「离已有事件都超过 gap」的就开一个新事件，
凑够 top 个就停。仍然按分数降序，所以先到的仍然是高分事件。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

TOL = 20.0
GT = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
rows = json.loads((ROOT / "_ml" / "features_v2.json").read_text(encoding="utf-8"))["rows"]
works = sorted({r["rj"] for r in rows if r["rj"] in GT})
bywork = {w: [r for r in rows if r["rj"] == w] for w in works}


def merge_old(cands, top=3, gap=30.0, pool=None):
    """现在的实现：只考察分数最高的 pool 个候选"""
    if not cands:
        return []
    if pool is None:
        pool = top + 1
    cands = sorted(cands, key=lambda c: -c["score"])[:max(1, pool)]
    events = []
    for c in cands:
        hit = next((e for e in events if abs(c["time"] - e["time"]) <= gap), None)
        if hit is None:
            e = dict(c); e["_t"] = [c["time"]]; e["merged"] = 1
            events.append(e)
        else:
            hit["_t"].append(c["time"])
            hit["time"] = float(np.mean(hit["_t"]))
            hit["merged"] += 1
            if c["score"] > hit["score"]:
                keep = dict(c); keep["_t"] = hit["_t"]; keep["merged"] = hit["merged"]
                events[events.index(hit)] = keep; hit = keep
            hit["score"] = max(hit["score"], c["score"])
    for e in events:
        e.pop("_t", None)
    events.sort(key=lambda e: -e["score"])
    return events[:top]


def merge_new(cands, top=3, gap=30.0, scan=None):
    """新实现：往下扫，凑够 top 个不同事件就停"""
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: -c["score"])
    if scan is None:
        scan = len(cands)
    events = []
    for c in cands[:scan]:
        hit = next((e for e in events if abs(c["time"] - e["time"]) <= gap), None)
        if hit is None:
            if len(events) >= top:
                continue          # 已经够了，但继续扫下去把中点修准
            e = dict(c); e["_t"] = [c["time"]]; e["merged"] = 1
            events.append(e)
        else:
            hit["_t"].append(c["time"])
            hit["time"] = float(np.mean(hit["_t"]))
            hit["merged"] += 1
            if c["score"] > hit["score"]:
                keep = dict(c); keep["_t"] = hit["_t"]; keep["merged"] = hit["merged"]
                events[events.index(hit)] = keep; hit = keep
            hit["score"] = max(hit["score"], c["score"])
    for e in events:
        e.pop("_t", None)
    events.sort(key=lambda e: -e["score"])
    return events[:top]


def snap(events, pool, key, window=12.0):
    if not window:
        return events
    for e in events:
        near = [c for c in pool if abs(c["time"] - e["time"]) <= window]
        if not near:
            continue
        best = max(near, key=lambda c: c.get(key, -1e9))
        if abs(best["time"] - e["time"]) > 1e-6:
            e = dict(e); e["time"] = best["time"]
    return events


# 用上次缓存好的样本外分数（含精修分）
CACHE = ROOT / "_ml" / "_final_oof2.pkl"
if not CACHE.exists():
    print("  缺少 _final_oof2.pkl，先跑 _ml/threshold_metrics.py")
    sys.exit(1)
OOF, _LAB = __import__("pickle").load(open(CACHE, "rb"))


def evaluate(merger, name, min_prob=0.0):
    per_p, per_r = [], []
    for hold in works:
        by = defaultdict(list)
        for s, g, tk, t, pk in OOF[hold]:
            by[tk].append({"time": t, "score": s, "peak": pk, "_ref": g})
        ftp = fn = 0; cov = set(); gtot = 0
        for tk, cand in by.items():
            g = GT.get(hold, {}).get(tk)
            if g is None:
                continue
            gtot += len(g)
            kept = [c for c in cand if c.get("_ref") is not None]
            base = merger([{k: v for k, v in c.items() if k != "_ref"} for c in cand],
                          top=3)
            base = snap(base, cand, "_ref", 12.0)
            for e in base:
                fn += 1
                h = [x for x in g if abs(e["time"] - x) <= TOL]
                if h:
                    ftp += 1
                    for x in h:
                        cov.add(x)
            del kept
        per_p.append(ftp / fn if fn else 0.0)
        per_r.append(len(cov) / max(1, gtot))
    P, R = float(np.mean(per_p)), float(np.mean(per_r))
    return P, R, 2 * P * R / max(1e-9, P + R)


print("  %-34s %9s %9s %8s %9s" % ("做法", "精确率", "召回率", "F1", "每轨候选"))
print("  " + "-" * 74)
nt = sum(len(GT[w]) for w in works)
for lbl, fn in (("旧：只考察分数最高的 4 个候选", merge_old),
                ("新：往下扫，凑够 3 个事件", merge_new)):
    P, R, F = evaluate(fn, lbl)
    print("  %-34s %8.1f%% %8.1f%% %7.3f" % (lbl, P * 100, R * 100, F))

# 每轨候选数
for lbl, fn in (("旧", merge_old), ("新", merge_new)):
    tot = 0
    for hold in works:
        by = defaultdict(list)
        for s, g, tk, t, pk in OOF[hold]:
            by[tk].append({"time": t, "score": s, "peak": pk, "_ref": g})
        for tk, cand in by.items():
            if GT.get(hold, {}).get(tk) is None:
                continue
            tot += len(fn([{k: v for k, v in c.items() if k != "_ref"} for c in cand],
                          top=3))
    print("  %s 每轨候选 %.2f 个" % (lbl, tot / nt))
