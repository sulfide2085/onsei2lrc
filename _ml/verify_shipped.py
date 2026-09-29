# -*- coding: utf-8 -*-
"""验证：交付的工具是否复现 10 折成绩 + 检查 isotonic 是否过冲"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

from climax_finder import ClimaxModel, MODEL_FEATS

print("=" * 96)
print("① isotonic 映射检查：原始分 → 校准概率")
print("=" * 96)
with open(ROOT / "climax_model.pkl", "rb") as fh:
    blob = pickle.load(fh)
iso = blob["iso"]
print(f"  {'原始分':>10}{'校准概率':>12}")
print("  " + "-" * 24)
for v in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
    print(f"  {v:>10.3f}{float(iso.predict([v])[0])*100:>11.1f}%")

print()
print("=" * 96)
print("② 工具端到端跑全部 69 轨，与 10 折成绩对照")
print("=" * 96)
from climax_finder import find_climaxes

meta = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))
GT, FILES = meta["gt"], meta["files"]
ASR_EXTRA = {"RJ324692": ROOT / "_climax" / "asr", "RJ362169": ROOT / "_climax2" / "asr"}
ASR_ML = ROOT / "_ml" / "asr"
TOL = 20.0
model = ClimaxModel()

per_work = defaultdict(lambda: [0, 0, set(), 0])   # tp_cand, n_cand, cov, gt_total
rows_log = []
for rj in sorted(GT):
    asrdir = ASR_EXTRA.get(rj, ASR_ML / rj)
    for tk, gts in GT[rj].items():
        fp = FILES[rj].get(tk)
        if not fp or not Path(fp).exists():
            continue
        r = find_climaxes(Path(fp), top=3, verbose=False, model="ml",
                          model_obj=model, transcript_dirs=[asrdir])
        if r.get("error"):
            continue
        for c in r["candidates"]:
            per_work[rj][1] += 1
            hit = [g for g in gts if abs(c["time"] - g) <= TOL]
            if hit:
                per_work[rj][0] += 1
                for g in hit:
                    per_work[rj][2].add((tk, g))
        per_work[rj][3] += len(gts)
    a = per_work[rj]
    print(f"  {rj:<12} 候选 {a[1]:>3}  命中 {a[0]:>3}  "
          f"精确率 {a[0]/max(1,a[1])*100:5.1f}%  "
          f"覆盖 {len(a[2])}/{a[3]} = {len(a[2])/max(1,a[3])*100:5.1f}%")

tot_tp = sum(a[0] for a in per_work.values())
tot_n = sum(a[1] for a in per_work.values())
tot_cov = set()
for rj, a in per_work.items():
    tot_cov |= {(rj,) + x for x in a[2]}
tot_gt = sum(a[3] for a in per_work.values())
print("  " + "-" * 80)
print(f"  {'合计':<12} 候选 {tot_n:>3}  命中 {tot_tp:>3}  "
      f"精确率 {tot_tp/max(1,tot_n)*100:5.1f}%  "
      f"覆盖 {len(tot_cov)}/{tot_gt} = {len(tot_cov)/max(1,tot_gt)*100:5.1f}%")
print()
print(f"  对照模型记录：精确率 {blob['meta']['cv_prec']*100:.1f}%  "
      f"召回率 {blob['meta']['cv_rec']*100:.1f}%")
print(f"  ⚠ 工具的模型是在**全部 10 部**上训练的，上表是**训练集内**成绩，必然偏高。")
print(f"     真实预期应看 10 折（模型元信息里的数字）。")

print()
print("=" * 96)
print("③ 校准概率的可靠性：最高概率区间实际命中率")
print("=" * 96)
probs, labels = [], []
for rj in sorted(GT):
    asrdir = ASR_EXTRA.get(rj, ASR_ML / rj)
    for tk, gts in GT[rj].items():
        fp = FILES[rj].get(tk)
        if not fp or not Path(fp).exists():
            continue
        r = find_climaxes(Path(fp), top=30, verbose=False, model="ml",
                          model_obj=model, transcript_dirs=[asrdir])
        for c in r["candidates"]:
            if "probability" not in c:
                continue
            probs.append(c["probability"])
            labels.append(any(abs(c["time"] - g) <= TOL for g in gts))
probs = np.array(probs); labels = np.array(labels, bool)
print(f"  {'概率区间':<16}{'候选':>7}{'命中':>6}{'实际命中率':>12}")
print("  " + "-" * 44)
for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
    m = (probs >= lo) & (probs < hi)
    if m.sum() == 0:
        continue
    k = int(labels[m].sum())
    print(f"  [{lo:.1f}, {hi:.1f})".ljust(16) + f"{int(m.sum()):>7}{k:>6}{k/m.sum()*100:11.1f}%")
