# -*- coding: utf-8 -*-
"""在候选点附近换一个「声学地标」，看能不能比当前候选更接近真值

当前候选 = 0.3 秒平滑能量包络的局部极大值。真高潮是语义事件，
能量峰未必落在它上面。所以试试别的定义：
  ① 能量最高点（局部窗口内）
  ② 上升最陡的点（一阶导最大）
  ③ 起爆点（能量首次超过 峰值−3dB）
  ④ 能量重心（超过阈值的部分按能量加权）
  ⑤ 下降最陡的点
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

from climax_finder import (SR, HOP, decode_audio_mono, energy_envelope, _smooth,
                           find_peaks)

GT = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
FILES = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["files"]
TOL = 20.0
WORKS = ["RJ01586001", "RJ01589742", "RJ01610459", "RJ324692", "RJ362169"]


def landmarks(s, hop, t, half=30.0):
    """候选时刻 t 附近的几种地标时刻"""
    n = len(s)
    i = int(t / hop)
    a = max(0, i - int(half / hop))
    b = min(n, i + int(half / hop))
    if b - a < 5:
        return {}
    w = s[a:b]
    med = float(np.median(w))
    out = {}
    out["① 窗口内能量最高点"] = (a + int(np.argmax(w))) * hop
    d = np.diff(w)
    out["② 上升最陡"] = (a + int(np.argmax(d))) * hop
    out["⑤ 下降最陡"] = (a + int(np.argmin(d))) * hop
    pk = w.max()
    thr = pk - 3.0                          # 峰值下 3 dB
    above = np.where(w >= thr)[0]
    out["③ 起爆点（首次超过峰值−3dB）"] = (a + int(above[0])) * hop if len(above) else t
    e = np.clip(w - med, 0, None)
    out["④ 能量重心"] = (a + float((np.arange(len(w)) * e).sum() / max(1e-9, e.sum()))) * hop
    # 相对候选点的偏移
    out["_raw_peak"] = (a + int(np.argmax(w))) * hop
    return out


res = defaultdict(list)
cur_err = []
for rj in WORKS:
    for tk, gts in GT[rj].items():
        fp = FILES[rj].get(tk)
        if not fp or not Path(fp).exists():
            continue
        try:
            x = decode_audio_mono(Path(fp))
        except Exception as e:
            print(f"  ✗ {rj}/{tk}: {e}")
            continue
        db, hop = energy_envelope(x)
        s = _smooth(db, int(0.3 / hop))
        peaks = find_peaks(s, hop)
        if not peaks:
            continue
        for g in gts:
            # 当前工具会选的候选 = 离真值最近的那个峰
            if not peaks:
                continue
            i = min(peaks, key=lambda k: abs(k * hop - g))
            t = i * hop
            cur_err.append(abs(t - g))
            for name, v in landmarks(s, hop, t).items():
                if name.startswith("_"):
                    continue
                res[name].append(abs(v - g))
            res["_cand"].append(t)
        print(f"  {rj}/{tk}  真值 {len(gts)} 个  峰 {len(peaks)} 个", flush=True)

print()
print("=" * 88)
print(f"  共 {len(cur_err)} 个真值点，对比各种地标的时间误差")
print("=" * 88)
print("  %-28s %9s %9s %8s %8s %8s" % ("地标", "平均误差", "中位误差", "≤5秒", "≤10秒", "≤15秒"))
print("  " + "-" * 76)
cur = np.array(cur_err)
print("  %-28s %8.1f秒 %8.1f秒 %7.0f%% %7.0f%% %7.0f%%" % (
    "【现状】候选（能量局部极大）", cur.mean(), np.median(cur),
    (cur <= 5).mean()*100, (cur <= 10).mean()*100, (cur <= 15).mean()*100))
for name in sorted(res):
    if name.startswith("_"):
        continue
    e = np.array(res[name])
    print("  %-28s %8.1f秒 %8.1f秒 %7.0f%% %7.0f%% %7.0f%%" % (
        name, e.mean(), np.median(e),
        (e <= 5).mean()*100, (e <= 10).mean()*100, (e <= 15).mean()*100))
print()
print("  说明：这里是「离真值最近的峰」作为候选（理想情况），")
print("        实际工具选出的候选还要经过模型打分，误差只会更大。")
