# -*- coding: utf-8 -*-
"""拆解「分析高潮点」的耗时构成"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

import climax_finder as CF
from climax_finder import (SR, HOP, ClimaxModel, decode_audio_mono, energy_envelope,
                           _smooth, find_peaks, model_features, load_transcript,
                           text_score_at, DEFAULT_CUES)

CASES = [
    (r"D:\库\音声\含高潮时间\RJ362169\01_mp3\track04_没关系，我会用魔法来隐藏的…！.mp3",
     r"D:\pyitme\onsei2lrc\_climax2\asr"),
    (r"D:\库\音声\含高潮时间\RJ362169\01_mp3\track05_我，想要知道.mp3",
     r"D:\pyitme\onsei2lrc\_climax2\asr"),
]

model = ClimaxModel()

for fp, asr in CASES:
    p = Path(fp)
    print("=" * 92)
    print(f"  {p.name[:56]}   {p.stat().st_size/1048576:.1f} MB")
    print("=" * 92)
    T = {}

    t0 = time.time(); x = decode_audio_mono(p); T["① 解码 16kHz 单声道"] = time.time() - t0
    dur = len(x) / SR
    T["_dur"] = dur

    t0 = time.time(); db, hop = energy_envelope(x); T["② 算能量包络"] = time.time() - t0
    t0 = time.time(); s = _smooth(db, int(0.3 / hop)); T["③ 平滑"] = time.time() - t0
    t0 = time.time(); med = float(np.median(s)); T["④ 中位数"] = time.time() - t0
    t0 = time.time(); peaks = find_peaks(s, hop); T["⑤ 找候选峰"] = time.time() - t0

    t0 = time.time(); segs = load_transcript(p, [Path(asr)]); T["⑥ 读转写"] = time.time() - t0

    t0 = time.time()
    feats = [model_features(x, s, hop, i, med) for i in peaks]
    T["⑦ 提特征"] = time.time() - t0

    t0 = time.time()
    for i in peaks:
        text_score_at(segs, i * hop, DEFAULT_CUES)
    T["⑧ 文本匹配"] = time.time() - t0

    order = sorted(range(len(peaks)), key=lambda n: -s[peaks[n]])
    rank_of = {n: r for r, n in enumerate(order)}
    for n, f in enumerate(feats):
        f["rank_energy"] = rank_of[n]
        f["gap_prev"] = 0.0 if n == 0 else float((peaks[n] - peaks[n - 1]) * hop)
        f["txt"] = 0.0

    t0 = time.time(); raw = model.predict_raw(feats); T["⑨ 模型预测"] = time.time() - t0
    t0 = time.time(); prob = model.predict_prob(raw); T["⑩ 校准"] = time.time() - t0

    tot = sum(v for k, v in T.items() if not k.startswith("_"))
    n_frames = len(s)
    print(f"  时长 {dur/60:.1f} 分钟    包络帧数 {n_frames:,}    候选峰 {len(peaks)} 个")
    print()
    print(f"  {'阶段':<22}{'耗时':>10}{'占比':>9}{'相对实时':>12}")
    print("  " + "-" * 56)
    for k in sorted(T):
        if k.startswith("_"):
            continue
        v = T[k]
        print(f"  {k:<22}{v*1000:>8.1f} ms{v/tot*100:>8.1f}%{dur/max(v,1e-9):>11.0f}×")
    print("  " + "-" * 56)
    print(f"  {'合计':<22}{tot*1000:>8.1f} ms{'':>9}{dur/tot:>11.0f}×")
    print()

    # 关键对比：如果对每一帧都打分要多久
    per_cand = T["⑦ 提特征"] / max(1, len(peaks))
    print(f"  候选池只有 {len(peaks)} 个，而包络有 {n_frames:,} 帧 —— 压缩了 "
          f"{n_frames/max(1,len(peaks)):.0f} 倍")
    print(f"  若对每帧都提特征：约 {per_cand*n_frames:.0f} 秒（实际只花 {T['⑦ 提特征']:.1f} 秒）")
    print()
