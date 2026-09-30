# -*- coding: utf-8 -*-
import os
"""生成 JS 对拍的参考数据

用**真实音频**跑一遍完整 Python 流程，把每一步的中间结果都存下来：
  ① 解码后的 16 kHz 单声道样本（原始 float32 字节）
  ② 能量包络 + 平滑 + 中位数
  ③ 峰列表
  ④ 每个峰的 21 个特征
  ⑤ 模型原始分 / 校准概率 / 精修分
  ⑥ 最终候选（合并 + 精修之后）

JS 那边读同一份音频，逐步比对。哪一步不一致就一目了然。
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np
import climax_finder as CF

OUT = ROOT / "_offline"
OUT.mkdir(exist_ok=True)

BASE = Path(os.environ.get("ONSEI_AUDIO_BASE",
                        r"D:\库\音声\含高潮时间"))
# 挑一个够长的轨（短的标题轨会被 MIN_DURATION 挡掉，测不到东西）
AUDIO = None
for d in sorted(BASE.iterdir()):
    if not d.is_dir() or not d.name.startswith("RJ362169"):
        continue
    for f in sorted((d / "01_mp3").glob("track04*.mp3")):
        AUDIO = f
if AUDIO is None:
    AUDIO = max(BASE.rglob("*.mp3"), key=lambda p: p.stat().st_size)
print(f"  测试音频: {AUDIO.name}")

x = CF.decode_audio_mono(AUDIO)
print(f"  解码: {len(x)} 样本 = {len(x)/CF.SR:.1f} 秒")
(OUT / "ref_audio.f32").write_bytes(np.asarray(x, dtype=np.float32).tobytes())

db, hop = CF.energy_envelope(x)
s = CF._smooth(db, int(0.3 / hop))
med = float(np.median(s))
# find_peaks 返回 numpy int，转成 int 才能 JSON 序列化
peaks = [int(i) for i in CF.find_peaks(s, hop)]
print(f"  包络 {len(db)} 帧，峰 {len(peaks)} 个")

# 每个峰的特征（按 find_climaxes 里的顺序补齐轨内上下文）
feats = []
for i in peaks:
    f = CF.model_features(x, s, hop, i, med)
    f["txt"] = 0.0
    feats.append(f)
order = sorted(range(len(peaks)), key=lambda n: -s[peaks[n]])
rank_of = {n: r for r, n in enumerate(order)}
for n, f in enumerate(feats):
    f["rank_energy"] = float(rank_of[n])
    f["gap_prev"] = 0.0 if n == 0 else float((peaks[n] - peaks[n - 1]) * hop)

model = CF.ClimaxModel()
raw = model.predict_raw(feats)
prob = model.predict_prob(raw)
ref = model.predict_refine(feats)

# 完整流程（用 find_climaxes 的真实链路，但喂已经解好的样本）
res = CF.find_climaxes(AUDIO, top=3, verbose=False, model="ml", model_obj=model,
                       min_score=CF.MIN_PROB)

# 逐模型的分数（不含集成）—— JS 对拍时能直接指出是哪个模型不一致
Xm = model._matrix(feats)
per_model = {}
for _n, _m in model.models.items():
    # ⚠️ 不能只看 type(_m).__name__ —— LR/Ridge 是 Pipeline，
    # 名字里没有 "Classifier"，会掉进 predict() 分支拿到硬标签 0/1。
    # 要看 Pipeline 的**最后一步**。
    # 能力检测：分类器都有 predict_proba，回归器都没有。
    # 不能靠类名判断 —— sklearn 的逻辑回归叫 LogisticRegression，名字里没有 Classifier。
    if hasattr(_m, "predict_proba"):
        per_model[_n] = [float(v) for v in _m.predict_proba(Xm)[:, 1]]
    else:
        per_model[_n] = [float(v) for v in _m.predict(Xm)]

ref_out = {
    "per_model": per_model,
    # 特征矩阵转 float32 —— JS 那边就是这么喂给树的
    "X32": [[float(v) for v in row] for row in Xm.astype('float32')],
    "audio": AUDIO.name,
    "n_samples": len(x),
    "duration": len(x) / CF.SR,
    "median": med,
    "n_frames": len(db),
    "peaks": peaks,
    "features": [{k: float(v) for k, v in f.items()} for f in feats],
    "raw": [float(v) for v in raw],
    "prob": [float(v) for v in prob],
    "refine": [float(v) for v in ref],
    "candidates": [
        {"time": c["time"], "mmss": c["mmss"], "score": c["score"],
         "probability": c.get("probability"), "peak": c.get("peak"),
         "merged": c.get("merged"), "snapped_from": c.get("snapped_from")}
        for c in res["candidates"]],
}
(OUT / "ref_out.json").write_text(
    json.dumps(ref_out, ensure_ascii=False), encoding="utf-8")

# 包络单独存（浮点多，放 JSON 里太大）
np.save(OUT / "ref_env.npy", np.stack([db, s]))
print(f"  已写出 ref_audio.f32（{len(x)*4/1048576:.1f} MB）、ref_out.json、ref_env.npy")
print(f"  最终候选: {[c['mmss'] for c in res['candidates']]}")
print(f"  对应概率: {[round(c.get('probability', 0), 3) for c in res['candidates']]}")
