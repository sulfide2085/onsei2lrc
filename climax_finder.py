# -*- coding: utf-8 -*-
"""climax_finder.py —— 从日语音频里找高潮（射精）候选点

设计取向：**候选生成器**，不是自动打标机。
输出的时间点需要你确认——实测精确率约 53%，自动标会错一半。

原理（详见 README 第 11 节）：高潮不是一个「响」的时刻，而是一个**形状**——
    持续高涨的平台 → 爆发 → 迅速脱力下坠
普通响亮时刻是从安静里突然一响、之后维持原状，两者形状相反。

评分 = 1.0 × 声学(signature) + 1.0 × 声学(recover) + 0.5 × 语义(拟声/倒计时)

单独依赖：
    numpy              （已有）
    faster-whisper     （已有，只用它的 decode_audio 解码）
转写文本可选：同目录下的 *.segments.json 或 *.lrc。没有就退化为纯声学模式。

用法：
    python climax_finder.py "C:\\音声\\RJ324692\\01_mp3"
    python climax_finder.py track01.mp3 --top 5
    python climax_finder.py "目录" --json out.json
    python climax_finder.py "目录" --acoustic-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["find_climaxes", "find_in_path", "CueRules", "DEFAULT_CUES"]

# ======================================================================================
# 归一化常数：用 5 轨 / 9 个官方标注点（RJ324692 的 readme Finish Time）拟合
# 换作品后建议重新拟合，见 README 第 11 节的复现步骤
# ======================================================================================
NORM = {
    "signature": {"mu": -0.250, "sd": 7.702},
    "recover":   {"mu": 1.173,  "sd": 0.889},
    "txt":       {"mu": 0.280,  "sd": 0.694},
}
W_ACOUSTIC = 1.0        # signature 权重
W_TXT = 0.5             # 语义权重
SR = 16000              # 解码采样率
HOP = 0.05              # 帧步长（秒）
WIN = 0.10              # 帧长（秒）

# ======================================================================================
# 置信度标定：把分数映射成经验精确率
#
# 在 15 轨 / 67 个候选上实测（9 轨有高潮、6 轨没有，标注点来自两部作品的 readme）：
#     分数 <  +2.5  →  经验精确率  0% ~ 12%   （基本是噪声）
#     分数 >= +2.5  →  经验精确率 54% ~ 64%   （可用）
# 门槛非常清晰，所以下面就按这条线分档。
# ======================================================================================
CONF_HIGH = 2.5         # 高于此分：实测精确率 54~64%
CONF_MID = 1.5          # 1.5~2.5：实测 7~12%
# 低于 CONF_MID：实测 0%

# 轨级判断：整轨最高分低于此值 → 提示「这一轨可能没有高潮」
# 实测（排除 <1 分钟的短轨后）轨级准确率 92%
TRACK_MIN_SCORE = 2.3
# 短于此时长的轨，特征不稳定（实测 5 秒的标题轨能拿到 +3.90 分），直接跳过
MIN_DURATION = 60.0


def confidence_of(score: float) -> Tuple[str, str]:
    """formula 模式的分数 → (档位, 经验精确率说明)。数字来自那批 2 部作品的实测标定。"""
    if score >= CONF_HIGH:
        return "高", "实测精确率 54~64%"
    if score >= CONF_MID:
        return "中", "实测精确率 7~12%"
    return "低", "实测精确率 0%"


# ======================================================================================
# ml 模式的置信度：直接用 isotonic 校准后的概率
#
# 校准器在 10 折**样本外**预测上拟合（3981 个候选）。样本外实测命中率：
#     原始分 [0.85, 0.95) → 75.6%（78 个）      [0.95, 1.00] → 94.1%（17 个）
#     原始分 [0.70, 0.85) → 55.4%（112 个）     [0.50, 0.70) → 33.0%（106 个）
#     原始分 [0.30, 0.50) → 15.3%（359 个）     [0.10, 0.30) →  4.3%（1012 个）
#     原始分 [0.00, 0.10) →  0.7%（2297 个）
# Brier 分数 0.0565 → 0.0412
# ======================================================================================
# 上限：样本外最高区间的实测命中率是 94.1%（只有 17 个样本）。
# isotonic 会在极少数全正样本块上映射到 100%，那是在给用户「绝对确定」的错觉。
# 所以封顶 95%——不影响任何合理阈值的判断，但不再声称 100%。
PROB_CAP = 0.95


def confidence_of_prob(p: float) -> Tuple[str, str]:
    """校准概率 → (档位, 说明)。概率本身已是命中率，所以直接引用实测区间。"""
    if p >= 0.6:
        return "高", f"校准概率 {p*100:.0f}%　（此区间样本外实测命中率 55~94%）"
    if p >= 0.3:
        return "中", f"校准概率 {p*100:.0f}%　（此区间样本外实测命中率 15~33%）"
    return "低", f"校准概率 {p*100:.0f}%　（此区间样本外实测命中率 0.7~4%）"


# 轨级判断（ml 模式）：整轨最高校准概率低于此值 → 提示可能没有高潮
TRACK_MIN_PROB = 0.30


# ======================================================================================
# 语义线索
# ======================================================================================
class CueRules:
    """可替换的线索规则。默认这套是从实测语料里总结的，换作品建议自行调整。"""

    def __init__(self, onomatopoeia: str, done: str, countdown: str, negative: str):
        self.ono = re.compile(onomatopoeia)
        self.done = re.compile(done)
        self.cnt = re.compile(countdown)
        self.neg = re.compile(negative)

    def score(self, text: str) -> Tuple[int, List[str]]:
        w, tags = 0, []
        if self.ono.search(text):
            w += 2; tags.append("拟声")
        if self.done.search(text):
            w += 2; tags.append("已然")
        if self.cnt.search(text):
            w += 1; tags.append("倒计时")
        if self.neg.search(text):
            w -= 3; tags.append("负面")
        return w, tags


# 射精拟声（ぴゅ/どぴゅ/びゅく/びゅる 系）
DEFAULT_CUES = CueRules(
    onomatopoeia=r"ぴゅ|ピュ|どぴゅ|ドピュ|びゅ|ビュ|びゅく|びゅる|ぴちゅ",
    # 已然/进行：注意 `イク` 和 `イッた` 是两种不同写法，早期版本只匹配后者，
    # 导致换到用「イクイクイク」的作品时完全失效（实测 RJ362169 track06 漏掉）。
    done=r"出(た|てる|ちゃ|てしま)|イッ?(た|ちゃ|たの|てる)|イ[クキ]|イキ?(ます|そう|たい)|射精(し)?(た|てる)",
    countdown=r"[5５][…\s、.]*[4４][…\s、.]*[3３]|[3３][…\s、.]*[2２][…\s、.]*[1１]|ぜー?ろ",
    # 负面：未然 / 祈使 / 回顾 / 泛指 —— 这些时刻文本有线索但不是高潮
    # `出して` 在祈使语境下多为「还没射」，但若紧跟拟声则是正在射——
    # `text_score_at` 里正面与负面同时命中时按和计分，靠拟声把分数拉回来。
    negative=r"(し|に)?そう|ないように|寸前|したい|したから|出して(ください|ぇぇ)|ばかり|準備|支度",
)


# ======================================================================================
# 音频特征
# ======================================================================================
def _smooth(a: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k)
    return np.convolve(a, np.ones(k) / k, mode="same")


def energy_envelope(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """逐帧 RMS（dBFS），帧长 WIN、步长 HOP。"""
    step, wlen = int(SR * HOP), int(SR * WIN)
    n = max(1, (len(x) - wlen) // step)
    out = np.empty(n)
    for i in range(n):
        w = x[i * step:i * step + wlen]
        out[i] = 10.0 * np.log10(max(float(np.mean(w ** 2)), 1e-12))
    return out, HOP


def _win(arr: np.ndarray, hop: float, i: int, a: float, b: float) -> np.ndarray:
    """取 [i+a, i+b] 秒的窗口。轨首/轨尾不足时自动收缩，绝不返回空数组
    ——空数组会让 nanmean 产生 NaN 并顺着评分传播下去。"""
    n = len(arr)
    lo = min(max(0, i + int(a / hop)), n - 1)
    hi = min(max(lo + 1, i + int(b / hop)), n)
    return arr[lo:hi]


def find_peaks(arr: np.ndarray, hop: float, min_gap: float = 8.0,
               pct: float = 95.0) -> List[int]:
    """候选池：全轨 95 分位以上的局部极大值，再做最小间隔抑制。

    早期版本用「固定取前 12 个峰」，在样本外测试里崩了——
    某些音轨（如 RJ362169 的 track06）后半段挤满极响的音效，
    固定名额被占满，真高潮（全轨 99.8% 分位）反而进不了候选池，
    后面的评分再准也没用。
    改用分位数阈值后不再依赖「一轨里最多有几个响点」，自适应。
    """
    if len(arr) < 3:
        return []
    thr = float(np.percentile(arr, pct))
    gap = max(1, int(min_gap / hop))
    idx = [i for i in range(1, len(arr) - 1)
           if arr[i] >= arr[i - 1] and arr[i] > arr[i + 1] and arr[i] >= thr]
    kept: List[int] = []
    for i in sorted(idx, key=lambda j: -arr[j]):
        if all(abs(i - j) > gap for j in kept):
            kept.append(i)
    return kept


def acoustic_features(x: np.ndarray, med: float, i: int, s: np.ndarray,
                      hop: float) -> Dict[str, float]:
    """signature = 前平台 − 释放后；recover = 掉到全轨中位数以下的用时(秒)。"""
    pre = float(np.nanmean(_win(s, hop, i, -32, -10)) - med)
    post = float(np.nanmean(_win(s, hop, i, 1.5, 4)) - med)
    after = s[i:min(len(s), i + int(60 / hop))]
    below = np.where(after < med)[0]
    rec = float(below[0] * hop) if len(below) else 60.0
    return {"signature": pre - post, "pre30": pre, "post4": post, "recover": rec,
            "peak": float(s[i]) - med}


# ======================================================================================
# 机器学习模型（--model ml）
#
# 这是训练与推理**共用**的特征实现。任何改动都必须重新训练模型，否则特征错位。
# 训练脚本 _ml/train_model.py 反过来 import 本函数，保证两边绝对一致。
# ======================================================================================
# 模型使用的 21 个特征，顺序即训练时的列顺序，不可改动
MODEL_FEATS = ["peak", "pre60", "pre30", "pre10", "post2", "post10", "post30",
               "signature", "contrast", "rise", "decay", "recover", "prominence",
               "plateau", "rel_time", "rank_energy", "gap_prev", "zcr", "centroid",
               "flatness", "txt"]

MODEL_FILE = Path(__file__).with_name("climax_model.pkl")


def _winmean(s: np.ndarray, hop: float, i: int, a: float, b: float, med: float) -> float:
    """峰值 i 前后 [a, b] 秒窗口的平均能量（相对全轨中位数）。"""
    n = len(s)
    lo = min(max(0, i + int(a / hop)), n - 1)
    hi = min(max(lo + 1, i + int(b / hop)), n)
    return float(np.mean(s[lo:hi]) - med)


def model_features(x: np.ndarray, s: np.ndarray, hop: float, i: int,
                   med: float) -> Dict[str, float]:
    """围绕峰值 i 提取 21 个模型特征（不含 rank_energy/gap_prev/txt，那三个需要轨内上下文）。"""
    t = i * hop
    pk = float(s[i] - med)
    f: Dict[str, float] = {}

    # ---- 能量水平（相对全轨中位数，dB） ----
    f["peak"] = pk
    f["pre60"] = _winmean(s, hop, i, -62, -35, med)
    f["pre30"] = _winmean(s, hop, i, -32, -10, med)
    f["pre10"] = _winmean(s, hop, i, -10, -2, med)
    f["post2"] = _winmean(s, hop, i, 1.5, 4, med)
    f["post10"] = _winmean(s, hop, i, 4, 10, med)
    f["post30"] = _winmean(s, hop, i, 10, 30, med)
    f["signature"] = f["pre30"] - f["post2"]
    f["contrast"] = f["peak"] - f["pre10"]

    # ---- 形状 ----
    j = i
    thr_r = med + f["pre10"] * 0.5 + (pk - f["pre10"]) * 0.5
    while j > 0 and s[j] > thr_r:
        j -= 1
    f["rise"] = (i - j) * hop
    k = i
    thr_d = med + f["post2"] + (pk - f["post2"]) * 0.5
    while k < len(s) - 1 and s[k] > thr_d:
        k += 1
    f["decay"] = (k - i) * hop
    after = s[i:min(len(s), i + int(60 / hop))]
    below = np.where(after < med)[0]
    f["recover"] = float(below[0] * hop) if len(below) else 60.0
    a0 = max(0, i - int(30 / hop))
    b0 = min(len(s), i + int(30 / hop))
    f["prominence"] = pk - float(np.median(s[a0:b0]) - med)
    w = s[max(0, i - int(3 / hop)):min(len(s), i + int(3 / hop) + 1)]
    f["plateau"] = float((w > med + pk * 0.5).mean()) if len(w) else 0.0

    # ---- 轨内位置 ----
    f["rel_time"] = t / max(1e-6, len(s) * hop)

    # ---- 频谱（实测重要性低，但保留——删掉会略降指标） ----
    step = int(hop * SR)
    seg = x[max(0, i - int(0.5 / hop)) * step:min(len(x), i + int(0.5 / hop)) * step]
    if len(seg) > 512:
        seg = seg - seg.mean()
        f["zcr"] = float((np.diff(np.sign(seg)) != 0).mean())
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-12
        fr = np.fft.rfftfreq(len(seg), 1 / SR)
        f["centroid"] = float((sp * fr).sum() / sp.sum())
        f["flatness"] = float(np.exp(np.log(sp).mean()) / sp.mean())
    else:
        f["zcr"] = f["centroid"] = f["flatness"] = 0.0
    return f


class ClimaxModel:
    """加载 climax_model.pkl，给出原始分与校准概率。

    设计要点（见 README 第 11 节）：
      · **排序用原始分**——isotonic 校准是单调变换，会产生大量并列值，
        用它排序反而略降 AUC（实测 −0.0018）。
      · **显示用校准概率**——让「62%」真的意味着 62%（Brier 0.078 → 0.044）。
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else MODEL_FILE
        if not self.path.exists():
            raise FileNotFoundError(
                f"找不到模型文件 {self.path}\n"
                f"请先运行：python _ml\\train_model.py")
        import pickle
        with open(self.path, "rb") as fh:
            blob = pickle.load(fh)
        self.models: Dict[str, object] = blob["models"]
        self.weights: List[float] = blob["weights"]
        self.iso = blob["iso"]                    # isotonic 校准器（可为 None）
        self.feat_order: List[str] = blob["feats"]
        self.meta: Dict = blob.get("meta", {})
        if self.feat_order != MODEL_FEATS:
            raise ValueError(
                f"模型文件的特征顺序与本代码不一致——模型是用旧版特征训练的，"
                f"请重新运行 _ml\\train_model.py。\n"
                f"  模型: {self.feat_order}\n  代码: {MODEL_FEATS}")

    def _matrix(self, feats: List[Dict[str, float]]) -> "np.ndarray":
        return np.array([[float(f.get(k, 0.0)) for k in self.feat_order]
                         for f in feats], float)

    def predict_raw(self, feats: List[Dict[str, float]]) -> "np.ndarray":
        """三等权平均的原始分（用于排序）。"""
        if not feats:
            return np.zeros(0)
        X = self._matrix(feats)
        ps = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        w = np.asarray(self.weights, float)
        w = w / w.sum()
        return np.average(ps, axis=0, weights=w)

    def predict_prob(self, raw: "np.ndarray") -> "np.ndarray":
        """isotonic 校准后的概率（用于显示与阈值），封顶 PROB_CAP。

        没有校准器时原样返回。
        """
        raw = np.asarray(raw, float)
        if self.iso is None or raw.size == 0:
            return raw
        return np.clip(self.iso.predict(raw), 0.0, PROB_CAP)

    def describe(self) -> str:
        m = self.meta
        names = " + ".join(f"{k}×{v:.2f}" for k, v in zip(self.models, self.weights))
        s = [f"模型：{names}", f"特征：{len(self.feat_order)} 个"]
        if m:
            s.append(f"训练：{m.get('n_works', '?')} 部作品 / "
                     f"{m.get('n_tracks', '?')} 轨 / {m.get('n_pos', '?')} 正例")
            s.append(f"10 折留一作品实测：AUC {m.get('cv_auc', float('nan')):.3f}　"
                     f"精确率 {m.get('cv_prec', float('nan'))*100:.1f}%　"
                     f"召回率 {m.get('cv_rec', float('nan'))*100:.1f}%")
        return "\n".join("  " + x for x in s)


# ======================================================================================
# 转写文本
# ======================================================================================
def load_transcript(audio: Path, extra_dirs: Optional[List[Path]] = None) -> Optional[List[dict]]:
    """找转写：先看音频旁边，再看 extra_dirs。优先 .segments.json，其次 .lrc。

    支持在 extra_dirs 里用「名字包含」匹配——ASR 输出常带前缀/后缀
    （如 `track01_童贞君与小穴值日生同学.segments.json`），而音频名可能被改过。
    """
    dirs = [audio.parent] + list(extra_dirs or [])
    for d in dirs:
        if not d.is_dir():
            continue
        for pattern in (f"{audio.stem}.segments.json", f"{audio.stem}.lrc",
                        f"{audio.stem}.ja.lrc"):
            p = d / pattern
            if p.exists():
                r = _parse_transcript(p, audio.stem)
                if r:
                    return r
    # 退一步：按名字前缀模糊匹配（ASR 输出目录常这样）
    for d in (extra_dirs or []):
        if not d.is_dir():
            continue
        key = audio.stem.split("_")[0].lower()
        for p in sorted(d.glob("*.segments.json")):
            if p.name.lower().startswith(key):
                r = _parse_transcript(p, audio.stem)
                if r:
                    return r
    return None


def _parse_transcript(p: Path, stem: str) -> Optional[List[dict]]:
    try:
        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))["segments"]
        out = []
        for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            m = re.match(r"\[(\d+):([\d.]+)\](.*)", line.strip())
            if m:
                out.append({"start": int(m.group(1)) * 60 + float(m.group(2)),
                            "ja": m.group(3).strip()})
        return out or None
    except (OSError, ValueError, KeyError):
        return None


def text_score_at(segs: List[dict], t: float, cues: CueRules,
                  radius: float = 6.0) -> Tuple[int, List[str], str]:
    """候选点 ±radius 秒内最强的一条线索。返回 (权重, 标签, 那句原文)。

    `text` 一定返回**最近的那句台词**（即使没有任何线索命中）——
    用户需要看到候选点上到底在说什么，而不是只有命中关键词时才有内容。
    """
    best, tags = 0, []
    best_text, best_d = "", radius + 1e-9
    for s in segs:
        d = abs(s["start"] - t)
        ja = s.get("ja") or ""
        if d < best_d and ja:
            best_d, best_text = d, ja
        if d > radius:
            continue
        w, tg = cues.score(ja)
        if w > best:
            best, tags = w, tg
    return best, tags, best_text


def decode_audio_mono(path: Path, sr: int = SR) -> np.ndarray:
    """解码成 sr 采样率单声道 float32。

    **优先用 ffmpeg 子进程，不是 faster_whisper** —— 这是实测结论：

        方式                          解码耗时     导入开销
        faster_whisper.decode_audio    915 ms      6811 ms   ← 首次调用白等 6.8 秒
        ffmpeg 子进程                   779 ms         0 ms   ← 更快且无导入
        PyAV 直接用                    1199 ms        72 ms

    faster_whisper 的 decode_audio 只是 PyAV 的一层包装，但 import 它会连带
    把 ctranslate2 拉进来（6.2 秒）。本工具不需要 ASR 引擎，只解音频。
    ffmpeg 已经是本项目的外部依赖，所以没有新增依赖。
    """
    import subprocess
    try:
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                            "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                           capture_output=True)
    except FileNotFoundError:
        p = None
    if p is not None and p.returncode == 0 and p.stdout:
        return np.frombuffer(p.stdout, dtype=np.float32).copy()

    # 没有 ffmpeg（或该文件 ffmpeg 解不开）→ 退回 PyAV
    try:
        import av
    except ImportError:
        raise RuntimeError("ffmpeg 不可用，且未安装 PyAV —— 无法解码音频")
    res = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=sr)
    chunks = []
    with av.open(str(path), mode="r", metadata_errors="ignore") as c:
        for frame in c.decode(audio=0):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):          # 冲刷尾部
            chunks.append(f.to_ndarray().reshape(-1))
    del res
    import gc
    gc.collect()      # 不手动回收会漏（faster-whisper#390）
    if not chunks:
        raise RuntimeError("解码结果为空")
    return np.concatenate(chunks).astype(np.float32)


# ======================================================================================
# 合并相邻候选
#
# 问题（用户实测发现）：模型有时把**一次**高潮拆成两个相距 ~20 秒的候选，
# 两者置信度还差不多。用户听完说「准确的点在他们正中间」。
#
# 在 10 折样本外数据上核查，相距 ≤60 秒的成对候选共 110 对：
#     一次高潮被拆成两个   66 对（60%）
#     其中含误报           42 对（38%）
#     真的是两次高潮        2 对（1.8%）
# 所以「相近的两个候选多半是同一次高潮」成立。
#
# 但**不能**对整个候选池做合并 —— 单链式合并会让簇沿着时间轴漂移：
# 实测 20 秒窗口能把 100 多个候选串成一大簇（合并 779 个），指标直接崩掉。
# 所以只考察分数最高的 pool 个，再按「与簇代表点的距离」判断。
#
# 参数是实测选出来的（10 折留一作品，±20 秒容差，每轨取 3 个）：
#     规则                          精确率   召回率    F1     每轨候选
#     旧的间隔抑制（15 秒，拉满 3 个）   41.5%   68.5%   0.517   2.92
#     取前 3 → 合并 30s              54.3%   62.0%   0.579   1.48
#     取前 4 → 合并 30s  ← 采用       47.0%   68.5%   0.558   1.89
#     取前 5 → 合并 30s              40.8%   69.6%   0.514   2.21
# 取前 4 在**每个维度**上都优于旧实现：精确率 +5.5 点、召回率持平、
# 候选数少 35%、且不再出现「同一次高潮两个条目」。
# ======================================================================================
MERGE_GAP = 30.0        # 相距不超过此值的候选视为同一次高潮
POOL_EXTRA = 1          # 只考察分数最高的 top + POOL_EXTRA 个

# ======================================================================================
# 时间吸附：把选中的候选挪到附近能量最大的峰
#
# 问题（用户听着发现的）：候选池里明明有很准的峰，模型却挑了个偏的。
# 实测（10 折留一作品，分作品看「离真值最近的峰」的中位误差）：
#     RJ01583845 1.3 秒   RJ362169 1.6 秒   RJ324692 2.0 秒
#     RJ01610459 2.2 秒   RJ01586001 2.4 秒
# 而模型**实际选中**的候选，中位误差是 4~11 秒。差距全在「挑哪个峰」上。
#
# 近邻峰之间有可分的特征（真值 ±15 秒内，「最近峰」对「其它峰」的标准化均值差）：
#     prominence  18.96 vs 15.59   （+0.65）
#     peak        24.20 vs 21.27   （+0.56）
#     rank_energy 15.5  vs 26.5    （−0.47，越小越靠前）
# 也就是**最近的那个峰更突出、更响、轨内能量排名更高**。
#
# 所以选完之后再吸附一次。窗口扫描（精确率 / 召回率 / ≤5 秒比例 / 平均误差）：
#     不吸附         57.8% / 68.5% / 35% / 7.9 秒   ← 现状
#     ±6 秒         57.8% / 68.5% / 43% / 7.6 秒
#     ±12 秒 ←采用   57.8% / 68.5% / 52% / 6.8 秒
#     ±20 秒         56.9% / 65.2% / 58% / 6.4 秒
#     ±25 秒         59.6% / 66.3% / 60% / 6.0 秒   ← 均值更好，但 10 部里 3 部变差
# ±12 是「免费」的：精确率与召回率**都不变**，≤5 秒的比例从 35% 提到 52%。
# 而且 12 秒 < 合并窗口 30 秒 —— 吸附不会跳到相邻的另一次高潮上去。
# ======================================================================================
SNAP_WINDOW = 12.0


def snap_to_peak(events: List[dict], pool: List[dict],
                 window: float = SNAP_WINDOW) -> List[dict]:
    """把每个事件的时间挪到 ±window 秒内**能量最大**的峰。

    pool 里的候选必须带 `peak`（相对全轨中位数的 dB）。
    """
    if window <= 0 or not events:
        return events
    for e in events:
        near = [c for c in pool if abs(c["time"] - e["time"]) <= window]
        if not near:
            continue
        best = max(near, key=lambda c: c.get("peak", -1e9))
        if abs(best["time"] - e["time"]) > 1e-6:
            e["snapped_from"] = e["time"]
            e["time"] = best["time"]
            e["mmss"] = f"{int(best['time'])//60:02d}:{int(best['time'])%60:02d}"
    return events


def merge_candidates(cands: List[dict], *, top: int = 3,
                     merge_gap: float = MERGE_GAP, pool: Optional[int] = None
                     ) -> List[dict]:
    """把相近的候选合并成「事件」，再按分数取前 top 个。

    合并时代表点取**成员时间的中点**（用户实测：真值在两个候选中间），
    其余字段沿用分数最高的那个成员，并记录合并了几个。
    """
    if not cands:
        return []
    if pool is None:
        pool = top + POOL_EXTRA
    cands = sorted(cands, key=lambda c: -c["score"])[:max(1, pool)]

    events: List[dict] = []
    for c in cands:
        hit = None
        for e in events:
            if abs(c["time"] - e["time"]) <= merge_gap:
                hit = e
                break
        if hit is None:
            e = dict(c)
            e["_times"] = [c["time"]]
            e["merged"] = 1
            events.append(e)
        else:
            hit["_times"].append(c["time"])
            hit["time"] = float(np.mean(hit["_times"]))     # 取中点
            hit["merged"] += 1
            if c["score"] > hit["score"]:                    # 沿用分最高的成员
                keep = dict(c)
                keep["_times"] = hit["_times"]
                keep["merged"] = hit["merged"]
                events[events.index(hit)] = keep
                hit = keep
            hit["score"] = max(hit["score"], c["score"])
            if c.get("probability") is not None:
                hit["probability"] = max(hit.get("probability", 0.0), c["probability"])

    for e in events:
        t = e["time"]
        e["mmss"] = f"{int(t)//60:02d}:{int(t)%60:02d}"
        e.pop("_times", None)
    events.sort(key=lambda e: -e["score"])
    return events[:top]
def find_climaxes(audio: Path, *, top: int = 3, cues: Optional[CueRules] = None,
                  acoustic_only: bool = False, verbose: bool = True,
                  transcript_dirs: Optional[List[Path]] = None,
                  min_score: Optional[float] = None,
                  model: str = "ml",
                  model_obj: Optional["ClimaxModel"] = None,
                  merge_gap: float = MERGE_GAP,
                  pool: Optional[int] = None,
                  snap: float = SNAP_WINDOW) -> dict:
    """分析单个音频，返回候选点。不抛异常——失败信息放在结果的 error 字段。

    min_score: 只保留分数不低于此值的候选。None = 不过滤。
    model:     "ml"     用训练好的集成模型（默认）
               "formula" 用旧的手工权重公式（可解释，但实测差距很大）
    merge_gap: 相距不超过此值的候选视为同一次高潮，合并取中点（默认 30 秒）
    pool:      只考察分数最高的几个候选（默认 top+1）
    snap:      选完后把时间吸附到 ±snap 秒内能量最大的峰（默认 12；0 = 关闭）
    """
    cues = cues or DEFAULT_CUES
    t0 = time.time()
    res: Dict = {"audio": str(audio), "name": audio.name, "candidates": [],
                 "transcript": False, "error": "", "has_climax": None,
                 "model": model}
    try:
        x = decode_audio_mono(audio)
    except Exception as e:
        res["error"] = f"解码失败：{type(e).__name__}: {e}"
        return res
    if len(x) < SR * 3:
        res["error"] = f"音频过短（{len(x)/SR:.1f}s），无法分析"
        return res

    segs = None if acoustic_only else load_transcript(audio, transcript_dirs)
    res["transcript"] = segs is not None
    res["duration"] = len(x) / SR

    # 极短轨的特征不稳定（实测 5 秒的标题轨能拿到 +3.90 分，比多数真高潮还高），直接跳过
    if res["duration"] < MIN_DURATION:
        res["error"] = (f"时长 {res['duration']:.0f} 秒，短于 {MIN_DURATION:.0f} 秒阈值——"
                        f"短轨的能量特征不稳定，跳过（这类轨通常也没有高潮段）")
        if verbose:
            _print_one(res)
        return res

    db, hop = energy_envelope(x)
    s = _smooth(db, int(0.3 / hop))
    med = float(np.median(s))
    peaks = find_peaks(s, hop)

    # 语义线索（两种模型都用得上）
    texts: Dict[int, tuple] = {}
    for i in peaks:
        texts[i] = (text_score_at(segs, i * hop, cues) if segs else (0, [], ""))

    if model == "ml":
        if model_obj is None:
            model_obj = ClimaxModel()
        res["model_meta"] = model_obj.meta
        # 21 个模型特征 + 轨内上下文
        feats: List[Dict[str, float]] = []
        for i in peaks:
            f = model_features(x, s, hop, i, med)
            f["txt"] = texts[i][0]
            feats.append(f)
        # 轨内上下文：能量排名与到上一个候选的间距
        order = sorted(range(len(peaks)), key=lambda n: -s[peaks[n]])
        rank_of = {n: r for r, n in enumerate(order)}
        for n, f in enumerate(feats):
            f["rank_energy"] = rank_of[n]
            f["gap_prev"] = 0.0 if n == 0 else float((peaks[n] - peaks[n - 1]) * hop)
        raw = model_obj.predict_raw(feats)          # 排序用原始分
        prob = model_obj.predict_prob(raw)          # 显示用校准概率
        scored = [(peaks[n] * hop, float(raw[n]), float(prob[n]), feats[n])
                  for n in range(len(peaks))]
    else:
        scored = []
        for i in peaks:
            f = acoustic_features(x, med, i, s, hop)
            za = ((f["signature"] - NORM["signature"]["mu"]) / NORM["signature"]["sd"]
                  - (f["recover"] - NORM["recover"]["mu"]) / NORM["recover"]["sd"])
            zt = (texts[i][0] - NORM["txt"]["mu"]) / NORM["txt"]["sd"]
            sc = W_ACOUSTIC * za + W_TXT * zt
            scored.append((i * hop, sc, None, f))

    cands: List[dict] = []
    for t, rawsc, prob, f in scored:
        i = int(round(t / hop))
        tx, tags, txt = texts.get(i, (0, [], ""))
        c = {
            "time": round(t, 2),
            "mmss": f"{int(t)//60:02d}:{int(t)%60:02d}",
            "score": round(rawsc, 4),               # 原始分 → 排序
            "text": txt, "semantic": tx, "cue_tags": tags,
        }
        if prob is not None:
            c["probability"] = round(float(prob), 4)   # 校准概率 → 显示/阈值
            conf, why = confidence_of_prob(float(prob))
            c["confidence"] = conf
            c["confidence_note"] = why
        else:
            conf, why = confidence_of(rawsc)
            c["confidence"] = conf
            c["confidence_note"] = why
            c["acoustic"] = round(
                ((f["signature"] - NORM["signature"]["mu"]) / NORM["signature"]["sd"]
                 - (f["recover"] - NORM["recover"]["mu"]) / NORM["recover"]["sd"]), 3)
        c.update({k: round(float(v), 3) for k, v in f.items()})
        cands.append(c)

    # 过滤：ml 模式按校准概率，formula 模式按原始分
    if min_score is not None:
        key = "probability" if model == "ml" else "score"
        cands = [c for c in cands if c.get(key) is not None and c[key] >= min_score]

    picked = merge_candidates(cands, top=top, merge_gap=merge_gap, pool=pool)
    picked = snap_to_peak(picked, cands, window=snap)
    picked.sort(key=lambda c: -c["score"])
    res["candidates"] = picked
    res["merge_gap"] = merge_gap
    res["snap"] = snap
    res["merged_total"] = sum(c.get("merged", 1) - 1 for c in picked)
    # 轨级判断：ml 模式看最高校准概率，formula 模式看最高分
    res["max_score"] = round(picked[0]["score"], 4) if picked else None
    if model == "ml":
        mp = max((c.get("probability", 0.0) for c in picked), default=0.0)
        res["max_prob"] = round(mp, 4)
        res["has_climax"] = bool(picked and mp >= TRACK_MIN_PROB)
    else:
        res["has_climax"] = bool(picked and picked[0]["score"] >= TRACK_MIN_SCORE)
    res["elapsed"] = round(time.time() - t0, 1)
    if verbose:
        _print_one(res)
    return res


def _print_one(res: dict) -> None:
    name = res["name"]
    if res.get("error"):
        print(f"\n  【{name}】跳过：{res['error']}")
        return
    dur = res.get("duration", 0)
    is_ml = res.get("model", "ml") == "ml"
    src = "机器学习模型" if is_ml else "手工权重公式"
    if not res["transcript"]:
        src += "（无转写，语义特征置零）"
    print(f"\n{'=' * 78}")
    print(f"  {name}   {dur/60:.1f} 分钟   [{src}]   {res.get('elapsed', 0)}s")
    print(f"{'=' * 78}")
    if not res["candidates"]:
        print("    （没有候选点）")
        return

    if res.get("has_climax") is False:
        if is_ml:
            print(f"  ⚠ 这一轨可能**没有高潮**：最高校准概率 "
                  f"{res.get('max_prob', 0)*100:.0f}% 低于阈值 {TRACK_MIN_PROB*100:.0f}%")
        else:
            print(f"  ⚠ 这一轨可能**没有高潮**：最高分 {res.get('max_score'):+.2f} "
                  f"低于阈值 {TRACK_MIN_SCORE:+.2f}")
        print()
    elif is_ml:
        print(f"  最高校准概率 {res.get('max_prob', 0)*100:.0f}%\n")

    for i, c in enumerate(res["candidates"], 1):
        tags = "，".join(c["cue_tags"]) if c["cue_tags"] else "无文本线索"
        if is_ml:
            print(f"  #{i}  [{c['mmss']}]  校准概率 {c.get('probability', 0)*100:5.1f}%   "
                  f"置信度 {c['confidence']}")
            print(f"        {c['confidence_note']}")
        else:
            print(f"  #{i}  [{c['mmss']}]  分数 {c['score']:+6.2f}   "
                  f"置信度 {c['confidence']}（{c['confidence_note']}）")
            print(f"        声学 {c.get('acoustic', 0):+5.2f} · "
                  f"文本 {c['semantic']:+d}（{tags}）")
        print(f"        形状: 前平台 {c['pre30']:+6.1f} dB · 峰值 {c['peak']:+6.1f} dB · "
              f"释放后 {c.get('post2', c.get('post4', 0.0)):+6.1f} dB · "
              f"脱力 {c['recover']:.1f}s")
        if not is_ml:
            print(f"        文本线索 {c['semantic']:+d}（{tags}）")
        if c["text"]:
            print(f"        转写: {c['text'][:62]}")
        print()


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


def find_in_path(target: Path, **kw) -> List[dict]:
    """target 可以是音频文件，也可以是目录（递归找音频）。"""
    if target.is_file():
        return [find_climaxes(target, **kw)]
    files = sorted(p for p in target.rglob("*")
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"  ✗ {target} 里没有找到音频文件")
        return []
    print(f"\n  共 {len(files)} 个音频文件")
    return [find_climaxes(p, **kw) for p in files]


# ======================================================================================
# 命令行
# ======================================================================================
def _cli() -> int:
    ap = argparse.ArgumentParser(
        description="从日语音频里找高潮（射精）候选点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python climax_finder.py \"C:\\音声\\RJ324692\\01_mp3\"\n"
               "  python climax_finder.py track01.mp3 --top 5\n"
               "  python climax_finder.py \"目录\" --json out.json\n"
               "  python climax_finder.py \"目录\" --acoustic-only\n")
    ap.add_argument("target", help="音频文件或目录（目录会递归）")
    ap.add_argument("-n", "--top", type=int, default=3, help="每轨输出几个候选（默认 3）")
    ap.add_argument("--model", choices=("ml", "formula"), default="ml",
                    help="ml = 训练好的集成模型（默认，10 折实测精确率 60.2%%/召回 69.8%%）；"
                         "formula = 旧的手工权重公式（可解释，但实测精确率仅 18.5%%）")
    ap.add_argument("--merge-gap", type=float, default=MERGE_GAP, metavar="秒",
                    help=f"相距不超过此值的候选视为同一次高潮，合并取中点（默认 {MERGE_GAP:.0f}）")
    ap.add_argument("--pool", type=int, default=None, metavar="N",
                    help="只考察分数最高的 N 个候选（默认 top+1）")
    ap.add_argument("--snap", type=float, default=SNAP_WINDOW, metavar="秒",
                    help=f"选完后把时间吸附到 ±此值内能量最大的峰（默认 {SNAP_WINDOW:.0f}；0 = 关闭）")
    ap.add_argument("--json", help="把结果写到 JSON")
    ap.add_argument("--acoustic-only", action="store_true", help="只用声学，不读转写")
    ap.add_argument("--cues", help="自定义线索规则文件（每行一个正则，# 开头为注释）")
    ap.add_argument("-m", "--min-score", type=float, default=None,
                    help="ml 模式：只输出校准概率不低于此值的候选（如 0.5）。"
                         "formula 模式：只输出分数不低于此值的候选（如 +2.5）。"
                         "不加则全输出并标置信度")
    ap.add_argument("--transcript-dir", "--asr-dir", action="append", default=[],
                    dest="transcript_dir", metavar="DIR",
                    help="转写文件所在目录（可重复；默认只在音频旁边找）")
    ap.add_argument("-q", "--quiet", action="store_true", help="只输出 JSON，不打屏")
    a = ap.parse_args()

    cues = DEFAULT_CUES
    if a.cues:
        try:
            lines = [l.strip() for l in Path(a.cues).read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.strip().startswith("#")]
            if len(lines) >= 4:
                cues = CueRules(*lines[:4])
                print(f"  已载入自定义线索规则：{a.cues}")
        except OSError as e:
            print(f"  ✗ 线索规则读取失败：{e}")

    target = Path(a.target).expanduser()
    if not target.exists():
        print(f"  ✗ 路径不存在：{target}")
        return 1

    mobj = None
    print("=" * 78)
    print("  climax_finder —— 高潮候选点检测")
    if a.model == "ml":
        try:
            mobj = ClimaxModel()
        except (FileNotFoundError, ValueError) as e:
            print(f"  ✗ {e}")
            return 2
        print("  模型 = 三等权集成（随机森林 + 梯度提升 + 逻辑回归）")
        print(mobj.describe())
        print(f"  排序用集成原始分；显示的「校准概率」经 isotonic 校准，可直接当命中率读。")
        print(f"  轨级门槛 校准概率 {TRACK_MIN_PROB*100:.0f}%　最短时长 {MIN_DURATION:.0f}s")
    else:
        print(f"  评分 = {W_ACOUSTIC} × 声学(signature, recover) + {W_TXT} × 语义")
        print(f"  置信度门槛 {CONF_HIGH:+.1f}（以上实测精确率 54~64%）　"
              f"轨级门槛 {TRACK_MIN_SCORE:+.1f}　最短时长 {MIN_DURATION:.0f}s")
        print("  ⚠ formula 是早期版本，10 折实测精确率仅 18.5% —— 仅供对照")
    print("=" * 78)

    tdirs = [Path(d).expanduser() for d in a.transcript_dir]
    kw = dict(top=a.top, cues=cues, acoustic_only=a.acoustic_only,
              verbose=not a.quiet, transcript_dirs=tdirs or None,
              min_score=a.min_score, model=a.model, model_obj=mobj,
              merge_gap=a.merge_gap, pool=a.pool, snap=a.snap)
    results = find_in_path(target, **kw)

    if not a.quiet:
        print("=" * 78)
        print("  提示：这是候选生成器，请按「置信度」人工确认后再使用")
        print("=" * 78)
    if a.json:
        Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print(f"  已写入 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
