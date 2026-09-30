#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onsei2lrc.py —— 日语音频 → 中日双语 LRC

流水线：
    wav/mp3 ──ffmpeg──> anime-whisper(CT2) 日语转写 ──> 断句清洗 ──> LLM 翻译 ──> .lrc

针对「语音作品 / ASMR / galgame 语音」做了这些专门处理：
  * 默认 ASR 模型 litagin/anime-whisper 的 CTranslate2 版（专为动漫调演技语音微调，
    对轻声、气声等低能量台词也能照实转写，且吐息/笑声/喘ぎ等非语言发声不会被丢掉）
  * 绝不传 initial_prompt（anime-whisper 官方警告 initial prompt 会导致幻觉）
  * no_repeat_ngram_size=5 + condition_on_previous_text=False，抑制长静音段的复读幻觉
  * 幻觉黑名单过滤（「ご視聴ありがとうございました」等）+ 连续重复段去重
  * 默认关闭 VAD（VAD 会把吐息/喘ぎ当静音切掉），长音效段可加 --vad 打开
  * 翻译走 OpenAI 兼容接口，内置 Sakura 官方 prompt 格式（多行对齐批量翻译，最省显存最快）
  * 断句按 LRC 阅读习惯切分（LRC 只有起始时间，无需结束时间）

示例：
    # 只转写（不翻译），输出日语 LRC
    python onsei2lrc.py "RJ123456/track01.mp3" --translator none --lrc-mode ja

    # 本地 Sakura（llama.cpp server 或 Ollama）翻译成中日双语 LRC
    python onsei2lrc.py track01.mp3 --preset sakura --lrc-mode both

    # 用 DeepSeek API 翻译
    set DEEPSEEK_API_KEY=sk-xxx
    python onsei2lrc.py track01.mp3 --preset deepseek --lrc-mode zh

作者：为自用场景生成，仅供个人学习/翻译使用。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------------------

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}

_MODEL_CACHE: dict = {}          # (模型, device, compute_type) → WhisperModel，跨文件复用

DEFAULT_ASR_MODEL = "quantumcookie/anime-whisper-ct2-int8"

# Whisper 在日语长静音 / 非语音段最经典的复读幻觉，命中且「可疑」时才丢
HALLUCINATION_PATTERNS = [
    r"ご視聴ありがとうございました",
    r"ご視聴ありがとうございます",
    r"最後までご視聴",
    r"ご覧いただきありがとうございます",
    r"チャンネル登録",
    r"高評価",
    r"おやすみなさい",
    r"おやすみなさいませ",
    r"お疲れ様でした",
    r"ありがとうございました$",
    r"^字幕",
    r"^日本語字幕",
    r"^翻訳",
    r"^ん+$",
    r"^[、。．，！？!?…‥・ー\s]+$",          # 纯标点
    r"^[A-Za-z0-9\s\.\,\!\?\'\"-]+$",         # 纯英文/数字（日语音声里几乎必为幻觉）
]
HALLUCINATION_RE = [re.compile(p) for p in HALLUCINATION_PATTERNS]

# Sakura v1.0 官方 prompt（轻小说风格），不要改动
SAKURA_SYSTEM = (
    "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
)
SAKURA_USER_PREFIX = "将下面的日文文本翻译成中文："
SAKURA_USER_PREFIX_GLOSSARY = "将下面的日文文本根据对应关系和备注翻译成中文："

# Sakura GalTransl v3/v4 官方 prompt（视觉小说风格），与 v1.0 完全不同，不可混用
GALTRANSL_SYSTEM = (
    "你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，"
    "不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。"
)
GALTRANSL_USER_PREFIX = "根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文翻译成简体中文："

# 提示词模板 → (system, user 前缀, 官方推荐解码参数)
# 两代模型的 prompt 与参数**不可混用**，混用会明显掉质量
PROMPT_STYLES = {
    "v1": dict(
        label="v1.0 轻小说风格（配 Sakura-7B/14B-Qwen2.5-v1.0）",
        system=SAKURA_SYSTEM, prefix=SAKURA_USER_PREFIX,
        glossary_header="根据以下术语表（可以为空）：",
        prefix_glossary=SAKURA_USER_PREFIX_GLOSSARY,
        always_glossary_header=False,
        temperature=0.1, top_p=0.3, frequency_penalty=0.15,
    ),
    "v3": dict(
        label="v3/v4 视觉小说风格（配 GalTransl-7B-v3.7 / v4-4B）",
        system=GALTRANSL_SYSTEM, prefix=GALTRANSL_USER_PREFIX,
        glossary_header="参考以下术语表（可为空，格式为src->dst #备注）：",
        prefix_glossary=GALTRANSL_USER_PREFIX,
        always_glossary_header=True,
        temperature=0.3, top_p=0.8, frequency_penalty=0.0,
    ),
}

# 通用 LLM（DeepSeek / Qwen / GPT …）用的 JSON 协议
JSON_SYSTEM = (
    "你是专业的日语到简体中文字幕翻译。要求：1) 逐条对应翻译，条数必须与输入完全一致；"
    "2) 保留语气词、拟声词、喘ぎ等非语言发声的语感（如「んっ」→「嗯…」），不要省略、不要合并；"
    "3) 口语化、自然，符合音声作品的对白风格；4) 只输出 JSON，不要任何解释。"
)

PRESETS = {
    # 本地 Sakura：llama.cpp server（默认 8080）或 Ollama（默认 11434）
    # 注意：这里只放「连接方式」相关字段；采样参数由 --prompt-style 决定（v1/v3 不同）
    "sakura-llamacpp": dict(
        base_url="http://127.0.0.1:8080/v1", model_name="sakura", protocol="sakura", max_tokens=2048,
    ),
    "sakura-ollama": dict(
        base_url="http://127.0.0.1:11434/v1", model_name="sakura", protocol="sakura", max_tokens=2048,
    ),
    # LM Studio 本地服务（默认端口 1234）
    "lmstudio": dict(
        base_url="http://127.0.0.1:1234/v1", model_name="sakura", protocol="sakura", max_tokens=2048,
    ),
    # 云端：注意内容审核风险
    "deepseek": dict(
        base_url="https://api.deepseek.com", model_name="deepseek-chat", protocol="json",
        max_tokens=4096,
        sampling=dict(temperature=1.3, top_p=0.95, frequency_penalty=0.0),
    ),
}


# --------------------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------------------

@dataclass
class Seg:
    start: float
    end: float
    ja: str
    zh: str = ""
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    ja_raw: str = ""          # 被 ASR 修正表改过之前的原始转写（留档）

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


# --------------------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def fmt_ts(t: float) -> str:
    """LRC 时间戳 [mm:ss.xx]"""
    t = max(0.0, round(t, 2))
    m = int(t // 60)
    s = t - m * 60
    if s >= 59.995:            # 避免出现 [00:60.00]
        m += 1
        s = 0.0
    return f"[{m:02d}:{s:05.2f}]"


def ensure_cuda_dlls(verbose: bool = True) -> None:
    """Windows 上 ctranslate2 4.x 需要 CUDA 12 的 cublas/cudnn DLL。

    很多机器只装了 torch 的 cu118 版（只有 cublas64_11.dll），于是报
    "Library cublas64_12.dll is not found"。这里自动在常见位置找 CUDA 12 运行时
    （conda 环境里的 torch/lib、Ollama、LM Studio、pip 的 nvidia-* 包）并注册到
    进程的 DLL 搜索路径，避免必须重装 CUDA。
    """
    if os.name != "nt":
        return
    import site

    cand: List[Path] = []
    env = os.environ.get("ONSEI2LRC_CUDA_DLL_DIRS")
    if env:
        cand += [Path(x) for x in env.split(os.pathsep) if x]
    for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
        cand += sorted(Path(sp).glob("nvidia/*/bin"))
    for root in (Path.home() / ".conda" / "envs", Path("C:/ProgramData/miniconda3/envs"),
                 Path("C:/ProgramData/anaconda3/envs")):
        if root.is_dir():
            cand += sorted(root.glob("*/Lib/site-packages/torch/lib"))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cand.append(Path(local) / "Programs" / "Ollama" / "lib" / "ollama" / "cuda_v12")
    lm = Path.home() / ".lmstudio" / "extensions" / "backends"
    if lm.is_dir():
        cand += [f.parent for f in lm.rglob("cublas64_12.dll")]

    cublas_dirs = [d for d in cand if (d / "cublas64_12.dll").exists()]
    if not cublas_dirs:
        return
    # 优先选同时带完整 cuDNN 9 子库的目录（cublas 与 cudnn 版本配套最省事）
    chosen: List[Path] = []
    full = next((d for d in cublas_dirs if (d / "cudnn_ops64_9.dll").exists()), None)
    chosen.append(full or cublas_dirs[0])
    if not full:
        extra = next((d for d in cand if (d / "cudnn64_9.dll").exists() and d not in chosen), None)
        if extra:
            chosen.append(extra)
    for d in chosen:
        try:
            os.add_dll_directory(str(d))
        except Exception:
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    if verbose:
        log("[CUDA] 已注册 CUDA 12 运行时目录：" + " | ".join(str(d) for d in chosen))


def load_asr_fixes(path) -> List[tuple]:
    """读取 ASR 修正表：每行 `错误=>正确`（也支持 -> 和 →），`#` 开头为注释。

    用途：ASR 听错的专有名词/术语，翻译环节救不了（错字已经在日文原文里了），
    必须在转写之后、翻译之前替换掉。
    """
    fixes: List[tuple] = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():   # utf-8-sig 吃掉 BOM
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        s = re.split(r"\s+#", s, maxsplit=1)[0].strip()      # 去掉行内 " # 注释"
        if not s:
            continue
        for sep in ("=>", "->", "→"):
            if sep in s:
                a, b = (x.strip() for x in s.split(sep, 1))
                if a:
                    fixes.append((a, b))
                break
    return fixes


def apply_asr_fixes(segs: List[Seg], fixes: List[tuple]) -> int:
    """对日文原文做替换，返回被改动的段数（原始文本存进 ja_raw）。

    被改动的段会清空已有译文——原文变了，旧译文就失效了，需要重译。
    配合 --reuse-translation 时只会重译这些段，不用整轨重跑。
    """
    if not fixes:
        return 0
    n = 0
    for s in segs:
        orig = s.ja
        for a, b in fixes:
            if a in s.ja:
                s.ja = s.ja.replace(a, b)
        if s.ja != orig:
            n += 1
            if not s.ja_raw:
                s.ja_raw = orig
            s.zh = ""                     # 原文变了，译文作废
    return n


def looks_like_hallucination(seg: Seg) -> bool:
    text = seg.ja.strip()
    if not text:
        return True
    for rx in HALLUCINATION_RE:
        if rx.search(text):
            # 只有在「短 + 模型自己都觉得不像语音」时才丢，避免误杀正常短台词
            if seg.dur < 2.5 or seg.no_speech_prob > 0.35:
                return True
    return False


def dedupe_runs(segs: List[Seg], max_run: int = 3) -> List[Seg]:
    """连续完全相同的行超过 max_run 次时只保留 max_run 次（典型复读幻觉）。"""
    out: List[Seg] = []
    run_text, run_n = None, 0
    for s in segs:
        t = s.ja.strip()
        if t and t == run_text:
            run_n += 1
            if run_n > max_run:
                continue
        else:
            run_text, run_n = t, 1
        out.append(s)
    return out


def split_text(text: str, max_chars: int) -> List[str]:
    """把长句切成 <= max_chars 的片段。

    分隔符按优先级依次尝试：句末标点 → 顿号/逗号 → 空白。
    某个片段切完仍超长时，**换更低优先级的分隔符再切一层**，而不是按字数硬切
    —— 硬切会把词劈开（实测把「満足」切成「満」+「足しない」，翻译模型看到残句
    会把两行并成一行，导致行数不匹配、触发拆半重试）。
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    for sep in (r"(?<=[。！？!?…‥])", r"(?<=[、，,])", r"(?<=[\s])"):
        parts = [p for p in re.split(sep, text) if p.strip()]
        if len(parts) <= 1:
            continue
        chunks: List[str] = []
        cur = ""
        for p in parts:
            if len(cur) + len(p) <= max_chars:
                cur += p
                continue
            if cur:
                chunks.append(cur)
                cur = ""
            if len(p) > max_chars:
                chunks.extend(split_text(p, max_chars))     # 换分隔符再切一层
            else:
                cur = p
        if cur:
            chunks.append(cur)
        if chunks and all(len(c) <= max_chars for c in chunks):
            return [c for c in chunks if c.strip()]

    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def resplit_segments(segs: List[Seg], max_chars: int, min_gap: float = 0.05) -> List[Seg]:
    """按字符比例把过长段切成多段（LRC 只需要起始时间，按长度比例分配即可）。"""
    out: List[Seg] = []
    for s in segs:
        chunks = split_text(s.ja, max_chars)
        if len(chunks) <= 1:
            out.append(s)
            continue
        total = sum(len(c) for c in chunks) or 1
        cursor = s.start
        for i, c in enumerate(chunks):
            share = (s.end - s.start) * len(c) / total
            end = s.end if i == len(chunks) - 1 else min(s.end, cursor + share)
            if end - cursor < min_gap:
                end = min(s.end, cursor + min_gap)
            out.append(Seg(start=cursor, end=end, ja=c.strip(),
                           no_speech_prob=s.no_speech_prob, avg_logprob=s.avg_logprob))
            cursor = end
    return out


def merge_short_segments(segs: List[Seg], gap: float, max_chars: int) -> List[Seg]:
    """把间隔很小的相邻段并起来，读起来更像一整句。

    合并只依据**间隔**（语义判断），不依据长度：
    间隔 0 秒的两段本来就是一句话被切开的；若因为「合并后超过 max_chars」而拒绝合并，
    断口就会留在句子中间，翻译模型看到残句会把两行并成一行 → 行数不匹配 → 触发拆半重试。
    长度问题交给后面的 resplit_segments 按标点重新分配即可。

    仍保留一个宽松上限（max_chars×2）防止无限累积：合并段的时间戳是按字符比例
    分摊的，跨度过大时时间会不准。
    """
    if gap <= 0:
        return segs
    ceiling = max_chars * 2
    out: List[Seg] = []
    for s in segs:
        if out:
            prev = out[-1]
            if (s.start - prev.end) <= gap and len(prev.ja) + len(s.ja) <= ceiling:
                prev.ja = (prev.ja + s.ja).strip()
                prev.end = s.end
                continue
        out.append(Seg(**asdict(s)))
    return out


# --------------------------------------------------------------------------------------
# 1) 转写
# --------------------------------------------------------------------------------------

def _asr_kwargs(args) -> dict:
    """faster-whisper 的转写参数（两种分块模式共用）。"""
    hallu = args.hallucination_silence if args.hallucination_silence > 0 else None
    # hallucination_silence_threshold 需要 word_timestamps=True 才生效；
    # 但部分 CT2 转换模型开 word_timestamps 会崩进程（0xC0000005），所以默认关闭
    word_ts = bool(hallu) or args.word_timestamps
    if word_ts:
        log("[警告] 已启用 word_timestamps：部分 CT2 转换模型会崩溃，如遇 0xC0000005 请去掉 --word-timestamps")
    return dict(
        language="ja",
        task="transcribe",              # Whisper 的 translate 只能出英文，这里必须转写
        beam_size=args.beam_size,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        condition_on_previous_text=False,   # 防止幻觉跨段传播
        no_speech_threshold=args.no_speech_threshold,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        hallucination_silence_threshold=hallu,
        word_timestamps=word_ts,
        vad_filter=args.vad,                 # 仅整文件模式生效；VAD 分块模式会强制关闭
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200) if args.vad else None,
        initial_prompt=None,                # anime-whisper：绝对不要设置 initial prompt！
    )


def _transcribe_whole(args, model, audio: Path) -> List[Seg]:
    """整文件交给 faster-whisper（标准做法）。"""
    segments, info = model.transcribe(str(audio), **_asr_kwargs(args))
    raw: List[Seg] = []
    t0 = time.time()
    for i, s in enumerate(segments, 1):
        seg = Seg(start=s.start, end=s.end, ja=(s.text or "").strip(),
                  no_speech_prob=getattr(s, "no_speech_prob", 0.0) or 0.0,
                  avg_logprob=getattr(s, "avg_logprob", 0.0) or 0.0)
        raw.append(seg)
        if i % 20 == 0 or i == 1:
            log(f"  …{i} 段 / {seg.end/60:.1f} 分钟  {fmt_ts(seg.start)} {seg.ja[:28]}")
    dur = info.duration or (raw[-1].end if raw else 0.0)
    speed = dur / max(1e-6, time.time() - t0)
    log(f"[ASR] 完成：{len(raw)} 段，音频 {dur/60:.1f} 分钟，速度约 {speed:.1f}x 实时")
    # 时间戳体检：某些 CT2 转换模型的时间戳是坏的（会把 13 秒语音标成 0.4 秒）
    if raw and dur > 5 and raw[-1].end < dur * 0.5:
        log(f"[严重警告] 时间戳疑似损坏：音频 {dur:.1f}s，最后一段只到 {raw[-1].end:.2f}s。"
            f"\n            建议改用 --chunk-mode vad（按静音切块后逐块转写，时间轴由 VAD 提供）。")
    return raw


def _pick_vad_chunks(data, args, dur: float, sr: int = 16000):
    """挑一个合适的 VAD 阈值。

    耳语/舔耳类音声能量很低，固定 0.4 会漏掉大量轻声台词（实测某轨覆盖率仅 17%、
    时间轴命中率 40%）。默认按 0.4 → 0.25 → 0.15 逐级下探，直到语音覆盖率达到
    --vad-min-coverage，兼顾「不漏台词」和「不把音效当人声」。
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    thresholds = [args.vad_threshold] if args.vad_threshold else [0.4, 0.25, 0.15]
    best = None
    for th in thresholds:
        opts = VadOptions(threshold=th, min_speech_duration_ms=args.vad_min_speech,
                          max_speech_duration_s=args.max_chunk,
                          min_silence_duration_ms=args.vad_min_silence,
                          speech_pad_ms=args.vad_pad)
        chunks = get_speech_timestamps(data, opts, sampling_rate=sr)
        cov = sum(c["end"] - c["start"] for c in chunks) / sr if chunks else 0.0
        log(f"[VAD] threshold={th}: {len(chunks)} 块，覆盖 {cov/max(dur,1e-6)*100:.0f}%")
        if best is None or cov > best[1]:
            best = (chunks, cov, th)
        if cov >= dur * args.vad_min_coverage:
            break
    return best


def _energy_split(data, sr: int, lo: int, hi: int, target_s: float, floor_db: float):
    """在样本区间 [lo, hi) 内按能量切块：按目标时长切，切点吸附到局部能量最低处。"""
    import numpy as np

    frame = int(sr * 0.02)
    target = max(frame, int(target_s * sr))
    out, i = [], lo
    while i < hi:
        start = i
        end = min(hi, i + target)
        if end < hi:
            a, b = max(start + sr, end - int(2 * sr)), min(hi, end + int(2 * sr))
            f0, f1 = a // frame, max(a // frame + 1, b // frame)
            seg = data[f0 * frame: min(len(data), f1 * frame)]
            if len(seg) >= frame:
                frames = seg[: (len(seg) // frame) * frame].astype(np.float64).reshape(-1, frame)
                db = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-7))
                end = max(start + sr, min(hi, (f0 + int(np.argmin(db))) * frame))
        piece = data[start:end]
        if len(piece):
            peak = 20 * np.log10(max(1e-7, float(np.abs(piece).max())))
            if peak >= floor_db:
                out.append({"start": start, "end": end})
        i = max(end, start + frame)
    return out


def _energy_chunks(data, args, dur: float, sr: int = 16000):
    """按能量切块，覆盖**全部**音频（含非语言段）。

    音声作品里大量内容是耳语、水音、吐息这类非语言音，Silero VAD 会把它们判成非语音
    整段跳过（实测某轨 2 分钟内容被跳过，而那段能量其实比台词段还高）。但 anime-whisper
    本来就擅长转写这类声音（れろれろ / ちゅぱ / んっ、はぁ），所以这类作品要用能量切块。
    """
    chunks = _energy_split(data, sr, 0, len(data), args.energy_chunk, args.energy_floor_db)
    covered = sum(c["end"] - c["start"] for c in chunks) / sr
    log(f"[能量切块] {len(chunks)} 块（目标 {args.energy_chunk}s/块），覆盖 {covered/max(dur,1e-6)*100:.0f}%")
    return chunks, covered, f"energy({args.energy_chunk}s)"


def _hybrid_chunks(data, args, dur: float, sr: int = 16000):
    """混合切块（推荐）：台词段用 VAD 边界（时间准），空隙用能量块补齐（不漏非语言内容）。"""
    vad, _, th = _pick_vad_chunks(data, args, dur, sr)
    vad = sorted((c for c in vad if c["end"] > c["start"]), key=lambda c: c["start"])
    gaps, prev = [], 0
    for c in vad:
        if c["start"] - prev >= int(args.gap_min * sr):
            gaps.append((prev, c["start"]))
        prev = max(prev, c["end"])
    if len(data) - prev >= int(args.gap_min * sr):
        gaps.append((prev, len(data)))
    extra = []
    for lo, hi in gaps:
        extra += _energy_split(data, sr, lo, hi, args.energy_chunk, args.energy_floor_db)
    allc = sorted(vad + extra, key=lambda c: c["start"])
    covered = sum(c["end"] - c["start"] for c in allc) / sr
    log(f"[混合切块] VAD {len(vad)} 块 + 空隙补 {len(extra)} 块 = {len(allc)} 块，"
        f"覆盖 {covered/max(dur,1e-6)*100:.0f}%（VAD 阈值 {th}，空隙阈值 {args.gap_min}s）")
    return allc, covered, f"hybrid(vad={th}+energy)"


def _transcribe_vad(args, model, audio: Path) -> List[Seg]:
    """按静音切块后逐块转写：时间戳来自 VAD 边界，不依赖模型的时间戳预测。

    对音声作品特别有效——通常是一句一停顿，且不少 CT2 转换模型的时间戳
    预测是坏的（只在 30 秒窗口边界给时间），逐块转写可以完全绕开这个问题。
    """
    from faster_whisper import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    sr = 16000
    data = decode_audio(str(audio), sampling_rate=sr)
    dur = len(data) / sr
    # 空文件 / 解码失败 / 极短音频：早点报清楚，别让它走到后面除零崩掉
    if dur < 0.1:
        raise ValueError(
            f"音频时长过短（{dur:.3f} 秒），无法转写。"
            f"文件是否为空、损坏，或截取范围超出了原文件长度？")
    if args.chunk_mode == "energy":
        chunks, covered, desc = _energy_chunks(data, args, dur, sr)
    elif args.chunk_mode == "hybrid":
        chunks, covered, desc = _hybrid_chunks(data, args, dur, sr)
    else:
        chunks, covered, th = _pick_vad_chunks(data, args, dur, sr)
        desc = f"vad(threshold={th})"
        log(f"[VAD] 采用 {desc}：{len(chunks)} 个语音块，覆盖 {covered:.0f}s / {dur:.0f}s"
            f"（{covered/max(dur,1e-6)*100:.0f}%）")
    if not chunks or covered < dur * 0.1:
        log("[切块] 语音覆盖过低（可能是纯耳语/环境音作品），自动回退到整文件模式")
        return _transcribe_whole(args, model, audio)

    kw = _asr_kwargs(args)
    kw["vad_filter"] = False            # 分块已由我们自己做，避免二次切割丢掉吐息/喘ぎ
    raw: List[Seg] = []
    t0 = time.time()
    for ci, c in enumerate(chunks, 1):
        s0, s1 = c["start"], c["end"]
        cstart, cend = s0 / sr, s1 / sr
        subs = list(model.transcribe(data[s0:s1], **kw)[0])
        subs = [x for x in subs if (x.text or "").strip()]
        if not subs:
            continue
        if len(subs) == 1:
            raw.append(Seg(start=cstart, end=cend, ja=subs[0].text.strip(),
                           no_speech_prob=getattr(subs[0], "no_speech_prob", 0.0) or 0.0))
        else:
            rel_last = getattr(subs[-1], "end", 0.0) or 0.0
            if rel_last > 0.6 * (cend - cstart) and all(x.end >= x.start for x in subs):
                for x in subs:                      # 模型块内时间戳可信 → 用偏移后的
                    raw.append(Seg(start=cstart + x.start, end=min(cend, cstart + x.end),
                                   ja=x.text.strip()))
            else:                                   # 否则按字数比例分配块内时间
                total = sum(len(x.text.strip()) for x in subs) or 1
                cur = cstart
                for x in subs:
                    share = (cend - cstart) * len(x.text.strip()) / total
                    raw.append(Seg(start=cur, end=min(cend, cur + share), ja=x.text.strip()))
                    cur += share
        if ci % 20 == 0 or ci == 1:
            log(f"  …{ci}/{len(chunks)} 块  {fmt_ts(cstart)} {raw[-1].ja[:26] if raw else ''}")
    speed = dur / max(1e-6, time.time() - t0)
    log(f"[ASR] 完成：{len(raw)} 段，音频 {dur/60:.1f} 分钟，速度约 {speed:.1f}x 实时")
    return raw


def _get_model(args):
    """加载（并缓存）faster-whisper 模型。

    同一个进程里处理多个音轨时，模型只加载一次——否则每个文件都要重新加载
    1.5~11s，一个 8 轨作品光加载就白花十几秒到一分多钟。
    """
    from faster_whisper import WhisperModel

    key = (str(args.model), args.device, args.compute_type)
    model = _MODEL_CACHE.get(key)
    if model is None:
        if args.device == "cuda":
            ensure_cuda_dlls()
        log(f"[ASR] 加载模型 {args.model}（device={args.device}, compute_type={args.compute_type}）")
        t0 = time.time()
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type,
                             cpu_threads=args.cpu_threads)
        log(f"[ASR] 模型就绪，用时 {time.time() - t0:.1f}s")
        _MODEL_CACHE[key] = model
    else:
        log(f"[ASR] 复用已加载的模型 {args.model}（跳过重复加载）")
    return model


def transcribe(args, audio: Path) -> List[Seg]:
    model = _get_model(args)
    log(f"[ASR] 开始转写 {audio.name}")

    raw = _transcribe_whole(args, model, audio) if args.chunk_mode == "whole" \
        else _transcribe_vad(args, model, audio)

    before = len(raw)
    segs = [s for s in raw if not looks_like_hallucination(s)]
    segs = dedupe_runs(segs, args.max_repeat)
    if before != len(segs):
        log(f"[清洗] 丢弃幻觉/空段 {before - len(segs)} 段，保留 {len(segs)} 段")
    segs = merge_short_segments(segs, args.merge_gap, args.max_chars)
    segs = resplit_segments(segs, args.max_chars)
    log(f"[清洗] 断句后共 {len(segs)} 行")
    if getattr(args, "asr_fixes", None):
        k = apply_asr_fixes(segs, args.asr_fixes)
        log(f"[修正] ASR 修正表改动 {k} 段（原文留档在 ja_raw）")
    return segs


# --------------------------------------------------------------------------------------
# 2) 翻译
# --------------------------------------------------------------------------------------

def _chat(client, args, messages) -> str:
    kw = dict(model=args.model_name, messages=messages)
    kw["max_tokens"] = args.max_tokens or 2048
    if args.temperature is not None:
        kw["temperature"] = args.temperature
    if args.top_p is not None:
        kw["top_p"] = args.top_p
    if args.frequency_penalty:
        kw["frequency_penalty"] = args.frequency_penalty
    try:
        r = client.chat.completions.create(**kw)
    except Exception as e:                     # 某些后端不认 top_p / frequency_penalty
        msg = str(e)
        if any(k in msg for k in ("top_p", "frequency_penalty", "max_tokens", "Unsupported", "400")):
            kw.pop("top_p", None)
            kw.pop("frequency_penalty", None)
            r = client.chat.completions.create(**kw)
        else:
            raise
    return (r.choices[0].message.content or "").strip()


def _clean_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*(译文|翻译|中文)\s*[:：]\s*", "", s)
    s = s.strip().strip('"').strip("“”")
    return s


def _parse_json_array(content: str, n: int) -> Optional[List[str]]:
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(data, list) or len(data) != n:
        return None
    out = []
    for item in data:
        if isinstance(item, dict):
            out.append(_clean_line(str(item.get("zh") or item.get("translation") or item.get("text") or "")))
        else:
            out.append(_clean_line(str(item)))
    return out


def _prompt_style(args) -> dict:
    return PROMPT_STYLES.get(getattr(args, "prompt_style", "v1") or "v1", PROMPT_STYLES["v1"])


def _build_user_prompt(args, body: str) -> str:
    """按当前提示词模板拼 user 内容（术语表可选）。"""
    st = _prompt_style(args)
    if args.glossary_text or st["always_glossary_header"]:
        head = f"{st['glossary_header']}\n{args.glossary_text}\n"
        return f"{head}{st['prefix_glossary']}{body}"
    return st["prefix"] + body


def _translate_once(client, args, lines: List[str],
                    attempts: Optional[int] = None) -> Optional[List[str]]:
    """一次批量请求；成功返回与 lines 等长的译文列表，否则 None。

    attempts 显式指定重发次数；不传则用 args.retries + 1。
    """
    n = len(lines)
    tries = (args.retries + 1) if attempts is None else max(1, attempts)
    if args.protocol == "sakura":
        # 多行一次性输入，输出按行对齐（两代 Sakura 都是按行对齐训练的）
        messages = [{"role": "system", "content": _prompt_style(args)["system"]},
                    {"role": "user", "content": _build_user_prompt(args, "\n".join(lines))}]
        for attempt in range(tries):
            content = _chat(client, args, messages)
            out = [_clean_line(x) for x in content.splitlines() if x.strip()]
            if len(out) == n and all(out):
                return out
            if attempt < tries - 1:     # 最后一次不再等待——马上要拆半了，白等没意义
                time.sleep(0.4 * (attempt + 1))
        return None

    payload = json.dumps([{"i": i + 1, "ja": t} for i, t in enumerate(lines)], ensure_ascii=False)
    user = ("把下面 JSON 里每条 ja 翻译成简体中文，逐条对应，返回 JSON 数组，"
            "元素形如 {\"i\": 序号, \"zh\": \"译文\"}，条数必须为 " + str(n) + "：\n" + payload)
    messages = [{"role": "system", "content": JSON_SYSTEM}, {"role": "user", "content": user}]
    for attempt in range(tries):
        content = _chat(client, args, messages)
        out = _parse_json_array(content, n)
        if out and all(out):
            return out
        if attempt < tries - 1:
            time.sleep(0.4 * (attempt + 1))
    return None


def translate_batch(client, args, lines: List[str], depth: int = 0) -> List[str]:
    """翻译一批行。

    行数不匹配的根因是「按停顿分的行 ≠ 按语义分的句子」，**同一批原样重发基本无效**
    （实测 3 次尝试全失败），所以默认 `--retries 0`：**一失败就拆半**，
    让模型在更小的上下文里重新对齐。

      * **顶层**（完整批次）试 `--retries + 1` 次；默认 0 → 试 1 次；
      * **拆半之后**的子批次固定只试 1 次；
      * 拆到**单行**时模型无处可并，必定 1 进 1 出，对齐由此得到保证。

    把 `--retries` 调大只会让顶层多试几次，深层不受影响。
    """
    out = _translate_once(client, args, lines,
                          attempts=(args.retries + 1) if depth == 0 else 1)
    if out is not None:
        return out
    if len(lines) > 1:
        mid = len(lines) // 2
        log(f"  [重试] {len(lines)} 行结果不匹配，拆成 {mid}+{len(lines) - mid} 行重试")
        return (translate_batch(client, args, lines[:mid], depth + 1)
                + translate_batch(client, args, lines[mid:], depth + 1))

    # 最后兜底：单行重发。这里跟批量的 --retries 解耦——它是最后一道保险，
    # 单行请求又快又便宜，多试几次避免留下空译文。
    log("  [警告] 单行翻译仍失败，改为逐行重发")
    t = lines[0]
    tries = max(3, args.retries + 1)
    for k in range(tries):
        try:
            if args.protocol == "sakura":
                msgs = [{"role": "system", "content": _prompt_style(args)["system"]},
                        {"role": "user", "content": _build_user_prompt(args, t)}]
                first = [x for x in _chat(client, args, msgs).splitlines() if x.strip()]
                if first:
                    return [_clean_line(first[0])]
            else:
                msgs = [{"role": "system", "content": JSON_SYSTEM},
                        {"role": "user", "content": f"把下面这条日文翻译成简体中文，只输出译文：\n{t}"}]
                got = _clean_line(_chat(client, args, msgs))
                if got:
                    return [got]
        except Exception as e:
            log(f"  [错误] 单行翻译失败：{e}")
        if k < tries - 1:
            time.sleep(0.5)
    return [""]


def make_batches(lines: List[str], batch_size: int, max_chars: int) -> List[List[int]]:
    batches, cur, cur_chars = [], [], 0
    for i, t in enumerate(lines):
        if cur and (len(cur) >= batch_size or cur_chars + len(t) > max_chars):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(i)
        cur_chars += len(t)
    if cur:
        batches.append(cur)
    return batches


def _make_client(args):
    """构造 OpenAI 兼容客户端。

    关键坑：httpx 默认 trust_env=True，会读取 Windows 注册表里的系统代理设置，
    于是连 http://127.0.0.1:1234 的本地模型请求也会被丢给代理，表现为 502 Bad Gateway。
    本地地址一律 trust_env=False 直连。
    """
    from urllib.parse import urlparse

    from openai import OpenAI

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or \
        os.environ.get("OPENAI_API_KEY") or "sk-noauth"
    kwargs = dict(base_url=args.base_url, api_key=api_key, timeout=args.timeout)
    host = (urlparse(args.base_url or "").hostname or "").lower()
    if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        try:
            import httpx
            kwargs["http_client"] = httpx.Client(trust_env=False, timeout=args.timeout)
        except Exception:
            pass
    return OpenAI(**kwargs)


def translate_all(args, segs: List[Seg]) -> None:
    # --reuse-translation：已有译文的段直接复用，只翻译缺失的
    # （配合 ASR 修正表时，只有被改动的段会重译）
    todo = [i for i, s in enumerate(segs)
            if s.ja.strip() and not (args.reuse_translation and s.zh.strip())]
    if not todo:
        log("[翻译] 所有段落都已有译文，无需翻译")
        return
    if "deepseek" in (args.base_url or "") and not (
            args.api_key or os.environ.get("DEEPSEEK_API_KEY")):
        log("[警告] 未找到 DEEPSEEK_API_KEY 环境变量，请求可能失败")
    client = _make_client(args)

    batches = make_batches([segs[i].ja for i in todo], args.batch_size, args.batch_chars)
    log(f"[翻译] {len(todo)} 行 → {len(batches)} 批（protocol={args.protocol}, model={args.model_name}）")

    def work(b: List[int]) -> None:
        lines = [segs[i].ja for i in b]
        outs = translate_batch(client, args, lines)
        for i, zh in zip(b, outs):
            segs[i].zh = zh

    t0 = time.time()
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, batches))
    else:
        for k, b in enumerate(batches, 1):
            work(b)
            if k % 5 == 0 or k == len(batches):
                log(f"  …{k}/{len(batches)} 批完成（{time.time() - t0:.0f}s）")
    log(f"[翻译] 完成，用时 {time.time() - t0:.0f}s")

    empty = [i for i in todo if not segs[i].zh]
    if empty:
        log(f"[警告] {len(empty)} 行翻译为空，已回退为原文")


# --------------------------------------------------------------------------------------
# 2.5) 显存编排：ASR 与 Sakura 分时进出显存
#
# 为什么要把流程切成两段：Sakura-7B 常驻要吃 6.25 GB，而卡只有 8 GB。
# 两者同时在场时显存贴顶，WDDM 开始换页 + GPU 发热降频，ASR 直接掉速
# （本机实测同一批音频：显存干净时 20.0x 实时，Sakura 在场时 3.7x —— 慢 5.4 倍）。
# 所以改成：ASR 阶段显存里只有 whisper，跑完就放掉；翻译阶段显存里只有 Sakura，译完也放掉。
# --------------------------------------------------------------------------------------

# 标识符 → LM Studio 模型键。
# `lms load` 收的是**磁盘上的模型键**，而 --identifier 起的名字是 API 里用的别名，
# 两者通常对不上（sakura37 ≠ sakura-galtransl-7b-v3.7）。不映射会直接找不到模型。
MODEL_KEY_HINTS = {
    "sakura37":    "sakura-galtransl-7b-v3.7",
    "sakura":      "sakura-7b-qwen2.5-v1.0",
    "galtransl4b": "galtransl-v4-4b-2601",
}


def find_lms() -> Optional[str]:
    """定位 lms 命令行（PATH → ~/.lmstudio/bin → LM Studio 安装目录）。"""
    exe = "lms.exe" if os.name == "nt" else "lms"
    cands: List[str] = []
    w = shutil.which("lms")
    if w:
        cands.append(w)
    cands.append(str(Path.home() / ".lmstudio" / "bin" / exe))
    if os.name == "nt":
        for root in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles")):
            if root:
                cands.append(str(Path(root) / "LM Studio" / "resources" / "app" / ".webpack" / exe))
    for c in cands:
        try:
            if c and Path(c).is_file():
                return c
        except OSError:
            continue
    return None


def run_lms(argv: List[str], timeout: float = 300.0) -> Tuple[int, str]:
    """跑一条 lms 命令，返回 (返回码, 合并输出)。超时/异常都变成返回码，不抛。"""
    lms = find_lms()
    if not lms:
        return 127, "找不到 lms 命令行（LM Studio 未安装或没生成 lms）"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        p = subprocess.run([lms] + argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, creationflags=flags)
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, f"`lms {' '.join(argv)}` 超过 {timeout:.0f}s 没返回"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def gpu_mem_used() -> Optional[float]:
    """当前显存占用（GB，多卡取最大）。拿不到就返回 None——只是展示用，不能因此中断流程。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        vals = [float(x) / 1024 for x in out.split() if x.strip().replace(".", "").isdigit()]
        return max(vals) if vals else None
    except Exception:
        return None


def _vram_suffix() -> str:
    v = gpu_mem_used()
    return f"，显存 {v:.1f} GB" if v is not None else ""


def backend_loaded() -> List[str]:
    """当前 LM Studio 里已加载实例的标识符列表（拿不到就返回空）。"""
    rc, out = run_lms(["ps"], timeout=60)
    if rc != 0:
        return []
    ids: List[str] = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts and not parts[0].startswith("-"):
            ids.append(parts[0])
    return ids


def backend_load(args) -> bool:
    """翻译阶段开始前把 Sakura 加载进显存。返回是否就绪。

    只有 LM Studio + --manage-backend 才由我们接管；其它后端（llama.cpp / Ollama /
    DeepSeek）各有各的常驻方式，不该越权去动。
    """
    if not getattr(args, "manage_backend", False) or args.translator == "none":
        return True
    if (getattr(args, "preset", None) or "") != "lmstudio":
        return True
    ident = (args.model_name or "").strip()
    if not ident:
        return True
    if ident in backend_loaded():
        log(f"[后端] {ident} 已在显存中，跳过加载")
        return True
    key = MODEL_KEY_HINTS.get(ident, ident)
    log(f"[后端] 加载 Sakura：{key}（API 标识符 {ident}）{_vram_suffix()}")
    t0 = time.time()
    rc, out = run_lms(["load", key, "--gpu", "max", "--context-length", "8192",
                       "--identifier", ident, "-y"], timeout=600)
    if rc != 0:
        log(f"[后端] ✗ 加载失败（返回码 {rc}）：{out[-300:]}")
        return False
    log(f"[后端] ✓ 就绪，用时 {time.time() - t0:.1f}s{_vram_suffix()}")
    return True


def backend_unload(args, reason: str = "") -> None:
    """把模型从显存卸掉。失败只提示，不影响产出。"""
    if not getattr(args, "manage_backend", False):
        return
    if (getattr(args, "preset", None) or "") != "lmstudio":
        return
    ident = (args.model_name or "").strip()
    if not ident or ident not in backend_loaded():
        return
    rc, out = run_lms(["unload", ident], timeout=180)
    tail = f"（{reason}）" if reason else ""
    if rc == 0:
        log(f"[后端] 已卸载 {ident}{tail}{_vram_suffix()}")
    else:
        log(f"[后端] 卸载 {ident} 失败（返回码 {rc}）：{out[-200:]}")


def free_asr_model() -> None:
    """把 ASR 模型从显存里放掉。

    ctranslate2 没有显式 unload 接口，唯一办法是丢掉引用 + gc，让析构去释放。
    必须把 _MODEL_CACHE 整表清掉——留着的话下一个文件又会命中缓存把它拉回来。
    """
    if not _MODEL_CACHE:
        return
    before = gpu_mem_used()
    _MODEL_CACHE.clear()
    gc.collect()
    try:                                    # torch 只用来归还它的缓存分配器；没有也不影响
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    after = gpu_mem_used()
    if before is not None and after is not None:
        log(f"[ASR] 已卸载转写模型，显存 {before:.1f} GB → {after:.1f} GB")
    else:
        log("[ASR] 已卸载转写模型")


# --------------------------------------------------------------------------------------
# 2.6) 文件名翻译（原本在 webui.py 的下载阶段，挪到这里是为了赶上 Sakura 在场的那段时间）
# --------------------------------------------------------------------------------------

_NAME_PROMPT = (
    "把下面的日文音频文件名逐行翻译成简体中文。要求：\n"
    "1. 输出行数必须与输入完全一致，一行一个，不要合并或拆分\n"
    "2. 编号与章号**原样照抄**：「第1章」就是「第1章」，不要改成「序章1」或「第一章」；\n"
    "   括号与分隔符（『』 / 【】）也保留\n"
    "3. 译名要简洁，像文件名而不是句子；不要加引号、序号或任何解释\n"
    "4. 逐词直译，不要省略或概括（长词就完整译出，不要缩写成更短的近义词）\n"
    "5. 这些是音声作品的固定用语，必须按此翻译：\n"
    "   トラック = 音轨（**不是「卡车」**）；パート = part；プロローグ = 序章；\n"
    "   エピローグ = 尾声；添い寝 = 陪睡；おまけ = 附赠；本編 = 正篇\n"
    "6. 人名一律保留原文汉字：「零」「舞」这类单字名不要当成数字或普通词翻译\n"
    "7. 数字与英文字母一律用半角（4 而不是 ４）"
)

# 全角 → 半角（模型偶尔会输出全角数字/字母，文件名里会显得很怪）
_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# 文件名开头的编号：第2-2章 / 第1章 / 03 / 2-1 …
_PREFIX_RE = re.compile(
    r"^\s*(第\s*[0-9０-９]+\s*(?:[-－ー][0-9０-９]+)?\s*[章話回]"
    r"|[0-9０-９]+\s*(?:[-－ー][0-9０-９]+)?)")


def _keep_prefix(orig: str, trans: str) -> str:
    """确保译文保留原名的编号前缀。

    模型很爱把「第1章…プロローグ…」整条改写成「序章…」——编号一丢，
    文件排序就乱了。编号是结构信息不是内容，这里用确定性规则兜住。
    """
    m = _PREFIX_RE.match(orig)
    if not m:
        return trans
    pre = m.group(1).strip().translate(_FULLWIDTH)
    if not pre:
        return trans
    head = trans.translate(_FULLWIDTH)[:len(pre) + 4]
    return trans if pre in head else f"{pre} {trans}"


def _balance(s: str) -> str:
    """补齐模型漏掉的右括号。

    实测模型会把「第2章　『お寺内』」译成「第2章 『寺庙里」——左括号留着，
    右括号丢了。括号是结构信息，用确定性规则补齐。
    """
    for l, r in (("『", "』"), ("「", "」"), ("【", "】"), ("（", "）"), ("《", "》")):
        n = s.count(l) - s.count(r)
        if n > 0:
            s += r * n
    return s


def _clean_name(s: str) -> str:
    """清掉模型可能带上的序号/引号，并做 Windows 文件名安全化。"""
    s = s.strip().strip('"\'“”')
    s = re.sub(r"^\s*\d+\s*[.、):：]\s*", "", s)          # 去掉 "1. " / "1、"
    s = s.translate(_FULLWIDTH)                           # ４ → 4
    s = _balance(s)                                       # 补右括号
    s = s.replace("　", " ")                              # 全角空格在部分工具里会出问题
    s = re.sub(r'[\\/:*?"<>|]', "", s)                    # Windows 非法字符
    s = re.sub(r"\s+", " ", s).strip().rstrip(". ")       # 结尾的点和空格 Windows 不允许
    if s.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL",
                                   *[f"COM{i}" for i in range(1, 10)],
                                   *[f"LPT{i}" for i in range(1, 10)]}:
        s = "_" + s
    return s[:120] or "unnamed"


def filenames_json_path(outdir) -> Path:
    """文件名译名落盘的位置（放在输出目录根下）。"""
    return Path(outdir) / "_filenames.json"


def translate_filenames(stems: List[str], cfg: dict, log_cb=None) -> Dict[str, str]:
    """把音频文件名（不含扩展名）翻译成中文，返回 {原名: 译名}。

    cfg 需要：base_url / api_key / model_name / protocol / max_tokens / glossary_text。
    采样温度**必须压低**：文件名翻译是「翻译」不是「创作」，用服务端默认温度（通常 0.8）
    实测同一批跑两遍有 85% 的条目不一样，纯噪声。
    """
    say = log_cb or (lambda _m: None)
    if not stems:
        return {}
    if not cfg.get("base_url"):
        say("[文件名] 未配置后端，跳过文件名翻译")
        return {}

    args = argparse.Namespace(
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key") or "",
        timeout=cfg.get("timeout", 180.0),
        model_name=cfg.get("model_name") or "sakura",
        protocol=cfg.get("protocol", "sakura"),
        max_tokens=cfg.get("max_tokens", 2048),
        temperature=0.2, top_p=0.9, frequency_penalty=0.0,
    )
    client = _make_client(args)

    system = _NAME_PROMPT
    gl = (cfg.get("glossary_text") or "").strip()
    if gl:
        # 只取规则行，去掉注释，免得把说明文字也喂进去
        rules = [l for l in gl.splitlines() if l.strip() and not l.strip().startswith("#")]
        if rules:
            system += "\n\n参考术语表（务必遵守）：\n" + "\n".join(rules[:40])

    out: Dict[str, str] = {}
    B = 10                                                 # 一次 10 条，行数对齐更稳
    for i in range(0, len(stems), B):
        chunk = stems[i:i + B]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": "\n".join(chunk)}]
        got = None
        for _ in range(3):                                 # 行数不匹配就重发，最多 3 次
            try:
                txt = _chat(client, args, msgs)
            except Exception as e:
                say(f"[文件名] 调用失败：{type(e).__name__}: {e}")
                break
            lines = [x for x in (txt or "").splitlines() if x.strip()]
            if len(lines) == len(chunk):
                got = lines
                break
        if got:
            for src, dst in zip(chunk, got):
                out[src] = _keep_prefix(src, _clean_name(dst))
        else:
            say(f"[文件名] 第 {i // B + 1} 批对齐失败，这批保留原名")
    # 去重：译名撞车时补序号，否则后一个会覆盖前一个
    seen: dict = {}
    for k, v in list(out.items()):
        base, n = v, 2
        while v in seen:
            v = f"{base} ({n})"
            n += 1
        seen[v] = k
        out[k] = v
    say(f"[文件名] 已翻译 {len(out)}/{len(stems)} 个文件名")
    return out


def _do_names(args, files) -> Optional[dict]:
    """文件名翻译：蹭 Sakura 还在显存里的这段时间一起做掉。

    结果写到 <outdir>/_filenames.json，WebUI 打包时直接读，
    省掉「为了几个文件名再加载一次 6 GB 模型」。
    """
    stems: List[str] = []
    for f, _root in files:
        if f.stem not in stems:
            stems.append(f.stem)
    if not stems:
        return None
    cfg = {
        "base_url": args.base_url,
        "api_key": args.api_key or "",
        "model_name": args.model_name,
        "protocol": args.protocol,
        "max_tokens": args.max_tokens,
        "glossary_text": getattr(args, "glossary_text", "") or "",
        "timeout": args.timeout,
    }
    t0 = time.time()
    try:
        nm = translate_filenames(stems, cfg, log_cb=log)
    except Exception as e:
        log(f"[文件名] 翻译失败，保留原文件名：{type(e).__name__}: {e}")
        return None
    el = time.time() - t0
    if not nm:
        return None
    if args.outdir:
        p = filenames_json_path(args.outdir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(nm, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"[文件名] 译名已存 {p.name}（{p.parent}）")
    else:
        log("[文件名] 未指定 --outdir，译名不落盘（WebUI 靠它做下载改名）")
    return {"count": len(nm), "sec": el}


# --------------------------------------------------------------------------------------
# 3) 输出
# --------------------------------------------------------------------------------------

def write_lrc(path: Path, segs: List[Seg], mode: str, header: Optional[str]) -> None:
    lines: List[str] = []
    if header:
        for h in header.splitlines():
            lines.append(f"[ti:{h}]" if h == header.splitlines()[0] else f"[by:{h}]")
    for s in segs:
        ja = s.ja.strip()
        zh = s.zh.strip() or ja
        ts = fmt_ts(s.start)
        if mode == "ja":
            lines.append(f"{ts}{ja}")
        elif mode == "zh":
            lines.append(f"{ts}{zh}")
        elif mode == "both":
            lines.append(f"{ts}{ja}")
            lines.append(f"{ts}{zh}")
        elif mode == "inline":
            lines.append(f"{ts}{ja} ｜ {zh}")
        else:
            raise ValueError(f"未知 lrc-mode: {mode}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[输出] {path}（{len(lines)} 行）")


def write_srt(path: Path, segs: List[Seg], bilingual: bool = True) -> None:
    def t(x: float) -> str:
        ms = int(round(x * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out = []
    for i, s in enumerate(segs, 1):
        text = f"{s.ja}\n{s.zh}" if (bilingual and s.zh) else (s.zh or s.ja)
        out.append(f"{i}\n{t(s.start)} --> {t(s.end)}\n{text}\n")
    path.write_text("\n".join(out), encoding="utf-8")
    log(f"[输出] {path}")


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def _signature(args) -> dict:
    """翻译配置指纹：术语表/提示词模板/模型/采样参数任一变化，缓存译文即失效。"""
    import hashlib
    g = getattr(args, "glossary_text", "") or ""
    return {
        "prompt_style": getattr(args, "prompt_style", "v1"),
        "model_name": getattr(args, "model_name", "") or "",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "glossary": hashlib.md5(g.encode("utf-8")).hexdigest()[:10] if g else "",
    }


def _write_cache(cache: Path, audio: Path, args, segs: List[Seg]) -> None:
    cache.write_text(json.dumps(
        {"audio": str(audio), "model": args.model, "meta": _signature(args),
         "segments": [asdict(s) for s in segs]}, ensure_ascii=False, indent=1), encoding="utf-8")


def probe_duration(audio: Path) -> float:
    """用 ffprobe 取音频时长（秒）。拿不到返回 0——只是用于报速度，不能因此中断。"""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(audio)],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _out_base(args, audio: Path, root: Optional[Path]) -> Path:
    """输出目录：镜像输入的相对子目录结构，避免不同子目录里的同名音轨互相覆盖
    （例如 1.本編\4-1.xxx.wav 与 3.SEなし\4-1.xxx.wav 同名）。"""
    base = Path(args.outdir) if args.outdir else audio.parent
    if args.outdir and root is not None:
        try:
            base = base / audio.parent.relative_to(root)
        except ValueError:
            pass
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cache_path(args, audio: Path, root: Optional[Path]) -> Path:
    # 注意：不能用 Path.with_suffix()——像 "1.イントロ.wav" 这种文件名，Path 会把
    # ".イントロ" 当成扩展名替换掉，导致输出变成 "1.lrc"（描述丢了）。一律字符串拼接。
    return _out_base(args, audio, root) / (audio.stem + ".segments.json")


def _read_cache(args, cache: Path) -> Optional[List[Seg]]:
    """读回 segments.json，顺带做配置指纹校验。文件不存在返回 None。"""
    if not cache.exists():
        return None
    blob = json.loads(cache.read_text(encoding="utf-8"))
    segs = [Seg(**d) for d in blob["segments"]]
    old_meta = blob.get("meta")
    if args.reuse_translation and old_meta and old_meta != _signature(args):
        log("[翻译] 检测到术语表/模型/参数已变化 → 忽略缓存译文，本次全部重译")
        args.reuse_translation = False
    elif args.reuse_translation and not old_meta and not getattr(args, "_meta_warned", False):
        log("[提示] 该缓存来自旧版本、没有配置指纹，本次按原样复用译文；"
            "若要应用新术语表/新参数，请去掉 --reuse-translation 重跑一次")
        args._meta_warned = True
    return segs


# --------------------------------------------------------------------------------------
# 阶段一：只做 ASR（把所有音轨都转写完，才轮到翻译）
# --------------------------------------------------------------------------------------

def asr_one(args, audio: Path, root: Optional[Path] = None) -> dict:
    """转写一轨并落盘缓存。返回该轨的统计（时长/耗时/行数）用于收尾报告。"""
    base = _out_base(args, audio, root)
    cache = base / (audio.stem + ".segments.json")
    t0 = time.time()

    segs: Optional[List[Seg]] = None
    cached = False
    if args.from_json:
        segs = [Seg(**d) for d in
                json.loads(Path(args.from_json).read_text(encoding="utf-8"))["segments"]]
        log(f"[缓存] 从 {args.from_json} 载入 {len(segs)} 段")
        cached = True
    elif args.retranslate:
        segs = _read_cache(args, cache)
        if segs is not None:
            log(f"[缓存] 复用 {cache.name} 的 {len(segs)} 段转写结果")
            cached = True

    if segs is None:
        segs = transcribe(args, audio)
        _write_cache(cache, audio, args, segs)
        log(f"[缓存] 转写结果已存 {cache.name}（下次可用 --retranslate 跳过 ASR）")
    elif getattr(args, "asr_fixes", None) and (args.from_json or args.retranslate):
        # 从缓存载入时也要套用修正表，否则 --retranslate 会漏掉修正
        k = apply_asr_fixes(segs, args.asr_fixes)
        if k:
            log(f"[修正] ASR 修正表改动 {k} 段（原文留档在 ja_raw）")
            _write_cache(cache, audio, args, segs)

    el = time.time() - t0
    dur = probe_duration(audio)
    # 命中缓存时耗时就该是 ~0，算出来的「x 实时」会是几万倍，纯噪声——直接不给速度
    speed = (dur / el) if (not cached and dur > 0 and el > 0.05) else 0.0
    tail = "（命中缓存，未重新转写）" if cached else (f" / {speed:.1f}x 实时" if speed else "")
    log(f"[计时] ASR {audio.name}：{dur / 60:.1f} 分钟音频 / 用时 {el:.1f}s{tail}")
    return {"name": audio.name, "dur": dur, "sec": el, "lines": len(segs),
            "speed": speed, "cached": cached}


# --------------------------------------------------------------------------------------
# 阶段二：只做翻译 + 出 LRC（此时显存里只有 Sakura，ASR 已经放掉了）
# --------------------------------------------------------------------------------------

def mt_one(args, audio: Path, root: Optional[Path] = None) -> dict:
    """翻译一轨（载入阶段一的缓存）并输出 LRC。返回统计。"""
    base = _out_base(args, audio, root)
    cache = base / (audio.stem + ".segments.json")
    segs = _read_cache(args, cache)
    if segs is None:
        raise RuntimeError(f"缺少转写缓存 {cache.name}，请先跑 ASR 阶段")
    if args.retranslate:
        log(f"[缓存] 复用 {cache.name} 的 {len(segs)} 段转写结果")

    n_lines = sum(1 for s in segs if s.ja.strip())
    t0 = time.time()
    reused = False
    if args.translator != "none":
        if args.reuse_translation and all(s.zh.strip() or not s.ja.strip() for s in segs):
            log("[翻译] 缓存里已有全部译文，跳过翻译（换 --lrc-mode 重新输出时可省时间）")
            reused = True
        else:
            translate_all(args, segs)
        _write_cache(cache, audio, args, segs)      # 两种情况都写，保证配置指纹落盘
    el = time.time() - t0

    n_zh = sum(1 for s in segs if s.zh.strip())
    speed = (n_zh / el) if (not reused and el > 0.05 and n_zh) else 0.0
    tail = "（命中缓存，未重新翻译）" if reused else (f" / {speed:.1f} 行每秒" if speed else "")
    log(f"[计时] 翻译 {audio.name}：{n_zh} 行 / 用时 {el:.1f}s{tail}")

    stem_name = audio.stem
    header = args.title or stem_name
    if args.lrc_mode != "none":
        # 纯中文是主产物，直接用 <名字>.lrc（播放器可直接识别）；
        # 只有非中文的产物才加标记：.ja / .zh-ja / .inline
        suffix = {"zh": "", "ja": ".ja", "both": ".zh-ja", "inline": ".inline"}[args.lrc_mode]
        write_lrc(base / (stem_name + suffix + ".lrc"), segs, args.lrc_mode, header)
    # 额外输出日文原文 LRC：用于校对，也方便换模型重译时对照
    if args.keep_ja and args.lrc_mode not in ("ja", "both", "inline"):
        write_lrc(base / (stem_name + ".ja.lrc"), segs, "ja", header)
    if args.srt:
        write_srt(base / (stem_name + ".srt"), segs, bilingual=args.translator != "none")

    return {"name": audio.name, "lines": n_zh, "ja_lines": n_lines, "sec": el,
            "speed": speed, "reused": reused}


def _fmt_sec(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}分{s % 60:02d}秒" if s >= 60 else f"{s}秒"


def print_report(asr_stats: List[dict], mt_stats: List[dict], name_stat: Optional[dict],
                 total_sec: float, vram_asr: Optional[float], vram_mt: Optional[float],
                 phases: str) -> None:
    """收尾报告：整体速度与用时 + 逐轨明细。"""
    log("")
    log("=" * 74)
    log("完成报告")
    log("=" * 74)

    if asr_stats:
        fresh = [s for s in asr_stats if not s.get("cached")]
        log("")
        if fresh:
            dur = sum(s["dur"] for s in fresh)
            sec = sum(s["sec"] for s in fresh)
            sp = (dur / sec) if sec > 0 else 0
            extra = f"（{len(asr_stats) - len(fresh)} 轨命中缓存）" if len(fresh) < len(asr_stats) else ""
            log(f"ASR 阶段 ：{len(asr_stats)} 轨 / {dur / 60:.1f} 分钟音频{extra}"
                f" / 用时 {_fmt_sec(sec)} / 平均 {sp:.1f}x 实时")
        else:
            log(f"ASR 阶段 ：{len(asr_stats)} 轨全部命中缓存，无需重新转写")
        log(f"{'':2}{'音轨':<36}{'音频':>10}{'用时':>11}{'速度':>10}")
        for s in asr_stats:
            v = "缓存" if s.get("cached") else (f"{s['speed']:.1f}x" if s["speed"] else "-")
            log(f"{'':2}{s['name'][:35]:<36}{s['dur'] / 60:>8.1f}分{s['sec']:>10.1f}s{v:>10}")

    if mt_stats:
        fresh_mt = [s for s in mt_stats if not s.get("reused")]
        log("")
        if fresh_mt:
            n_zh = sum(s["lines"] for s in fresh_mt)
            sec = sum(s["sec"] for s in fresh_mt)
            sp = (n_zh / sec) if sec > 0 else 0
            extra = f"（{len(mt_stats) - len(fresh_mt)} 轨命中缓存）" if len(fresh_mt) < len(mt_stats) else ""
            log(f"翻译阶段 ：{len(mt_stats)} 轨 / {n_zh} 行{extra}"
                f" / 用时 {_fmt_sec(sec)} / 平均 {sp:.1f} 行每秒")
        else:
            log(f"翻译阶段 ：{len(mt_stats)} 轨全部命中缓存，无需重新翻译")
        log(f"{'':2}{'音轨':<36}{'行数':>10}{'用时':>11}{'速度':>10}")
        for s in mt_stats:
            v = "缓存" if s.get("reused") else (f"{s['speed']:.1f}/s" if s["speed"] else "-")
            log(f"{'':2}{s['name'][:35]:<36}{s['lines']:>10}{s['sec']:>10.1f}s{v:>10}")

    if name_stat:
        log("")
        log(f"文件名   ：{name_stat['count']} 个 / 用时 {_fmt_sec(name_stat['sec'])}")

    if phases == "all":
        log("")
        log(f"总计     ：{_fmt_sec(total_sec)}")
        if vram_asr is not None and vram_mt is not None:
            log(f"显存驻留 ：ASR 阶段 {vram_asr:.1f} GB → 翻译阶段 {vram_mt:.1f} GB"
                f"（分时复用，不再叠加成 {vram_asr + vram_mt:.1f} GB）")
    log("=" * 74)


def collect_inputs(paths: Sequence[str]) -> List[tuple]:
    """返回 [(音频路径, 输入根目录)]，根目录用于在 --outdir 下镜像子目录结构。"""
    out: List[tuple] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out += [(f, path) for f in sorted(f for f in path.rglob("*")
                                              if f.suffix.lower() in AUDIO_EXTS)]
        elif path.is_file():
            out.append((path, path.parent))
        else:                                     # 支持通配符
            out += [(f, Path(".")) for f in sorted(Path(".").glob(p))
                    if f.suffix.lower() in AUDIO_EXTS]
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="日语音频 → 中日双语 LRC（anime-whisper + LLM）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("示例：")[-1],
    )
    p.add_argument("inputs", nargs="+", help="音频文件 / 目录 / 通配符")
    p.add_argument("--preset", choices=sorted(PRESETS), help="翻译后端预设（sakura-llamacpp / sakura-ollama / deepseek）")

    g = p.add_argument_group("ASR（转写）")
    g.add_argument("--model", default=DEFAULT_ASR_MODEL, help=f"faster-whisper 模型或本地目录（默认 {DEFAULT_ASR_MODEL}）")
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    g.add_argument("--compute-type", default="int8_float16", help="cuda 建议 int8_float16 / float16；cpu 用 int8")
    g.add_argument("--cpu-threads", type=int, default=4)
    g.add_argument("--cuda-dll-dir", help="手动指定含 cublas64_12.dll 的目录（自动找不到 CUDA 12 运行时用）")
    g.add_argument("--beam-size", type=int, default=5)
    g.add_argument("--no-repeat-ngram-size", type=int, default=5, help="抑制复读幻觉（anime-whisper 官方建议 5-10）")
    g.add_argument("--repetition-penalty", type=float, default=1.0)
    g.add_argument("--no-speech-threshold", type=float, default=0.6)
    g.add_argument("--hallucination-silence", type=float, default=0.0,
                   help="幻觉时跳过的静音长度（秒），0=关闭（默认）。注意：开启会自动启用 word_timestamps，"
                        "而部分 CT2 转换模型开 word_timestamps 会崩溃")
    g.add_argument("--word-timestamps", action="store_true", help="强制启用词级时间戳（更慢）")
    g.add_argument("--vad", action="store_true", help="整文件模式下启用 VAD 切静音（默认关闭）")
    g.add_argument("--chunk-mode", default="hybrid", choices=["hybrid", "vad", "energy", "whole"],
                   help="hybrid=台词段用 VAD 边界、空隙用能量块补齐（默认，最稳）；"
                        "vad=只按静音切块；energy=全部按能量切块；whole=整文件交给模型")
    g.add_argument("--gap-min", type=float, default=6.0,
                   help="hybrid 模式下超过该秒数的空隙才用能量块补（默认 6s）")
    g.add_argument("--energy-chunk", type=float, default=10.0, help="energy/hybrid 模式的目标块长（秒）")
    g.add_argument("--energy-floor-db", type=float, default=-55.0, help="energy 模式下丢弃低于该峰值的块（dBFS）")
    g.add_argument("--max-chunk", type=float, default=15.0, help="VAD 模式下单个语音块最长秒数")
    g.add_argument("--vad-threshold", type=float, default=None,
                   help="VAD 语音判定阈值；默认 auto（0.4→0.25→0.15 下探直到覆盖率达 --vad-min-coverage）")
    g.add_argument("--vad-min-coverage", type=float, default=0.25,
                   help="自动下探阈值时的目标语音覆盖率（默认 0.25）")
    g.add_argument("--vad-min-silence", type=int, default=600, help="切块的最短静音时长(ms)")
    g.add_argument("--vad-min-speech", type=int, default=250, help="最短语音时长(ms)")
    g.add_argument("--vad-pad", type=int, default=200, help="语音块两端补白(ms)")
    g.add_argument("--max-repeat", type=int, default=3, help="连续相同行最多保留几次")
    g.add_argument("--merge-gap", type=float, default=0.8,
                   help="间隔小于该秒数的相邻行合并，0=不合并。"
                        "0.8 是实测值：与人工字幕的粒度更接近（官方译文显示时长中位 6.8s，"
                        "0.8→4.8s，0.4→3.0s），且行数不匹配重试从 2 次降到 0 次")
    g.add_argument("--max-chars", type=int, default=32, help="单行最大字符数（超出按标点切分）")

    g = p.add_argument_group("翻译")
    g.add_argument("--translator", default="openai", choices=["openai", "none"], help="none=只转写不翻译")
    g.add_argument("--base-url", help="OpenAI 兼容接口地址，如 http://127.0.0.1:8080/v1")
    g.add_argument("--api-key", help="默认读环境变量 DEEPSEEK_API_KEY / OPENAI_API_KEY")
    g.add_argument("--model-name", help="翻译模型名，如 sakura / deepseek-chat")
    g.add_argument("--protocol", choices=["sakura", "json"], help="sakura=官方多行对齐格式；json=通用 JSON 协议")
    g.add_argument("--prompt-style", default="v1", choices=sorted(PROMPT_STYLES),
                   help="提示词模板：v1=轻小说风格（配 Sakura-7B/14B-Qwen2.5-v1.0）；"
                        "v3=视觉小说风格（配 GalTransl-7B-v3.7 / v4-4B）。两代不可混用！")
    g.add_argument("--temperature", type=float, help="采样温度（v1 官方推荐 0.1；v3 官方推荐 0.3）")
    g.add_argument("--top-p", type=float, help="top_p（v1 官方推荐 0.3；v3 官方推荐 0.8）")
    g.add_argument("--frequency-penalty", type=float, default=None,
                   help="频率惩罚；Sakura 官方建议仅在出现退化时设 0.1~0.2（预设默认 0.15）")
    g.add_argument("--max-tokens", type=int, default=None,
                   help="单次回复上限（默认 2048；一次翻 8 行需要足够大，别用 512）")
    g.add_argument("--batch-size", type=int, default=8, help="每次请求翻译几行")
    g.add_argument("--batch-chars", type=int, default=600, help="单次请求的原文总字符上限")
    g.add_argument("--workers", type=int, default=4,
                   help="翻译并发数，应与 LM Studio 的 PARALLEL 设置一致。"
                        "实测 156 行：并发 1 → 59.7s，并发 4 → 25.8s（快 57%%）。"
                        "服务端 slot 不够时请求自动排队，不会出错，只是没有加速")
    g.add_argument("--retries", type=int, default=0,
                   help="**顶层**整批翻译失败后、拆半之前先原样重发几次（默认 0 = 不重发，直接拆半）。"
                        "行数不匹配的根因是「按停顿分的行≠句子」，同一批原样重发基本无效"
                        "（实测 3 次尝试全失败），所以默认一失败就拆。"
                        "拆半后的子批次同样只试 1 次。"
                        "最后兜底的逐行重发不受此值影响，至少试 3 次")
    g.add_argument("--timeout", type=float, default=180.0)
    g.add_argument("--glossary", help="术语表文件，每行 原文->译文 #备注（Sakura GPT 字典格式）")
    g.add_argument("--fix-asr", dest="fix_asr_path",
                   help="ASR 修正表文件，每行 `听错的=>正确的`；在转写之后、翻译之前替换日文原文"
                        "（术语表救不了 ASR 错字，必须在这一步修）")

    g = p.add_argument_group("输出 / 缓存")
    g.add_argument("--lrc-mode", default="both", choices=["zh", "ja", "both", "inline", "none"],
                   help="both=同一时间戳两行（日文在上）；inline=一行内用｜分隔")
    g.add_argument("--title", help="LRC 标题（默认文件名）")
    g.add_argument("--keep-ja", action="store_true",
                   help="额外输出日文原文 LRC（<名字>.ja.lrc），用于校对/换模型重译对照；"
                        "转写+译文本身始终保存在 <名字>.segments.json 里")
    g.add_argument("--outdir", help="输出目录（默认与音频同目录）")
    g.add_argument("--srt", action="store_true", help="额外输出 SRT")
    g.add_argument("--retranslate", action="store_true", help="复用已有 .segments.json 的转写，只重跑翻译")
    g.add_argument("--reuse-translation", action="store_true", help="缓存里已有译文时直接复用，不重复调用模型")
    g.add_argument("--from-json", help="直接从指定 segments.json 开始（跳过 ASR）")

    g = p.add_argument_group("显存编排（避免 ASR 与 Sakura 抢显存导致降频）")
    g.add_argument("--phases", default="all", choices=["all", "asr", "translate"],
                   help="all=先跑完所有 ASR 再统一翻译（默认，显存分时复用）；"
                        "asr/translate=只跑其中一段（调试或单独重跑用）")
    g.add_argument("--manage-backend", action="store_true",
                   help="由本程序在翻译阶段加载 Sakura、跑完卸载（仅 LM Studio）。"
                        "WebUI 会自动带上；命令行默认不开，免得干预你自己常驻的服务")
    g.add_argument("--translate-names", action="store_true",
                   help="顺便把音频文件名也译了，译名写到 <outdir>/_filenames.json。"
                        "蹭 Sakura 在场的时间做，不用为了几个文件名再加载一次模型")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.preset:
        preset = PRESETS[args.preset]
        # 优先级：命令行显式指定 > 预设自带的 sampling（如 DeepSeek 官方 1.3）> 提示词模板推荐值 > 预设其余字段
        for k, v in preset.get("sampling", {}).items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)
    style = PROMPT_STYLES[args.prompt_style]
    for k in ("temperature", "top_p", "frequency_penalty"):
        if getattr(args, k, None) is None:
            setattr(args, k, style[k])
    if args.preset:
        for k, v in PRESETS[args.preset].items():
            if k == "sampling":
                continue
            if getattr(args, k, None) is None:
                setattr(args, k, v)
        log(f"[配置] 预设 {args.preset}: {args.base_url} model={args.model_name} protocol={args.protocol}")
    log(f"[配置] 提示词模板 {args.prompt_style}（{style['label']}）"
        f" temperature={args.temperature} top_p={args.top_p}"
        f" frequency_penalty={args.frequency_penalty} max_tokens={args.max_tokens}")

    if args.protocol is None:
        args.protocol = "sakura" if "sakura" in (args.model_name or "").lower() else "json"
    if args.translator != "none" and not args.base_url:
        args.base_url = PRESETS["sakura-llamacpp"]["base_url"]
        args.model_name = args.model_name or "sakura"
        log(f"[配置] 未指定 --base-url，默认按本地 Sakura 处理：{args.base_url}")

    if args.glossary:
        args.glossary_text = Path(args.glossary).read_text(encoding="utf-8-sig").strip()
    else:
        args.glossary_text = ""

    args.asr_fixes = load_asr_fixes(args.fix_asr_path) if args.fix_asr_path else []
    if args.asr_fixes:
        log(f"[配置] ASR 修正表 {args.fix_asr_path}：{len(args.asr_fixes)} 条规则")

    if args.device == "auto":
        args.device = "cuda"          # faster-whisper 会在无卡时报错，交给用户显式指定 cpu

    if args.cuda_dll_dir:
        os.environ["ONSEI2LRC_CUDA_DLL_DIRS"] = args.cuda_dll_dir + os.pathsep + \
            os.environ.get("ONSEI2LRC_CUDA_DLL_DIRS", "")

    files = collect_inputs(args.inputs)
    if not files:
        log("没有找到音频文件"); return 2
    log(f"[输入] 共 {len(files)} 个文件")

    t_all = time.time()
    asr_stats: List[dict] = []
    mt_stats: List[dict] = []
    name_stat: Optional[dict] = None
    vram_asr = vram_mt = None

    # ---- 阶段 1：ASR。显存里只有 whisper，全部转写完再放掉 ----
    if args.phases in ("all", "asr"):
        log("\n" + "=" * 74)
        log(f"阶段 1/2：ASR 转写（{len(files)} 个文件）")
        log("=" * 74)
        for i, (f, root) in enumerate(files, 1):
            log(f"\n----- [{i}/{len(files)}] {f} -----")
            try:
                asr_stats.append(asr_one(args, f, root))
            except KeyboardInterrupt:
                log("已中断"); return 130
            except Exception as e:
                log(f"[错误] {f} 转写失败：{type(e).__name__}: {e}")
        vram_asr = gpu_mem_used()
        free_asr_model()          # 腾空显存，给 Sakura 让位（这一步是整次提速的关键）

    # ---- 阶段 2：翻译。此时才加载 Sakura，译完正文和文件名再放掉 ----
    if args.phases in ("all", "translate"):
        if args.translator != "none":
            log("\n" + "=" * 74)
            log(f"阶段 2/2：翻译（{len(files)} 个文件）")
            log("=" * 74)
            backend_load(args)
        for i, (f, root) in enumerate(files, 1):
            log(f"\n----- [{i}/{len(files)}] {f} -----")
            try:
                mt_stats.append(mt_one(args, f, root))
            except KeyboardInterrupt:
                log("已中断"); return 130
            except Exception as e:
                log(f"[错误] {f} 翻译失败：{type(e).__name__}: {e}")
        # 文件名翻译搭 Sakura 在场的顺风车，省掉「为几个文件名再加载一次 6 GB 模型」
        if args.translator != "none" and args.translate_names:
            log("")
            name_stat = _do_names(args, files)
        vram_mt = gpu_mem_used()
        backend_unload(args, reason="正文与文件名都已完成")

    print_report(asr_stats, mt_stats, name_stat, time.time() - t_all,
                 vram_asr, vram_mt, args.phases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
