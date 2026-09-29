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
    done=r"出(た|てる|ちゃ|てしま)|イッ?(た|ちゃ)|射精(し)?(た|てる)",
    countdown=r"[5５][…\s、.]*[4４][…\s、.]*[3３]|[3３][…\s、.]*[2２][…\s、.]*[1１]|ぜー?ろ",
    # 负面：未然 / 祈使 / 回顾 / 泛指 —— 这些时刻文本有线索但不是高潮
    negative=r"(し|に)?そう|ないように|寸前|したい|したから|出して(ください|ぇぇ)|ばかり",
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


def find_peaks(arr: np.ndarray, hop: float, min_gap: float = 15.0,
               top: int = 8) -> List[int]:
    """局部极大值，按高度排序后做最小间隔抑制。"""
    gap = int(min_gap / hop)
    idx = [i for i in range(1, len(arr) - 1)
           if arr[i] >= arr[i - 1] and arr[i] > arr[i + 1]]
    idx.sort(key=lambda i: -arr[i])
    kept: List[int] = []
    for i in idx:
        if all(abs(i - j) > gap for j in kept):
            kept.append(i)
        if len(kept) >= top:
            break
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
    """候选点 ±radius 秒内最强的一条线索。返回 (权重, 标签, 那句原文)。"""
    best, tags, text = 0, [], ""
    for s in segs:
        if abs(s["start"] - t) > radius:
            continue
        ja = s.get("ja") or ""
        w, tg = cues.score(ja)
        if w > best:
            best, tags, text = w, tg, ja
    return best, tags, text


def decode_audio_mono(path: Path, sr: int = SR) -> np.ndarray:
    """解码成 sr 采样率单声道 float32。

    优先用 faster-whisper 的 decode_audio（PyAV，无外部进程）；
    不可用时退回 ffmpeg——这样本工具不强制依赖 faster-whisper。
    """
    try:
        from faster_whisper import decode_audio
        return decode_audio(str(path), sampling_rate=sr)
    except ImportError:
        pass
    import subprocess
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[:200]
                           or "ffmpeg 解码失败")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


# ======================================================================================
# 主流程
# ======================================================================================
def find_climaxes(audio: Path, *, top: int = 3, cues: Optional[CueRules] = None,
                  acoustic_only: bool = False, verbose: bool = True,
                  transcript_dirs: Optional[List[Path]] = None) -> dict:
    """分析单个音频，返回候选点。不抛异常——失败信息放在结果的 error 字段。"""
    cues = cues or DEFAULT_CUES
    t0 = time.time()
    res: Dict = {"audio": str(audio), "name": audio.name, "candidates": [],
                 "transcript": False, "error": ""}
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

    db, hop = energy_envelope(x)
    s = _smooth(db, int(0.3 / hop))
    med = float(np.median(s))

    cands = []
    for i in find_peaks(s, hop, min_gap=15.0, top=max(top * 3, 12)):
        t = i * hop
        f = acoustic_features(x, med, i, s, hop)
        tx, tags, txt = (text_score_at(segs, t, cues) if segs else (0, [], ""))
        # 标准化后加权
        za = ((f["signature"] - NORM["signature"]["mu"]) / NORM["signature"]["sd"]
              - (f["recover"] - NORM["recover"]["mu"]) / NORM["recover"]["sd"])
        zt = (tx - NORM["txt"]["mu"]) / NORM["txt"]["sd"]
        cands.append({
            "time": round(t, 2),
            "mmss": f"{int(t)//60:02d}:{int(t)%60:02d}",
            "score": round(W_ACOUSTIC * za + W_TXT * zt, 3),
            "acoustic": round(za, 3), "semantic": tx, "cue_tags": tags,
            "text": txt,
            **{k: round(v, 2) for k, v in f.items()},
        })

    # 最小间隔再抑制一次（Top-N 之间至少隔 15 秒）
    cands.sort(key=lambda c: -c["score"])
    picked: List[dict] = []
    for c in cands:
        if all(abs(c["time"] - p["time"]) > 15.0 for p in picked):
            picked.append(c)
        if len(picked) >= top:
            break
    res["candidates"] = picked
    res["elapsed"] = round(time.time() - t0, 1)
    if verbose:
        _print_one(res)
    return res


def _print_one(res: dict) -> None:
    name = res["name"]
    if res.get("error"):
        print(f"\n  【{name}】✗ {res['error']}")
        return
    dur = res.get("duration", 0)
    src = "声学+语义" if res["transcript"] else "纯声学（无转写）"
    print(f"\n{'=' * 78}")
    print(f"  {name}   {dur/60:.1f} 分钟   [{src}]   {res.get('elapsed', 0)}s")
    print(f"{'=' * 78}")
    if not res["candidates"]:
        print("    （没有找到候选点）")
        return
    for i, c in enumerate(res["candidates"], 1):
        tags = "，".join(c["cue_tags"]) if c["cue_tags"] else "无文本线索"
        print(f"  #{i}  [{c['mmss']}]  分数 {c['score']:+6.2f}   "
              f"声学 {c['acoustic']:+5.2f}  文本 {c['semantic']:+d}（{tags}）")
        print(f"        形状: 前平台 {c['pre30']:+6.1f} dB · 峰值 {c['peak']:+6.1f} dB · "
              f"释放后 {c['post4']:+6.1f} dB · 脱力 {c['recover']:.1f}s")
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
    ap.add_argument("--json", help="把结果写到 JSON")
    ap.add_argument("--acoustic-only", action="store_true", help="只用声学，不读转写")
    ap.add_argument("--cues", help="自定义线索规则文件（每行一个正则，# 开头为注释）")
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

    print("=" * 78)
    print("  climax_finder —— 高潮候选点检测")
    print(f"  评分 = {W_ACOUSTIC} × 声学(signature, recover) + {W_TXT} × 语义")
    print("=" * 78)

    tdirs = [Path(d).expanduser() for d in a.transcript_dir]
    kw = dict(top=a.top, cues=cues, acoustic_only=a.acoustic_only,
              verbose=not a.quiet, transcript_dirs=tdirs or None)
    results = find_in_path(target, **kw)

    if not a.quiet:
        print("=" * 78)
        print("  提示：这是候选生成器，精确率约 53%，请人工确认后再使用")
        print("=" * 78)
    if a.json:
        Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print(f"  已写入 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
