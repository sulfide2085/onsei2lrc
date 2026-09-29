#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wav2mp3 — 音频转高品质 MP3 的工具库（同时可当命令行用）

单文件、无第三方依赖，只有 ffmpeg / ffprobe 两个外部程序。
直接把本文件复制到你自己的项目里就能用。

================================================================================
作为库使用
================================================================================

    from wav2mp3 import to_mp3, to_mp3_batch, probe, find_audio, summarize

    # 1) 转单个文件
    r = to_mp3("input.wav", "output.mp3")
    if r.ok:
        print(f"{r.dst_bytes} 字节，压缩到 1/{r.ratio:.1f}")
    else:
        print("失败:", r.error)

    # 2) 目标路径可以省略（默认同目录同名 .mp3）
    r = to_mp3("input.wav")

    # 3) 批量：输入可以是文件、目录，或两者的混合
    results = to_mp3_batch(
        ["D:/音声/RJ00000000", "D:/另一个.flac"],
        outdir="D:/输出",          # 省略则各自写在源文件旁边
        quality="vbr0",            # cbr320(默认) / vbr0 / vbr2 / cbr192
        jobs=8,                    # 并行数，默认 min(4, CPU核数)
        on_progress=lambda i, n, r: print(f"[{i}/{n}] {r.src.name}"),
    )
    s = summarize(results)
    print(s["done"], s["failed"], s["src_bytes"], s["dst_bytes"])

    # 4) 只看信息，不转码
    info = probe("input.wav")
    print(info.sample_rate, info.channels, info.duration)

    # 5) 只收集待处理文件
    files = find_audio(["D:/音声"], recursive=True)

================================================================================
作为命令行使用
================================================================================

    python wav2mp3.py "D:/音声/RJ00000000"
    python wav2mp3.py a.wav b.flac -o D:/out -q vbr0 -j 8
    python wav2mp3.py "D:/音声" -n                  # dry-run

================================================================================
设计要点（都是踩过坑才这么定的）
================================================================================

* **保持采样率与声道，绝不 downmix** —— 音声/ASMR 多为バイノーラル或ダミヘ录音，
  立体声里编码的是空间方位信息，压成单声道会把整个 3D 声场拍平。
* **已是 MP3 的默认跳过** —— 重新编码只会劣化，不会提升。见 AUDIO_EXTS 的说明。
* **ID3v2.3** —— 默认的 v2.4 在 Windows 资源管理器、老播放器、车机上对
  中日文标签支持很差，会显示乱码；v2.3 兼容性最好。
* **不覆盖已有产物**（除非 overwrite=True）—— 批量任务中断后重跑不会重做已完成部分。
* **采样率处理**：在 MP3 支持表内（8/11.025/12/16/22.05/24/32/44.1/48 kHz）原样保留；
  超出（如 96k）才被 ffmpeg 重采样到最接近的有效值，并在 result.notes 里标注。
* **低采样率码率封顶**：≤24 kHz 属 MPEG-2 档位，码率上限 160k，
  LAME 会静默封顶，result.notes 里会写出来。
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "AudioInfo", "ConvertResult", "AudioError", "FFmpegNotFound",
    "QUALITY_PRESETS", "AUDIO_EXTS", "MP3_SAMPLE_RATES",
    "probe", "to_mp3", "to_mp3_batch", "find_audio", "summarize",
    "human", "check_ffmpeg", "set_ffmpeg_path",
]

log = logging.getLogger("wav2mp3")

PathLike = Union[str, os.PathLike]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 质量预设 -> (ffmpeg 参数, 名义码率 kbps, 说明)。VBR 的名义码率填 0。
QUALITY_PRESETS: Dict[str, Tuple[List[str], int, str]] = {
    "cbr320": (["-b:a", "320k"], 320, "CBR 320k（MP3 最高码率，最保险）"),
    "vbr0":   (["-q:a", "0"],      0, "VBR 最高质量，体积比 320k 小约 14%"),
    "vbr2":   (["-q:a", "2"],      0, "VBR ≈190k，体积更小"),
    "cbr192": (["-b:a", "192k"], 192, "CBR 192k，最省体积"),
}

#: 默认处理的扩展名。**故意不含 mp3** —— MP3 转 MP3 只会劣化。
#: 需要重新编码 MP3 时请显式传入 exts 参数。
AUDIO_EXTS: Tuple[str, ...] = (
    "wav", "flac", "aiff", "aif", "aifc", "m4a", "alac", "ogg", "opus", "wma", "ape", "wv",
)

#: MP3 支持的采样率，按 MPEG 版本分档（每档码率上限不同）：
#:   MPEG-1 (32/44.1/48k) 最高 320k ｜ MPEG-2 (16/22.05/24k) 最高 160k ｜ MPEG-2.5 (8/11.025/12k) 最高 64k
MP3_SAMPLE_RATES: Tuple[int, ...] = (48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000)

_FFMPEG = "ffmpeg"
_FFPROBE = "ffprobe"


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class AudioError(Exception):
    """单个文件处理失败。to_mp3 不抛它，而是放进 ConvertResult.error；
    to_mp3_batch 同样。想抛异常请调 result.raise_for_error()。"""


class FFmpegNotFound(AudioError):
    """找不到 ffmpeg / ffprobe。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class AudioInfo:
    """一个音频文件的流信息。"""
    path: Path
    sample_rate: int = 0
    channels: int = 0
    codec: str = ""
    duration: float = 0.0
    bit_rate: int = 0

    @property
    def is_lossless(self) -> bool:
        return self.codec.lower() in ("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_u8",
                                      "flac", "alac", "aiff", "wavpack", "ape")

    def __str__(self) -> str:
        return (f"{self.path.name}: {self.codec} {self.sample_rate}Hz "
                f"{self.channels}ch {self.duration:.1f}s")


@dataclass
class ConvertResult:
    """一次转换的结果。所有字段都可安全读取，失败时 ok=False 且 error 有说明。"""
    src: Path
    dst: Path
    ok: bool = False
    skipped: bool = False              # 目标已存在且未开 overwrite
    error: str = ""
    notes: List[str] = field(default_factory=list)   # resampled:... / capped:...
    src_bytes: int = 0
    dst_bytes: int = 0
    elapsed: float = 0.0
    src_info: Optional[AudioInfo] = None

    @property
    def ratio(self) -> float:
        """源体积 / 产物体积。未成功或产物为空时返回 0。"""
        return self.src_bytes / self.dst_bytes if self.dst_bytes else 0.0

    @property
    def resampled(self) -> bool:
        return any(n.startswith("resampled") for n in self.notes)

    @property
    def bitrate_capped(self) -> bool:
        return any(n.startswith("capped") for n in self.notes)

    def raise_for_error(self) -> "ConvertResult":
        """失败时抛 AudioError，方便在单文件场景下用 try/except。"""
        if not self.ok:
            raise AudioError(f"{self.src}: {self.error}")
        return self

    def __str__(self) -> str:
        if self.skipped:
            return f"跳过 {self.src.name}（已存在）"
        if not self.ok:
            return f"失败 {self.src.name}: {self.error}"
        return (f"{self.src.name} → {self.dst.name}  "
                f"{human(self.src_bytes)} → {human(self.dst_bytes)} (1/{self.ratio:.1f})")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def human(n: float) -> str:
    """1536 -> '1.5 KB'"""
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def set_ffmpeg_path(ffmpeg: PathLike, ffprobe: Optional[PathLike] = None) -> None:
    """指定 ffmpeg / ffprobe 的位置（不在 PATH 里时用）。"""
    global _FFMPEG, _FFPROBE
    _FFMPEG = str(ffmpeg)
    _FFPROBE = str(ffprobe) if ffprobe else str(Path(ffmpeg).with_name(
        "ffprobe" + (".exe" if str(ffmpeg).lower().endswith(".exe") else "")))


def check_ffmpeg() -> Tuple[str, str]:
    """确认 ffmpeg / ffprobe 可用，返回它们的路径。不可用抛 FFmpegNotFound。"""
    ff = shutil.which(_FFMPEG) or (str(_FFMPEG) if Path(_FFMPEG).is_file() else None)
    fp = shutil.which(_FFPROBE) or (str(_FFPROBE) if Path(_FFPROBE).is_file() else None)
    if not ff or not fp:
        raise FFmpegNotFound(
            "找不到 ffmpeg / ffprobe，请安装并加入 PATH，"
            "或用 set_ffmpeg_path() 指定完整路径")
    return ff, fp


def probe(path: PathLike) -> Optional[AudioInfo]:
    """读音频流信息。读不出（非音频、文件损坏）返回 None，不抛异常。"""
    p = Path(path)
    try:
        r = subprocess.run(
            [_FFPROBE, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels,codec_name,bit_rate",
             "-show_entries", "format=duration,bit_rate",
             "-of", "default=nw=1", str(p)],
            capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    d = dict(l.split("=", 1) for l in r.stdout.strip().splitlines() if "=" in l)

    def _int(k: str) -> int:
        try:
            return int(float(d.get(k) or 0))
        except ValueError:
            return 0

    info = AudioInfo(path=p, sample_rate=_int("sample_rate"), channels=_int("channels"),
                     codec=d.get("codec_name", ""), duration=0.0,
                     bit_rate=_int("bit_rate") or _int("bit_rate"))
    try:
        info.duration = float(d.get("duration") or 0)
    except ValueError:
        info.duration = 0.0
    return info


def find_audio(paths: Union[PathLike, Sequence[PathLike]], *,
               recursive: bool = True,
               exts: Sequence[str] = AUDIO_EXTS) -> List[Path]:
    """把「文件 / 目录 / 混合列表」摊平成待处理文件列表（已去重、已排序）。

    >>> find_audio("D:/音声", exts=("wav",))
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    want = {e.lower().lstrip(".") for e in exts}
    out: List[Path] = []
    seen = set()
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_file():
            cand = [p] if p.suffix.lower().lstrip(".") in want else []
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            cand = [f for f in it if f.is_file() and f.suffix.lower().lstrip(".") in want]
        else:
            log.warning("路径不存在，已跳过：%s", raw)
            continue
        for f in cand:
            k = str(f.resolve()).lower()
            if k not in seen:
                seen.add(k)
                out.append(f)
    return sorted(out)


# ---------------------------------------------------------------------------
# 核心：单个文件转换
# ---------------------------------------------------------------------------
def to_mp3(src: PathLike, dst: Optional[PathLike] = None, *,
           quality: str = "cbr320",
           overwrite: bool = False,
           tags: bool = True,
           title: Optional[str] = None,
           timeout: float = 3600) -> ConvertResult:
    """把一个音频文件转成 MP3。

    参数
    ----
    src        源文件
    dst        目标路径；省略则写在源文件旁边、同名 .mp3
    quality    QUALITY_PRESETS 里的键，默认 "cbr320"
    overwrite  目标已存在时是否覆盖，默认 False（跳过）
    tags       是否继承源文件标签（-map_metadata 0）
    title      ID3 标题；省略则用源文件名
    timeout    单个文件的转码超时（秒）

    返回
    ----
    ConvertResult —— **不抛异常**，失败信息在 .error 里。想抛就调 .raise_for_error()。

    说明
    ----
    采样率与声道原样保留（绝不 downmix）；已是 MP3 的源请勿调用本函数。
    产物采样率被降级或码率被封顶时，会在 .notes 里标注。
    """
    s = Path(src)
    d = Path(dst) if dst is not None else s.with_suffix(".mp3")
    res = ConvertResult(src=s, dst=d)
    if quality not in QUALITY_PRESETS:
        res.error = f"未知质量预设 {quality!r}，可选：{', '.join(QUALITY_PRESETS)}"
        return res
    if not s.is_file():
        res.error = "源文件不存在"
        return res

    res.src_bytes = s.stat().st_size
    if d.exists() and not overwrite:
        res.ok = True
        res.skipped = True
        res.dst_bytes = d.stat().st_size
        return res

    q_opts, nominal, _ = QUALITY_PRESETS[quality]
    res.src_info = probe(s)

    try:
        check_ffmpeg()
    except FFmpegNotFound as e:
        res.error = str(e)
        return res

    try:
        d.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        res.error = f"无法创建输出目录：{e}"
        return res

    cmd = [_FFMPEG, "-v", "error", "-y", "-i", str(s),
           "-c:a", "libmp3lame", *q_opts,
           "-id3v2_version", "3",              # 兼容性最好的 ID3 版本
           "-metadata", f"title={title if title is not None else s.stem}"]
    if tags:
        cmd += ["-map_metadata", "0"]          # 继承源标签
    cmd.append(str(d))

    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        res.error = f"转码超时（>{timeout:.0f}s）"
        return res
    except OSError as e:
        res.error = f"无法执行 ffmpeg：{e}"
        return res
    res.elapsed = time.time() - t0

    if r.returncode != 0:
        res.error = r.stderr.decode("utf-8", "replace").strip()[:200] or "ffmpeg 返回非零"
        return res
    if not d.exists() or d.stat().st_size == 0:
        res.error = "产物为空"
        return res

    res.ok = True
    res.dst_bytes = d.stat().st_size

    # 跟源对比，标注格式层面的降级（不是错误，但调用方有权知道）
    if res.src_info and res.src_info.sample_rate:
        out = probe(d)
        out_sr = out.sample_rate if out else 0
        if out_sr and out_sr != res.src_info.sample_rate:
            res.notes.append(f"resampled:{res.src_info.sample_rate}->{out_sr}")
        elif nominal and out_sr and out_sr <= 24000 and nominal > 160:
            res.notes.append(f"capped:{nominal}->160")   # MPEG-2 档位上限
    return res


# ---------------------------------------------------------------------------
# 核心：批量转换
# ---------------------------------------------------------------------------
def to_mp3_batch(inputs: Union[PathLike, Sequence[PathLike]], *,
                 outdir: Optional[PathLike] = None,
                 quality: str = "cbr320",
                 jobs: int = 0,
                 overwrite: bool = False,
                 tags: bool = True,
                 recursive: bool = True,
                 exts: Sequence[str] = AUDIO_EXTS,
                 dry_run: bool = False,
                 on_progress: Optional[Callable[[int, int, ConvertResult], None]] = None,
                 ) -> List[ConvertResult]:
    """批量转换。输入可以是文件、目录，或两者的混合。

    参数
    ----
    inputs      文件/目录，或它们的列表
    outdir      输出目录；省略则各自写在源文件旁边。给了 outdir 时会**保持相对结构**
                （以第一个输入目录为基准），避免不同子目录的同名文件互相覆盖
    jobs        并行数；0 = min(4, CPU 核数)
    dry_run     True 时只算出目标路径，不实际转码（结果里 ok=True、notes 含 "dry-run"）
    on_progress 回调 (已完成数, 总数, ConvertResult)，按**完成顺序**调用；
                返回的列表则严格按**输入顺序**

    返回
    ----
    List[ConvertResult]，与 find_audio 的输出顺序一一对应。
    """
    files = find_audio(inputs, recursive=recursive, exts=exts)
    if not files:
        return []

    n_jobs = jobs or min(4, os.cpu_count() or 1)
    base = Path(inputs[0]).expanduser() if isinstance(inputs, (list, tuple)) and inputs \
        else (Path(inputs).expanduser() if isinstance(inputs, (str, os.PathLike)) else None)

    def dst_of(f: Path) -> Path:
        if not outdir:
            return f.with_suffix(".mp3")
        if base and base.is_dir():
            try:
                return Path(outdir) / f.relative_to(base).with_suffix(".mp3")
            except ValueError:
                pass
        return Path(outdir) / f.with_suffix(".mp3").name

    if dry_run:
        out = []
        for f in files:
            d = dst_of(f)
            res = ConvertResult(src=f, dst=d, ok=True, skipped=d.exists() and not overwrite,
                                notes=["dry-run"], src_bytes=f.stat().st_size,
                                dst_bytes=d.stat().st_size if d.exists() else 0)
            out.append(res)
        if on_progress:
            for i, r in enumerate(out, 1):
                on_progress(i, len(out), r)
        return out

    # 预分配结果槽位，保证返回顺序 = 输入顺序
    slots: List[Optional[ConvertResult]] = [None] * len(files)
    done = 0

    def work(idx: int) -> Tuple[int, ConvertResult]:
        return idx, to_mp3(files[idx], dst_of(files[idx]), quality=quality,
                           overwrite=overwrite, tags=tags)

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        for idx, res in ex.map(work, range(len(files))):
            slots[idx] = res
            done += 1
            if on_progress:
                on_progress(done, len(files), res)

    return [r for r in slots if r is not None]


def summarize(results: Iterable[ConvertResult]) -> Dict[str, object]:
    """把一批结果汇总成统计字典，方便打印或上报。

    返回键：total / done / skipped / failed / src_bytes / dst_bytes /
            ratio / elapsed / resampled / capped / errors
    """
    rs = list(results)
    done = [r for r in rs if r.ok and not r.skipped]
    src_b = sum(r.src_bytes for r in done)
    dst_b = sum(r.dst_bytes for r in done)
    return {
        "total": len(rs),
        "done": len(done),
        "skipped": sum(1 for r in rs if r.skipped),
        "failed": sum(1 for r in rs if not r.ok),
        "src_bytes": src_b,
        "dst_bytes": dst_b,
        "ratio": src_b / dst_b if dst_b else 0.0,
        "elapsed": sum(r.elapsed for r in rs),
        "resampled": [str(r.src) for r in rs if r.resampled],
        "capped": [str(r.src) for r in rs if r.bitrate_capped],
        "errors": [(str(r.src), r.error) for r in rs if not r.ok],
    }


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="wav2mp3",
        description="批量把音频转成高品质 MP3（本文件同时是可 import 的库）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<8} {v[2]}" for k, v in QUALITY_PRESETS.items()))
    ap.add_argument("inputs", nargs="+", help="音频文件或文件夹（可多个）")
    ap.add_argument("-o", "--outdir", help="输出目录（默认与源文件同目录）")
    ap.add_argument("-q", "--quality", default="cbr320", choices=list(QUALITY_PRESETS),
                    help="编码模式（默认 cbr320）")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="并行数（默认 min(4, CPU 核数)）")
    ap.add_argument("--ext", default=",".join(AUDIO_EXTS),
                    help="要处理的扩展名，逗号分隔")
    ap.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    ap.add_argument("-f", "--force", action="store_true", help="覆盖已存在的 MP3")
    ap.add_argument("--no-tags", action="store_true", help="不继承源文件标签")
    ap.add_argument("-n", "--dry-run", action="store_true", help="只列出计划，不实际转码")
    ap.add_argument("--delete-source", action="store_true",
                    help="转码成功后删除源文件（危险，建议先 --dry-run）")
    ap.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    try:
        check_ffmpeg()
    except FFmpegNotFound as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    files = find_audio(a.inputs, recursive=not a.no_recursive,
                       exts=[x for x in a.ext.split(",") if x.strip()])
    if not files:
        print("没有找到可处理的文件。")
        return 1

    _, _, q_desc = QUALITY_PRESETS[a.quality]
    print(f"编码模式 : {a.quality} —— {q_desc}")
    print(f"待处理   : {len(files)} 个文件")
    print(f"输出位置 : {a.outdir or '（与源文件同目录）'}")
    print(f"并行数   : {a.jobs or min(4, os.cpu_count() or 1)}")

    if a.dry_run:
        print("\n[dry-run] 只列出计划，不实际转码：\n")
        rs = to_mp3_batch(a.inputs, outdir=a.outdir, quality=a.quality,
                          recursive=not a.no_recursive, dry_run=True,
                          exts=[x for x in a.ext.split(",") if x.strip()])
        for r in rs:
            mark = "已存在，将跳过" if r.skipped else "将转码"
            print(f"  {r.src}  →  {r.dst}    [{mark}]")
        return 0

    print()
    t0 = time.time()

    def show(i: int, n: int, r: ConvertResult) -> None:
        flag = "·" if r.skipped else ("✓" if r.ok else "✗")
        print(f"  [{i}/{n}] {flag} {r.src.name}", flush=True)

    rs = to_mp3_batch(a.inputs, outdir=a.outdir, quality=a.quality, jobs=a.jobs,
                      overwrite=a.force, tags=not a.no_tags,
                      recursive=not a.no_recursive, on_progress=show,
                      exts=[x for x in a.ext.split(",") if x.strip()])
    s = summarize(rs)
    el = time.time() - t0

    print("\n" + "=" * 66)
    print(f"  完成 {s['done']}   跳过 {s['skipped']}   失败 {s['failed']}   耗时 {el:.1f}s")
    if s["done"]:
        print(f"  体积 {human(s['src_bytes'])} → {human(s['dst_bytes'])}"
              f"   （压到 1/{s['ratio']:.1f}）")
    if s["resampled"]:
        print(f"\n  ⚠ {len(s['resampled'])} 个文件的采样率不在 MP3 支持范围内，已重采样：")
        print("    （MP3 只支持 8/11.025/12/16/22.05/24/32/44.1/48 kHz；")
        print("      48kHz 的奈奎斯特频率是 24kHz，已完整覆盖人耳可听范围）")
        for n in s["resampled"][:5]:
            print(f"      {n}")
    if s["capped"]:
        print(f"\n  ℹ {len(s['capped'])} 个文件是低采样率（≤24kHz），属 MPEG-2 档位，")
        print("    码率上限 160k，LAME 已自动封顶（这不是错误）")
    if s["errors"]:
        print("\n  失败清单：")
        for name, why in s["errors"][:10]:
            print(f"    {name}: {why}")
        if len(s["errors"]) > 10:
            print(f"    …还有 {len(s['errors'])-10} 个")

    if a.delete_source and s["done"]:
        n = 0
        for r in rs:
            if r.ok and not r.skipped and r.dst.exists() and r.dst.stat().st_size > 0:
                try:
                    r.src.unlink()
                    n += 1
                except OSError as e:
                    print(f"    删除失败 {r.src.name}: {e}")
        print(f"\n  已删除 {n} 个源文件")

    return 0 if not s["failed"] else 1


if __name__ == "__main__":
    sys.exit(_cli())
