#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onsei2lrc WebUI —— 拖拽导入音频/压缩包，一键转写+翻译，导出 LRC 或 LRC+音频压缩包

启动：
    python webui.py            # 默认 http://127.0.0.1:7860
    python webui.py --port 8000

功能：
  * 拖拽/选择 音频文件（wav/mp3/m4a/flac…）或 zip 压缩包（自动解包，兼容日文 CP932 文件名）
  * 调用 onsei2lrc.py 做「静音+能量混合切块 → anime-whisper 日语转写 → Sakura 翻译」
  * 输出：LRC 压缩包 / LRC+音频 压缩包（LRC 与音频同名，播放器可自动加载）
  * WAV → MP3 320kbps 转换（可单独使用，也可与翻译一起做）
  * 实时日志与进度、健康检查（ffmpeg / ASR 模型 / 翻译服务）

只在本机 127.0.0.1 监听，不上传任何数据到外部。
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# --------------------------------------------------------------------------------------
# 常量与全局状态
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "onsei2lrc.py"
RUNS_DIR = ROOT / "webui_runs"
RUNS_DIR.mkdir(exist_ok=True)
# 个人术语表：存在就填进文本框当默认内容（空表 = 不启用术语表，会让术语一致性收益丢掉）。
# 本仓库不附带成品术语表——内容取决于素材，见 glossary_template.txt。
GLOSSARY_DEFAULT = ROOT / "glossary_common.txt"

log = logging.getLogger("onsei2lrc.webui")

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}
ARCHIVE_EXTS = {".zip"}
LRC_SUFFIX = {"zh": ".lrc", "ja": ".ja.lrc", "both": ".zh-ja.lrc", "inline": ".inline.lrc"}

ASR_MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--quantumcookie--anime-whisper-ct2-int8"

app = FastAPI(title="onsei2lrc WebUI")
RUNS: Dict[str, dict] = {}
RUNS_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------------------

def log_to(st: dict, msg: str) -> None:
    """写日志。时间戳在这里统一加 —— 流水线子进程的输出只有内容没有时刻，
    事后想回答「哪一步慢、等了多久」就只能靠它。

    多行消息按行拆开，每行各自带前缀；空行保留（当分隔用），不加前缀。
    """
    ts = time.strftime("[%H:%M:%S] ")
    parts = msg.rstrip("\n").split("\n")
    with RUNS_LOCK:
        for ln in parts:
            st["log"].append((ts + ln) if ln.strip() else "")
        if len(st["log"]) > 5000:            # 丢弃最旧的 1000 行，但用 offset 保住绝对行号，
            del st["log"][:1000]             # 否则前端传上来的 since 会错位
            st["log_offset"] = st.get("log_offset", 0) + 1000


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def audio_duration(p: Path) -> float:
    """用 ffprobe 取时长（毫秒级开销），用于按时长加权算总进度。"""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(p)],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


# 当前文件内部各阶段占的比重（用于把「转写 40/338 块」折算成文件完成度）
STAGE_RANGE = {
    "准备中": (0.00, 0.02),
    "转写": (0.02, 0.58),
    "翻译": (0.58, 0.94),
    "收尾": (0.94, 1.00),
    "转码": (0.00, 1.00),
}


def update_progress(st: dict, line: str) -> None:
    """从流水线输出里解析出「当前文件 / 当前阶段 / 阶段内进度」。"""
    p = st["progress"]
    m = re.match(r"^===== \[(\d+)/(\d+)\]\s+(.+?)\s*=====$", line)
    if m:
        idx = int(m.group(1))
        p["file_index"] = idx
        p["file_name"] = Path(m.group(3)).name
        p["files_total"] = int(m.group(2))
        p["file_stage"], p["stage_frac"], p["stage_detail"] = "准备中", 0.0, "加载音频"
        p["done_dur"] = sum(st.get("durations", [])[: idx - 1])
        return
    m = re.search(r"…(\d+)/(\d+) 块", line)
    if m:
        p["file_stage"] = "转写"
        p["stage_frac"] = int(m.group(1)) / max(1, int(m.group(2)))
        p["stage_detail"] = f"转写 {m.group(1)}/{m.group(2)} 块"
        return
    if "[ASR] 加载模型" in line:
        p["file_stage"], p["stage_frac"] = "准备中", 0.0
        p["stage_detail"] = "加载 ASR 模型（首次 5~10 秒）"
        return
    if "[ASR] 复用已加载的模型" in line:
        p["file_stage"], p["stage_frac"] = "准备中", 0.0
        p["stage_detail"] = "复用已加载模型"
        return
    if "[混合切块]" in line or "[VAD] 采用" in line or "[能量切块]" in line:
        p["file_stage"], p["stage_frac"] = "准备中", 0.01
        p["stage_detail"] = "切分语音块"
        return
    if "[ASR] 完成" in line:
        p["file_stage"], p["stage_frac"], p["stage_detail"] = "翻译", 0.0, "转写完成，准备翻译"
        return
    m = re.search(r"…(\d+)/(\d+) 批完成", line)
    if m:
        p["file_stage"] = "翻译"
        p["stage_frac"] = int(m.group(1)) / max(1, int(m.group(2)))
        p["stage_detail"] = f"翻译 {m.group(1)}/{m.group(2)} 批"
        return
    if "[翻译] 完成" in line:
        p["file_stage"], p["stage_frac"], p["stage_detail"] = "收尾", 1.0, "写入字幕"
    elif "[转码]" in line:
        p["file_stage"], p["stage_frac"], p["stage_detail"] = "转码", 0.5, "转 MP3 320k"


def progress_view(st: dict) -> dict:
    """把原始进度折算成两个 0~1 的百分比（当前文件 / 总任务）。"""
    p = dict(st["progress"])
    lo, hi = STAGE_RANGE.get(p.get("file_stage", ""), (0.0, 1.0))
    file_frac = lo + (hi - lo) * float(p.get("stage_frac", 0.0))
    durs = st.get("durations") or []
    total_dur = sum(durs)
    done_dur = float(p.get("done_dur", 0.0))
    idx = int(p.get("file_index", 0))
    cur_dur = durs[idx - 1] if 0 < idx <= len(durs) else 0.0
    if total_dur > 0:
        task_frac = min(1.0, (done_dur + cur_dur * file_frac) / total_dur)
    else:                                    # 拿不到时长就按文件个数算
        task_frac = (p.get("files_done", 0) + file_frac) / max(1, p.get("files_total", 1))
    el = time.time() - st["t0"] if st.get("t0") else 0.0
    eta = el / task_frac * (1 - task_frac) if task_frac > 0.03 else None
    p.update(file_frac=round(file_frac, 4), task_frac=round(task_frac, 4),
             elapsed=round(el, 1), eta=round(eta, 1) if eta else None,
             total_dur=round(total_dur, 1))
    return p


def extract_archive(zpath: Path, dest: Path, st: dict) -> int:
    """解包 zip。日文同人作品的 zip 常见「非 UTF-8 标志 + CP932 文件名」，
    Python 默认按 CP437 解会变乱码，这里手动按 CP932/GBK 回退解码。"""
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = info.filename
            if not (info.flag_bits & 0x800):          # 没有 UTF-8 标志位
                for enc in ("cp932", "gbk", "utf-8"):
                    try:
                        name = name.encode("cp437").decode(enc)
                        break
                    except Exception:
                        continue
            name = name.replace("\\", "/")
            if name.startswith("__MACOSX") or Path(name).name in (".DS_Store", "Thumbs.db"):
                continue
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest.resolve())):   # 防路径穿越
                log_to(st, f"  ⚠ 跳过可疑路径: {name}")
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def collect_audio(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def count_audio_bounded(root: Path, cap: int = 30000, budget: float = 0.45) -> int:
    """有上限的递归音频计数。

    双上限：既限制遍历条目数，也限制**耗时**。只限条目数是不够的——
    实测在 C:\\ 上遍历 3 万个条目就要 5 秒，整个接口被拖到不可用。
    这里每 256 个条目检查一次时间，超预算就停（返回的是下界，但对「这个文件夹
    里有没有音频」这个用途足够了）。
    """
    n = 0
    t0 = time.time()
    try:
        for i, p in enumerate(root.rglob("*")):
            if i > cap:
                break
            if (i & 255) == 0 and time.time() - t0 > budget:
                break
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                n += 1
    except (PermissionError, OSError):
        pass
    return n


# 目录浏览结果缓存：来回点同一层时不必重复扫盘。
# key = 路径，value = (写入时间, 结果)。TTL 短一点，避免用户新建文件夹后看不到。
_BROWSE_CACHE: Dict[str, tuple] = {}
_BROWSE_TTL = 45.0
_BROWSE_LOCK = threading.Lock()


def list_drives() -> List[str]:
    import string
    out = []
    for c in string.ascii_uppercase:
        d = Path(f"{c}:\\")
        try:
            if d.exists():
                out.append(str(d))
        except OSError:
            pass
    return out


def browse_dir(raw: str) -> dict:
    """列出盘符或某目录下的子目录（含各自的音频文件数）。结果带缓存。"""
    key = raw.strip()
    now = time.time()
    with _BROWSE_LOCK:
        hit = _BROWSE_CACHE.get(key)
        if hit and now - hit[0] < _BROWSE_TTL:
            return hit[1]

    res = _browse_dir_uncached(raw)

    with _BROWSE_LOCK:
        _BROWSE_CACHE[key] = (now, res)
        if len(_BROWSE_CACHE) > 512:                 # 简单的容量控制
            for k in sorted(_BROWSE_CACHE, key=lambda k: _BROWSE_CACHE[k][0])[:128]:
                _BROWSE_CACHE.pop(k, None)
    return res


def _browse_dir_uncached(raw: str) -> dict:
    if not raw.strip():
        dirs = []
        for d in list_drives():
            # 盘符的递归计数最贵（等于扫整个盘），预算给得更紧
            dirs.append({"name": d, "path": d,
                         "audio": count_audio_bounded(Path(d), 4000, 0.30)})
        return {"path": "", "parent": "", "is_root": True, "dirs": dirs,
                "audio": 0, "audio_recursive": 0, "exists": True}

    p = Path(raw).expanduser()
    if not p.exists() or not p.is_dir():
        return {"path": str(p), "parent": str(p.parent), "is_root": False, "dirs": [],
                "audio": 0, "audio_recursive": 0, "exists": False,
                "error": "路径不存在或不是文件夹"}

    dirs = []
    try:
        for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            try:
                if not c.is_dir() or c.name.startswith("."):
                    continue
                n = sum(1 for f in c.iterdir()
                        if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
                dirs.append({"name": c.name, "path": str(c), "audio": n})
            except (PermissionError, OSError):
                dirs.append({"name": c.name, "path": str(c), "audio": -1})
    except (PermissionError, OSError) as e:
        return {"path": str(p), "parent": str(p.parent), "is_root": False, "dirs": [],
                "audio": 0, "audio_recursive": 0, "exists": True, "error": f"无权限读取：{e}"}

    direct = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
    return {"path": str(p), "parent": str(p.parent), "is_root": False, "dirs": dirs,
            "audio": direct, "audio_recursive": count_audio_bounded(p), "exists": True}


def build_cmd(inp: Path, out: Path, o: dict) -> List[str]:
    cmd = [sys.executable, str(PIPELINE), str(inp), "--outdir", str(out),
           "--lrc-mode", o["lrc_mode"], "--chunk-mode", o["chunk_mode"],
           "--max-chars", str(o["max_chars"]), "--merge-gap", str(o["merge_gap"]),
           "--batch-size", str(o["batch_size"]),
           "--workers", str(o.get("workers") or 4),
           "--prompt-style", o.get("prompt_style") or "v1"]
    for key, flag in (("temperature", "--temperature"), ("top_p", "--top-p"),
                      ("frequency_penalty", "--frequency-penalty"), ("max_tokens", "--max-tokens")):
        v = o.get(key)
        if v is not None and v != "":
            cmd += [flag, str(v)]
    # .ja.lrc 恒定生成（很便宜，只是把缓存里的原文再写一份），
    # 要不要放进下载包由下载区的「附带日文 lrc」勾选决定
    cmd += ["--keep-ja"]
    # 断点继续：复用 .segments.json 里已有的转写与译文，只补没做完的部分
    if o.get("resume"):
        cmd += ["--retranslate", "--reuse-translation"]
    if o.get("glossary_path"):
        cmd += ["--glossary", o["glossary_path"]]
    if o.get("fix_path"):
        cmd += ["--fix-asr", o["fix_path"]]
    backend = o["backend"]
    model = o.get("model_name") or "sakura"
    # 显存编排交给流水线自己：翻译阶段它才加载 Sakura、译完（含文件名）再卸载。
    # 这样 ASR 阶段显存里只有 whisper，不会出现「两个模型同时驻留把 8 GB 撑爆 → 换页降频」。
    if backend == "lmstudio":
        cmd += ["--manage-backend", "--translate-names"]
    if backend == "lmstudio":
        cmd += ["--preset", "lmstudio", "--model-name", model]
    elif backend == "llamacpp":
        cmd += ["--preset", "sakura-llamacpp", "--model-name", model]
    elif backend == "ollama":
        cmd += ["--preset", "sakura-ollama", "--model-name", model]
    elif backend == "deepseek":
        cmd += ["--preset", "deepseek", "--model-name", model or "deepseek-chat"]
        if o.get("api_key"):
            cmd += ["--api-key", o["api_key"]]
    elif backend == "custom":
        cmd += ["--base-url", o["base_url"], "--model-name", model,
                "--protocol", o.get("protocol") or "json"]
        if o.get("api_key"):
            cmd += ["--api-key", o["api_key"]]
    return cmd


def convert_mp3(src: Path, dst: Path, st: dict) -> bool:
    """WAV → MP3 320kbps（保持采样率与声道数，绝不 downmix）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
           "-c:a", "libmp3lame", "-b:a", "320k", "-map_metadata", "0",
           "-id3v2_version", "3", "-metadata", f"title={src.stem}", str(dst)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        log_to(st, f"  ✗ 转码失败 {src.name}: {e.stderr.decode('utf-8', 'replace')[:200]}")
        return False


def make_zip(pairs, zpath: Path) -> None:
    """pairs: [(源文件, 压缩包内路径)]。直接读源文件写进 zip，不做暂存复制。"""
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in pairs:
            z.write(src, arc)


# --------------------------------------------------------------------------------------
# 文件名翻译（下载时的可选项）
#
# 实现已挪到 onsei2lrc.py：那样才能赶上「Sakura 还在显存里」的翻译阶段顺手做掉，
# 不必为了几个文件名再把 6 GB 模型加载一次。这里只留一个薄适配层给下载用。
# --------------------------------------------------------------------------------------

def translate_filenames(stems: List[str], o: dict, st: dict) -> dict:
    """把音频文件名（不含扩展名）翻译成中文。返回 {原名: 译名}。

    优先用流水线在翻译阶段顺手产出的 `<outdir>/_filenames.json`——
    那时候 Sakura 本来就在显存里，等于白捡；这里再去调模型反而要重新
    加载一次 6 GB 的模型。只有拿不到缓存（老任务 / 上传模式）才自己翻。
    """
    import onsei2lrc as P                                  # 顶层只有标准库，import 很快

    cached = P.filenames_json_path((st.get("dir") or Path(".")) / "output")
    if cached.is_file():
        try:
            nm = json.loads(cached.read_text(encoding="utf-8"))
            hit = {s: nm[s] for s in stems if s in nm}
            if hit:
                log_to(st, f"[文件名] 复用流水线译名 {len(hit)}/{len(stems)} 个（免去重新加载模型）")
                return hit
            log_to(st, "[文件名] 流水线译名对不上本次文件，改为现翻")
        except Exception as e:
            log_to(st, f"[文件名] 读译名缓存失败（{type(e).__name__}: {e}），改为现翻")

    preset = {"lmstudio": "lmstudio", "llamacpp": "sakura-llamacpp",
              "ollama": "sakura-ollama", "deepseek": "deepseek"}.get(o.get("backend"))
    cfg = dict(P.PRESETS.get(preset, {})) if preset else {}
    if o.get("base_url"):
        cfg["base_url"] = o["base_url"]
    if o.get("model_name"):
        cfg["model_name"] = o["model_name"]
    if o.get("api_key"):
        cfg["api_key"] = o["api_key"]
    cfg["glossary_text"] = o.get("glossary_text") or ""
    return P.translate_filenames(stems, cfg, log_cb=lambda m: log_to(st, m))


# --------------------------------------------------------------------------------------
# 任务状态持久化（断点继续）
#
# 每个任务把自身状态写到 webui_runs/<run_id>/task.json。服务重启后：
#   * 已完成/失败的 → 恢复出来，产出仍可下载
#   * 跑了一半被中断的 → 重新入队，并带 --retranslate --reuse-translation
#     跳过已完成的转写与翻译（.segments.json 就是天然的检查点）
# --------------------------------------------------------------------------------------

TASK_FILE = "task.json"


def _save_task(st: dict) -> None:
    """把任务状态落盘。只存可序列化的字段，proc/线程之类不能存。"""
    try:
        d = {
            "run_id": st.get("id"),
            "src_dir": st.get("src_dir") or "",
            "state": st.get("state"),
            "t0": st.get("t0"),
            "elapsed": st.get("elapsed") or 0,
            "error": st.get("error"),
            "resume": bool(st.get("resume")),
            "options": st.get("options") or {},
            "outputs": st.get("outputs") or [],
            "files": st.get("files") or [],
        }
        (st["dir"] / TASK_FILE).write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except (OSError, TypeError, ValueError) as e:
        log_to(st, f"[状态] 落盘失败（不影响本次运行）：{type(e).__name__}: {e}")


def _load_tasks() -> List[dict]:
    out = []
    for p in sorted(RUNS_DIR.glob("*/" + TASK_FILE)):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def _rebuild_pairs(st: dict) -> None:
    """从磁盘重建下载所需的文件清单。

    恢复任务时 `lrc_pairs` / `audio_src` 这些内存字段已经没了，但产物还在
    output/ 和源目录里，扫一遍就能重建，这样老任务的下载链接照样能用。
    """
    work = st["dir"]
    out = work / "output"
    lrc = []
    if out.is_dir():
        for p in sorted(out.rglob("*.lrc")):
            lrc.append((p, p.relative_to(out).as_posix()))
    audio_src: List[tuple] = []
    mp3_src: List[tuple] = []
    mp3_todo: List[tuple] = []
    extras: List[tuple] = []
    src = Path(st["src_dir"]) if st.get("src_dir") else None
    if src and src.is_dir():
        existing = {p for p in src.rglob("*")
                    if p.is_file() and p.suffix.lower() == ".mp3"}
        for a in collect_audio(src):
            rel = a.relative_to(src)
            arc = rel.as_posix()
            audio_src.append((a, arc))
            is_mp3 = a.suffix.lower() == ".mp3"
            twin = (a.parent / (a.stem + ".mp3")) if not is_mp3 else None
            if is_mp3:
                mp3_src.append((a, arc))
            elif twin is not None and twin in existing:
                mp3_src.append((twin, rel.with_suffix(".mp3").as_posix()))
            else:
                mp3_todo.append((a, rel.with_suffix(".mp3").as_posix()))
        # 封面/文档等：原样带上，不改名
        for f in sorted(src.rglob("*")):
            if f.is_file() and f.suffix.lower() not in AUDIO_EXTS and f.suffix.lower() != ".lrc":
                extras.append((f, f.relative_to(src).as_posix()))
    st["lrc_pairs"] = lrc
    st["audio_src"] = audio_src
    st["mp3_src"] = mp3_src
    st["mp3_todo"] = mp3_todo
    st["extras"] = extras
    st["zip_cache"] = {}
    st["name_map"] = None
    st["has_audio"] = bool(audio_src)
    st["need_mp3"] = bool(mp3_todo)


def _blank_progress() -> dict:
    return {"files_total": 0, "files_done": 0, "file_index": 0, "file_name": "",
            "file_stage": "", "stage_frac": 0.0, "stage_detail": "", "done_dur": 0.0}


def _restore_tasks() -> None:
    """服务启动时恢复上次的任务。"""
    n_done = n_resume = 0
    for d in _load_tasks():
        rid = d.get("run_id")
        if not rid or rid in RUNS:
            continue
        work = RUNS_DIR / rid
        if not work.is_dir():
            continue
        state = d.get("state") or "error"
        st = {
            "id": rid, "dir": work, "state": state, "log": [], "log_offset": 0,
            "t0": d.get("t0") or time.time(), "durations": [], "proc": None,
            "src_dir": d.get("src_dir") or "", "options": d.get("options") or {},
            "outputs": d.get("outputs") or [], "error": d.get("error"),
            "elapsed": d.get("elapsed") or 0, "files": d.get("files") or [],
            "progress": _blank_progress(), "resume": bool(d.get("resume")),
        }
        RUNS[rid] = st
        if state in ("done", "error"):
            _rebuild_pairs(st)
            log_to(st, "[恢复] 上次的任务已结束，产出仍可下载")
            n_done += 1
        else:
            # 跑了一半被中断 → 重新入队，并标记 resume 以跳过已完成的部分
            st["state"] = "queued"
            st["resume"] = True
            log_to(st, "[恢复] 上次被中断，已重新入队继续（跳过已完成的转写与翻译）")
            enqueue(rid)
            n_resume += 1
    if n_done or n_resume:
        log.info("恢复任务：%d 个已完成，%d 个继续", n_done, n_resume)


# --------------------------------------------------------------------------------------
# 任务队列
#
# 一个文件夹 = 一个任务。任务串行执行（GPU 只有一个，并发只会互相拖慢）。
# 队列是内存里的 run_id 列表 + 一个常驻工作线程轮询；支持从中间移除待跑任务。
# --------------------------------------------------------------------------------------

QUEUE: List[str] = []                    # 待跑的 run_id，FIFO
QUEUE_LOCK = threading.Lock()
_CURRENT: Optional[str] = None           # 正在跑的 run_id
_WAITING: Optional[str] = None           # 正卡在队首等翻译后端的 run_id
_WORKER: Optional[threading.Thread] = None


def _backend_base(o: dict) -> str:
    """解析翻译后端的 base_url：预设优先，其次用户自定义的那个。"""
    preset = {"lmstudio": "lmstudio", "llamacpp": "sakura-llamacpp",
              "ollama": "sakura-ollama", "deepseek": "deepseek"}.get(o.get("backend"))
    if preset:
        import onsei2lrc as P
        return o.get("base_url") or (P.PRESETS.get(preset) or {}).get("base_url") or ""
    return o.get("base_url") or ""


def _probe_backend(base: str, timeout: float = 2.0) -> bool:
    """端口探活：HTTP 200 就算服务在跑（不看模型有没有加载）。"""
    if not base:
        return False
    import httpx
    try:
        return httpx.get(base.rstrip("/") + "/models", timeout=timeout,
                         trust_env=False).status_code == 200
    except Exception:
        return False


def _backend_ready(o: dict, require_model: bool = True) -> Tuple[bool, str]:
    """确认翻译后端可用。

    require_model=False 只确认「服务在跑」——因为流水线现在自己管模型的
    加载/卸载（见 onsei2lrc.py 的 --manage-backend）：ASR 阶段 Sakura 本来
    就该是不在显存里的，这时候要求「模型已加载」会把任务永远卡在队首。
    """
    import httpx
    base = _backend_base(o)
    if not base:
        return True, ""                     # 判断不了就放行，别挡着
    try:
        r = httpx.get(base.rstrip("/") + "/models", timeout=2.0, trust_env=False)
    except Exception as e:
        return False, f"翻译后端连不上（{type(e).__name__}）"
    if r.status_code != 200:
        return False, f"翻译后端返回 HTTP {r.status_code}"
    if not require_model:
        return True, ""
    try:
        ids = [m.get("id") for m in (r.json().get("data") or [])]
    except Exception:
        return True, ""
    want = o.get("model_name")
    if want and ids and want not in ids:
        return False, f"后端在线但没加载模型 {want}（现有：{', '.join(ids[:3])}）"
    return True, ""


def _needs_model_ready(o: dict) -> bool:
    """这一跑是不是还需要「模型已经加载」才算就绪。

    LM Studio + --manage-backend 的跑法由流水线在翻译阶段自己 load，
    所以队首检查只确认服务在跑。其它后端（llama.cpp / Ollama / DeepSeek）
    是常驻的，仍然要检查模型。
    """
    return (o.get("backend") or "lmstudio") != "lmstudio"


# --------------------------------------------------------------------------------------
# 拉起翻译后端（LM Studio）
#
# 为什么要有这一块：LM Studio 的模型会被**外部**弄掉——空闲 TTL 自动 eject、
# 或者另一个项目跑 `lms load` 把当前模型挤下去（LM Studio 同一时刻只驻留一个）。
# 之前只会「等」，等待期间流水线照跑 ASR，每一轨都白烧几分钟到几十分钟。
# 现在改成「先自己拉起来，拉不动再等」。
#
# 实测注意：`lms` 只是客户端，它叫不醒 LM Studio 守护进程
# （会打印 "Waking up LM Studio service..." 然后 60 秒超时），
# 所以必须先把桌面程序拉起来，再 `lms server start` / `lms load`。
# --------------------------------------------------------------------------------------

# 标识符 → LM Studio 模型键。定义在 onsei2lrc.py（流水线 load 模型时也要用），
# 这里只做转发，别在两处各维护一份。
def _model_keys() -> dict:
    import onsei2lrc as P
    return dict(P.MODEL_KEY_HINTS)

AUTO_START_BACKEND = True       # 队列等待 / 运行中后端掉线时，自动尝试拉起
_BACKEND_COOLDOWN = 90.0        # 同一个任务两次自动拉起之间的最小间隔（秒）
_BACKEND_TRIES: Dict[str, float] = {}
_LOCATE: Dict[str, Optional[str]] = {"lms": None, "app": None, "lms_done": False, "app_done": False}

if os.name == "nt":
    _NO_WINDOW = subprocess.CREATE_NO_WINDOW
    _DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
else:
    _NO_WINDOW = 0
    _DETACHED = 0


def _find_lms() -> Optional[str]:
    """定位 lms 命令行。实现共用 onsei2lrc.py 的那份，避免两处各写一遍。

    流水线自己也要 load/unload 模型（--manage-backend），不能只有 WebUI 认识 lms。
    """
    import onsei2lrc as P
    return P.find_lms()


def _find_lmstudio_app() -> Optional[str]:
    """定位 LM Studio 桌面程序。

    不能写死安装路径——本机就装在 D:\\LM_Studio 而不是默认的 %LOCALAPPDATA%。
    优先读注册表卸载项里的 DisplayIcon（跟着用户实际装在哪），再退回常见位置。
    """
    if _LOCATE["app_done"]:
        return _LOCATE["app"]
    _LOCATE["app_done"] = True
    cands: List[str] = []
    if os.name == "nt":
        try:
            import winreg
            subs = ((winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"))
            for hive, sub in subs:
                try:
                    with winreg.OpenKey(hive, sub) as k:
                        for i in range(winreg.QueryInfoKey(k)[0]):
                            try:
                                with winreg.OpenKey(k, winreg.EnumKey(k, i)) as sk:
                                    if "LM Studio" not in str(winreg.QueryValueEx(sk, "DisplayName")[0]):
                                        continue
                                    icon = str(winreg.QueryValueEx(sk, "DisplayIcon")[0])
                                    cands.append(icon.split(",")[0].strip().strip('"'))
                            except OSError:
                                continue
                except OSError:
                    continue
        except Exception:
            pass
        for root in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles")):
            if root:
                cands += [str(Path(root) / "Programs" / "LM Studio" / "LM Studio.exe"),
                          str(Path(root) / "LM Studio" / "LM Studio.exe")]
    else:
        cands.append("/Applications/LM Studio.app/Contents/MacOS/LM Studio")
    for c in cands:
        try:
            if c and c.lower().endswith(".exe") and Path(c).is_file():
                _LOCATE["app"] = c
                return c
        except OSError:
            continue
    return None


def _run_lms(argv: List[str], timeout: float = 180.0) -> Tuple[int, str]:
    """跑一条 lms 命令。共用 onsei2lrc.py 的实现（流水线也要调 lms）。"""
    import onsei2lrc as P
    return P.run_lms(argv, timeout)


def _launch_app(path: str) -> str:
    """启动 LM Studio 桌面程序，返回实际用的方式（用于日志）。

    为什么不是一句 subprocess.Popen 就完事（实测踩到的坑）：
    如果本服务本身是被别的进程以 job object 托管起来的（被上层工具/沙箱拉起时
    很常见），Popen 出来的 LM Studio 会在 2 秒内**静默退出**——退出码 0 或 9，
    自己的 main.log 一行都不写，看起来就像「拉不起来」。
    同一个上下文里改用 WMI 的 Win32_Process.Create 就正常：WMI 由 WmiPrvSE 代建进程，
    不受父进程 job 约束。所以 Windows 上优先走 WMI，失败再退回 Popen。
    """
    if os.name == "nt":
        import base64
        ps = ('$r = ([wmiclass]"Win32_Process").Create(\'"%s"\'); exit $r.ReturnValue' % path)
        try:
            enc = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
            p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc],
                               capture_output=True, timeout=90, creationflags=_NO_WINDOW)
            if p.returncode == 0:
                return "WMI"
        except Exception:
            pass
    subprocess.Popen([path], cwd=str(Path(path).parent), creationflags=_DETACHED,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    return "Popen"


def start_backend(o: dict, log_cb=None, load_model: bool = True) -> Tuple[bool, str]:
    """把翻译后端拉起来（LM Studio）。

    顺序：已就绪 → 拉桌面程序 → lms server start →（可选）lms load --identifier → 等就绪。
    load_model=False 只保证「服务在跑」——流水线自己管模型的时候用这个，
    免得在 ASR 阶段就把 6 GB 的 Sakura 塞进显存，正好抵消分时复用的意义。
    全程不抛异常；每一步都通过 log_cb 汇报，失败也返回人话原因。
    """
    say = log_cb or (lambda _m: None)
    if (o.get("backend") or "lmstudio") != "lmstudio":
        return False, "只有 LM Studio 支持由本工具拉起（其它后端请自行常驻）"
    base = _backend_base(o)
    if not base:
        return False, "没有配置后端地址，无法拉起"

    if _probe_backend(base) and (not load_model or _backend_ready(o)[0]):
        return True, "后端已就绪"

    m = re.search(r":(\d+)", base)
    port = int(m.group(1)) if m else 1234

    # ① 先把 LM Studio 本体拉起来（lms 自己叫不醒守护进程）
    if not _probe_backend(base):
        app = _find_lmstudio_app()
        if not app:
            return False, "找不到 LM Studio 程序，请手动启动它（或确认已正确安装）"
        say(f"[后端] 启动 LM Studio：{app}")
        try:
            how = _launch_app(app)
            say(f"[后端] 已启动（{how}），等 HTTP 服务就绪…")
        except Exception as e:
            return False, f"启动 LM Studio 失败：{type(e).__name__}: {e}"
        say("[后端] 等待 HTTP 服务就绪…")
        for _ in range(45):                     # 冷启动实测 20~60 秒
            time.sleep(2)
            if _probe_backend(base):
                break
        else:
            return False, "LM Studio 起来了但 HTTP 服务没响应（在 LM Studio 里确认已开启开发者服务器）"

    # ② 让服务器监听（幂等；已经在跑会立刻返回）
    rc, out = _run_lms(["server", "start", "--port", str(port)], timeout=120)
    if rc != 0:
        say(f"[后端] lms server start 返回 {rc}：{out[:200]}")

    if not load_model:
        return True, "服务已就绪（模型由流水线在翻译阶段自己加载）"

    # ③ 加载模型
    ident = (o.get("model_name") or "").strip()
    if not ident:
        return False, "没填模型名，无法拉起"
    key = _model_keys().get(ident, ident)
    say(f"[后端] 加载模型 {key}（API 标识符 {ident}）…首次加载 6 GB 模型要 30~90 秒")
    rc, out = _run_lms(["load", key, "--gpu", "max", "--context-length", "8192",
                        "--identifier", ident, "-y"], timeout=420)
    if rc != 0:
        tail = out[-300:] if out else "(无输出)"
        return False, f"lms load 失败（返回码 {rc}）：{tail}"

    # ④ 等标识符真的出现在 /v1/models 上
    for _ in range(30):
        ok, why = _backend_ready(o)
        if ok:
            say(f"[后端] ✅ 就绪：{ident}")
            return True, f"已就绪（{ident}）"
        time.sleep(2)
        last = why
    return False, f"load 命令已返回，但 {ident} 仍未就绪：{last}"


def _ensure_worker() -> None:
    """确保队列工作线程在跑（只起一个）。"""
    global _WORKER
    if _WORKER is None or not _WORKER.is_alive():
        _WORKER = threading.Thread(target=_queue_worker, name="queue-worker", daemon=True)
        _WORKER.start()


def _queue_worker() -> None:
    """串行消费队列。空闲时轮询等待——比条件变量简单，且天然支持从中间移除。

    开跑前会检查翻译后端；不可用时把任务**留在队首**并等一会儿重试，
    而不是让它跑完 ASR 再失败（那要几十分钟到几小时）。
    """
    global _CURRENT, _WAITING
    while True:
        rid = None
        with QUEUE_LOCK:
            if QUEUE:
                rid = QUEUE[0]                    # 先看队首，不弹出
        if rid is None:
            time.sleep(0.4)
            continue
        st = RUNS.get(rid)
        if not st or st.get("state") != "queued":
            with QUEUE_LOCK:                      # 已被移除/取消
                if QUEUE and QUEUE[0] == rid:
                    QUEUE.pop(0)
            continue
        o = st.get("options") or {}
        # LM Studio 的跑法里模型由流水线在翻译阶段自己加载，队首只确认「服务在跑」；
        # 其它常驻后端仍然要求模型已加载。
        ok, why = _backend_ready(o, require_model=_needs_model_ready(o))
        if not ok:
            _WAITING = rid                        # 让前端能显示「等待翻译后端」
            if not st.get("_waiting"):
                st["_waiting"] = True
                log_to(st, f"[等待] {why} —— 先尝试自动拉起，失败则每 20 秒重试")
            # 自动拉起（带冷却）。这里只保证服务在跑，不预加载模型——
            # 模型由流水线在翻译阶段 load，ASR 阶段显存要留给 whisper。
            if AUTO_START_BACKEND and time.time() - _BACKEND_TRIES.get(rid, 0.0) > _BACKEND_COOLDOWN:
                _BACKEND_TRIES[rid] = time.time()
                log_to(st, "[后端] 尝试自动拉起翻译后端…")
                done, msg = start_backend(o, log_cb=lambda m: log_to(st, m), load_model=False)
                log_to(st, f"[后端] {'✅ ' if done else '❌ '}{msg}")
                if done:
                    continue                      # 立刻回到循环重新判一次
            time.sleep(20)
            continue
        st.pop("_waiting", None)
        _WAITING = None
        with QUEUE_LOCK:
            if QUEUE and QUEUE[0] == rid:
                QUEUE.pop(0)
            else:
                continue
        _CURRENT = rid
        try:
            threading.Thread(target=_backend_watchdog, args=(rid,), daemon=True).start()
            run_pipeline(rid)
        except Exception as e:            # run_pipeline 自己会兜异常，这里是双保险
            st["state"] = "error"
            st["error"] = f"{type(e).__name__}: {e}"
            log_to(st, f"\n[错误] {st['error']}")
        finally:
            _CURRENT = None


def _backend_watchdog(rid: str) -> None:
    """任务运行期间盯着翻译后端**服务**，掉线就自动拉起来。

    这一条是补 _backend_ready() 的缺口：那个检查只在**开跑前**做一次，
    后端要是跑到一半没了，流水线会把后面每一轨的 ASR 全跑完再在翻译那步失败
    （实测第 1 轨报 400 Model is unloaded，第 2~4 轨报 APIConnectionError，
    十几分钟 ASR 全白跑）。这里每 20 秒复查一次。

    注意只看「服务在不在」，不碰模型：LM Studio 的模型归流水线管
    （ASR 阶段故意不加载 Sakura，翻译阶段才 load）。老版本在这里主动 load 模型，
    正好会把分时复用废掉——ASR 跑到一半 Sakura 被塞进显存，又回到抢显存降频。
    """
    while _CURRENT == rid:
        time.sleep(20)
        if _CURRENT != rid:
            return
        st = RUNS.get(rid)
        if not st or st.get("state") != "running":
            return
        o = st.get("options") or {}
        if _probe_backend(_backend_base(o)):
            continue
        if not AUTO_START_BACKEND:
            if not st.get("_warned_down"):
                st["_warned_down"] = True
                log_to(st, "[后端] ⚠ 翻译后端服务掉线，自动拉起已关闭，请手动启动")
            continue
        if time.time() - _BACKEND_TRIES.get(rid, 0.0) < _BACKEND_COOLDOWN:
            continue
        _BACKEND_TRIES[rid] = time.time()
        log_to(st, "[后端] ⚠ 翻译后端服务掉线，自动拉起…")
        done, msg = start_backend(o, log_cb=lambda m: log_to(st, m), load_model=False)
        log_to(st, f"[后端] {'✅ ' if done else '❌ '}{msg}")


def enqueue(run_id: str) -> int:
    """把任务加入队列，返回它的排队位置（1 = 下一个就跑）。"""
    with QUEUE_LOCK:
        if run_id in QUEUE:
            return QUEUE.index(run_id) + 1
        QUEUE.append(run_id)
        pos = len(QUEUE)
    _ensure_worker()
    return pos


def dequeue(run_id: str) -> bool:
    """把还没开跑的任务从队列里拿掉。已在跑或已完成返回 False。"""
    with QUEUE_LOCK:
        if run_id in QUEUE:
            QUEUE.remove(run_id)
            return True
    return False


def queue_view() -> dict:
    """队列快照：待跑列表 + 当前正在跑（或正等后端）的任务 + 已结束的。"""
    # 卡在队首等后端时，那个任务已经报成 current 了，pending 里要排除，否则前端显示两遍
    waiting_id = _WAITING if not _CURRENT else None
    with QUEUE_LOCK:
        pending = [r for r in QUEUE if r != waiting_id]
    items = []
    for i, rid in enumerate(pending, 1):
        st = RUNS.get(rid) or {}
        prog = st.get("progress") or {}
        items.append({
            "id": rid, "position": i, "state": st.get("state", "queued"),
            "src": st.get("src_dir") or "",
            "files_total": prog.get("files_total") or 0,
            "total_dur": prog.get("total_dur") or 0,
        })
    # 卡在队首等翻译后端时也报成 current，前端才能显示「等待翻译后端…」
    cur = _CURRENT or _WAITING
    cst = RUNS.get(cur) if cur else None
    return {
        "pending": items,
        "current": ({"id": cur, "src": (cst or {}).get("src_dir") or "",
                     "state": (cst or {}).get("state"),
                     "resume": bool((cst or {}).get("resume")),
                     "waiting": bool(_WAITING and not _CURRENT)} if cur else None),
        # finished 里同时含 done 和 error，前端按 state 区分显示
        "finished": sorted([{"id": k, "src": v.get("src_dir") or "",
                             "state": v.get("state"), "t0": v.get("t0") or 0,
                             "resume": bool(v.get("resume")),
                             "elapsed": v.get("elapsed") or 0,
                             "outputs": v.get("outputs") or []}
                            for k, v in RUNS.items()
                            if v.get("state") in ("done", "error")],
                           key=lambda x: x["t0"], reverse=True),
    }


# --------------------------------------------------------------------------------------
# 流水线（在队列工作线程里跑）
# --------------------------------------------------------------------------------------

def run_pipeline(run_id: str) -> None:
    st = RUNS[run_id]
    o = st["options"]
    # resume 是任务级状态，但 build_cmd 从 options 里读参数——同步过去，
    # 否则 --retranslate --reuse-translation 根本不会出现在命令行上（踩过这个坑）
    o["resume"] = bool(st.get("resume"))
    t0 = time.time()
    try:
        st["state"] = "running"
        _save_task(st)          # 立刻落盘成 running：崩了也知道这是被中断的任务
        work = st["dir"]
        src_dir = st.get("src_dir")

        # 术语表 / ASR 修正表：写进本次运行的目录，再以文件路径传给流水线
        def _rules(text: str) -> int:
            return len([l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")])

        if o.get("glossary_text", "").strip():
            gp = work / "glossary.txt"
            gp.write_text(o["glossary_text"], encoding="utf-8")
            o["glossary_path"] = str(gp)
            log_to(st, f"[术语表] {_rules(o['glossary_text'])} 条")
        else:
            log_to(st, "[术语表] 未填写 —— 本次不使用术语表（术语不会统一）")
        if o.get("fix_text", "").strip():
            fp = work / "asr_fixes.txt"
            fp.write_text(o["fix_text"], encoding="utf-8")
            o["fix_path"] = str(fp)
            log_to(st, f"[ASR 修正表] {_rules(o['fix_text'])} 条")

        if src_dir:
            # ---- 本机文件夹模式：直接读源目录，绝不往里面写中间文件 ----
            inp = Path(src_dir)
            out = work / "output"
            out.mkdir(exist_ok=True)
            log_to(st, f"[输入] 本机文件夹：{inp}")
            if any(p.is_file() and p.suffix.lower() in ARCHIVE_EXTS
                   for p in inp.rglob("*")):
                log_to(st, "[提示] 该文件夹里有压缩包，文件夹模式下不自动解包（避免写进你的目录）；"
                           "需要解包请改用「上传」模式")
        else:
            inp, out = work / "input", work / "output"
            out.mkdir(exist_ok=True)
            # ---- 1) 解包 ----
            archives = [p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in ARCHIVE_EXTS]
            for a in archives:
                dest = inp / (a.stem + "_unpacked")
                dest.mkdir(parents=True, exist_ok=True)
                n = extract_archive(a, dest, st)
                log_to(st, f"[解包] {a.name} → {n} 个文件")

        audio = collect_audio(inp)
        if not audio:
            raise RuntimeError("没有找到音频文件（支持 wav/mp3/m4a/flac…，也可上传 zip）")
        log_to(st, f"[输入] 共 {len(audio)} 个音频文件")
        # 同目录同时存在 X.wav 与 X.mp3 时，X.mp3 视为已转好的成品：
        # 否则两者都输出 X.mp3，WAV 的转码产物会把现成的 MP3 覆盖掉（静默丢文件）
        existing_mp3 = {a.parent / (a.stem + ".mp3") for a in audio if a.suffix.lower() == ".mp3"}
        st["durations"] = [audio_duration(a) for a in audio]
        total_min = sum(st["durations"]) / 60
        log_to(st, f"[输入] 音频总时长 {total_min:.1f} 分钟")
        st["progress"] = {"files_total": len(audio), "files_done": 0, "file_index": 0,
                          "file_name": "", "file_stage": "", "stage_frac": 0.0,
                          "stage_detail": "", "done_dur": 0.0}

        # ---- 2) 转写 + 翻译（这两步恒定执行，界面上不再给开关）----
        cmd = build_cmd(inp, out, o)
        log_to(st, f"[执行] {' '.join(cmd[1:])}")
        # 必须显式让子进程按 UTF-8 写 stdout。
        # 子进程的 stdout 是管道，Python 会退回 locale 首选编码（中文 Windows = cp936/GBK），
        # 而这边按 encoding="utf-8" 解码 —— 于是流水线每一行的中文标签都变成乱码
        # （[清洗] → [��ϴ]），日志面板没法看，按标签着色也会失效。
        # 实测字节：b'[\xc7\xe5\xcf\xb4]' 就是 GBK 的「清洗」。
        # 只灌 PYTHONIOENCODING（只管 stdio）——不用 PYTHONUTF8，那会顺带改掉
        # 子进程 open() 的默认编码，可能影响它读写素材文件。
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env,
                                encoding="utf-8", errors="replace", bufsize=1)
        st["proc"] = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            log_to(st, line)
            update_progress(st, line)
        proc.wait()
        st["proc"] = None
        if proc.returncode != 0:
            raise RuntimeError(f"转写/翻译进程退出码 {proc.returncode}")

        # ---- 3) 整理产出 ----
        # produced 存 (源文件, 压缩包内路径)：音频**不复制到暂存目录**，打包时直接读源文件。
        # 上传模式下 input/ 里已经有一份了，再复制一份纯属浪费磁盘和时间。
        produced: List[tuple] = []
        audio_src: List[tuple] = []     # 「含音频」包：原始文件原样
        mp3_src: List[tuple] = []       # MP3 包里可直接引用的（已是 MP3 / 有同名 MP3）
        mp3_todo: List[tuple] = []      # MP3 包里需要下载时转码的（WAV 等）
        for idx, a in enumerate(audio, 1):
            rel = a.relative_to(inp)
            # 跳过解包出来的中间目录前缀（保持结构但不带 _unpacked 后缀）
            rel = Path(*[p[:-9] if p.endswith("_unpacked") else p for p in rel.parts])
            stem = rel.stem
            _d = rel.parent.as_posix()
            prefix = "" if _d == "." else _d + "/"
            arc = lambda name: prefix + name          # noqa: E731

            sub = a.relative_to(inp).parent
            src_lrc = out / sub / (a.stem + LRC_SUFFIX[o["lrc_mode"]])
            if not src_lrc.exists():                       # 兜底：找同名任意 lrc
                cand = list((out / sub).glob(a.stem + "*.lrc"))
                src_lrc = cand[0] if cand else None
            if src_lrc and src_lrc.exists():
                produced.append((src_lrc, arc(stem + ".lrc")))
                # 日文原文（下载时可选是否附带）+ 翻译缓存（只留在输出目录，不进下载包）
                src_ja = out / sub / (a.stem + ".ja.lrc")
                if src_ja.exists():
                    produced.append((src_ja, arc(stem + ".ja.lrc")))
                src_cache = out / sub / (a.stem + ".segments.json")
                if src_cache.exists():
                    produced.append((src_cache, arc(stem + ".segments.json")))
            else:
                log_to(st, f"  ⚠ 未找到 {a.name} 的字幕")

            # ---- 4) 音频：运行时只登记引用，不转码 ----
            # 转码和打包都挪到「下载」时按需触发：不下载就不浪费时间与磁盘。
            # 已是 mp3 的一律原样，绝不重新编码（实测 320k 与 128k 输出与源文件逐字节相同）。
            is_mp3 = a.suffix.lower() == ".mp3"
            twin = (a.parent / (a.stem + ".mp3")) if not is_mp3 else None
            has_twin = twin is not None and twin in existing_mp3
            audio_src.append((a, arc(a.name)))               # 「含音频」包：原始文件原样
            if is_mp3:
                mp3_src.append((a, arc(stem + ".mp3")))      # 已是 MP3，直接引用
            elif has_twin:
                mp3_src.append((twin, arc(stem + ".mp3")))   # 同目录已有同名 MP3，用现成的
                log_to(st, f"  [跳过转码] {a.name}：同目录已有同名 MP3，下载 MP3 包时直接用")
            else:
                mp3_todo.append((a, arc(stem + ".mp3")))     # WAV 等：下载时才转码

        if not produced:
            raise RuntimeError("没有产出任何文件")
        st["progress"]["files_done"] = len(audio)
        st["progress"]["done_dur"] = sum(st.get("durations", []))

        # ---- 4b) 其余素材原样带上（封面 / 插图 / readme / Finishtime / .vtt …）----
        # 打包只改**音频和歌词**的文件名；别的一律不改名、不转换，原封不动搬进包里。
        # 以前这些被整个丢掉，出来的包缺封面和说明文档，一个作品集不完整。
        extras: List[tuple] = []
        for f in sorted(inp.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() in AUDIO_EXTS or f.suffix.lower() == ".lrc":
                continue                       # 音频和歌词走后面的重命名逻辑
            rel = f.relative_to(inp)
            rel = Path(*[p[:-9] if p.endswith("_unpacked") else p for p in rel.parts])
            extras.append((f, rel.as_posix()))
        if extras:
            log_to(st, f"  （另有 {len(extras)} 个封面/文档等素材，原样放进包内）")

        # ---- 5) 登记下载素材（真正的打包在下载时按选项组合生成）----
        # 只登记 .lrc / .ja.lrc：.segments.json 是本工具的翻译缓存，不进任何下载包。
        lrc_pairs = [(s, n) for s, n in produced if n.lower().endswith(".lrc")]
        aux_pairs = [(s, n) for s, n in produced if n.lower().endswith(".json")]
        st["lrc_pairs"] = lrc_pairs
        st["audio_src"] = audio_src
        st["mp3_src"] = mp3_src
        st["mp3_todo"] = mp3_todo
        st["extras"] = extras
        st["zip_cache"] = {}
        st["name_map"] = None
        st["has_audio"] = bool(audio_src)
        st["need_mp3"] = bool(mp3_todo)
        st["outputs"] = ([
            {"kind": "lrc", "label": "下载字幕 LRC",
             "size": human(sum(s.stat().st_size for s, _ in lrc_pairs))},
            {"kind": "all", "label": "下载含音频压缩包",
             "size": human(sum(s.stat().st_size for s, _ in audio_src)), "est": True},
        ] if lrc_pairs else [])
        if aux_pairs:
            log_to(st, f"  （{len(aux_pairs)} 个翻译缓存留在输出目录，不放进下载包）")

        st["elapsed"] = time.time() - t0
        st["state"] = "done"
        st["resume"] = False            # 跑完了，下次不用再续
        log_to(st, f"\n[完成] 耗时 {st['elapsed']:.1f}s，产出 {len(produced)} 个文件")
        _save_task(st)
    except Exception as e:
        st["state"] = "error"
        st["error"] = f"{type(e).__name__}: {e}"
        st["elapsed"] = time.time() - t0
        # 失败也要存：这样重启后能看出是失败任务，产物也还能下载
        _save_task(st)
        log_to(st, f"\n[错误] {st['error']}")


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    # 把个人术语表填进文本框作为默认内容。空表 = 不启用术语表，
    # 会让术语一致性的收益白白丢掉。文件不存在时留空即可。
    try:
        g = GLOSSARY_DEFAULT.read_text(encoding="utf-8-sig").strip()
    except OSError:
        g = ""
    return INDEX_HTML.replace("__GLOSSARY_DEFAULT__", _html.escape(g))


# 健康检查缓存。探测本地服务要发 HTTP 请求；未启动的服务不是被拒绝而是**连接超时**，
# 实测 Ollama / llama.cpp 各等 1.7 秒 → 整个接口 3.7 秒。
# 前端每 8 秒调一次，如果这个接口跑在事件循环上，就会把整个服务冻结 3.7 秒，
# 表现就是「进度条计时不走秒、一跳好几秒」。所以：同步 def（走线程池）+ 缓存 + 短超时。
_HEALTH = {"t": 0.0, "v": None, "asr": None}
_HEALTH_TTL = 15.0          # 前端每 8 秒问一次；TTL 设 15 秒让约一半请求命中缓存


@app.get("/api/health")
def health() -> dict:
    """同步 def → FastAPI 丢到线程池执行，不阻塞事件循环。"""
    import httpx

    now = time.time()
    if _HEALTH["v"] is not None and now - _HEALTH["t"] < _HEALTH_TTL:
        return _HEALTH["v"]

    def probe(url: str) -> bool:
        try:
            # 本机端口 0.3 秒没应答就是没在跑，不必等满 1.5 秒
            return httpx.get(url, timeout=0.3, trust_env=False).status_code == 200
        except Exception:
            return False

    if _HEALTH["asr"] is None:              # 文件系统遍历只做一次
        _HEALTH["asr"] = ASR_MODEL_DIR.is_dir() and any(ASR_MODEL_DIR.rglob("model.bin"))
    out = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "asr_model": bool(_HEALTH["asr"]),
        "lmstudio": probe("http://127.0.0.1:1234/v1/models"),
        "ollama": probe("http://127.0.0.1:11434/v1/models"),
        "llamacpp": probe("http://127.0.0.1:8080/v1/models"),
    }
    _HEALTH.update(t=now, v=out)
    return out


@app.get("/api/backend/info")
def backend_info() -> dict:
    """告诉前端：能不能自动拉起（找不找得到 LM Studio），以及自动拉起开没开。"""
    return {"lms": _find_lms() or "", "app": _find_lmstudio_app() or "",
            "auto": AUTO_START_BACKEND, "keys": _model_keys()}


@app.post("/api/backend/start")
def backend_start(payload: dict) -> dict:
    """手动拉起翻译后端。同步 def → FastAPI 丢线程池，不阻塞事件循环。

    冷启动可能要 1~3 分钟（拉桌面程序 + 加载 6 GB 模型），所以前端要给出等待提示。
    """
    o = payload.get("options")
    if not isinstance(o, dict):
        o = {"backend": payload.get("backend") or "lmstudio",
             "model_name": payload.get("model_name") or "",
             "base_url": payload.get("base_url") or ""}
    steps: List[str] = []
    try:
        ok, msg = start_backend(o, log_cb=steps.append)
    except Exception as e:                  # 兜底：绝不把 500 抛给前端
        ok, msg = False, f"{type(e).__name__}: {e}"
    _HEALTH["t"] = 0.0                      # 让健康徽章下一次立刻反映真实状态
    return {"ok": bool(ok), "msg": msg, "steps": steps,
            "lms": _find_lms() or "", "app": _find_lmstudio_app() or ""}


@app.post("/api/backend/auto")
def backend_auto(payload: dict) -> dict:
    """开关「队列等待/运行中自动拉起后端」。只作用于当前服务进程。"""
    global AUTO_START_BACKEND
    AUTO_START_BACKEND = bool(payload.get("on"))
    _BACKEND_TRIES.clear()          # 关掉再打开时，别让旧冷却卡住立即拉起
    return {"ok": True, "on": AUTO_START_BACKEND}


@app.get("/api/browse")
def browse(path: str = "") -> dict:
    """服务端目录浏览（WebUI 与音频在同一台机器上，直接读盘，不用上传）。"""
    return browse_dir(path)


# PowerShell 调用 Windows 自带的文件夹选择对话框。
# 独立进程 + -STA：既拿到系统原生对话框，又不会和 Web 服务的线程纠缠。
# ---- 新式文件夹选择框（IFileDialog，带地址栏，可直接粘贴路径回车跳转）----
# 老式的 FolderBrowserDialog（SHBrowseForFolder）是纯树形控件，没有地址栏，
# 只能一层层点；新式的 IFileDialog 才有地址栏/面包屑/搜索。这里用 C# COM 互操作调它。
_PICK_PS_MODERN = r"""
$ErrorActionPreference = 'Stop'

$cs = @'
using System;
using System.Runtime.InteropServices;

public static class ModernFolderPicker
{
    [ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
    private class FileOpenDialogRCW { }

    [ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IFileDialog
    {
        [PreserveSig] int Show(IntPtr parent);
        void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
        void SetFileTypeIndex(uint iFileType);
        void GetFileTypeIndex(out uint piFileType);
        void Advise(IntPtr pfde, out uint pdwCookie);
        void Unadvise(uint dwCookie);
        void SetOptions(uint fos);
        void GetOptions(out uint pfos);
        void SetDefaultFolder(IShellItem psi);
        void SetFolder(IShellItem psi);
        void GetFolder(out IShellItem ppsi);
        void GetCurrentSelection(out IShellItem ppsi);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
        void GetResult(out IShellItem ppsi);
        void AddPlace(IShellItem psi, int fdap);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
        [PreserveSig] int Close(int hr);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(IntPtr pFilter);
    }

    [ComImport, Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IShellItem
    {
        void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
        void GetParent(out IShellItem ppsi);
        void GetDisplayName(uint sigdnName, out IntPtr ppszName);
        void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
        void Compare(IShellItem psi, uint hint, out int piOrder);
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    private static extern void SHCreateItemFromParsingName(
        [MarshalAs(UnmanagedType.LPWStr)] string pszPath, IntPtr pbc,
        [MarshalAs(UnmanagedType.LPStruct)] Guid riid,
        [MarshalAs(UnmanagedType.Interface)] out IShellItem ppv);

    public static string Pick(string title, string initial)
    {
        var dlg = (IFileDialog)new FileOpenDialogRCW();
        const uint FOS_PICKFOLDERS   = 0x00000020;
        const uint FOS_FORCEFILESYSTEM = 0x00000040;
        const uint FOS_PATHMUSTEXIST = 0x00000800;
        dlg.SetOptions(FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
        dlg.SetTitle(title);
        if (!string.IsNullOrEmpty(initial))
        {
            try
            {
                IShellItem item;
                SHCreateItemFromParsingName(initial, IntPtr.Zero,
                    new Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"), out item);
                dlg.SetFolder(item);
            }
            catch { }
        }
        if (dlg.Show(IntPtr.Zero) != 0) return "";
        IShellItem res;
        dlg.GetResult(out res);
        IntPtr p;
        res.GetDisplayName(0x80058000, out p);          // SIGDN_FILESYSPATH
        return Marshal.PtrToStringUni(p) ?? "";
    }
}
'@

Add-Type -TypeDefinition $cs -Language CSharp | Out-Null
$picked = [ModernFolderPicker]::Pick('选择音频文件夹', '{START}')
if ($picked) {
    [IO.File]::WriteAllText('{OUT}', $picked, (New-Object Text.UTF8Encoding $false))
}
"""

# ---- 老式文件夹选择框（兜底：新式调用失败时用）----
_PICK_PS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = '选择音频文件夹'
$dlg.ShowNewFolderButton = $true
# 关键：把根节点限制在「此电脑」。否则老式对话框会枚举整个外壳命名空间
# （网络位置、云盘…），实测要 30 秒以上才可用；限制后 1~3 秒。
$dlg.RootFolder = [System.Environment+SpecialFolder]::MyComputer
{START}
# 用一个隐藏的 TopMost 窗体当 owner，保证对话框弹在最前面
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.WindowState = 'Minimized'
$res = $dlg.ShowDialog($owner)
$owner.Dispose()
if ($res -eq [System.Windows.Forms.DialogResult]::OK) {
    [IO.File]::WriteAllText('{OUT}', $dlg.SelectedPath, (New-Object Text.UTF8Encoding $false))
}
"""


def _run_pick_script(template: str, start: str, tmp: Path, result: Path) -> str:
    """把模板写成临时 .ps1（UTF-8 带 BOM）并执行，返回用户选中的路径（空=取消）。"""
    script = tmp / "pick.ps1"
    if template is _PICK_PS_MODERN:
        start_line = start.replace("'", "''") if start else ""
    else:
        start_line = f"$dlg.SelectedPath = '{start}'" if start and Path(start).is_dir() else ""
    script.write_text(template.replace("{START}", start_line).replace("{OUT}", str(result)),
                      encoding="utf-8-sig")          # BOM：PowerShell 5.1 才认 UTF-8
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip()[:300])
    return result.read_text(encoding="utf-8-sig").strip() if result.exists() else ""


@app.post("/api/pick-folder")
def pick_folder(payload: dict) -> dict:
    """弹出系统文件夹选择框（新式 IFileDialog，带地址栏，可粘贴路径回车跳转）。

    优先用新式对话框；互操作不可用时退回老式 FolderBrowserDialog。
    走临时 .ps1 文件（UTF-8 **带 BOM**）而不是 -Command：
    Windows PowerShell 5.1 读无 BOM 的脚本会按 ANSI 解码，脚本里的中文会变乱码。
    """
    if os.name != "nt":
        raise HTTPException(501, "系统对话框仅支持 Windows，请手动填写路径")
    start = (payload.get("path") or "").strip().strip('"')
    tmp = Path(tempfile.mkdtemp(prefix="onsei2lrc_pick_"))
    result = tmp / "picked.txt"
    try:
        try:
            picked = _run_pick_script(_PICK_PS_MODERN, start, tmp, result)
            return {"path": picked, "cancelled": not picked, "dialog": "modern"}
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            if isinstance(e, subprocess.TimeoutExpired):
                raise HTTPException(504, "对话框超时未操作")
            log = f"新式对话框不可用（{e}），已退回老式对话框"
            picked = _run_pick_script(_PICK_PS, start, tmp, result)
            return {"path": picked, "cancelled": not picked, "dialog": "classic", "note": log}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "对话框超时未操作")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/use-dir")
def use_dir(payload: dict) -> dict:
    """选定本机文件夹，创建一个绑定该目录的运行记录。"""
    raw = (payload.get("path") or "").strip().strip('"')
    if not raw:
        raise HTTPException(400, "请先选择文件夹")
    p = Path(raw).expanduser()
    if not p.exists() or not p.is_dir():
        raise HTTPException(400, f"路径不存在或不是文件夹：{p}")
    audio = collect_audio(p)
    if not audio:
        raise HTTPException(400, f"该文件夹（含子目录）里没有找到音频文件：{p}")
    run_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    work = RUNS_DIR / run_id
    work.mkdir(parents=True, exist_ok=True)
    saved = [{"name": str(a.relative_to(p)), "size": human(a.stat().st_size)} for a in audio[:50]]
    with RUNS_LOCK:
        RUNS[run_id] = {"id": run_id, "dir": work, "state": "uploaded", "log": [],
                        "log_offset": 0, "t0": time.time(), "durations": [],
                        "progress": {"files_total": 0, "files_done": 0, "file_index": 0,
                                     "file_name": "", "file_stage": "", "stage_frac": 0.0,
                                     "stage_detail": "", "done_dur": 0.0},
                        "outputs": [], "error": None, "files": saved, "proc": None,
                        "options": {}, "elapsed": 0.0, "src_dir": str(p)}
    return {"run_id": run_id, "src_dir": str(p), "count": len(audio), "files": saved}


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)) -> dict:
    run_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    work = RUNS_DIR / run_id
    inp = work / "input"
    inp.mkdir(parents=True, exist_ok=True)
    saved = []
    root = inp.resolve()
    for f in files:
        # 文件夹上传时 filename 带相对路径（a/b/c.wav），保留结构；否则只有文件名
        raw = (f.filename or "unnamed").replace("\\", "/")
        rel = Path(*[p for p in raw.split("/") if p not in ("", ".", "..")])
        if not rel.parts:
            rel = Path("unnamed")
        dst = (inp / rel)
        # 防路径穿越：解析后必须仍在 input/ 之内
        if not str(dst.resolve()).startswith(str(root)):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as fh:
            while chunk := await f.read(1 << 20):
                fh.write(chunk)
        saved.append({"name": rel.as_posix(), "size": human(dst.stat().st_size)})
    with RUNS_LOCK:
        RUNS[run_id] = {"id": run_id, "dir": work, "state": "uploaded", "log": [],
                        "log_offset": 0, "t0": time.time(), "durations": [],
                        "progress": {"files_total": 0, "files_done": 0, "file_index": 0,
                                     "file_name": "", "file_stage": "", "stage_frac": 0.0,
                                     "stage_detail": "", "done_dur": 0.0},
                        "outputs": [], "error": None, "files": saved, "proc": None,
                        "options": {}, "elapsed": 0.0}
    return {"run_id": run_id, "files": saved}


@app.post("/api/run")
async def run(payload: dict) -> dict:
    run_id = payload.get("run_id", "")
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(404, "run_id 不存在，请重新上传文件")
    if st["state"] in ("running", "queued"):
        raise HTTPException(409, f"该任务已在队列里（{st['state']}）")
    # 转写 + 翻译恒定执行（界面上不再给开关）；转码与打包改到下载时按需触发
    opts = {
        "lrc_mode": payload.get("lrc_mode", "zh"),
        "chunk_mode": payload.get("chunk_mode", "hybrid"),
        "backend": payload.get("backend", "lmstudio"),
        "model_name": payload.get("model_name", "sakura"),
        "base_url": payload.get("base_url", ""),
        "protocol": payload.get("protocol", "json"),
        "api_key": payload.get("api_key", ""),
        "max_chars": int(payload.get("max_chars", 32)),
        "merge_gap": float(payload.get("merge_gap", 0.4)),
        "batch_size": int(payload.get("batch_size", 8)),
        "workers": int(payload.get("workers", 4)),
        "prompt_style": payload.get("prompt_style", "v1"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "frequency_penalty": payload.get("frequency_penalty"),
        "max_tokens": payload.get("max_tokens"),
        "glossary_text": payload.get("glossary_text", "") or "",
        "fix_text": payload.get("fix_text", "") or "",
    }
    if opts["backend"] == "custom" and not opts["base_url"]:
        raise HTTPException(400, "自定义后端需要填接口地址")
    st["options"] = opts
    st["log"].clear()
    st["log_offset"] = 0
    st["t0"] = time.time()
    st["durations"] = []
    st["progress"] = {"files_total": 0, "files_done": 0, "file_index": 0, "file_name": "",
                      "file_stage": "", "stage_frac": 0.0, "stage_detail": "", "done_dur": 0.0}
    st["outputs"] = []
    st["error"] = None
    st["elapsed"] = 0.0
    # 不再直接起线程——交给队列串行执行（GPU 只有一个，并发只会互相拖慢）
    st["state"] = "queued"
    pos = enqueue(run_id)
    _save_task(st)
    return {"ok": True, "queued": True, "position": pos}


@app.post("/api/write-back")
async def write_back(payload: dict) -> dict:
    """把某个任务产出的 .lrc 写回音频所在文件夹。

    改成按钮触发（原来是跑完自动执行），因为写回会改动用户的素材目录，
    应该由用户明确地按一下。只**新增** .lrc，绝不改动或删除任何原有文件；
    同名文件会被覆盖，前端会先提示。

    `ja` 与下载区的「附带日文 lrc」同一个开关：下载会给你什么，写回就写什么。
    """
    run_id = payload.get("run_id", "")
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(404, "run_id 不存在")
    src_dir = st.get("src_dir")
    if not src_dir:
        raise HTTPException(400, "这个任务不是文件夹模式，没有可写回的目标目录")
    if not Path(src_dir).is_dir():
        raise HTTPException(400, f"音频目录已不存在：{src_dir}")
    want_ja = bool(payload.get("ja", 0))

    pairs = [(s, a) for s, a in (st.get("lrc_pairs") or []) if a.lower().endswith(".lrc")]
    if not want_ja:
        pairs = [(s, a) for s, a in pairs if not a.lower().endswith(".ja.lrc")]
    if not pairs:
        raise HTTPException(400, "这次任务还没有产出字幕")

    n, failed = 0, []
    for src, arc in pairs:
        dst = Path(src_dir) / arc
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
        except OSError as e:
            failed.append(f"{arc}: {e}")
    log_to(st, f"[写回] 已把 {n} 个字幕文件写入 {src_dir}")
    return {"ok": True, "written": n, "failed": failed,
            "total": len(pairs), "dir": src_dir}


@app.get("/api/queue")
async def queue_api() -> dict:
    """任务队列快照：待跑列表 + 正在跑的 + 已完成的（含下载项）。"""
    return queue_view()


@app.post("/api/queue/remove")
async def queue_remove(payload: dict) -> dict:
    """把还没开跑的任务移出队列。已在跑的请用 /api/cancel。"""
    run_id = payload.get("run_id", "")
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(404, "run_id 不存在")
    if st.get("state") == "running":
        raise HTTPException(409, "任务已经在跑，请用「取消」")
    if not dequeue(run_id):
        raise HTTPException(409, "该任务不在队列里")
    st["state"] = "uploaded"
    log_to(st, "[队列] 已移出队列")
    return {"ok": True}


@app.post("/api/queue/retry")
async def queue_retry(payload: dict) -> dict:
    """把失败/中断的任务重新入队，并开启断点继续（跳过已完成的转写与翻译）。"""
    run_id = payload.get("run_id", "")
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(404, "run_id 不存在")
    if st.get("state") in ("running", "queued"):
        raise HTTPException(409, f"该任务已在队列里（{st['state']}）")
    if not st.get("src_dir") and not (st["dir"] / "input").is_dir():
        raise HTTPException(400, "找不到源目录，无法继续")
    st["state"] = "queued"
    st["error"] = None
    st["resume"] = True
    st["elapsed"] = 0.0
    st["progress"] = _blank_progress()
    log_to(st, "[继续] 重新入队，跳过已完成的转写与翻译")
    pos = enqueue(run_id)
    _save_task(st)
    return {"ok": True, "position": pos}


@app.post("/api/queue/clear")
async def queue_clear() -> dict:
    """清空所有待跑任务（不影响正在跑的）。"""
    with QUEUE_LOCK:
        n = len(QUEUE)
        for rid in QUEUE:
            st = RUNS.get(rid)
            if st:
                st["state"] = "uploaded"
                log_to(st, "[队列] 已移出队列")
        QUEUE.clear()
    return {"ok": True, "removed": n}


@app.get("/api/status")
async def status(run_id: str, since: int = 0) -> dict:
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(404, "run_id 不存在")
    with RUNS_LOCK:
        off = st.get("log_offset", 0)
        start = max(0, since - off)
        lines = st["log"][start:]
        nxt = off + len(st["log"])
    pairs = st.get("lrc_pairs") or []
    return {"state": st["state"], "log": lines, "next": nxt,
            "progress": progress_view(st), "outputs": st["outputs"],
            "has_audio": bool(st.get("has_audio")), "need_mp3": bool(st.get("need_mp3")),
            # 供「字幕写回」按钮判断是否可用（只有文件夹模式才有目标目录）
            "src_dir": st.get("src_dir") or "",
            "lrc_count": len(pairs),
            "lrc_ja_count": sum(1 for _, a in pairs if a.lower().endswith(".ja.lrc")),
            "error": st["error"], "elapsed": round(st.get("elapsed", 0), 1)}


@app.get("/api/runs")
async def runs() -> dict:
    """列出最近的任务。

    用于「刷新浏览器后找回正在跑的任务」：流水线是独立进程，刷新不会中断它，
    但前端状态会丢。前端优先用 localStorage 里存的 run_id，丢了就退回这里查。
    """
    with RUNS_LOCK:
        items = []
        now = time.time()
        for k, v in RUNS.items():
            prog = v.get("progress") or {}
            # elapsed 只在任务结束时才写进 st，进行中的要现算
            el = v.get("elapsed") or (now - v["t0"] if v.get("t0") else 0)
            items.append({
                "id": k,
                "state": v.get("state"),
                "t0": v.get("t0") or 0,
                "src": v.get("src_dir") or "",
                "file": prog.get("file_name") or "",
                "files_total": prog.get("files_total") or 0,
                "elapsed": round(el, 1),
            })
    items.sort(key=lambda x: x["t0"], reverse=True)
    return {"runs": items[:20]}


@app.post("/api/cancel")
async def cancel(payload: dict) -> dict:
    """取消任务：排队中的移出队列，正在跑的终止其子进程。"""
    run_id = payload.get("run_id", "")
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(404, "run_id 不存在")
    if st.get("state") == "queued":
        if dequeue(run_id):
            st["state"] = "uploaded"
            log_to(st, "[队列] 已取消排队")
            return {"ok": True, "was": "queued"}
    proc = st.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        st["state"] = "error"
        st["error"] = "已手动取消"
        log_to(st, "[取消] 已终止任务")
        return {"ok": True, "was": "running"}
    return {"ok": False, "msg": "没有正在运行的进程"}


@app.get("/api/download/{run_id}/{kind}")
def download(run_id: str, kind: str, ja: int = 1, mp3: int = 0, tr: int = 0):
    """下载产出。三个复选项作为查询参数，按需组合打包（结果缓存）。

    kind=lrc  仅字幕：<名字>.lrc（+ .ja.lrc 若 ja=1）
    kind=all  含音频：上面 + 音频（mp3=1 时把 WAV 等转成 MP3；tr=1 时文件名翻成中文）

    `.segments.json` 是本工具的翻译缓存，**不进任何下载包**。
    含音频的包可能几 GB，所以不在运行时预先打，首次点击才生成。
    """
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(404, "run_id 不存在")
    if kind not in ("lrc", "all"):
        raise HTTPException(404, "没有该产出")
    if st.get("state") != "done":
        raise HTTPException(409, "任务尚未完成")

    work = st["dir"]
    key = f"{kind}|ja{int(ja)}|mp3{int(mp3)}|tr{int(tr)}"
    cache = st.setdefault("zip_cache", {})
    hit = cache.get(key)
    if hit and Path(hit).exists():
        p = Path(hit)
        return FileResponse(p, filename=p.name, media_type="application/zip")

    subs = list(st.get("lrc_pairs") or [])          # 已只含 .lrc / .ja.lrc
    if not ja:
        subs = [(s, n) for s, n in subs if not n.lower().endswith(".ja.lrc")]
    if not subs:
        raise HTTPException(404, "没有字幕")

    # 文件名翻译：拿所有音频的原名去翻，再按原名替换。
    # 放进 try 里——这只是可选功能，失败就退回原名，不该让整个下载 500。
    name_map: dict = {}
    if tr:
        try:
            stems = []
            for s, n in (list(st.get("mp3_src") or []) + list(st.get("mp3_todo") or [])
                         + list(st.get("audio_src") or [])):
                base = Path(n).stem
                if base not in stems:
                    stems.append(base)
            if stems:
                nm = st.get("name_map")
                if nm is None:
                    log_to(st, f"[文件名] 开始翻译 {len(stems)} 个文件名…")
                    nm = translate_filenames(stems, st.get("options") or {}, st)
                    st["name_map"] = nm
                name_map = nm
        except Exception as e:
            log_to(st, f"[文件名] 翻译失败，改用原名：{type(e).__name__}: {e}")
            name_map = {}

    def rename(arc: str) -> str:
        """把压缩包内路径的 stem 换成译名，保留目录前缀与扩展名。

        注意 `X.ja.lrc` 的 `Path.stem` 是 `X.ja`，直接查表会查不到——
        必须把 `.ja` 拆出来，否则中文 LRC 改了名、日文 LRC 没改，两者就配不上对了。
        """
        if not name_map:
            return arc
        p = PurePosixPath(arc)
        stem, extra = p.stem, ""
        if stem.endswith(".ja"):
            stem, extra = stem[:-3], ".ja"
        new = name_map.get(stem)
        return str(p.parent / (new + extra + p.suffix)) if new else arc

    stamp = time.strftime("%Y%m%d_%H%M%S")
    try:
        pairs = [(s, rename(n)) for s, n in subs]
        if kind == "all":
            audio = list(st.get("audio_src") or [])
            if not audio:
                raise HTTPException(404, "本次没有音频")
            if mp3:
                ready = list(st.get("mp3_src") or [])
                todo = list(st.get("mp3_todo") or [])
                conv = []
                if todo:
                    log_to(st, f"[下载] 开始转码 {len(todo)} 个文件为 MP3 320k…")
                    for i, (s, name) in enumerate(todo, 1):
                        dst = work / "audio" / name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if convert_mp3(s, dst, st):
                            conv.append((dst, name))
                        log_to(st, f"  [转码] {i}/{len(todo)} {s.name}")
                pairs += [(s, rename(n)) for s, n in ready + conv]
            else:
                pairs += [(s, rename(n)) for s, n in audio]
            # 封面 / 插图 / readme / Finishtime / .vtt 等：原样进包，不改名
            extras = list(st.get("extras") or [])
            if extras:
                log_to(st, f"[下载] 附上 {len(extras)} 个封面/文档等素材（保持原名）")
            pairs += extras
            log_to(st, f"[下载] 正在打包 {len(pairs)} 个文件…")
        z = work / f"{kind}_{stamp}.zip"
        make_zip(pairs, z)
    except HTTPException:
        raise
    except Exception as e:
        log_to(st, f"[下载] 打包失败：{type(e).__name__}: {e}")
        raise HTTPException(500, f"打包失败：{e}")

    cache[key] = str(z)
    log_to(st, f"[下载] 已生成 {z.name}（{human(z.stat().st_size)}）")
    return FileResponse(z, filename=z.name, media_type="application/zip")


# --------------------------------------------------------------------------------------
# 前端页面
# --------------------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>日语音频翻译 · onsei2lrc</title>
<link rel="icon" href="data:,"><!-- 不请求 favicon，省掉一条 404 -->
<style>
  :root{--bg:#0f1115;--card:#171a21;--card2:#1e222b;--line:#2a2f3a;--fg:#e6e8ee;--dim:#9aa3b2;
        --acc:#6ea8fe;--ok:#3ddc97;--warn:#ffcc66;--err:#ff6b6b}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:24px 20px 60px}
  h1{font-size:20px;margin:0 0 14px}
  .badges{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
  .badge{font-size:12px;padding:3px 9px;border-radius:20px;background:var(--card2);border:1px solid var(--line);color:var(--dim)}
  button.badge{font-family:inherit;cursor:pointer}
  .badge.act{background:#243447;border-color:#3a5a86;color:#cfe0ff}
  .badge.act:hover{background:#2d4160}
  .badge.act:disabled{cursor:default;opacity:.75}
  .badge.on{color:var(--ok);border-color:#23503c}
  .badge.off{color:var(--err);border-color:#55292b}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
  .card h2{font-size:14px;margin:0 0 12px;color:var(--dim);font-weight:600;letter-spacing:.3px}
  #drop{border:2px dashed var(--line);border-radius:12px;padding:34px 16px;text-align:center;cursor:pointer;
        transition:.15s;background:var(--card2)}
  #drop.over{border-color:var(--acc);background:#1b2536}
  #drop .big{font-size:15px;margin-bottom:6px}
  #drop .small{color:var(--dim);font-size:12.5px}
  .files{margin-top:12px;max-height:170px;overflow:auto;font-size:13px}
  .files div{padding:4px 8px;border-radius:6px;background:var(--card2);margin-bottom:4px;display:flex;justify-content:space-between}
  .files span{color:var(--dim)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px}
  label.f{display:block;font-size:12.5px;color:var(--dim);margin-bottom:5px}
  select,input[type=text],input[type=password],input[type=number]{width:100%;padding:7px 9px;border-radius:8px;
        border:1px solid var(--line);background:var(--card2);color:var(--fg);font-size:13px}
  select:focus,input:focus{outline:none;border-color:var(--acc)}
  .checks{display:flex;gap:20px;flex-wrap:wrap;margin-top:4px}
  .checks label{display:flex;align-items:center;gap:7px;font-size:13.5px;cursor:pointer}
  input[type=checkbox]{width:16px;height:16px;accent-color:var(--acc)}
  .row{display:flex;gap:10px;align-items:center;margin-top:16px;flex-wrap:wrap}
  button{padding:10px 22px;border-radius:9px;border:1px solid var(--line);background:var(--card2);color:var(--fg);
         font-size:14px;cursor:pointer;transition:.15s}
  button:hover:not(:disabled){border-color:var(--acc);color:#fff}
  button.primary{background:var(--acc);border-color:var(--acc);color:#08111f;font-weight:700}
  button.primary:hover:not(:disabled){background:#8ab8ff}
  button:disabled{opacity:.45;cursor:not-allowed}
  .bar{height:8px;background:var(--card2);border-radius:6px;overflow:hidden;margin:6px 0 4px;border:1px solid var(--line)}
  .bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#4b7fd6,#6ea8fe);transition:width .3s}
  .bar i.task{background:linear-gradient(90deg,#2f9e6e,#3ddc97)}
  .prog{margin-top:14px;padding:12px 14px;background:var(--card2);border:1px solid var(--line);border-radius:10px}
  .plabel{display:flex;justify-content:space-between;font-size:12.5px;color:var(--dim);gap:12px}
  .plabel span:last-child{color:var(--fg);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:70%}
  .meta{font-size:12.5px;color:var(--dim);display:flex;justify-content:space-between}
  #log{background:#0b0d11;border:1px solid var(--line);border-radius:10px;padding:10px 12px;height:340px;overflow:auto;
       font:12px/1.6 "Cascadia Mono",Consolas,monospace;color:#c8d0dd}
  /* 一行一个 div + 悬挂缩进：折行的部分正好对齐到正文起点（时间戳 8 字符 + 6px） */
  #log .ln{white-space:pre-wrap;word-break:break-word;padding-left:64px;text-indent:-64px}
  #log .ts{color:#5a6678;margin-right:6px}
  #log .lg-err{color:#ff8a8a}
  #log .lg-warn{color:#ffcf6b}
  #log .lg-ok{color:#7ee2a8}
  #log .lg-stage{color:#8ab4ff;font-weight:600;margin-top:6px}
  #log .lg-step{color:#9fb2cc}
  #log .lg-prog{color:#5f6b7d}
  #log.compact .lg-prog{display:none}
  .logbar{display:flex;align-items:center;gap:16px;margin:2px 0 8px;font-size:12.5px;color:var(--dim)}
  .logbar label{display:flex;align-items:center;gap:6px;cursor:pointer}
  .dl{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px}
  .dl a{display:block;padding:12px 18px;border-radius:10px;background:#1b2536;
        border:1px solid #2f4a6e;color:#cfe0ff;text-decoration:none;font-size:13.5px}
  .dl a:hover{background:#22304a}
  .dl small{display:block;color:var(--dim);font-size:11.5px;margin-top:2px}
  .hide{display:none}
  .hint{font-size:12px;color:var(--dim);margin-top:8px}
  code{background:var(--card2);padding:1px 6px;border-radius:5px;font-size:12px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
  @media(max-width:760px){.grid2{grid-template-columns:1fr}}
  .tabs{display:flex;gap:8px;margin:6px 0 10px}
  .tab{padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card2);
       color:var(--dim);cursor:pointer;font-size:13px}
  .tab.on{background:var(--acc);border-color:var(--acc);color:#fff}
  .dirrow{display:flex;gap:8px}
  .dirrow input{flex:1}
  .modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;
         justify-content:center;z-index:99}
  .modalbox{background:var(--card);border:1px solid var(--line);border-radius:14px;width:min(720px,92vw);
            max-height:82vh;display:flex;flex-direction:column;overflow:hidden}
  .modalhead{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;
             border-bottom:1px solid var(--line)}
  .modalhead button{background:none;border:none;color:var(--dim);font-size:20px;cursor:pointer;line-height:1}
  .pathbar{padding:8px 16px;font:12px/1.5 "Cascadia Mono",Consolas,monospace;color:var(--acc);
           background:var(--card2);border-bottom:1px solid var(--line);word-break:break-all}
  .dirlist{overflow:auto;padding:6px 0;flex:1;min-height:200px}
  .diritem{display:flex;justify-content:space-between;gap:10px;padding:7px 16px;cursor:pointer;font-size:13px}
  .diritem:hover{background:var(--card2)}
  .diritem .cnt{color:var(--dim);font-size:12px;white-space:nowrap}
  .diritem.has .cnt{color:var(--ok)}
  .modalfoot{display:flex;align-items:center;gap:10px;padding:12px 16px;border-top:1px solid var(--line)}
  .modalfoot #browseInfo{flex:1;color:var(--dim);font-size:12px}
  .hide{display:none!important}
  details.adv{margin-top:14px;border:1px solid var(--line);border-radius:10px;background:var(--card2)}
  details.adv>summary{padding:9px 14px;cursor:pointer;font-size:13px;color:var(--dim);
                      list-style:none;user-select:none}
  details.adv>summary::-webkit-details-marker{display:none}
  details.adv>summary::before{content:"▸ ";display:inline-block;transition:.15s}
  details.adv[open]>summary::before{content:"▾ "}
  details.adv>summary:hover{color:var(--fg)}
  details.adv>.grid{padding:4px 14px 14px}
  /* 任务队列 */
  #queue{display:flex;flex-direction:column;gap:8px}
  .qempty{color:var(--dim);font-size:13px;padding:6px 2px}
  .qitem{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:9px;
         background:var(--card2);border:1px solid var(--line)}
  .qitem.run{border-color:#2f4f7f;background:#182234}
  .qitem.ok{border-color:#23503c}
  .qitem.err{border-color:#5a2b2b}
  .qidx{color:var(--dim);font-size:12px;min-width:22px;text-align:right}
  .qname{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
  .qstat{font-size:12px;color:var(--dim);white-space:nowrap}
  .qitem.run .qstat{color:var(--acc)}
  .qitem.ok .qstat{color:var(--ok)}
  .qitem.err .qstat{color:var(--err)}
  .qbar{width:120px;height:5px;border-radius:3px;background:#0d1016;overflow:hidden;flex:none}
  .qbar>i{display:block;height:100%;width:0;background:var(--acc);transition:width .3s}
  .qitem.ok .qbar>i{background:var(--ok)}
  .qbtn{font-size:11px;padding:2px 8px;border-radius:6px;flex:none}
  .qdl{display:flex;gap:6px;flex:none}
  .qdl a{font-size:11px;padding:2px 8px;border-radius:6px;background:#22303f;
         border:1px solid #2f4f7f;color:var(--acc);text-decoration:none}
  .qdl a:hover{background:#2a3d52}
  textarea{width:100%;padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card2);
           color:var(--fg);font:12px/1.5 "Cascadia Mono",Consolas,monospace;resize:vertical}
  textarea:focus{outline:none;border-color:var(--acc)}
  button.mini{padding:1px 8px;font-size:11px;border-radius:6px;margin-left:6px;vertical-align:middle}
</style>
</head>
<body>
<div class="wrap">
  <h1>日语音频翻译 · onsei2lrc</h1>
  <div class="badges" id="badges"></div>
  <div class="logbar" id="backendBar">
    <label><input type="checkbox" id="autoBackend" checked> 后端掉线时自动拉起 LM Studio</label>
    <span id="backendHint"></span>
  </div>

  <div class="card">
    <h2>① 导入</h2>
    <div class="tabs">
      <button class="tab on" data-srcmode="dir">本机文件夹</button>
      <button class="tab" data-srcmode="upload">上传文件</button>
    </div>

    <div id="dirBox">
      <div class="dirrow">
        <input type="text" id="src_dir" placeholder="粘贴或输入文件夹路径，回车直接确定">
        <button id="btnBrowse">浏览…</button>
        <button id="btnCheck">检查</button>
      </div>
      <div class="hint" id="dirInfo"></div>
    </div>

    <div id="dropBox" class="hide">
      <div id="drop">
        <div class="big">拖入音频、zip 或整个文件夹</div>
        <div class="small">
          或
          <button class="mini" id="btnPickFiles">选择文件</button>
          <button class="mini" id="btnPickDir">选择文件夹</button>
        </div>
      </div>
      <input type="file" id="pick" multiple class="hide">
      <input type="file" id="pickDir" webkitdirectory class="hide">
      <div class="files" id="files"></div>
    </div>
  </div>

  <div class="card">
    <h2>② 设置</h2>
    <div class="grid">
      <div>
        <label class="f">LRC 样式</label>
        <select id="lrc_mode">
          <option value="zh">纯中文</option>
          <option value="both">中日双语</option>
          <option value="ja">纯日文</option>
          <option value="inline">一行内 日文｜中文</option>
        </select>
      </div>
    </div>

    <details class="adv">
      <summary>高级设置</summary>
      <div class="grid">
      <div>
        <label class="f">翻译后端</label>
        <select id="backend">
          <option value="lmstudio">LM Studio</option>
          <option value="llamacpp">llama.cpp server</option>
          <option value="ollama">Ollama</option>
          <option value="deepseek">DeepSeek API</option>
          <option value="custom">自定义接口</option>
        </select>
      </div>
      <div>
        <label class="f">翻译模型</label>
        <select id="model_preset">
          <option value="v1-7b">Sakura-7B-Qwen2.5-v1.0</option>
          <option value="v37-7b" selected>Sakura-GalTransl-7B-v3.7</option>
          <option value="v4-4b">GalTransl-v4-4B-2601</option>
        </select>
      </div>
      <div>
        <label class="f">模型标识符</label>
        <input type="text" id="model_name" value="sakura37">
      </div>
      <div>
        <label class="f">提示词模板</label>
        <select id="prompt_style">
          <option value="v1">v1.0</option>
          <option value="v3" selected>v3</option>
        </select>
      </div>
      <div>
        <label class="f">切块模式</label>
        <select id="chunk_mode">
          <option value="hybrid">hybrid</option>
          <option value="vad">vad</option>
          <option value="energy">energy</option>
          <option value="whole">whole</option>
        </select>
      </div>
      <div>
        <label class="f">单行最大字数</label>
        <input type="number" id="max_chars" value="32" min="12" max="80">
      </div>
      <div>
        <label class="f">合并间隔（秒）</label>
        <input type="number" id="merge_gap" value="0.8" step="0.1" min="0" max="3">
      </div>
      <div>
        <label class="f">每批翻译行数</label>
        <input type="number" id="batch_size" value="8" min="1" max="32">
      </div>
      <div>
        <label class="f">翻译并发数 workers</label>
        <input type="number" id="workers" value="4" min="1" max="8">
      </div>
      <div>
        <label class="f">温度 temperature</label>
        <input type="number" id="temperature" value="0.3" step="0.05" min="0" max="2">
      </div>
      <div>
        <label class="f">top_p</label>
        <input type="number" id="top_p" value="0.8" step="0.05" min="0" max="1">
      </div>
      <div>
        <label class="f">频率惩罚 frequency_penalty</label>
        <input type="number" id="frequency_penalty" value="0" step="0.05" min="0" max="2">
      </div>
      <div>
        <label class="f">单次输出上限 max_tokens</label>
        <input type="number" id="max_tokens" value="2048" step="256" min="128" max="8192">
      </div>
      <div id="custom_url_box" class="hide">
        <label class="f">接口地址（OpenAI 兼容）</label>
        <input type="text" id="base_url" placeholder="http://127.0.0.1:8080/v1">
      </div>
      <div id="key_box" class="hide">
        <label class="f">API Key</label>
        <input type="password" id="api_key" placeholder="sk-...">
      </div>
      </div>
    </details>

    <div class="grid2">
      <div>
        <label class="f">
          术语表
          <button class="mini" data-load="glossary_text" data-file="glossary_file">载入</button>
          <button class="mini" data-clear="glossary_text">清空</button>
        </label>
        <textarea id="glossary_text" rows="6" spellcheck="false"
          placeholder="# 每行一条：日文→中文 #备注">__GLOSSARY_DEFAULT__</textarea>
        <input type="file" id="glossary_file" class="hide" accept=".txt,.csv,.md">
      </div>
      <div>
        <label class="f">
          ASR 修正表
          <button class="mini" data-load="fix_text" data-file="fix_file">载入</button>
          <button class="mini" data-clear="fix_text">清空</button>
        </label>
        <textarea id="fix_text" rows="6" spellcheck="false"
          placeholder="# 每行一条：听错的⇒正确的 #备注"></textarea>
        <input type="file" id="fix_file" class="hide" accept=".txt,.csv,.md">
      </div>
    </div>
    <div class="row">
      <button class="primary" id="start" disabled>加入队列</button>
      <button id="cancel" disabled>取消当前</button>
      <span class="meta" id="stateTxt"></span>
    </div>
    <div class="prog">
      <div class="plabel"><span>当前文件</span><span id="fileTxt">—</span></div>
      <div class="bar"><i id="fileBar"></i></div>
      <div class="plabel"><span>总进度</span><span id="taskTxt">—</span></div>
      <div class="bar"><i id="taskBar" class="task"></i></div>
    </div>
  </div>

  <div class="card">
    <h2>③ 任务队列</h2>
    <div id="queue"></div>
  </div>

  <div class="card">
    <h2>④ 日志</h2>
    <div class="logbar">
      <label><input type="checkbox" id="logCompact"> 精简（隐藏「…N/M 块」转写进度行）</label>
      <button class="badge" id="logClear">清空</button>
    </div>
    <div id="log"></div>
  </div>

  <div class="card hide" id="resultCard">
    <h2>⑤ 下载</h2>
    <div class="checks" id="dlOpts">
      <label><input type="checkbox" id="dl_ja"> 附带日文 lrc</label>
      <label><input type="checkbox" id="dl_mp3" checked> WAV 自动转 MP3（320K）</label>
      <label><input type="checkbox" id="dl_tr"> 翻译文件名</label>
    </div>
    <div class="row" id="wbRow">
      <button id="btnWriteBack" disabled>字幕写回音频文件夹</button>
      <span class="meta" id="wbInfo"></span>
    </div>
    <div class="dl" id="dl"></div>
  </div>
</div>

<div id="browseModal" class="modal hide">
  <div class="modalbox">
    <div class="modalhead">
      <b>选择音频文件夹</b>
      <button id="browseClose" title="关闭">×</button>
    </div>
    <div class="pathbar" id="browsePath">（盘符列表）</div>
    <div class="dirlist" id="browseList"></div>
    <div class="modalfoot">
      <span id="browseInfo"></span>
      <button id="browseUp">上一层</button>
      <button id="browsePick" class="primary">使用此文件夹</button>
    </div>
  </div>
</div>

<script>
// runId 只用于「文件已上传/文件夹已选定、还没加入队列」这个中间态。
// 任务一旦入队，一切都由服务端的队列状态驱动（见 tick()），
// 所以刷新浏览器不需要任何恢复逻辑——tick() 自己就能接上。
let runId = null, since = 0;

function setRunId(id){ runId = id; }
function clearRunId(){ runId = null; }

const $ = id => document.getElementById(id);
const drop = $('drop'), pick = $('pick');

let backendBusy = false, backendMsg = '';

async function loadHealth(){
  if (backendBusy) return;          // 拉起过程中别重建顶栏，否则按钮状态被冲掉
  try{
    const h = await (await fetch('/api/health')).json();
    const b = [];
    b.push(`<span class="badge ${h.ffmpeg?'on':'off'}">ffmpeg ${h.ffmpeg?'就绪':'缺失'}</span>`);
    b.push(`<span class="badge ${h.asr_model?'on':'off'}">ASR 模型 ${h.asr_model?'已下载':'未下载'}</span>`);
    b.push(`<span class="badge ${h.lmstudio?'on':'off'}">LM Studio ${h.lmstudio?'在线':'未启动'}</span>`);
    if (!h.lmstudio)
      b.push(`<button class="badge act" id="btnStartBackend" title="启动 LM Studio 并加载当前设定的模型">↑ 拉起 LM Studio</button>`);
    b.push(`<span class="badge ${h.llamacpp?'on':'off'}">llama.cpp ${h.llamacpp?'在线':'未启动'}</span>`);
    b.push(`<span class="badge ${h.ollama?'on':'off'}">Ollama ${h.ollama?'在线':'未启动'}</span>`);
    if (backendMsg) b.push(`<span class="badge ${backendMsg.startsWith('✅')?'on':'off'}">${backendMsg}</span>`);
    $('badges').innerHTML = b.join('');
  }catch(e){ $('badges').innerHTML = '<span class="badge off">健康检查失败</span>'; }
}
loadHealth(); setInterval(loadHealth, 8000);

// 手动拉起翻译后端。冷启动 = 拉桌面程序 + 加载 6 GB 模型，1~3 分钟是正常的。
$('badges').addEventListener('click', async e => {
  if (e.target.id !== 'btnStartBackend' || backendBusy) return;
  backendBusy = true;
  const btn = e.target;
  btn.disabled = true;
  btn.textContent = '正在拉起…（首次 1~3 分钟）';
  let msg;
  try {
    const r = await fetch('/api/backend/start', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({backend: ($('backend') || {}).value || 'lmstudio',
                            model_name: ($('model_name') || {}).value.trim()})});
    const j = await r.json();
    msg = (j.ok ? '✅ ' : '❌ ') + (j.msg || '失败');
    if (j.steps && j.steps.length) console.log('[拉起后端]\n' + j.steps.join('\n'));   // 明细进控制台
  } catch(err){ msg = '❌ ' + err; }
  backendBusy = false;
  backendMsg = msg.slice(0, 90);
  loadHealth();
});

// 后端能力与自动拉起开关
async function loadBackendInfo(){
  try{
    const j = await (await fetch('/api/backend/info')).json();
    $('autoBackend').checked = !!j.auto;
    if (j.app)      $('backendHint').textContent = '已找到 LM Studio，可自动拉起';
    else if (j.lms) $('backendHint').textContent = '只找到 lms，可能拉不起桌面程序';
    else { $('backendHint').textContent = '没找到 LM Studio，只能手动启动'; $('autoBackend').disabled = true; }
  }catch(e){ $('backendHint').textContent = ''; }
}
loadBackendInfo();
$('autoBackend').onchange = () => fetch('/api/backend/auto', {method:'POST',
  headers:{'Content-Type':'application/json'},
  body: JSON.stringify({on: $('autoBackend').checked})});

// 日志工具：精简（CSS 折叠转写进度行）+ 清空
$('logCompact').onchange = () => $('log').classList.toggle('compact', $('logCompact').checked);
$('logClear').onclick = () => { $('log').textContent = ''; };

drop.onclick = e => { if (e.target.tagName !== 'BUTTON') pick.click(); };
$('btnPickFiles').onclick = e => { e.stopPropagation(); pick.click(); };
$('btnPickDir').onclick = e => { e.stopPropagation(); $('pickDir').click(); };

// 把拖进来的东西展开成文件列表：文件夹用 webkitGetAsEntry 递归，保留相对路径
async function filesFromDrop(dt){
  const out = [];
  const items = dt.items ? Array.from(dt.items) : [];
  const entries = items.filter(i => i.kind === 'file' && i.webkitGetAsEntry)
                        .map(i => i.webkitGetAsEntry()).filter(Boolean);
  if (!entries.length) return Array.from(dt.files);          // 不支持目录 API → 当普通文件
  const readAll = reader => new Promise(res => reader.readEntries(res, () => res([])));
  async function walk(entry, prefix){
    if (entry.isFile){
      const f = await new Promise((res, rej) => entry.file(res, rej));
      const rel = prefix + f.name;
      out.push(new File([f], rel, {type: f.type, lastModified: f.lastModified}));
    } else if (entry.isDirectory){
      const reader = entry.createReader();
      let batch;
      while ((batch = await readAll(reader)).length){
        for (const e of batch) await walk(e, prefix + entry.name + '/');
      }
    }
  }
  for (const en of entries) await walk(en, '');
  return out;
}

drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', async e => {
  e.preventDefault(); drop.classList.remove('over');
  const fs = await filesFromDrop(e.dataTransfer);
  if (fs.length) upload(fs);
});
pick.onchange = () => { if (pick.files.length) upload(pick.files); };
$('pickDir').onchange = () => { if ($('pickDir').files.length) upload($('pickDir').files); };

// ===== 来源模式：本机文件夹（默认，零复制）/ 上传 =====
let srcMode = 'dir';
function applySrcMode(){
  document.querySelectorAll('.tab[data-srcmode]').forEach(x =>
    x.classList.toggle('on', x.dataset.srcmode === srcMode));
  $('dirBox').classList.toggle('hide', srcMode !== 'dir');
  $('dropBox').classList.toggle('hide', srcMode !== 'upload');
  $('start').disabled = srcMode === 'dir' ? !$('src_dir').value.trim() : !runId;
}
document.querySelectorAll('.tab[data-srcmode]').forEach(t => {
  t.onclick = () => { srcMode = t.dataset.srcmode; applySrcMode(); };
});
applySrcMode();

// 路径输入框回车 = 直接确定这个路径（不弹对话框，最省事）
$('src_dir').addEventListener('keydown', e => {
  if (e.key === 'Enter'){
    e.preventDefault();
    const p = $('src_dir').value.trim();
    if (!p) return;
    $('start').disabled = false;
    checkDir();                 // 顺便校验并显示音频数量
  }
});

// ===== 服务端目录浏览器 =====
let browseCur = '';
function renderBrowse(d){
  browseCur = d.path || '';
  $('browsePath').textContent = browseCur || '（盘符列表）';
  const items = (d.dirs || []).map(x => {
    const has = x.audio > 0;
    const cnt = x.audio < 0 ? '无权限' : (x.audio > 0 ? `${x.audio} 个音频` : '');
    return `<div class="diritem ${has?'has':''}" data-p="${x.path.replace(/"/g,'&quot;')}">
              <span>${has?'🎵':'📁'} ${x.name}</span><span class="cnt">${cnt}</span></div>`;
  }).join('');
  $('browseList').innerHTML = items || '<div class="diritem"><span class="cnt">（没有子文件夹）</span></div>';
  document.querySelectorAll('.diritem[data-p]').forEach(el => {
    el.onclick = () => browseTo(el.dataset.p);
  });
  if (d.exists === false){
    $('browseInfo').textContent = d.error || '路径不存在';
  } else if (d.is_root){
    $('browseInfo').textContent = '选择一个盘符开始';
  } else {
    $('browseInfo').textContent = `本层 ${d.audio} 个音频 · 含子目录共 ${d.audio_recursive} 个`;
  }
}
async function browseTo(p){
  const r = await fetch('/api/browse?path=' + encodeURIComponent(p || ''));
  if (!r.ok) return;
  renderBrowse(await r.json());
}
$('btnBrowse').onclick = async () => {
  const btn = $('btnBrowse');
  const old = btn.textContent;
  btn.disabled = true; btn.textContent = '等待对话框…';
  try {
    const r = await fetch('/api/pick-folder', {method:'POST', headers:{'Content-Type':'application/json'},
                                               body: JSON.stringify({path: $('src_dir').value.trim()})});
    if (r.status === 501 || r.status === 500){
      // 系统对话框不可用（非 Windows / 调用失败）→ 退回内置目录树
      const e = await r.json().catch(() => ({}));
      $('browseModal').classList.remove('hide');
      await browseTo($('src_dir').value.trim() || '');
      $('browseInfo').textContent = e.detail || '系统对话框不可用，请用下面的目录树选择';
      return;
    }
    if (!r.ok) throw new Error(r.status);
    const j = await r.json();
    if (j.path && !j.cancelled){
      $('src_dir').value = j.path;
      $('start').disabled = false;
      checkDir();
    }
  } catch(e){
    $('browseModal').classList.remove('hide');
    await browseTo($('src_dir').value.trim() || '');
    $('browseInfo').textContent = '系统对话框调用失败，请用下面的目录树选择';
  } finally {
    btn.disabled = false; btn.textContent = old;
  }
};
// 输入框里直接回车 = 用系统对话框选
$('src_dir').onkeydown = e => { if (e.key === 'Enter'){ e.preventDefault(); $('btnBrowse').click(); } };
$('browseClose').onclick = () => $('browseModal').classList.add('hide');
$('browseModal').onclick = e => { if (e.target === $('browseModal')) $('browseModal').classList.add('hide'); };
$('browseUp').onclick = () => {
  if (!browseCur) return;
  const par = browseCur.replace(/[\\/][^\\/]+[\\/]?$/, '');
  browseTo(par && par !== browseCur ? par : '');
};
$('browsePick').onclick = () => {
  if (!browseCur){ $('browseInfo').textContent = '请先进入一个文件夹'; return; }
  $('src_dir').value = browseCur;
  $('start').disabled = false;
  $('browseModal').classList.add('hide');
  checkDir();
};

// ===== 检查文件夹里的音频 =====
async function checkDir(){
  const p = $('src_dir').value.trim();
  if (!p) return;
  $('dirInfo').textContent = '检查中…';
  const r = await fetch('/api/browse?path=' + encodeURIComponent(p));
  const d = await r.json();
  if (d.exists === false){
    $('dirInfo').innerHTML = `<span style="color:var(--bad)">${d.error || '路径不存在'}</span>`;
    return;
  }
  const sub = (d.dirs || []).filter(x => x.audio > 0).length;
  $('dirInfo').innerHTML = `本层 <b>${d.audio}</b> 个音频 · 含子目录共 <b>${d.audio_recursive}</b> 个`
    + (sub ? ` · ${sub} 个子文件夹里有音频` : '')
    + (d.audio_recursive ? '' : ' <span style="color:var(--bad)">（没找到音频）</span>');
}
$('btnCheck').onclick = checkDir;
$('src_dir').oninput = () => { $('start').disabled = !$('src_dir').value.trim(); };

async function upload(files){
  const fd = new FormData();
  for (const f of files){
    // 文件夹上传时保留相对路径（webkitRelativePath 或拖拽时构造的 name），
    // 这样输出目录结构能跟源文件夹一致
    const rel = f.webkitRelativePath || f.name;
    fd.append('files', f, rel);
  }
  $('files').innerHTML = '<div>上传中…<span></span></div>';
  const r = await fetch('/api/upload', {method:'POST', body: fd});
  if (!r.ok){ $('files').innerHTML = '<div>上传失败<span></span></div>'; return; }
  const j = await r.json();
  setRunId(j.run_id);
  $('files').innerHTML = j.files.map(f => `<div>${f.name}<span>${f.size}</span></div>`).join('');
  $('start').disabled = false;
  $('resultCard').classList.add('hide');
  $('log').textContent = '';
  $('stateTxt').textContent = '';
  $('fileTxt').textContent = '—';
  $('taskTxt').textContent = '—';
  $('fileBar').style.width = '0%';
  $('taskBar').style.width = '0%';
}

$('backend').onchange = () => {
  const v = $('backend').value;
  $('custom_url_box').classList.toggle('hide', v !== 'custom');
  $('key_box').classList.toggle('hide', !(v === 'custom' || v === 'deepseek'));
  if (v === 'deepseek') $('model_name').value = 'deepseek-chat';
  else if (v === 'custom') { if(!$('base_url').value) $('base_url').value = 'http://127.0.0.1:8080/v1'; }
  else $('model_name').value = 'sakura';
};

// 切换提示词模板时，自动填入该代模型的官方推荐采样参数
const STYLE_DEFAULTS = {
  v1: {temperature: 0.1, top_p: 0.3, frequency_penalty: 0.15},
  v3: {temperature: 0.3, top_p: 0.8, frequency_penalty: 0.0}
};
function applyStyleDefaults(){
  const d = STYLE_DEFAULTS[$('prompt_style').value];
  if (!d) return;
  $('temperature').value = d.temperature;
  $('top_p').value = d.top_p;
  $('frequency_penalty').value = d.frequency_penalty;
}
$('prompt_style').onchange = applyStyleDefaults;

// 一键切换「模型 + 提示词模板 + 采样参数」（三者必须配套，混用会掉质量）
const MODEL_PRESETS = {
  'v1-7b':  {model: 'sakura',      style: 'v1'},
  'v37-7b': {model: 'sakura37',    style: 'v3'},
  'v4-4b':  {model: 'galtransl4b', style: 'v3'}
};
$('model_preset').onchange = () => {
  const p = MODEL_PRESETS[$('model_preset').value];
  if (!p) return;
  $('model_name').value = p.model;
  $('prompt_style').value = p.style;
  applyStyleDefaults();
};

// 首屏就同步一次：否则 HTML 里写死的采样参数会跟选中的模板不配套
applyStyleDefaults();
// 队列状态在服务端，页面加载后由末尾的 tick() 自动接上，无需恢复逻辑

// ===== 术语表 / ASR 修正表：文件载入 + 本地持久化 =====
['glossary_text', 'fix_text'].forEach(id => {
  const saved = localStorage.getItem('onsei2lrc_' + id);
  // 用 !== null 判断：用户显式「清空」后存的是空串，也要能覆盖服务端默认值
  if (saved !== null) $(id).value = saved;
  $(id).addEventListener('input', () => localStorage.setItem('onsei2lrc_' + id, $(id).value));
});
document.querySelectorAll('button[data-load]').forEach(btn => {
  btn.onclick = () => $(btn.dataset.file).click();
});
['glossary_file', 'fix_file'].forEach(fid => {
  $(fid).onchange = () => {
    const f = $(fid).files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = () => {
      const target = fid === 'glossary_file' ? 'glossary_text' : 'fix_text';
      $(target).value = r.result;
      localStorage.setItem('onsei2lrc_' + target, r.result);
    };
    r.readAsText(f, 'utf-8');
    $(fid).value = '';
  };
});
document.querySelectorAll('button[data-clear]').forEach(btn => {
  btn.onclick = () => {
    const id = btn.dataset.clear;
    $(id).value = '';
    localStorage.removeItem('onsei2lrc_' + id);
  };
});

// 转写与翻译恒定执行，翻译相关控件始终可用，不再需要联动灰化

$('start').onclick = async () => {
  if (srcMode === 'dir'){
    const p = $('src_dir').value.trim();
    if (!p){ $('log').textContent = '请先选择或填写文件夹路径'; return; }
    $('start').disabled = true;
    const r = await fetch('/api/use-dir', {method:'POST', headers:{'Content-Type':'application/json'},
                                           body: JSON.stringify({path: p})});
    if (!r.ok){
      const e = await r.json();
      $('log').textContent = '错误：' + (e.detail || r.status);
      $('start').disabled = false;
      return;
    }
    const j = await r.json();
    setRunId(j.run_id);
    $('log').textContent = '';
    $('resultCard').classList.add('hide');
  }
  if (!runId) return;
  const payload = {
    run_id: runId,
    lrc_mode: $('lrc_mode').value,
    chunk_mode: $('chunk_mode').value,
    backend: $('backend').value,
    model_name: $('model_name').value.trim(),
    base_url: $('base_url').value.trim(),
    api_key: $('api_key').value.trim(),
    max_chars: +$('max_chars').value,
    merge_gap: +$('merge_gap').value,
    batch_size: +$('batch_size').value,
    workers: +$('workers').value,
    temperature: +$('temperature').value,
    top_p: +$('top_p').value,
    frequency_penalty: +$('frequency_penalty').value,
    max_tokens: +$('max_tokens').value,
    prompt_style: $('prompt_style').value,
    glossary_text: $('glossary_text').value,
    fix_text: $('fix_text').value
  };
  $('start').disabled = true;
  $('resultCard').classList.add('hide');
  const r = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if (!r.ok){ const e = await r.json(); $('log').textContent = '错误：' + (e.detail || r.status); $('start').disabled = false; return; }
  const jj = await r.json();
  // 加完就清空输入，方便连续把多个文件夹丢进队列
  if (srcMode === 'dir'){
    $('src_dir').value = '';
    $('start').disabled = true;
  } else {
    runId = null;
    $('files').innerHTML = '';
    $('start').disabled = true;
  }
  clearRunId();
  $('log').textContent = '';
  since = 0;
  tick();
};

$('cancel').onclick = async () => {
  const id = Q.current && Q.current.id;
  if (!id) return;
  await fetch('/api/cancel', {method:'POST', headers:{'Content-Type':'application/json'},
                              body: JSON.stringify({run_id: id})});
  activeId = null; since = 0;
  tick();
};

// ===== 任务队列 =====
// 一个文件夹 = 一个任务。任务由服务端串行执行，前端只负责展示与增删。
// 日志/进度区跟随「当前正在跑的任务」；没有在跑时跟随最后完成的任务。
// 注意：since 已在上面声明过（let runId = null, since = 0），这里不能再声明一次
// ——同一作用域重复 let 会让整个 <script> 解析失败，所有按钮一起失效。
let Q = {pending: [], current: null, finished: []};
let activeId = null;

// 轮询循环只允许存在一条。
// tick() 会被好几个地方调用（加入队列 / 取消 / 继续 / 移除），而它自己又会
// setTimeout 排下一次——之前每调一次就多起一条永久循环，两条循环各自拿着
// **同一个** since 去请求日志，于是同一批行被追加两遍。
// 这就是「日志重复两遍」的根因。用一个自增代号解决：谁最后调用谁活着，
// 旧循环在它下一轮醒来时自己退出；配合 pollStatus 的在飞锁，双保险。
let tickGen = 0;

async function tick(){
  const gen = ++tickGen;
  while (gen === tickGen){
    try {
      Q = await (await fetch('/api/queue')).json();
    } catch(e){ await sleep(1200); continue; }
    if (gen !== tickGen) return;                 // 已被更新的循环取代
    renderQueue();
    // 正在跑的任务换了 → 日志区切到新任务，写回结果也清掉（那是上一个任务的）
    const cur = Q.current && Q.current.id;
    if (cur && cur !== activeId){ activeId = cur; since = 0; $('log').textContent = ''; wbMsg = ''; }
    if (activeId) await pollStatus(activeId);
    await sleep(800);
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

function fmtSec(s){
  if (s == null) return '';
  s = Math.round(s);
  return s >= 60 ? `${Math.floor(s/60)}分${String(s%60).padStart(2,'0')}秒` : `${s}秒`;
}

function qname(src){
  if (!src) return '(未知)';
  const p = src.replace(/[\\/]+$/, '').split(/[\\/]/);
  return p[p.length - 1] || src;
}

function renderQueue(){
  const el = $('queue');
  const rows = [];
  // 正在跑（或正在等翻译后端）
  if (Q.current){
    const w = Q.current.waiting, r = Q.current.resume;
    rows.push(`<div class="qitem ${w ? 'err' : 'run'}" data-id="${Q.current.id}">
      <span class="qidx">${w ? '⏸' : '▶'}</span>
      <span class="qname" title="${Q.current.src || ''}">${qname(Q.current.src)}</span>
      ${w ? '' : '<span class="qbar"><i id="qcur" style="width:0%"></i></span>'}
      <span class="qstat" id="qcurtxt">${w ? '等待翻译后端…' : (r ? '继续上次进度…' : '处理中…')}</span>
    </div>`);
  }
  // 待跑
  Q.pending.forEach(p => rows.push(`<div class="qitem" data-id="${p.id}">
      <span class="qidx">${p.position}</span>
      <span class="qname" title="${p.src || ''}">${qname(p.src)}</span>
      <span class="qstat">排队中${p.files_total ? ' · ' + p.files_total + ' 个文件' : ''}</span>
      <button class="qbtn" data-remove="${p.id}">移除</button>
    </div>`));
  // 已完成（最近的在前）
  Q.finished.slice(0, 8).forEach(d => {
    const ok = d.state === 'done';
    const dl = ok && d.outputs && d.outputs.length
      ? `<span class="qdl">${d.outputs.filter(o => o.kind === 'lrc' || o.kind === 'all')
          .map(o => `<a href="#" data-dl="${d.id}" data-kind="${o.kind}">${o.kind === 'lrc' ? '字幕' : '含音频'}</a>`).join('')}</span>`
      : '';
    rows.push(`<div class="qitem ${ok ? 'ok' : 'err'}" data-id="${d.id}">
      <span class="qidx">${ok ? '✓' : '✗'}</span>
      <span class="qname" title="${d.src || ''}">${qname(d.src)}</span>
      <span class="qstat">${ok ? '完成 · ' + fmtSec(d.elapsed) : '出错'}</span>
      ${dl}
      ${ok ? '' : `<button class="qbtn" data-retry="${d.id}">继续</button>`}
    </div>`);
  });
  el.innerHTML = rows.length ? rows.join('')
    : '<div class="qempty">队列为空。选好文件夹后点「加入队列」，可以连续加多个。</div>';

  el.querySelectorAll('[data-remove]').forEach(b => {
    b.onclick = async () => {
      await fetch('/api/queue/remove', {method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({run_id: b.dataset.remove})});
      tick();
    };
  });
  el.querySelectorAll('[data-dl]').forEach(a => {
    a.onclick = e => { e.preventDefault(); location.href = dlHref(a.dataset.kind, a.dataset.dl); };
  });
  el.querySelectorAll('[data-retry]').forEach(b => {
    b.onclick = async () => {
      b.disabled = true; b.textContent = '…';
      const r = await fetch('/api/queue/retry', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({run_id: b.dataset.retry})});
      if (!r.ok){ const e = await r.json(); alert('重试失败：' + (e.detail || r.status)); }
      tick();
    };
  });
  if ($('cancel')) $('cancel').disabled = !Q.current;
}

// ---- 日志渲染 ------------------------------------------------------------------------
// 一行一个 div：按前缀着色、长行折行，比原来一整块 textContent 好扫读得多。
// 「精简」用 CSS 把每 20 块一条的转写进度折掉——长任务里它们是绝大多数行。
const LOG_MAX_LINES = 2000;

const LOG_RULES = [
  [/^\[(错误|失败)\]|Traceback|Error[:：]/,                       'lg-err'],
  [/^\[(等待|警告)\]|^\[重试\]|^\[后端\][^\n]*[⚠❌]/,              'lg-warn'],
  [/^\[后端\][^\n]*✅|^\[(完成|写回)\]|✅/,                        'lg-ok'],
  [/^=+ ?\[/,                                                    'lg-stage'],
  [/^\s*…\d+\/\d+\s*块/,                                         'lg-prog'],
  [/^\[(ASR|VAD|混合切块|清洗|翻译|缓存|输入|输出|配置|执行|CUDA|术语表|下载|跳过转码)\]/, 'lg-step'],
];

function logClass(body){
  for (const [re, cls] of LOG_RULES) if (re.test(body)) return cls;
  return '';
}

function appendLog(lines){
  const el = $('log');
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  const frag = document.createDocumentFragment();
  for (const raw of lines){
    if (raw == null) continue;
    let ts = '', body = raw;
    const m = /^\[(\d{2}:\d{2}:\d{2})\]\s?/.exec(raw);   // 服务端加的时间戳单独成列
    if (m){ ts = m[1]; body = raw.slice(m[0].length); }
    const d = document.createElement('div');
    d.className = 'ln ' + logClass(body);
    if (ts){
      const s = document.createElement('span');
      s.className = 'ts';
      s.textContent = ts;
      d.appendChild(s);
    }
    d.appendChild(document.createTextNode(body));
    frag.appendChild(d);
  }
  el.appendChild(frag);
  while (el.childElementCount > LOG_MAX_LINES) el.removeChild(el.firstChild);
  if (atBottom) el.scrollTop = el.scrollHeight;
}

let polling = false;                  // 同一时刻只允许一个 /api/status 在飞

async function pollStatus(id){
  // 两个调用同时在飞时，两边都会用切换前读到的 since 去请求，
  // 拿到同一批行各追加一遍 → 日志重复。这里直接挡掉后来者。
  if (polling) return;
  polling = true;
  try { await pollStatusOnce(id); } finally { polling = false; }
}

async function pollStatusOnce(id){
  let j;
  try { j = await (await fetch(`/api/status?run_id=${encodeURIComponent(id)}&since=${since}`)).json(); }
  catch(e){ return; }
  if (j.detail) return;
  if (typeof j.next === 'number' && j.next < since){   // 服务端日志被截断过，重新同步
    since = 0; $('log').textContent = '';
    return;
  }
  if (j.log && j.log.length) appendLog(j.log);
  if (typeof j.next === 'number') since = j.next;

  const p = j.progress || {};
  const fPct = Math.round((p.file_frac || 0) * 100);
  const tPct = Math.round((p.task_frac || 0) * 100);
  $('fileBar').style.width = fPct + '%';
  $('taskBar').style.width = tPct + '%';
  $('fileTxt').textContent = p.file_name
      ? `${p.file_index}/${p.files_total} · ${p.file_name} · ${p.file_stage || ''} ${fPct}%${p.stage_detail ? ' · ' + p.stage_detail : ''}`
      : '—';
  $('taskTxt').textContent = p.files_total
      ? `${tPct}% · 已用 ${fmtSec(p.elapsed)}${p.eta ? ' · 预计剩余 ' + fmtSec(p.eta) : ''}`
      : '—';
  // 队列行里的进度条跟着当前任务走
  const qb = $('qcur'), qt = $('qcurtxt');
  if (qb && j.state === 'running'){
    qb.style.width = tPct + '%';
    if (qt) qt.textContent = `${tPct}% · 已用 ${fmtSec(p.elapsed)}`;
  }
  const map = {uploaded:'待处理', queued:'排队中', running:'处理中…',
               done:'✅ 完成', error:'❌ 出错'};
  $('stateTxt').textContent = map[j.state] || j.state;

  if (j.state === 'done'){
    $('fileBar').style.width = '100%'; $('taskBar').style.width = '100%';
    $('taskTxt').textContent = `100% · 总耗时 ${fmtSec(j.elapsed)}`;
    renderDownloads(j, id);
  } else if (j.state === 'error' && j.error){
    $('log').textContent += '\n\n[失败] ' + j.error;
  }
}

// ===== 下载区：三个复选项 + 两个下载项 =====
// 勾选只影响压缩包内容，所以直接拼进下载链接的查询参数，无需重跑任务。
// 队列模式下可能有多个已完成任务，链接里要带各自的 run_id。
let dlInfo = {has_audio: false, need_mp3: false};
let lastStatus = null;      // 最近一次已完成任务的状态，供写回提示复用
let wbMsg = '';             // 写回结果；非空时压过默认提示，改动选项或换任务时清掉

function dlHref(kind, rid){
  const ja = $('dl_ja').checked ? 1 : 0;
  const mp3 = (kind === 'all' && $('dl_mp3').checked) ? 1 : 0;
  const tr = $('dl_tr').checked ? 1 : 0;
  return `/api/download/${rid || activeId}/${kind}?ja=${ja}&mp3=${mp3}&tr=${tr}`;
}

function renderDownloads(j, rid){
  lastStatus = j;
  dlInfo = {has_audio: !!j.has_audio, need_mp3: !!j.need_mp3};
  // 没有非 MP3 音频时，「WAV 自动转 MP3」没有意义
  const mp3Box = $('dl_mp3');
  mp3Box.disabled = !dlInfo.need_mp3;
  if (!dlInfo.need_mp3) mp3Box.checked = false;
  mp3Box.parentElement.title = dlInfo.need_mp3 ? '' : '本次没有 WAV 等非 MP3 音频';

  const outs = (j.outputs || []).filter(o => o.kind === 'lrc' || (o.kind === 'all' && dlInfo.has_audio));
  const dl = $('dl');
  dl.innerHTML = outs.map(o =>
    `<a data-kind="${o.kind}" data-rid="${rid || activeId}" href="${dlHref(o.kind, rid)}">`
    + `${o.label}<small>${o.est ? '约 ' : ''}${o.size || ''} · 点击下载</small></a>`).join('');
  $('resultCard').classList.toggle('hide', !outs.length);

  // 写回按钮：只有「文件夹模式 + 已有字幕」才可用
  const rid2 = rid || activeId;
  const canWb = !!(j.src_dir && j.lrc_count && rid2);
  $('btnWriteBack').disabled = !canWb;
  $('btnWriteBack').dataset.rid = rid2 || '';
  // 写回结果优先显示——否则会被每 800ms 一次的轮询重渲染冲掉，用户看不到反馈
  $('wbInfo').textContent = wbMsg ? wbMsg
      : (canWb ? wbText(j) : (j.src_dir ? '' : '（仅文件夹模式可用）'));
  syncDlHrefs();
}

// 写回会写几个文件，跟「附带日文 lrc」保持一致
function wbCount(j){
  if (!$('dl_ja').checked) return Math.max(0, (j.lrc_count || 0) - (j.lrc_ja_count || 0));
  return j.lrc_count || 0;
}
function wbText(j){
  return `将 ${wbCount(j)} 个 .lrc 复制到 ${j.src_dir}`;
}

// 字幕写回：按钮触发，不再跑任务时自动执行
$('btnWriteBack').onclick = async () => {
  const b = $('btnWriteBack'), rid = b.dataset.rid;
  if (!rid) return;
  if (!confirm(`确定把字幕写回音频文件夹？\n\n${$('wbInfo').textContent}\n\n`
               + `只会新增 .lrc 文件，同名文件会被覆盖，不会改动或删除任何原有文件。`)) return;
  b.disabled = true;
  const old = b.textContent;
  b.textContent = '写入中…';
  try {
    const r = await fetch('/api/write-back', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({run_id: rid, ja: $('dl_ja').checked ? 1 : 0})});
    const j = await r.json();
    if (!r.ok){ wbMsg = ''; $('wbInfo').textContent = '失败：' + (j.detail || r.status); return; }
    let msg = `✓ 已写入 ${j.written}/${j.total} 个字幕`;
    if (j.failed && j.failed.length) msg += `，${j.failed.length} 个失败`;
    wbMsg = msg + ' → ' + j.dir;
    $('wbInfo').textContent = wbMsg;
  } catch(e){
    wbMsg = '';
    $('wbInfo').textContent = '失败：' + e;
  } finally {
    b.disabled = false; b.textContent = old;
  }
};

function syncDlHrefs(){
  document.querySelectorAll('#dl a[data-kind]').forEach(a => {
    a.href = dlHref(a.dataset.kind, a.dataset.rid);
  });
}

['dl_ja','dl_mp3','dl_tr'].forEach(id => { $(id).onchange = () => {
  syncDlHrefs();
  // 「附带日文 lrc」同时决定写回几个文件，提示要跟着变；写回结果也一并清掉
  wbMsg = '';
  if (lastStatus && $('btnWriteBack').dataset.rid) $('wbInfo').textContent = wbText(lastStatus);
}; });

// 启动队列轮询
tick();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------------------
# 启动
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="onsei2lrc WebUI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-restore", action="store_true",
                    help="不恢复上次的任务（默认会自动恢复并继续被中断的）")
    a = ap.parse_args()
    if not a.no_restore:
        try:
            _restore_tasks()
        except Exception as e:
            # 恢复失败绝不能挡住服务启动——顶多是丢一次断点继续
            print(f"[警告] 恢复上次任务失败，已跳过：{type(e).__name__}: {e}", flush=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"onsei2lrc WebUI → http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
