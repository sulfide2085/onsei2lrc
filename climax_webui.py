# -*- coding: utf-8 -*-
"""climax_webui —— 本地音频播放器（带高潮点提示与人工标注）

本地音频播放器：选文件 → 生成高潮点 → 播放时提示。

    python climax_webui.py                  # 默认 127.0.0.1:7861
    python climax_webui.py --port 8080

不复制文件、不上传——直接按路径读取本机音频。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import climax_finder as CF  # noqa: E402

AUDIO_EXTS = sorted(CF.AUDIO_EXTS)
WARN_LEAD = 3.0          # 提前几秒给「即将到达」的提示
BROWSE_TTL = 45.0
BROWSE_CACHE: Dict[str, Any] = {}
BROWSE_LOCK = threading.Lock()

# 人工标注落盘（追加式 JSONL，不入库）
FEEDBACK_FILE = ROOT / "climax_feedback.jsonl"      # ✓/✗ 判定（旧方式）
MARKS_FILE = ROOT / "climax_marks.jsonl"            # 精确高潮时刻（新方式）
DONE_FILE = ROOT / "climax_tracks_done.jsonl"       # 「本轨已标完」标记
FEEDBACK_LOCK = threading.Lock()


def track_key(name: str) -> str:
    """从文件名提取音轨编号，与官方标注的键一致（track04_xxx → "04"）。"""
    import re
    m = re.search(r"track[_\-\s]*(\d+)", name, re.I)
    if m:
        return m.group(1).zfill(2)
    m = re.search(r"(\d+)", name)
    return m.group(1).zfill(2) if m else ""


def work_key(path: str) -> str:
    """从路径提取作品编号（RJxxxxxxxx）。"""
    import re
    m = re.search(r"(RJ\d+)", str(path))
    return m.group(1) if m else ""

app = FastAPI(title="climax player")

# ---------------------------------------------------------------------------
# 模型（7 MB，首次用到时加载；分析是 CPU 密集，串行执行）
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_MTIME = 0.0
_MODEL_LOCK = threading.Lock()
_ANALYZE_LOCK = threading.Lock()


def get_model() -> "CF.ClimaxModel":
    """按需加载模型；**模型文件变了就自动重载**。

    这样你标完、跑完 `python _ml\\train_model.py` 之后，
    下一次点「分析高潮点」就用上新模型了，不用重启服务。
    """
    global _MODEL, _MODEL_MTIME
    with _MODEL_LOCK:
        try:
            mt = CF.MODEL_FILE.stat().st_mtime
        except OSError:
            mt = 0.0
        if _MODEL is None or mt != _MODEL_MTIME:
            _MODEL = CF.ClimaxModel()
            _MODEL_MTIME = mt
    return _MODEL


# ---------------------------------------------------------------------------
# 目录浏览
# ---------------------------------------------------------------------------
def _quick_count(d: Path) -> int:
    """只数**直接子文件**里的音频 —— 一次 iterdir，约 0 ms。

    以前这里做递归 rglob，还要给每个子目录 0.35 秒预算：
    D:\\ 有 8 个子目录就累积到 12.9 秒，完全不可用。
    实测 iterdir 任何目录都是 0~2 ms，所以只用它。
    """
    n = 0
    try:
        for e in d.iterdir():
            try:
                if e.is_file() and e.suffix.lower() in CF.AUDIO_EXTS:
                    n += 1
            except OSError:
                continue
    except (PermissionError, OSError):
        pass
    return n


@app.get("/api/browse")
def api_browse(path: str = ""):
    key = path or "__drives__"
    now = time.time()
    with BROWSE_LOCK:
        hit = BROWSE_CACHE.get(key)
        if hit and now - hit[0] < BROWSE_TTL:
            return JSONResponse(hit[1])

    if not path:
        import string
        drives = []
        for c in string.ascii_uppercase:
            d = Path(f"{c}:\\")
            try:
                if d.exists():
                    drives.append({"name": f"{c}:", "path": str(d), "type": "drive"})
            except OSError:
                pass
        for d in (Path.home(), Path.home() / "Music", Path.home() / "Desktop"):
            if d.is_dir():
                drives.append({"name": d.name or str(d), "path": str(d), "type": "dir"})
        out = {"path": "", "parent": None, "dirs": drives, "audio": [], "lrcs": []}
    else:
        p = Path(path)
        if not p.is_dir():
            raise HTTPException(404, f"不是目录：{path}")
        dirs, audio, lrcs = [], [], []
        try:
            entries = sorted(p.iterdir(), key=lambda q: (q.is_file(), q.name.lower()))
        except (PermissionError, OSError) as e:
            raise HTTPException(403, f"无法读取：{e}")
        for e in entries:
            try:
                if e.is_dir():
                    if e.name.startswith(".") or e.name in ("$RECYCLE.BIN", "System Volume Information"):
                        continue
                    n = _quick_count(e)                    # 只数直接子文件
                    has_sub = False
                    if n == 0:                             # 空的才多看一眼有没有下级目录
                        try:
                            has_sub = any(x.is_dir() for x in e.iterdir())
                        except OSError:
                            pass
                    dirs.append({"name": e.name, "path": str(e),
                                 "audio": n, "has_sub": has_sub})
                elif e.is_file():
                    sfx = e.suffix.lower()
                    if sfx in CF.AUDIO_EXTS:
                        audio.append({"name": e.name, "path": str(e),
                                      "size": e.stat().st_size})
                    elif sfx in (".lrc", ".srt"):
                        lrcs.append({"name": e.name, "path": str(e),
                                     "size": e.stat().st_size})
            except OSError:
                continue
        audio.sort(key=lambda x: x["name"].lower())
        lrcs.sort(key=lambda x: x["name"].lower())
        out = {"path": str(p), "parent": str(p.parent) if p.parent != p else "",
               "dirs": dirs, "audio": audio, "lrcs": lrcs}

    with BROWSE_LOCK:
        BROWSE_CACHE[key] = (now, out)
        if len(BROWSE_CACHE) > 300:
            for k in sorted(BROWSE_CACHE, key=lambda k: BROWSE_CACHE[k][0])[:100]:
                BROWSE_CACHE.pop(k, None)
    return JSONResponse(out)


# ---------------------------------------------------------------------------
# 音频 / 歌词
# ---------------------------------------------------------------------------
@app.get("/api/audio")
def api_audio(path: str):
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "文件不存在")
    if p.suffix.lower() not in CF.AUDIO_EXTS:
        raise HTTPException(400, "不是支持的音频格式")
    return FileResponse(p, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store"})


def _find_lrc(audio: Path) -> Optional[Path]:
    for pat in (f"{audio.stem}.ja.lrc", f"{audio.stem}.lrc",
                f"{audio.stem}.zh-ja.lrc"):
        c = audio.parent / pat
        if c.exists():
            return c
    c = list(audio.parent.glob(f"{audio.stem}*.lrc"))
    return c[0] if c else None


@app.get("/api/lrc")
def api_lrc(path: str, lrc: str = ""):
    a = Path(path)
    target = Path(lrc) if lrc else _find_lrc(a)
    if not target or not target.exists():
        return {"found": False, "lines": []}
    lines: List[Dict[str, Any]] = []
    import re
    for raw in target.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("["):
            continue
        m = re.match(r"^\[(\d{1,2}):(\d{2}(?:\.\d+)?)\]\s*(.*)$", line)
        if not m:
            continue
        t = int(m.group(1)) * 60 + float(m.group(2))
        txt = m.group(3).strip()
        if txt:
            lines.append({"time": round(t, 2), "text": txt})
    lines.sort(key=lambda x: x["time"])
    return {"found": True, "name": target.name, "lines": lines}


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------
class AnalyzeReq(BaseModel):
    path: str
    lrc: str = ""
    top: int = 6
    min_prob: float = 0.0


@app.post("/api/analyze")
def api_analyze(req: AnalyzeReq, request: Request):
    p = Path(req.path)
    if not p.is_file():
        raise HTTPException(404, "文件不存在")
    t0 = time.time()
    with _ANALYZE_LOCK:
        try:
            model = get_model()
        except Exception as e:
            raise HTTPException(500, f"模型加载失败：{e}")
        tdirs = [Path(req.lrc).parent] if req.lrc else None
        res = CF.find_climaxes(p, top=max(1, min(req.top, 20)), verbose=False,
                               model="ml", model_obj=model,
                               transcript_dirs=tdirs)
    if res.get("error"):
        return {"ok": False, "error": res["error"]}
    cands = []
    for c in res["candidates"]:
        prob = float(c.get("probability", 0.0))
        if prob < req.min_prob:
            continue
        # 带上 21 个模型特征：人工标注落盘时要一起存，
        # 否则音频一旦移动/改名，标好的数据就没法用于训练了。
        feats = {k: round(float(c.get(k, 0.0)), 4) for k in CF.MODEL_FEATS}
        cands.append({"time": c["time"], "mmss": c["mmss"], "prob": round(prob, 4),
                      "raw": c["score"], "conf": c["confidence"],
                      "text": c.get("text", ""), "note": c.get("confidence_note", ""),
                      "merged": int(c.get("merged", 1)),
                      "feats": feats})
    return {"ok": True, "duration": round(res.get("duration", 0), 2),
            "candidates": cands, "has_climax": res.get("has_climax"),
            "max_prob": res.get("max_prob"), "elapsed": round(time.time() - t0, 1),
            "merge_gap": res.get("merge_gap"),
            "merged_total": res.get("merged_total", 0),
            "lrc": _find_lrc(p).name if _find_lrc(p) else "",
            "model": model.meta}


@app.get("/api/status")
def api_status():
    try:
        m = get_model()
        try:
            mtime = CF.MODEL_FILE.stat().st_mtime
        except OSError:
            mtime = 0.0
        pend = sorted({r.get("audio", "") for r in _read_jsonl(DONE_FILE)
                       if r.get("done", True) and r.get("audio")})
        newer = [a for a in pend
                 if any(r.get("audio") == a and r.get("marked_at", "")
                        for r in _read_jsonl(MARKS_FILE))]
        trained_at = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else ""
        return {"ok": True, "meta": m.meta,
                "feedback": _feedback_stats(), "marks": _marks_stats(),
                "model_trained_at": trained_at,
                "trained_works": m.meta.get("works_user_gt", []),
                "pending_works": sorted({work_key(a) for a in newer
                                          if work_key(a) not in m.meta.get("works_user_gt", [])})}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# 人工标注（听的时候标对错，用于后续继续训练）
# ---------------------------------------------------------------------------
class FeedbackReq(BaseModel):
    audio: str
    time: float
    mmss: str = ""
    prob: float = 0.0
    raw: float = 0.0
    text: str = ""
    verdict: Optional[int] = None      # 1=真是高潮, 0=不是, None=撤销
    feats: Dict[str, float] = {}


def _read_feedback() -> List[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    out = []
    for line in FEEDBACK_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _feedback_stats() -> Dict[str, int]:
    rows = _read_feedback()
    latest: Dict[tuple, int] = {}
    for r in rows:                     # 后来的覆盖先前的
        latest[(r.get("audio"), round(r.get("time", 0), 1))] = r.get("verdict", 0)
    pos = sum(1 for v in latest.values() if v == 1)
    return {"total": len(latest), "pos": pos, "neg": len(latest) - pos}


@app.get("/api/feedback")
def api_feedback(audio: str = ""):
    """返回某个音频上已有的标注（用于刷新页面后恢复状态）。"""
    rows = _read_feedback()
    latest: Dict[str, int] = {}
    for r in rows:
        if audio and r.get("audio") != audio:
            continue
        latest[f"{r.get('time', 0):.1f}"] = r.get("verdict", 0)
    return {"ok": True, "verdicts": latest, "stats": _feedback_stats()}


@app.post("/api/feedback")
def api_feedback_post(req: FeedbackReq):
    if req.verdict not in (0, 1, None):
        raise HTTPException(400, "verdict 只能是 0 / 1 / null")
    rec = {
        "audio": req.audio,
        "name": Path(req.audio).name,
        "time": round(float(req.time), 2),
        "mmss": req.mmss,
        "prob": round(float(req.prob), 4),
        "raw": round(float(req.raw), 4),
        "text": req.text[:120],
        "verdict": req.verdict,
        "feats": {k: float(req.feats.get(k, 0.0)) for k in CF.MODEL_FEATS},
        "marked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with FEEDBACK_LOCK:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "stats": _feedback_stats()}


@app.get("/api/feedback/export")
def api_feedback_export():
    """导出训练用的 {features, label} 表（后来的标注覆盖先前的）。"""
    rows = _read_feedback()
    latest: Dict[tuple, dict] = {}
    for r in rows:
        latest[(r.get("audio"), round(r.get("time", 0), 1))] = r
    out = []
    for (audio, _), r in latest.items():
        if r.get("verdict") not in (0, 1):
            continue
        if not r.get("feats"):
            continue
        out.append({"audio": audio, "time": r["time"], "label": int(r["verdict"]),
                    "prob": r.get("prob"), "text": r.get("text", ""),
                    "feats": r["feats"]})
    return JSONResponse({"count": len(out),
                         "pos": sum(1 for x in out if x["label"] == 1),
                         "rows": out})


# ---------------------------------------------------------------------------
# 精确高潮时刻标注
#
# 比 ✓/✗ 好在三点：
#   1. 不依赖模型候选 —— 模型**漏掉**的高潮也能标出来（这是 ✓/✗ 永远做不到的）
#   2. 记录的是精确时刻，可以直接算时间误差
#   3. 标完的音轨可以直接当官方标注用，成为可评估的测试折
#
# 但有个前提：**只有标完的音轨才能当标注用**。
# 如果一轨有 3 次高潮只标了 2 次，第 3 次附近的候选就会被当成负例 ——
# 那比不标还糟。所以要有「本轨已标完」这个显式确认。
# ---------------------------------------------------------------------------
class MarkReq(BaseModel):
    audio: str
    time: Optional[float] = None       # 秒；删除时用
    action: str = "add"                # add | del | done | undone
    note: str = ""


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _marks_of(audio: str) -> List[float]:
    """某音频上现存的精确标注时刻（去重、排序）。"""
    live, dead = set(), set()
    for r in _read_jsonl(MARKS_FILE):
        if r.get("audio") != audio:
            continue
        t = round(float(r.get("time", 0)), 2)
        (live if r.get("action", "add") == "add" else dead).add(t)
    return sorted(live - dead)


def _done_of(audio: str) -> bool:
    v = False
    for r in _read_jsonl(DONE_FILE):
        if r.get("audio") == audio:
            v = bool(r.get("done", True))
    return v


@app.get("/api/marks")
def api_marks(audio: str = ""):
    """返回某音频的精确标注 + 一批统计。"""
    if audio:
        return {"ok": True, "times": _marks_of(audio), "done": _done_of(audio),
                "track": track_key(Path(audio).name), "work": work_key(audio)}
    # 不带参数时返回全部（供导出/概览）
    byfile: Dict[str, List[float]] = {}
    for r in _read_jsonl(MARKS_FILE):
        a = r.get("audio", "")
        if a:
            byfile.setdefault(a, [])
    return {"ok": True,
            "files": {a: {"times": _marks_of(a), "done": _done_of(a)}
                      for a in byfile}}


@app.post("/api/marks")
def api_marks_post(req: MarkReq):
    audio = req.audio
    if not audio:
        raise HTTPException(400, "缺少 audio")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with FEEDBACK_LOCK:
        if req.action in ("add", "del"):
            if req.time is None:
                raise HTTPException(400, "add/del 需要 time")
            rec = {"audio": audio, "time": round(float(req.time), 2),
                   "action": req.action, "note": req.note[:80], "marked_at": now}
            with open(MARKS_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        elif req.action in ("done", "undone"):
            rec = {"audio": audio, "done": req.action == "done", "marked_at": now}
            with open(DONE_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            raise HTTPException(400, f"未知 action：{req.action}")
    return {"ok": True, "times": _marks_of(audio), "done": _done_of(audio),
            "stats": _marks_stats()}


def _marks_stats() -> Dict[str, int]:
    """全局统计：多少音轨标完、共多少个点。"""
    files = {a for a in (_r.get("audio", "") for _r in _read_jsonl(MARKS_FILE)) if a}
    done = {a for a in (_r.get("audio", "") for _r in _read_jsonl(DONE_FILE)) if a
            and _done_of(a)}
    return {"tracks_marked": len(files), "tracks_done": len(done),
            "points": sum(len(_marks_of(a)) for a in files)}


@app.get("/api/marks/export")
def api_marks_export():
    """导出成和 gt_all.json 同构的格式，可直接给 train_model.py 用。

    只导出**已标完**的音轨 —— 没标完的会让模型把漏标的高潮学成负例。
    """
    files = {r.get("audio", "") for r in _read_jsonl(MARKS_FILE) if r.get("audio")}
    gt: Dict[str, Dict[str, List[float]]] = {}
    skipped = []
    for a in sorted(files):
        if not _done_of(a):
            skipped.append(a)
            continue
        rj, tk = work_key(a), track_key(Path(a).name)
        if not rj or not tk:
            skipped.append(a)
            continue
        gt.setdefault(rj, {})[tk] = _marks_of(a)
    return {"works": len(gt), "tracks": sum(len(v) for v in gt.values()),
            "points": sum(len(x) for v in gt.values() for x in v.values()),
            "skipped_incomplete": len(skipped), "gt": gt,
            "files": {a: {"work": work_key(a), "track": track_key(Path(a).name)}
                      for a in sorted(files)}}


# ===========================================================================
# 前端
# ===========================================================================
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>🎧</text></svg>">
<title></title>
<style>
:root{
  --bg:#0e1013; --panel:#16191f; --panel2:#1d2128; --line:#272c35;
  --fg:#e6e9ef; --dim:#8b93a3; --accent:#4da3ff; --ok:#35c88a;
  --warn:#ffb020; --hot:#ff4d6a;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--fg);
  font:14px/1.55 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;
  overflow:hidden}
button,input,select{font:inherit;color:inherit}
button{background:var(--panel2);border:1px solid var(--line);border-radius:7px;
  padding:6px 12px;cursor:pointer;transition:.12s}
button:hover{background:#262b34;border-color:#39404d}
button:disabled{opacity:.4;cursor:default}
button.primary{background:var(--accent);border-color:var(--accent);color:#06121f;font-weight:600}
button.primary:hover{background:#63b0ff}
button.ghost{background:transparent}

#app{display:grid;grid-template-columns:330px 1fr;height:100vh}

/* 左侧文件浏览 */
#side{border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
#pick{padding:10px 12px;border-bottom:1px solid var(--line);display:flex;gap:6px}
#pinput{flex:1;min-width:0;background:var(--panel2);border:1px solid var(--line);
  border-radius:7px;padding:6px 10px;font-size:12px;outline:none}
#pinput:focus{border-color:var(--accent)}
#pinput::placeholder{color:#5f6878}
#crumbs{padding:7px 12px;border-bottom:1px solid var(--line);display:flex;gap:6px;
  align-items:center;font-size:12px;color:var(--dim);min-height:32px}
#crumbs b{color:var(--fg);font-weight:500;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;direction:rtl;text-align:left}
#listing{flex:1;overflow:auto;padding:4px 0}
#listing.busy{opacity:.45}
.row{display:flex;align-items:center;gap:8px;padding:5px 14px;cursor:pointer;
  white-space:nowrap;font-size:13px}
.row:hover{background:var(--panel2)}
.row.sel{background:#1b3a5c}
.row .nm{overflow:hidden;text-overflow:ellipsis;flex:1}
.row .mt{color:var(--dim);font-size:11px;font-variant-numeric:tabular-nums;flex:none}
.row.dir .nm{color:#b9c6da}
.row.lrc .nm{color:#8fd4b0}
.row .hint{color:#4d5563;font-size:11px;flex:none}
.secn{padding:9px 14px 3px;font-size:11px;color:#5f6878;letter-spacing:.6px}
.empty{padding:20px 16px;color:var(--dim);font-size:12px}

/* 右侧播放器 */
#main{display:flex;flex-direction:column;min-height:0;position:relative}
#top{padding:14px 20px;border-bottom:1px solid var(--line);
  display:flex;align-items:center;gap:14px;flex-wrap:wrap}
#title{font-weight:600;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#title small{color:var(--dim);font-weight:400;margin-left:10px}
#stage{flex:1;overflow:auto;padding:18px 20px 26px}

/* 时间轴 */
#tl{position:relative;height:56px;background:var(--panel);border:1px solid var(--line);
  border-radius:10px;cursor:pointer;user-select:none;overflow:hidden}
#tlfill{position:absolute;inset:0 auto 0 0;background:rgba(77,163,255,.13);
  border-right:2px solid var(--accent);pointer-events:none}
#tlpos{position:absolute;top:0;bottom:0;width:2px;background:var(--accent);
  pointer-events:none;box-shadow:0 0 8px rgba(77,163,255,.8)}
#tl:hover #tlpos{width:3px}
.mk{position:absolute;top:0;bottom:0;width:3px;border-radius:2px;
  background:var(--warn);opacity:.85;transition:.2s;pointer-events:none}
.mk.hi{background:var(--hot);opacity:1;width:5px}
.mk.mid{background:var(--warn)}
.mk.lo{background:#5c6472}
.mk.done{opacity:.3}
/* 手动标注的点：绿色，比候选标记更宽更高 */
.mk.user{background:var(--ok);width:4px;opacity:1;top:0;bottom:0;
  box-shadow:0 0 7px rgba(53,200,138,.85)}
.mk.user::after{content:'';position:absolute;left:-3px;top:-4px;width:10px;height:6px;
  background:var(--ok);border-radius:2px}
.mk.pulse{animation:pulse .85s ease-out 3}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,77,106,.75)}
  100%{box-shadow:0 0 0 22px rgba(255,77,106,0)}}
#tl .lab{position:absolute;bottom:3px;font-size:10px;color:var(--dim);
  transform:translateX(-50%);pointer-events:none;font-variant-numeric:tabular-nums}
#tl .ghost{position:absolute;top:0;bottom:0;width:1px;background:#2f3540;pointer-events:none}

#clock{display:flex;justify-content:space-between;color:var(--dim);font-size:12px;
  margin-top:6px;font-variant-numeric:tabular-nums}

/* 控制条 */
#ctrl{display:flex;align-items:center;gap:10px;margin:14px 0 18px;flex-wrap:wrap}
#ctrl .sp{flex:1}
#rate{padding:5px 8px;background:var(--panel2);border:1px solid var(--line);
  border-radius:7px}
#vol{width:110px;accent-color:var(--accent)}

/* 面板 */
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px;margin-bottom:14px}
.card h2{margin:0 0 10px;font-size:12px;font-weight:600;color:var(--dim);
  letter-spacing:1px;text-transform:uppercase}
#hits{display:flex;flex-direction:column;gap:6px}
.hit{display:flex;align-items:center;gap:10px;padding:8px 11px;border-radius:8px;
  background:var(--panel2);cursor:pointer;border:1px solid transparent;transition:.12s}
.hit:hover{border-color:#39404d}
.hit.active{border-color:var(--hot);background:#2a1a20}
.hit .t{font-variant-numeric:tabular-nums;font-weight:600;min-width:52px}
.hit .p{font-variant-numeric:tabular-nums;font-size:12px;min-width:52px}
.hit .x{flex:1;color:var(--dim);font-size:12px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.hit.hi .p{color:var(--hot)} .hit.mid .p{color:var(--warn)} .hit.lo .p{color:var(--dim)}
.hit .vb{display:flex;gap:4px;flex:none}
.hit .mg{flex:none;font-size:11px;color:#8b93a3;background:#232a34;
  border:1px solid var(--line);border-radius:5px;padding:1px 6px;cursor:help}
.hit .vb button{padding:2px 9px;font-size:13px;line-height:1.5;border-radius:6px;
  background:transparent;border:1px solid var(--line);color:var(--dim)}
.hit .vb button:hover{background:#2c333e;color:var(--fg)}
.hit .vb button.y.on{background:#1c5c42;border-color:var(--ok);color:#8ff0c4}
.hit .vb button.n.on{background:#5c1c28;border-color:var(--hot);color:#ffb3c0}
.hit.marked-y{border-left:3px solid var(--ok)}
.hit.marked-n{border-left:3px solid var(--hot);opacity:.62}
.hit.marked-n .t{text-decoration:line-through;color:var(--dim)}
#fbstats{font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}
#fbstats b{color:var(--ok);font-weight:600}
#fbstats i{color:var(--hot);font-style:normal;font-weight:600}

/* 歌词 */
#lyr{max-height:230px;overflow:auto;font-size:13px;line-height:1.85}
#lyr div{padding:2px 6px;border-radius:5px;color:var(--dim);transition:.15s}
#lyr div.on{color:var(--fg);background:#1e2530;font-weight:500}
#lyr div em{color:#5f6878;font-style:normal;margin-right:8px;font-size:11px;
  font-variant-numeric:tabular-nums}

/* 提示层 */
#toast{position:fixed;top:18px;left:50%;transform:translate(-50%,-140%);
  background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:9px 16px;font-size:13px;transition:transform .3s cubic-bezier(.2,.9,.3,1.3);
  z-index:40;pointer-events:none;box-shadow:0 8px 26px rgba(0,0,0,.5)}
#toast.on{transform:translate(-50%,0)}

#flash{position:absolute;inset:0;pointer-events:none;z-index:30;opacity:0;
  background:radial-gradient(ellipse at 50% 0%,rgba(255,77,106,.28),transparent 62%)}
#flash.on{animation:fl 1.5s ease-out}
@keyframes fl{0%{opacity:1}100%{opacity:0}}

#banner{position:absolute;left:50%;top:74px;transform:translate(-50%,-26px) scale(.94);
  z-index:35;pointer-events:none;opacity:0;transition:.24s cubic-bezier(.2,.9,.3,1.3);
  background:linear-gradient(135deg,#ff4d6a,#c9294a);color:#fff;
  padding:13px 26px;border-radius:12px;text-align:center;
  box-shadow:0 12px 34px rgba(255,77,106,.42)}
#banner.on{opacity:1;transform:translate(-50%,0) scale(1)}
#banner b{display:block;font-size:21px;letter-spacing:1px;
  font-variant-numeric:tabular-nums}
#banner span{font-size:12px;opacity:.92}

.muted{color:var(--dim);font-size:12px}
#err{color:var(--hot);font-size:12px}

/* ===================== 移动端 ===================== */
#navbtn{display:none}
#scrim{display:none}

@media (max-width: 820px){
  /* 单栏；用 dvh 而不是 vh —— 手机地址栏会盖住 100vh 的底部 */
  #app{grid-template-columns:1fr;height:100dvh}

  /* 侧栏改成从左滑出的抽屉 */
  #side{
    position:fixed;top:0;bottom:0;left:0;width:min(82vw,330px);z-index:60;
    background:var(--bg);border-right:1px solid var(--line);
    transform:translateX(-103%);transition:transform .22s cubic-bezier(.3,.8,.4,1);
    box-shadow:0 0 44px rgba(0,0,0,.6);will-change:transform;
  }
  #side.open{transform:none}
  #scrim{
    display:block;position:fixed;inset:0;z-index:55;background:rgba(0,0,0,.55);
    opacity:0;pointer-events:none;transition:opacity .2s;
  }
  #scrim.on{opacity:1;pointer-events:auto}

  /* 汉堡按钮 */
  #navbtn{display:inline-flex;align-items:center;justify-content:center;
    width:40px;height:36px;flex:none;font-size:17px;line-height:1;padding:0}

  /* 顶栏：允许换行，标题独占一行 */
  #top{padding:9px 11px;gap:7px}
  #title{flex:1 1 100%;font-size:15px;white-space:normal;line-height:1.35}
  #err{flex:1 1 100%}
  #fbstats{display:none}          /* 手机上不显示这块统计 */

  #stage{padding:11px 11px 44px}

  /* 触控目标放大 */
  button{padding:9px 13px;border-radius:8px}
  .row{padding:11px 14px;font-size:15px}
  .row .mt{font-size:12px}
  #crumbs{padding:9px 12px}

  /* 输入框 >=16px，否则 iOS 聚焦时会自动放大整页 */
  #pinput{font-size:16px;padding:9px 11px}
  input,select,textarea{font-size:16px}
  #rate{padding:8px 9px}

  /* 时间轴加高，手指才好点 */
  #tl{height:78px}
  .mk{width:4px}
  .mk.hi{width:6px}
  .mk.user{width:6px}
  #tl .lab{font-size:11px}

  #ctrl{gap:8px;margin:11px 0 15px}
  #ctrl .sp{display:none}          /* 手机上不需要这个弹性占位 */
  #vol{width:100%;order:9}         /* 音量条独占一行 */

  /* 候选行：加大点击区 */
  .hit{padding:11px 12px;gap:9px;flex-wrap:wrap}
  .hit .t{min-width:50px;font-size:15px}
  .hit .p{min-width:48px}
  .hit .x{flex:1 1 100%;font-size:12px}
  .hit .vb{flex:1 1 100%;justify-content:flex-end}
  .hit .vb button{padding:7px 15px;font-size:15px}

  .card{padding:12px 13px;margin-bottom:12px}
  h2{font-size:14px}

  #banner{top:auto;bottom:26px;padding:12px 22px}
  #toast{left:11px;right:11px;transform:translateY(-140%)}
  #toast.on{transform:none}
}

/* 横屏且很矮时，进一步压缩 */
@media (max-width: 820px) and (max-height: 460px){
  #tl{height:52px}
  #top{padding:6px 10px}
  #title{font-size:13px}
}
</style>
</head>
<body>
<div id="app">
  <aside id="side">
    <button id="navclose" class="ghost" title="收起"
            style="display:none;position:absolute;top:8px;right:8px;z-index:2">✕</button>
    <div id="pick">
      <input id="pinput" spellcheck="false" placeholder="输入或粘贴路径，回车打开">
    </div>
    <div id="crumbs">位置</div>
    <div id="listing"></div>
  </aside>

  <div id="scrim"></div>
  <section id="main">
    <div id="top">
      <button id="navbtn" title="选择文件">☰</button>
      <div id="title">未选择文件</div>
      <label class="muted">候选数
        <select id="topn"><option>3</option><option selected>6</option><option>10</option></select>
      </label>
      <label class="muted" title="门槛越高误报越少、漏掉的越多。&#10;选项只列 isotonic 校准真实产生的台阶 —— 台阶之间的值行为完全相同，列了也没用。&#10;数字是 10 折留一作品的实测值（准=精确率，全=召回率，百分比）。">最低概率
        <select id="minp">
          <option value="0">不过滤 · 准49/全75</option>
          <option value="0.17">≥17% · 准61/全73</option>
          <option value="0.25">≥25% · 准67/全69</option>
          <option value="0.5" selected>≥50% · 准71/全68（F1 最优）</option>
          <option value="0.66">≥66% · 准84/全55</option>
          <option value="0.67">≥67% · 准92/全40</option>
        </select>
      </label>
      <button class="primary" id="go" disabled>分析高潮点</button>
      <button id="clr" disabled title="清除已导入的歌词">清除歌词</button>
      <span id="fbstats"></span>
      <button id="exp" class="ghost" title="导出精确标注（只含已标完的音轨）">导出标注</button>
      <span id="err"></span>
    </div>

    <div id="stage">
      <div id="tl"></div>
      <div id="clock"><span id="cur">00:00</span><span id="dur">00:00</span></div>

      <div id="ctrl">
        <button id="pp" disabled>▶ 播放</button>
        <button id="b10" disabled>−10s</button>
        <button id="f10" disabled>+10s</button>
        <select id="rate" disabled>
          <option value="0.75">0.75×</option><option value="1" selected>1×</option>
          <option value="1.25">1.25×</option><option value="1.5">1.5×</option>
          <option value="2">2×</option>
        </select>
        <input type="range" id="vol" min="0" max="1" step="0.02" value="1" disabled>
        <button id="mark" class="primary" disabled title="把当前播放位置记为高潮点（快捷键 M）">标记高潮点</button>
        <span class="sp"></span>
        <label class="muted"><input type="checkbox" id="beep" checked> 提示音</label>
        <label class="muted"><input type="checkbox" id="pauseat"> 到点暂停</label>
      </div>

      <div class="card" id="mkcard" style="display:none">
        <h2>手动标注的高潮点
          <span id="mkstat" class="muted"></span>
          <label class="muted" style="float:right;font-weight:400;text-transform:none">
            <input type="checkbox" id="mkdone"> 本轨已标完
          </label>
        </h2>
        <div id="mkhint" class="muted" style="margin:-4px 0 10px;font-size:12px"></div>
        <div id="mklist"></div>
      </div>

      <div class="card">
        <h2>高潮点</h2>
        <div id="hits"><div class="empty">选择音频后点「分析高潮点」</div></div>
      </div>

      <div class="card" id="lyrcard" style="display:none">
        <h2>歌词 <span id="lyrname" class="muted"></span></h2>
        <div id="lyr"></div>
      </div>
    </div>

    <div id="flash"></div>
    <div id="banner"><b id="bt">00:00</b><span id="bs"></span></div>
  </section>
</div>
<div id="toast"></div>
<audio id="au" preload="metadata"></audio>

<script>
"use strict";
const $ = id => document.getElementById(id);

/* ---------------- 移动端抽屉 ---------------- */
const _mq = window.matchMedia('(max-width: 820px)');
function isMobile(){ return _mq.matches; }
function drawer(open){
  if (!isMobile()){ $('side').classList.remove('open'); $('scrim').classList.remove('on');
    $('navclose').style.display='none'; document.body.style.overflow=''; return; }
  $('side').classList.toggle('open', open);
  $('scrim').classList.toggle('on', open);
  $('navclose').style.display = open ? 'inline-flex' : 'none';
  // 抽屉打开时锁住背景滚动，否则手指会滑到后面的列表
  document.body.style.overflow = open ? 'hidden' : '';
}
$('navbtn').onclick = () => drawer(true);
$('navclose').onclick = () => drawer(false);
$('scrim').onclick = () => drawer(false);
window.addEventListener('keydown', e => { if (e.key === 'Escape') drawer(false); });
_mq.addEventListener('change', () => drawer(false));
const au = $('au');
let cur = null;              // 当前音频路径
let curPath = '';            // 当前浏览的目录
let cands = [];              // [{time,mmss,prob,conf,text,note,fired,warned,verdict}]
let lyr = [];                // [{time,text}]
let pickedLrc = '';          // 用户手动导入的歌词路径
let lyrIdx = -1;

/* ---------- 工具 ---------- */
const fmt = s => {
  s = Math.max(0, s|0);
  return String(s/60|0).padStart(2,'0') + ':' + String(s%60).padStart(2,'0');
};
let toastTimer = null;
function toast(msg, kind){
  const t = $('toast');
  t.textContent = msg;
  t.style.borderColor = kind === 'err' ? 'var(--hot)' : 'var(--line)';
  t.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('on'), 2600);
}

/* 提示音：用 Web Audio 合成，不依赖音频文件 */
let actx = null;
function beep(kind){
  if (!$('beep').checked) return;
  try{
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const now = actx.currentTime;
    const seq = kind === 'hit' ? [[880,0,.13],[1320,.10,.20]]
                              : [[740,0,.07]];
    for (const [hz, at, dur] of seq){
      const o = actx.createOscillator(), g = actx.createGain();
      o.type = 'sine'; o.frequency.value = hz;
      g.gain.setValueAtTime(0, now+at);
      g.gain.linearRampToValueAtTime(.22, now+at+.012);
      g.gain.exponentialRampToValueAtTime(.001, now+at+dur);
      o.connect(g).connect(actx.destination);
      o.start(now+at); o.stop(now+at+dur+.03);
    }
  }catch(e){}
}

/* ---------- 文件浏览 ---------- */
const RECENT_KEY = 'climax_recent';
const getRecent = () => { try{ return JSON.parse(localStorage.getItem(RECENT_KEY)||'[]'); }catch(e){ return []; } };
function pushRecent(p){
  if (!p) return;
  let r = getRecent().filter(x => x !== p);
  r.unshift(p);
  try{ localStorage.setItem(RECENT_KEY, JSON.stringify(r.slice(0, 8))); }catch(e){}
}

let browseSeq = 0;
async function browse(path){
  const seq = ++browseSeq;
  const L = $('listing');
  L.classList.add('busy');
  if (!L.querySelector('.row')) L.innerHTML = '<div class="empty">读取中…</div>';

  let d;
  try{
    const r = await fetch('/api/browse?path=' + encodeURIComponent(path||''));
    if (!r.ok) throw new Error((await r.json()).detail || r.status);
    d = await r.json();
  }catch(e){
    L.classList.remove('busy');
    toast('无法读取目录：' + e.message, 'err');
    return;
  }
  if (seq !== browseSeq) return;          // 已经有更新的请求了，丢弃本次
  curPath = d.path || '';

  $('crumbs').innerHTML = d.path
    ? `<button class="ghost" id="up" title="上一级">↑</button><b>${esc(d.path)}</b>`
    : '<b>此电脑</b>';
  if (d.path){
    $('up').onclick = () => browse(d.parent);
    $('pinput').value = d.path;
    pushRecent(d.path);
  } else {
    $('pinput').value = '';
  }

  L.innerHTML = '';
  if (!d.path){
    const r = getRecent();
    if (r.length){
      L.appendChild(secn('最近'));
      for (const p of r) L.appendChild(dirRow(p.split('\\').filter(Boolean).pop() || p, p, null, false, true));
    }
    L.appendChild(secn('位置'));
  }
  for (const x of d.dirs){
    L.appendChild(dirRow(x.name, x.path, x.audio, x.has_sub, false));
  }
  for (const x of d.audio){
    const el = document.createElement('div');
    el.className = 'row' + (x.path === cur ? ' sel' : '');
    el.dataset.path = x.path;
    el.title = x.path;
    el.innerHTML = `<span class="nm">♪ ${esc(x.name)}</span>`
      + `<span class="mt">${(x.size/1048576).toFixed(1)}M</span>`;
    el.onclick = () => pick(x.path, x.name);
    L.appendChild(el);
  }
  if ((d.lrcs||[]).length){
    L.appendChild(secn('歌词'));
    for (const x of d.lrcs){
      const el = document.createElement('div');
      el.className = 'row lrc';
      el.title = cur ? '设为本音频的歌词' : '先选择一个音频文件';
      el.innerHTML = `<span class="nm">💬 ${esc(x.name)}</span>`;
      el.onclick = () => {
        if (!cur){ toast('先选择音频文件', 'err'); return; }
        useLrc(x.path, x.name);
      };
      L.appendChild(el);
    }
  }
  if (!d.dirs.length && !d.audio.length && !(d.lrcs||[]).length && d.path){
    L.innerHTML = '<div class="empty">此目录没有子目录或音频文件</div>';
  }
  L.classList.remove('busy');
  L.scrollTop = 0;
}

function secn(t){
  const e = document.createElement('div'); e.className = 'secn'; e.textContent = t; return e;
}

function dirRow(name, path, n, hasSub, recent){
  const el = document.createElement('div');
  el.className = 'row dir';
  el.dataset.path = path;
  el.title = path;
  const badge = (n === null || n === undefined) ? ''
    : (n > 0 ? `<span class="mt">${n}</span>`
             : (hasSub ? '<span class="hint">›</span>' : ''));
  el.innerHTML = `<span class="nm">${recent ? '🕘' : '📁'} ${esc(name)}</span>${badge}`;
  el.onclick = () => browse(path);
  return el;
}

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function pick(path, name){
  cur = path;
  cands = []; lyr = []; lyrIdx = -1; pickedLrc = '';
  document.querySelectorAll('.row.sel').forEach(e => e.classList.remove('sel'));
  const e = document.querySelector(`.row[data-path="${CSS.escape(path)}"]`);
  if (e) e.classList.add('sel');

  document.title = name;
  $('title').innerHTML = esc(name) + ' <small id="meta"></small>';
  au.src = '/api/audio?path=' + encodeURIComponent(path);
  au.load();
  $('cur').textContent = '00:00';
  $('dur').textContent = '00:00';
  $('go').disabled = false; $('clr').disabled = false; $('mark').disabled = false;
  marks = []; markDone = false;
  $('pp').disabled = $('b10').disabled = $('f10').disabled = $('rate').disabled = $('vol').disabled = false;
  $('hits').innerHTML = '<div class="empty">点「分析高潮点」</div>';
  $('lyrcard').style.display = 'none';
  drawMarkers();
  await loadMarks();
  toast('已选择：' + name);
}

$('clr').onclick = () => {
  pickedLrc = ''; lyr = []; lyrIdx = -1;
  $('lyrcard').style.display = 'none';
  toast('已清除导入的歌词');
};

$('exp').onclick = async () => {
  try{
    const r = await fetch('/api/marks/export');
    const d = await r.json();
    if (!d.points){
      toast('还没有标完的音轨可导出（勾上「本轨已标完」才算）', 'err');
      return;
    }
    const blob = new Blob([JSON.stringify(d.gt, null, 1)], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'climax_gt_user.json';
    a.click();
    URL.revokeObjectURL(a.href);
    toast(`已导出 ${d.works} 部 / ${d.tracks} 轨 / ${d.points} 个点`
      + (d.skipped_incomplete ? `（${d.skipped_incomplete} 轨未标完，已跳过）` : ''));
  }catch(e){ toast('导出失败：' + e.message, 'err'); }
};

/* ---------- 分析 ---------- */
$('go').onclick = async () => {
  if (!cur) return;
  const btn = $('go');
  btn.disabled = true; btn.textContent = '分析中…';
  $('err').textContent = '';
  try{
    const r = await fetch('/api/analyze', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({path: cur, lrc: pickedLrc, top: +$('topn').value,
                            min_prob: +$('minp').value})
    });
    const d = await r.json();
    if (!d.ok){ $('err').textContent = d.error || '分析失败'; toast(d.error || '分析失败','err'); return; }
    cands = (d.candidates||[]).map(c => ({...c, fired:false, warned:false, verdict:null}));
    $('meta').textContent = `　${fmt(d.duration)}　${cands.length} 个候选　`
      + `耗时 ${d.elapsed}s` + (d.has_climax === false ? '　（可能没有高潮）' : '');
    renderHits();
    await loadVerdicts();
    drawMarkers();
    await loadLrc(pickedLrc || d.lrc);
    if (!cands.length) toast('没有达到概率门槛的候选');
    else toast(`找到 ${cands.length} 个候选，最高 ${Math.round(d.max_prob*100)}%`);
  }catch(e){
    $('err').textContent = '分析出错：' + e.message;
  }finally{
    btn.disabled = false; btn.textContent = '分析高潮点';
  }
};

function renderHits(){
  const box = $('hits');
  box.innerHTML = '';
  if (!cands.length){ box.innerHTML = '<div class="empty">没有候选</div>'; return; }
  for (const c of cands){
    const el = document.createElement('div');
    el.className = 'hit ' + (c.conf === '高' ? 'hi' : c.conf === '中' ? 'mid' : 'lo');
    el.dataset.t = c.time;
    el.title = c.note || '';
    el.innerHTML = `<span class="t">${c.mmss}</span>`
      + `<span class="p">${Math.round(c.prob*100)}%</span>`
      + `<span class="x">${esc(c.text || '')}</span>`
      + (c.merged > 1
          ? `<span class="mg" title="附近 ${c.merged} 个候选合并为一点，取其中间时刻">合${c.merged}</span>`
          : '')
      + `<span class="vb">`
      +   `<button class="y" data-v="1" title="确认是高潮">✓</button>`
      +   `<button class="n" data-v="0" title="不是高潮">✗</button>`
      + `</span>`;
    el.onclick = e => {
      if (e.target.closest('.vb')) return;      // 点按钮不算跳转
      au.currentTime = Math.max(0, c.time - 2); au.play();
    };
    el.querySelectorAll('.vb button').forEach(b => {
      b.onclick = e => { e.stopPropagation(); mark(c, +b.dataset.v); };
    });
    box.appendChild(el);
    paintVerdict(c);
  }
}

/* ---------- 人工标注 ---------- */
function paintVerdict(c){
  const el = [...document.querySelectorAll('.hit')]
    .find(h => Math.abs(+h.dataset.t - c.time) < 1e-6);
  if (!el) return;
  el.classList.toggle('marked-y', c.verdict === 1);
  el.classList.toggle('marked-n', c.verdict === 0);
  el.querySelector('.vb .y').classList.toggle('on', c.verdict === 1);
  el.querySelector('.vb .n').classList.toggle('on', c.verdict === 0);
}

function setStats(s){
  if (!s) return;
  $('fbstats').innerHTML = s.total
    ? `已标 ${s.total} 条（<b>${s.pos}</b> 真 / <i>${s.neg}</i> 假）`
    : '';
}

async function mark(c, v){
  c.verdict = (c.verdict === v) ? null : v;      // 再点一次 = 撤销
  paintVerdict(c);
  try{
    const r = await fetch('/api/feedback', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({audio: cur, time: c.time, mmss: c.mmss,
                            prob: c.prob, raw: c.raw, text: c.text,
                            verdict: c.verdict, feats: c.feats || {}})
    });
    const d = await r.json();
    setStats(d.stats);
    if (c.verdict === 1) toast(`${c.mmss} 标为高潮`);
    else if (c.verdict === 0) toast(`${c.mmss} 标为不是`);
    else toast('已撤销标注');
  }catch(e){ toast('标注保存失败：' + e.message, 'err'); }
}

async function loadVerdicts(){
  if (!cur) return;
  try{
    const r = await fetch('/api/feedback?audio=' + encodeURIComponent(cur));
    const d = await r.json();
    setStats(d.stats);
    for (const c of cands){
      const v = d.verdicts[`${c.time.toFixed(1)}`];
      c.verdict = (v === 0 || v === 1) ? v : null;
      paintVerdict(c);
    }
  }catch(e){}
}

async function loadLrc(fullOrName){
  if (!cur) return;
  try{
    // fullOrName 可能是完整路径（手动导入），也可能只是文件名（自动发现）
    const q = fullOrName
      ? (fullOrName.includes('\\') || fullOrName.includes('/')
          ? '&lrc=' + encodeURIComponent(fullOrName) : '')
      : '';
    const r = await fetch('/api/lrc?path=' + encodeURIComponent(cur) + q);
    const d = await r.json();
    if (!d.found || !d.lines.length){ $('lyrcard').style.display='none'; lyr = []; return; }
    lyr = d.lines; lyrIdx = -1;
    $('lyrname').textContent = d.name;
    const box = $('lyr'); box.innerHTML = '';
    d.lines.forEach((l, i) => {
      const el = document.createElement('div');
      el.dataset.i = i;
      el.innerHTML = `<em>${fmt(l.time)}</em>${esc(l.text)}`;
      el.onclick = () => { au.currentTime = Math.max(0, l.time - .3); au.play(); };
      box.appendChild(el);
    });
    $('lyrcard').style.display = '';
  }catch(e){ $('lyrcard').style.display='none'; }
}

async function useLrc(path, name){
  pickedLrc = path;
  await loadLrc(path);
  if (lyr.length) toast(`已导入歌词：${name}（${lyr.length} 行）`);
  else toast('这个歌词文件没有解析出内容', 'err');
}

/* ---------- 手动标注精确高潮点 ---------- */
let marks = [];              // [秒]
let markDone = false;

async function loadMarks(){
  if (!cur){ marks = []; markDone = false; renderMarks(); return; }
  try{
    const r = await fetch('/api/marks?audio=' + encodeURIComponent(cur));
    const d = await r.json();
    marks = d.times || []; markDone = !!d.done;
  }catch(e){ marks = []; markDone = false; }
  renderMarks();
  drawMarkers();
}

async function addMark(){
  if (!cur || !au.duration) return;
  const t = Math.round(au.currentTime * 100) / 100;
  if (marks.some(x => Math.abs(x - t) < 1)){ toast('这个位置已经标过了'); return; }
  marks.push(t); marks.sort((a,b) => a - b);
  renderMarks(); drawMarkers();
  toast(`已标记 ${fmt(t)}`);
  await post({action:'add', time:t});
}

async function delMark(t){
  marks = marks.filter(x => Math.abs(x - t) > 1e-6);
  renderMarks(); drawMarkers();
  await post({action:'del', time:t});
}

async function setDone(v){
  markDone = v;
  await post({action: v ? 'done' : 'undone'});
  toast(v ? '已确认本轨标完 —— 可以作为训练标注使用' : '已取消「标完」');
}

async function post(body){
  try{
    const r = await fetch('/api/marks', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({audio: cur, ...body})
    });
    const d = await r.json();
    if (d.times){ marks = d.times; }
    renderMarks(); drawMarkers();
  }catch(e){ toast('保存失败：' + e.message, 'err'); }
}

function renderMarks(){
  const card = $('mkcard');
  const has = cur && (marks.length || markDone);
  card.style.display = has ? '' : 'none';
  $('mkstat').textContent = marks.length ? `　共 ${marks.length} 个` : '';
  $('mkdone').checked = markDone;
  $('mkhint').innerHTML = markDone
    ? '✓ 本轨已标完，会作为训练标注使用（模型漏掉的高潮也已被你补齐）'
    : '边听边按 <b>M</b> 标出每个高潮点。标完后勾上「本轨已标完」才会用于训练 —— '
      + '没标完就使用会让模型把你漏标的高潮学成负例。';
  const box = $('mklist'); box.innerHTML = '';
  marks.forEach((t, i) => {
    const el = document.createElement('div');
    el.className = 'hit';
    el.dataset.t = t;
    el.innerHTML = `<span class="t">${fmt(t)}</span>`
      + `<span class="x">手动标注 #${i+1}</span>`
      + `<span class="vb"><button class="n" title="删除这个标注">✗</button></span>`;
    el.onclick = e => {
      if (e.target.closest('.vb')) return;
      au.currentTime = Math.max(0, t - 1.5); au.play();
    };
    el.querySelector('.vb button').onclick = e => { e.stopPropagation(); delMark(t); };
    box.appendChild(el);
  });
  if (!marks.length){
    box.innerHTML = '<div class="empty">还没有标注。播放时按 M 记录当前时刻。</div>';
  }
}

$('mark').onclick = addMark;
$('mkdone').onchange = e => setDone(e.target.checked);

/* ---------- 时间轴 ---------- */
const tl = $('tl');
function pct(t){ return au.duration ? (t/au.duration*100) : 0; }

function drawMarkers(){
  tl.innerHTML = '';
  const fill = document.createElement('div'); fill.id='tlfill'; tl.appendChild(fill);
  const pos = document.createElement('div'); pos.id='tlpos'; tl.appendChild(pos);
  for (let m = 1; m < 12; m++){
    const g = document.createElement('div'); g.className='ghost';
    g.style.left = (m/12*100) + '%'; tl.appendChild(g);
  }
  const seen = [];
  // 手动标注的点画在最上层（绿色），候选标记在下
  for (const t of marks){
    if (!au.duration) break;
    const el = document.createElement('div');
    el.className = 'mk user';
    el.style.left = pct(t) + '%';
    el.dataset.mk = t;
    el.title = '手动标注 ' + fmt(t);
    tl.appendChild(el);
  }
  for (const c of cands){
    if (!au.duration) break;
    const el = document.createElement('div');
    el.className = 'mk ' + (c.conf==='高'?'hi':c.conf==='中'?'mid':'lo')
                 + (c.fired ? ' done' : '');
    el.style.left = pct(c.time) + '%';
    el.dataset.t = c.time;
    tl.appendChild(el);
    // 太近的时间戳只标一个文字
    if (!seen.some(x => Math.abs(x - c.time) < au.duration * .05)){
      seen.push(c.time);
      const lb = document.createElement('div');
      lb.className = 'lab'; lb.style.left = pct(c.time) + '%';
      lb.textContent = c.mmss; tl.appendChild(lb);
    }
  }
  updateBar();
}

function updateBar(){
  const f = $('tlfill'), p = $('tlpos');
  if (!f) return;
  const x = pct(au.currentTime) + '%';
  f.style.width = x; p.style.left = x;
}

tl.onclick = e => {
  if (!au.duration) return;
  const r = tl.getBoundingClientRect();
  const t = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  seek(au.duration * t);
};

function seek(t){
  au.currentTime = Math.max(0, Math.min(au.duration || 0, t));
  // 往回拖时重置标记，允许再次触发
  for (const c of cands){
    if (c.time > au.currentTime + 0.4){ c.fired = false; c.warned = false; }
  }
  highlightHit();
  updateBar();
}

/* ---------- 播放控制 ---------- */
$('pp').onclick = () => au.paused ? au.play() : au.pause();
au.onplay  = () => $('pp').textContent = '⏸ 暂停';
au.onpause = () => $('pp').textContent = '▶ 播放';
$('b10').onclick = () => seek(au.currentTime - 10);
$('f10').onclick = () => seek(au.currentTime + 10);
$('rate').onchange = () => au.playbackRate = +$('rate').value;
$('vol').oninput  = () => au.volume = +$('vol').value;

au.onloadedmetadata = () => {
  $('dur').textContent = fmt(au.duration);
  drawMarkers();
  loadMarks();          // 时长就绪后重绘（标注位置按百分比算，需要 duration）
};
au.ontimeupdate = () => { $('cur').textContent = fmt(au.currentTime); updateBar(); tick(); };
au.onseeking = () => { updateBar(); };
au.onerror = () => toast('音频无法播放（编码或路径问题）', 'err');

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space'){ e.preventDefault(); au.paused ? au.play() : au.pause(); }
  else if (e.code === 'ArrowLeft'){ e.preventDefault(); seek(au.currentTime - 5); }
  else if (e.code === 'ArrowRight'){ e.preventDefault(); seek(au.currentTime + 5); }
  else if (e.code === 'ArrowUp'){ e.preventDefault(); au.volume = Math.min(1, au.volume + .05); $('vol').value = au.volume; }
  else if (e.code === 'ArrowDown'){ e.preventDefault(); au.volume = Math.max(0, au.volume - .05); $('vol').value = au.volume; }
  else if (e.code === 'KeyM'){ e.preventDefault(); addMark(); }
});

/* ---------- 提示逻辑 ---------- */
let lastTick = 0;
function tick(){
  const t = au.currentTime;
  if (Math.abs(t - lastTick) < 1e-4) return;
  lastTick = t;

  for (const c of cands){
    if (!c.fired && t >= c.time && t < c.time + 4){
      c.fired = true;
      fire(c);
    } else if (!c.warned && t >= c.time - 3 && t < c.time){
      c.warned = true;
      beep('near');
      flashMarker(c.time, false);
      showBanner('即将…', c.mmss + ' 即将到达');
    }
  }
  syncLyrics(t);
  highlightHit();
}

function fire(c){
  beep('hit');
  flashMarker(c.time, true);
  showBanner(c.mmss, `置信度 ${c.conf}　${Math.round(c.prob*100)}%`
    + (c.text ? '　' + c.text.slice(0, 40) : ''));
  navigator.vibrate && navigator.vibrate([40, 60, 40]);
  if ($('pauseat').checked) au.pause();
}

let bannerTimer = null;
function showBanner(big, sub){
  $('bt').textContent = big; $('bs').textContent = sub;
  $('banner').classList.add('on');
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => $('banner').classList.remove('on'), 3400);
}

function flashMarker(time, strong){
  const el = [...tl.querySelectorAll('.mk')]
    .find(m => Math.abs(+m.dataset.t - time) < 1e-6);
  if (!el) return;
  el.classList.add('pulse');
  if (strong) el.classList.add('done');
  setTimeout(() => el.classList.remove('pulse'), 2700);
}

function highlightHit(){
  const t = au.currentTime;
  document.querySelectorAll('.hit').forEach(el => {
    el.classList.toggle('active', Math.abs(+el.dataset.t - t) < 2.4);
  });
}

function syncLyrics(t){
  if (!lyr.length) return;
  let i = -1;
  for (let k = 0; k < lyr.length; k++){ if (lyr[k].time <= t + .15) i = k; else break; }
  if (i === lyrIdx) return;
  lyrIdx = i;
  document.querySelectorAll('#lyr div').forEach(el => {
    const on = +el.dataset.i === i;
    el.classList.toggle('on', on);
    if (on) el.scrollIntoView({block:'nearest', behavior:'smooth'});
  });
}

/* ---------- 启动 ---------- */
$('pinput').onkeydown = e => {
  if (e.key !== 'Enter') return;
  const v = $('pinput').value.trim().replace(/^"|"$/g, '');
  if (!v){ browse(''); return; }
  browse(v);
};
$('pinput').onblur = () => { if (!curPath) $('pinput').value = ''; };

fetch('/api/status').then(r => r.json()).then(d => {
  if (!d.ok){
    $('err').textContent = d.error || '模型未就绪';
    return;
  }
  const m = d.meta || {};
  document.title = '';
  // 模型信息放进 hover 提示，不占界面
  const tip = `${m.n_works||'?'} 部作品 / ${m.n_tracks||'?'} 轨训练　`
    + `10 折 AUC ${(m.cv_auc||0).toFixed(3)}　`
    + `精确率 ${((m.cv_prec||0)*100).toFixed(1)}%　召回率 ${((m.cv_rec||0)*100).toFixed(1)}%\n`
    + `模型训练于 ${d.model_trained_at || '未知'}`;
  $('title').title = tip;
  $('pinput').title = tip;
  setStats(d.feedback);
  // 有标完但还没进模型的作品 → 提示需要手动重训（训练不会自动发生）
  const pend = d.pending_works || [];
  if (pend.length){
    $('exp').textContent = `导出标注（${pend.length} 部待训练）`;
    $('exp').title = `你标完了 ${pend.join('、')}，但模型还是旧的。\n`
      + '训练不会自动发生 —— 需要运行：python _ml\\train_model.py';
    toast(`有 ${pend.length} 部作品的标注还没用于训练，跑 train_model.py 后生效`);
  }
});
browse('');
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(PAGE)


def main():
    ap = argparse.ArgumentParser(description="本地音频播放器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    a = ap.parse_args()

    print("=" * 66)
    print("  本地音频播放器（高潮点提示 + 人工标注）")
    print(f"  http://{a.host}:{a.port}")
    try:
        m = get_model()
        meta = m.meta
        print(f"  模型：{meta.get('n_works','?')} 部作品 / {meta.get('n_tracks','?')} 轨 / "
              f"{meta.get('n_pos','?')} 正例")
        print(f"  10 折留一作品实测：AUC {meta.get('cv_auc',0):.3f}　"
              f"精确率 {meta.get('cv_prec',0)*100:.1f}%　"
              f"召回率 {meta.get('cv_rec',0)*100:.1f}%")
    except Exception as e:
        print(f"  ⚠ 模型未就绪：{e}")
    print("=" * 66)

    if a.open:
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()

    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
