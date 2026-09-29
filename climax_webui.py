# -*- coding: utf-8 -*-
"""climax_webui —— 高潮点播放器

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

app = FastAPI(title="climax player")

# ---------------------------------------------------------------------------
# 模型（7 MB，首次用到时加载；分析是 CPU 密集，串行执行）
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_LOCK = threading.Lock()
_ANALYZE_LOCK = threading.Lock()


def get_model() -> "CF.ClimaxModel":
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = CF.ClimaxModel()
    return _MODEL


# ---------------------------------------------------------------------------
# 目录浏览
# ---------------------------------------------------------------------------
def _count_audio(root: Path, cap: int = 4000, budget: float = 0.35) -> int:
    """有上限地数音频文件，避免在盘符根目录（如 C:\\）上卡住。"""
    n, t0 = 0, time.time()
    try:
        for i, p in enumerate(root.rglob("*")):
            if time.time() - t0 > budget or n >= cap:
                break
            if p.is_file() and p.suffix.lower() in CF.AUDIO_EXTS:
                n += 1
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
                    dirs.append({"name": e.name, "path": str(e),
                                 "audio": _count_audio(e)})
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
        cands.append({"time": c["time"], "mmss": c["mmss"], "prob": round(prob, 4),
                      "conf": c["confidence"], "text": c.get("text", ""),
                      "note": c.get("confidence_note", "")})
    return {"ok": True, "duration": round(res.get("duration", 0), 2),
            "candidates": cands, "has_climax": res.get("has_climax"),
            "max_prob": res.get("max_prob"), "elapsed": round(time.time() - t0, 1),
            "lrc": _find_lrc(p).name if _find_lrc(p) else "",
            "model": model.meta}


@app.get("/api/status")
def api_status():
    try:
        m = get_model()
        return {"ok": True, "meta": m.meta}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ===========================================================================
# 前端
# ===========================================================================
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>🎧</text></svg>">
<title>高潮点播放器</title>
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
#side h1{margin:0;padding:14px 16px;font-size:15px;font-weight:600;
  border-bottom:1px solid var(--line);letter-spacing:.4px}
#side h1 span{color:var(--dim);font-weight:400;font-size:12px;margin-left:8px}
#crumbs{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;gap:6px;
  align-items:center;font-size:12px;color:var(--dim);min-height:34px}
#crumbs b{color:var(--fg);font-weight:500}
#listing{flex:1;overflow:auto;padding:4px 0}
.row{display:flex;align-items:center;gap:8px;padding:5px 14px;cursor:pointer;
  white-space:nowrap;font-size:13px}
.row:hover{background:var(--panel2)}
.row.sel{background:#1b3a5c}
.row .nm{overflow:hidden;text-overflow:ellipsis;flex:1}
.row .mt{color:var(--dim);font-size:11px;font-variant-numeric:tabular-nums}
.row.dir .nm{color:#b9c6da}
.row.lrc .nm{color:#8fd4b0}
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
</style>
</head>
<body>
<div id="app">
  <aside id="side">
    <h1>高潮点播放器<span id="ver"></span></h1>
    <div id="crumbs">位置</div>
    <div id="listing"></div>
  </aside>

  <section id="main">
    <div id="top">
      <div id="title">未选择文件</div>
      <label class="muted">候选数
        <select id="topn"><option>3</option><option selected>6</option><option>10</option></select>
      </label>
      <label class="muted">最低概率
        <select id="minp">
          <option value="0">0</option><option value="0.3">30%</option>
          <option value="0.5">50%</option><option value="0.6">60%</option>
        </select>
      </label>
      <button class="primary" id="go" disabled>分析高潮点</button>
      <button id="clr" disabled title="清除已导入的歌词">清除歌词</button>
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
        <span class="sp"></span>
        <label class="muted"><input type="checkbox" id="beep" checked> 提示音</label>
        <label class="muted"><input type="checkbox" id="pauseat"> 到点暂停</label>
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
const au = $('au');
let cur = null;              // 当前音频路径
let cands = [];              // [{time,mmss,prob,conf,text,note,fired,warned}]
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
async function browse(path){
  let d;
  try{
    const r = await fetch('/api/browse?path=' + encodeURIComponent(path||''));
    if (!r.ok) throw new Error((await r.json()).detail || r.status);
    d = await r.json();
  }catch(e){ toast('无法读取目录：' + e.message, 'err'); return; }

  $('crumbs').innerHTML = d.path
    ? `<button class="ghost" id="up">↑</button><b>${esc(d.path)}</b>`
    : '<b>此电脑</b>';
  if (d.path) $('up').onclick = () => browse(d.parent);

  const L = $('listing');
  L.innerHTML = '';
  for (const x of d.dirs){
    const el = document.createElement('div');
    el.className = 'row dir';
    el.innerHTML = `<span class="nm">📁 ${esc(x.name)}</span>`
      + (x.audio ? `<span class="mt">${x.audio}</span>` : '');
    el.onclick = () => browse(x.path);
    L.appendChild(el);
  }
  for (const x of d.audio){
    const el = document.createElement('div');
    el.className = 'row' + (x.path === cur ? ' sel' : '');
    el.dataset.path = x.path;
    el.innerHTML = `<span class="nm">♪ ${esc(x.name)}</span>`
      + `<span class="mt">${(x.size/1048576).toFixed(1)}M</span>`;
    el.onclick = () => pick(x.path, x.name);
    L.appendChild(el);
  }
  for (const x of (d.lrcs || [])){
    const el = document.createElement('div');
    el.className = 'row lrc';
    el.innerHTML = `<span class="nm">💬 ${esc(x.name)}</span>`;
    el.title = cur ? '设为本音频的歌词' : '先选择一个音频文件';
    el.onclick = () => {
      if (!cur){ toast('先选择音频文件', 'err'); return; }
      useLrc(x.path, x.name);
    };
    L.appendChild(el);
  }
  if (!d.dirs.length && !d.audio.length && !(d.lrcs||[]).length){
    L.innerHTML = '<div class="empty">此目录没有子目录或音频文件</div>';
  }
}
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function pick(path, name){
  cur = path;
  cands = []; lyr = []; lyrIdx = -1; pickedLrc = '';
  document.querySelectorAll('.row.sel').forEach(e => e.classList.remove('sel'));
  const e = document.querySelector(`.row[data-path="${CSS.escape(path)}"]`);
  if (e) e.classList.add('sel');

  $('title').innerHTML = esc(name) + ' <small id="meta"></small>';
  au.src = '/api/audio?path=' + encodeURIComponent(path);
  au.load();
  $('cur').textContent = '00:00';
  $('dur').textContent = '00:00';
  $('go').disabled = false; $('clr').disabled = false;
  $('pp').disabled = $('b10').disabled = $('f10').disabled = $('rate').disabled = $('vol').disabled = false;
  $('hits').innerHTML = '<div class="empty">点「分析高潮点」</div>';
  $('lyrcard').style.display = 'none';
  drawMarkers();
  toast('已选择：' + name);
}

clr.onclick = () => {
  pickedLrc = ''; lyr = []; lyrIdx = -1;
  lyrcard.style.display = 'none';
  toast('已清除导入的歌词');
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
    cands = (d.candidates||[]).map(c => ({...c, fired:false, warned:false}));
    $('meta').textContent = `　${fmt(d.duration)}　${cands.length} 个候选　`
      + `耗时 ${d.elapsed}s` + (d.has_climax === false ? '　（可能没有高潮）' : '');
    renderHits();
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
      + `<span class="x">${esc(c.text || '')}</span>`;
    el.onclick = () => { au.currentTime = Math.max(0, c.time - 2); au.play(); };
    box.appendChild(el);
  }
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
$('flash').classList.remove('on');
fetch('/api/status').then(r => r.json()).then(d => {
  if (d.ok){
    const m = d.meta || {};
    $('ver').textContent = `${m.n_works||'?'}部作品训练　10折 AUC ${(m.cv_auc||0).toFixed(3)}`;
  } else {
    $('ver').textContent = '模型未就绪';
    $('err').textContent = d.error || '';
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
    ap = argparse.ArgumentParser(description="高潮点播放器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    a = ap.parse_args()

    print("=" * 66)
    print("  高潮点播放器")
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
