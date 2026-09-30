# -*- coding: utf-8 -*-
"""把模型 + 核心逻辑 + 播放器 UI 打包成一个自包含的 HTML

特点：
  · 单文件，双击就能用，不需要 Python / FFmpeg / 服务器 / 网络
  · 模型直接内嵌（JSON 放在 <script type="application/json"> 里，不走 base64）
  · 手机适配（顶栏换行、触控目标放大、dvh 高度、16px 输入框）
"""
import json
import re
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
OFF = ROOT / "_offline"

core = (OFF / "core.js").read_text(encoding="utf-8")
model = (OFF / "model.json").read_text(encoding="utf-8")
meta = json.loads(model)["meta"]

# 内嵌进 <script> 时必须防 </script> 提前闭合
model_safe = model.replace("</", "<\\/")

HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="dark">
<title>高潮点分析</title>
<style>
:root{--bg:#0e1013;--panel:#16191f;--panel2:#1d2128;--line:#272c35;
  --fg:#e6e9ef;--dim:#8b93a3;--accent:#4da3ff;--ok:#35c88a;--hot:#ff4d6a}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--fg);overflow:hidden;
  font:14px/1.55 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}
button,input{font:inherit;color:inherit}
button{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
  padding:8px 14px;cursor:pointer;transition:.12s}
button:hover:not(:disabled){background:#262b34;border-color:#39404d}
button:disabled{opacity:.38;cursor:default}
button.primary{background:var(--accent);border-color:var(--accent);color:#06121f;font-weight:600}

#app{display:flex;flex-direction:column;height:100dvh;height:100vh}
#top{display:flex;align-items:center;gap:9px;flex-wrap:wrap;
  padding:12px 16px;border-bottom:1px solid var(--line);flex:none}
#name{flex:1 1 100%;font-weight:600;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;color:var(--dim)}
#name.on{color:var(--fg)}
label.m{display:inline-flex;align-items:center;gap:5px;color:var(--dim);font-size:13px}
select{background:var(--panel2);border:1px solid var(--line);border-radius:7px;
  padding:6px 9px;color:inherit}

#body{flex:1;overflow:auto;padding:16px 16px 40px;min-height:0}
#drop{border:2px dashed var(--line);border-radius:14px;padding:46px 20px;
  text-align:center;color:var(--dim);transition:.15s;cursor:pointer}
#drop:hover,#drop.over{border-color:var(--accent);color:var(--fg);background:#141a22}
#drop b{display:block;font-size:16px;color:var(--fg);margin-bottom:6px;font-weight:600}
#drop small{font-size:12px;line-height:1.7}

#play{display:none}
#play.on{display:block}
audio{width:100%;margin-bottom:14px}
#tl{position:relative;height:64px;background:var(--panel);border:1px solid var(--line);
  border-radius:9px;overflow:hidden;cursor:pointer;touch-action:none}
#fill{position:absolute;inset:0 auto 0 0;background:rgba(77,163,255,.13)}
#pos{position:absolute;top:0;bottom:0;width:2px;background:var(--accent);pointer-events:none}
.mk{position:absolute;top:0;bottom:0;width:3px;border-radius:2px;
  background:var(--hot);opacity:.85;pointer-events:none}
.mk.hi{opacity:1;width:5px}
#clock{display:flex;justify-content:space-between;color:var(--dim);font-size:12px;
  font-variant-numeric:tabular-nums;margin:6px 2px 16px}

#ctrl{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:16px}
#ctrl .sp{flex:1}
#vol{width:120px;accent-color:var(--accent)}

#hits{display:flex;flex-direction:column;gap:7px}
.hit{display:flex;align-items:center;gap:11px;padding:11px 13px;border-radius:9px;
  background:var(--panel);border:1px solid var(--line);cursor:pointer;transition:.12s}
.hit:hover{border-color:#3a4351;background:#1a1e25}
.hit .t{font-variant-numeric:tabular-nums;font-weight:600;min-width:56px;font-size:15px}
.hit .p{font-variant-numeric:tabular-nums;font-size:12px;min-width:52px;color:var(--dim)}
.hit .lv{font-size:11px;padding:2px 7px;border-radius:5px;flex:none;font-weight:600}
.hit .lv.h{background:rgba(255,77,106,.16);color:#ff8298}
.hit .lv.m{background:rgba(255,176,32,.16);color:#ffc45c}
.hit .lv.l{background:rgba(139,147,163,.16);color:var(--dim)}
.hit .sp{flex:1}
.empty{color:var(--dim);padding:22px 4px;font-size:13px}
.note{color:var(--dim);font-size:12px;line-height:1.75;margin-top:14px}

#toast{position:fixed;left:50%;bottom:26px;transform:translate(-50%,180%);
  background:var(--hot);color:#fff;font-weight:700;font-size:15px;
  padding:11px 26px;border-radius:11px;transition:transform .28s cubic-bezier(.2,.9,.3,1.3);
  pointer-events:none;z-index:40;box-shadow:0 6px 24px rgba(255,77,106,.35)}
#toast.on{transform:translate(-50%,0)}
#err{color:#ff8298;font-size:13px;flex:1 1 100%}
#busy{display:none;position:fixed;inset:0;background:rgba(14,16,19,.82);z-index:50;
  align-items:center;justify-content:center;flex-direction:column;gap:14px;color:var(--dim)}
#busy.on{display:flex}
.spin{width:34px;height:34px;border:3px solid var(--line);border-top-color:var(--accent);
  border-radius:50%;animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}

@media (max-width:820px){
  #top{padding:10px 12px;gap:8px}
  #body{padding:12px 12px 40px}
  button{padding:10px 14px}
  select{padding:9px 10px;font-size:16px}
  #vol{width:100%;order:9}
  #ctrl .sp{display:none}
  #tl{height:78px}
  .mk{width:4px}.mk.hi{width:6px}
  .hit{padding:13px 13px}
  .hit .t{font-size:16px}
  #drop{padding:34px 16px}
}
</style>
</head>
<body>
<div id="app">
  <div id="top">
    <button class="primary" id="pick">选择音频</button>
    <label class="m" title="门槛越高误报越少、漏掉的越多。数字是 10 折留一作品的实测值（准=精确率，全=召回率）。">最低概率
      <select id="minp">
        <option value="0">不过滤 · 准50/全74</option>
        <option value="0.17">≥17% · 准59/全73</option>
        <option value="0.25">≥25% · 准69/全69</option>
        <option value="0.5" selected>≥50% · 准73/全69</option>
        <option value="0.66">≥66% · 准87/全57</option>
        <option value="0.67">≥67% · 准98/全41</option>
      </select>
    </label>
    <span id="name">未选择文件</span>
    <span id="err"></span>
  </div>
  <div id="body">
    <div id="drop">
      <b>把音频拖到这里，或点上面的「选择音频」</b>
      <small>MP3 / WAV / FLAC / M4A / OGG 都行<br>
      全部在本机浏览器里算，音频不会上传，也不联网</small>
    </div>
    <div id="play">
      <audio id="au" controls preload="metadata"></audio>
      <div id="tl"><div id="fill"></div><div id="pos"></div></div>
      <div id="clock"><span id="cur">00:00</span><span id="dur">00:00</span></div>
      <div id="ctrl">
        <button id="b10">−10s</button>
        <button id="f10">+10s</button>
        <select id="rate">
          <option value="0.75">0.75×</option><option value="1" selected>1×</option>
          <option value="1.25">1.25×</option><option value="1.5">1.5×</option>
          <option value="2">2×</option>
        </select>
        <input type="range" id="vol" min="0" max="1" step="0.02" value="1">
        <span class="sp"></span>
        <label class="m"><input type="checkbox" id="beep" checked> 提示音</label>
        <label class="m"><input type="checkbox" id="pauseat"> 到点暂停</label>
      </div>
      <div id="hits"><div class="empty">点上面的「选择音频」开始</div></div>
      <div class="note" id="foot"></div>
      <div class="note" id="about" style="margin-top:10px"></div>
    </div>
  </div>
</div>
<div id="toast">高潮点</div>
<div id="busy"><div class="spin"></div><div id="busytxt">正在分析…</div></div>
<input type="file" id="file" accept="audio/*,.mp3,.wav,.flac,.m4a,.ogg,.aac" hidden>

<script type="application/json" id="model">{MODEL}</script>
<script>{CORE}</script>
<script>
'use strict';
const $ = id => document.getElementById(id);
const C = globalThis.ClimaxCore;
const ENGINE = new C.Engine(JSON.parse($('model').textContent));

let CUES = [];          // 分析出的候选（秒）
let LAST = -99;         // 上一次提示的时间，避免连续触发
let URL_ = null;

/* ---------------------------------------------------------- 工具 */
const fmt = s => {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), q = s % 60;
  return (h ? h + ':' + String(m).padStart(2, '0') : String(m)) + ':' + String(q).padStart(2, '0');
};
function toast(msg) {
  const t = $('toast');
  if (msg) t.textContent = msg;
  t.classList.add('on');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove('on'), 2600);
}
function busy(on, txt) {
  $('busy').classList.toggle('on', !!on);
  if (txt) $('busytxt').textContent = txt;
}

/* ---------------------------------------------------- 音频解码 */
// 统一解码成 16 kHz 单声道 float64，与 Python 侧一致
async function decodeFile(file) {
  const buf = await file.arrayBuffer();
  let ctx = null, ab = null;
  try {
    ctx = new AudioContext({ sampleRate: C.SR });
    ab = await ctx.decodeAudioData(buf.slice(0));
  } catch (e) {
    if (ctx) { try { ctx.close(); } catch (_) {} }
    ctx = new AudioContext();                      // 退回原生采样率
    ab = await ctx.decodeAudioData(buf);
  }
  const n = ab.length;
  let out = new Float64Array(n);
  for (let ch = 0; ch < ab.numberOfChannels; ch++) {
    const d = ab.getChannelData(ch);
    for (let i = 0; i < n; i++) out[i] += d[i];
  }
  if (ab.numberOfChannels > 1) for (let i = 0; i < n; i++) out[i] /= ab.numberOfChannels;
  const rate = ab.sampleRate;
  try { ctx.close(); } catch (_) {}
  if (rate !== C.SR) out = resample(out, rate, C.SR);   // Safari 等不支持指定采样率
  return out;
}
// 线性插值重采样（只在浏览器不支持 16 kHz AudioContext 时才走到）
function resample(x, from, to) {
  const n = Math.max(1, Math.round(x.length * to / from));
  const out = new Float64Array(n), r = (x.length - 1) / Math.max(1, n - 1);
  for (let i = 0; i < n; i++) {
    const p = i * r, i0 = Math.floor(p), i1 = Math.min(x.length - 1, i0 + 1);
    out[i] = x[i0] + (x[i1] - x[i0]) * (p - i0);
  }
  return out;
}

/* ------------------------------------------------------ 分析 */
async function run(file) {
  $('err').textContent = '';
  busy(true, '正在解码音频…');
  await new Promise(r => setTimeout(r, 30));       // 让遮罩先画出来
  let x;
  try {
    x = await decodeFile(file);
  } catch (e) {
    busy(false);
    $('err').textContent = '这个文件解不开：' + (e && e.message ? e.message : e);
    return;
  }
  if (x.length / C.SR < C.MIN_DURATION) {
    busy(false);
    $('err').textContent = '时长 ' + (x.length / C.SR).toFixed(0) +
      ' 秒，短于 ' + C.MIN_DURATION + ' 秒——短轨的能量特征不稳定，跳过';
    return;
  }
  busy(true, '正在分析…');
  await new Promise(r => setTimeout(r, 30));
  const t0 = performance.now();
  let res;
  try {
    // top=6：候选池是 top+1=7 个名额。用 3 的话，同一处高潮散成的多个峰
    // 会把池子占满，别处真正的高潮根本进不来（实测少找到一处）。
    res = C.analyze(x, ENGINE, { top: 6, minProb: parseFloat($('minp').value) });
  } catch (e) {
    busy(false);
    $('err').textContent = '分析出错：' + (e && e.message ? e.message : e);
    return;
  }
  const ms = performance.now() - t0;
  busy(false);
  if (res.error) { $('err').textContent = res.error; return; }

  CUES = res.candidates.map(c => c.time);
  $('name').textContent = file.name;
  $('name').classList.add('on');
  $('name').title = file.name + '　' + fmt(res.duration) +
    '　' + res.nPeaks + ' 个候选峰';
  $('foot').textContent =
    '分析用时 ' + ms.toFixed(0) + ' 毫秒（包络 ' + res.timing.envelope.toFixed(0) +
    'ms　特征 ' + res.timing.features.toFixed(0) +
    'ms　模型 ' + res.timing.model.toFixed(0) + 'ms）　' +
    '全轨 ' + res.nPeaks + ' 个候选峰　共 ' + res.candidates.length + ' 个高潮点';
  render(res.candidates);
  drawMarks();
}

function render(cs) {
  const box = $('hits');
  if (!cs.length) {
    box.innerHTML = '<div class="empty">没找到够格的候选。' +
      '可以把上面的「最低概率」调低再看看。</div>';
    return;
  }
  box.innerHTML = '';
  cs.forEach(c => {
    const d = document.createElement('div');
    d.className = 'hit';
    const cls = c.probability >= 0.6 ? 'h' : (c.probability >= 0.3 ? 'm' : 'l');
    d.innerHTML = '<span class="t">' + c.mmss + '</span>' +
      '<span class="p">' + (c.probability * 100).toFixed(0) + '%</span>' +
      '<span class="lv ' + cls + '">' + c.confidence + '</span>' +
      '<span class="sp"></span>';
    d.onclick = () => { $('au').currentTime = c.time; $('au').play(); };
    box.appendChild(d);
  });
}

/* ------------------------------------------------------ 时间轴 */
function drawMarks() {
  const tl = $('tl');
  [...tl.querySelectorAll('.mk')].forEach(e => e.remove());
  const au = $('au');
  const dur = au.duration && isFinite(au.duration) ? au.duration : 0;
  if (!dur) return;
  CUES.forEach(t => {
    const m = document.createElement('div');
    m.className = 'mk' + (t / dur < 0.999 ? '' : '');
    m.style.left = (t / dur * 100) + '%';
    tl.appendChild(m);
  });
}
function tick() {
  const au = $('au');
  const d = au.duration && isFinite(au.duration) ? au.duration : 0;
  if (!d) return;
  $('pos').style.left = (au.currentTime / d * 100) + '%';
  $('fill').style.width = (au.currentTime / d * 100) + '%';
  $('cur').textContent = fmt(au.currentTime);
  $('dur').textContent = fmt(d);
  if (!au.paused) {
    for (const t of CUES) {
      if (au.currentTime >= t && au.currentTime - t < 1.2 &&
          Math.abs(au.currentTime - LAST) > 4) {
        LAST = au.currentTime;
        toast('高潮点 ' + fmt(t));
        if ($('beep').checked) beep();
        if ($('pauseat').checked) au.pause();
        break;
      }
    }
  }
}
let AC = null;
function beep() {
  try {
    AC = AC || new (window.AudioContext || window.webkitAudioContext)();
    const o = AC.createOscillator(), g = AC.createGain();
    o.frequency.value = 880; o.type = 'sine';
    g.gain.setValueAtTime(0.001, AC.currentTime);
    g.gain.exponentialRampToValueAtTime(0.25, AC.currentTime + 0.02);
    g.gain.exponentialRampToValueAtTime(0.001, AC.currentTime + 0.5);
    o.connect(g); g.connect(AC.destination);
    o.start(); o.stop(AC.currentTime + 0.52);
  } catch (_) {}
}

/* ------------------------------------------------------ 交互 */
function load(file) {
  if (!file) return;
  if (URL_) URL_.revokeObjectURL && URL.revokeObjectURL(URL_);
  URL_ = window.URL.createObjectURL(file);
  const au = $('au');
  au.src = URL_;
  $('drop').style.display = 'none';
  $('play').classList.add('on');
  $('name').textContent = file.name;
  $('name').classList.add('on');
  // **必须清空上一个文件的结果**：否则新文件分析完之前，
  // 界面上还挂着旧候选，外部读 CUES 也会拿到旧值。
  LAST = -99;
  CUES = [];
  drawMarks();
  $('hits').innerHTML = '<div class="empty">正在分析…</div>';
  $('foot').textContent = '';
  $('err').textContent = '';
  run(file);
}

$('pick').onclick = () => $('file').click();
$('file').onchange = e => load(e.target.files[0]);
$('drop').onclick = () => $('file').click();
['dragenter', 'dragover'].forEach(ev =>
  document.addEventListener(ev, e => {
    e.preventDefault();
    if (!$('play').classList.contains('on')) $('drop').classList.add('over');
  }));
['dragleave', 'drop'].forEach(ev =>
  document.addEventListener(ev, e => {
    e.preventDefault(); $('drop').classList.remove('over');
  }));
document.addEventListener('drop', e => {
  const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) load(f);
});

$('tl').onclick = e => {
  const au = $('au');
  if (!au.duration || !isFinite(au.duration)) return;
  const r = $('tl').getBoundingClientRect();
  au.currentTime = Math.min(au.duration, Math.max(0, (e.clientX - r.left) / r.width * au.duration));
};
$('b10').onclick = () => { $('au').currentTime = Math.max(0, $('au').currentTime - 10); };
$('f10').onclick = () => { $('au').currentTime = Math.min($('au').duration || 0, $('au').currentTime + 10); };
$('rate').onchange = e => { $('au').playbackRate = parseFloat(e.target.value); };
$('vol').oninput = e => { $('au').volume = parseFloat(e.target.value); };
$('au').onloadedmetadata = () => { drawMarks(); tick(); };
$('au').ontimeupdate = tick;
$('au').onerror = () => {
  $('err').textContent = '这个浏览器播不了这个格式（但仍可能分析得了）。';
};
$('minp').onchange = () => { const f = $('file').files[0]; if (f) run(f); };
setInterval(tick, 100);

(function init() {
  const m = {META};
  $('about').innerHTML =
    '模型用 ' + (m.n_works || '?') + ' 部作品 / ' + (m.n_tracks || '?') +
    ' 轨真实音声训练。10 折留一作品实测：AUC ' +
    (m.cv_auc || 0).toFixed(3) +
    '，门槛 ≥50% 时精确率 72.7%、召回率 69.0%。<br>' +
    '换一部没训过的作品，大致就是这个水平，不是 95%。' +
    '听到不对的地方属正常，把「最低概率」调高可以少些误报。<br>' +
    '全部计算在本机浏览器里完成，音频不上传、不联网。';
})();
</script>
</body>
</html>
"""

html = (HTML
        .replace("{MODEL}", model_safe)
        .replace("{CORE}", core)
        .replace("{META}", json.dumps({
            "n_works": meta.get("n_works"), "n_tracks": meta.get("n_tracks"),
            "n_rows": meta.get("n_rows"), "cv_auc": meta.get("cv_auc"),
        }, ensure_ascii=False)))

out = OFF / "高潮点分析.html"
out.write_text(html, encoding="utf-8")
print(f"  已生成 {out}")
print(f"    大小 {out.stat().st_size/1048576:.2f} MB")
print(f"    模型内嵌 {len(model)/1048576:.2f} MB，核心代码 {len(core)/1024:.0f} KB")
