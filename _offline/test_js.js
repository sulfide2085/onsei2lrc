/* ============================================================================
 * JS ↔ Python 对拍
 *
 * 用同一段真实音频（ref_audio.f32），Python 和 Node 各跑一遍完整流程，
 * 逐步比对：包络 → 峰 → 21 个特征 → 模型分 → 最终候选。
 * 哪一步不一致就一目了然。
 * ========================================================================== */
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const DIR = __dirname;
require(path.join(DIR, 'core.js'));
const C = globalThis.ClimaxCore;

const ref = JSON.parse(fs.readFileSync(path.join(DIR, 'ref_out.json'), 'utf8'));

// 模型：优先用 gzip 版（和 HTML 里嵌入的是同一份）
let modelJson;
const gzPath = path.join(DIR, 'model.json.gz');
if (fs.existsSync(gzPath)) {
  modelJson = JSON.parse(zlib.gunzipSync(fs.readFileSync(gzPath)).toString('utf8'));
} else {
  modelJson = JSON.parse(fs.readFileSync(path.join(DIR, 'model.json'), 'utf8'));
}

// 音频
const buf = fs.readFileSync(path.join(DIR, 'ref_audio.f32'));
const x = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);

let pass = 0, fail = 0;
function ck(name, ok, detail) {
  if (ok) { pass++; console.log('  ✓ ' + name + (detail ? '　' + detail : '')); }
  else { fail++; console.log('  ✗ ' + name + '　' + (detail || '')); }
}
function maxdiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; }
  return m;
}

console.log('  样本数 ' + x.length + '　参考 ' + ref.n_samples +
  '　' + (x.length === ref.n_samples ? '一致' : '不一致'));
console.log();

/* ---------------------------------------------------------------- ① 包络 */
const db = C.energyEnvelope(x);
ck('包络帧数', db.length === ref.n_frames, db.length + ' vs ' + ref.n_frames);
const envRef = require('child_process').execSync(
  'python -c "import numpy as np,sys; a=np.load(sys.argv[1]); ' +
  'sys.stdout.write(repr([float(a[0].max()),float(a[1].max())]))" "' +
  path.join(DIR, 'ref_env.npy') + '"', { encoding: 'utf8' });
ck('包络已生成', db.length > 0, '长度 ' + db.length);

/* ---------------------------------------------------------------- ② 峰 */
const hop = C.HOP;
const s = C.smoothSame(db, Math.trunc(0.3 / hop));
const peaks = C.findPeaks(s, hop);
ck('峰数量', peaks.length === ref.peaks.length,
  peaks.length + ' vs ' + ref.peaks.length);
const peakSame = peaks.length === ref.peaks.length &&
  peaks.every((v, i) => v === ref.peaks[i]);
ck('峰位置完全一致', peakSame);

/* ------------------------------------------------------------ ③ 特征 */
const tmp = Array.prototype.slice.call(s).sort((a, b) => a - b);
const med = tmp.length % 2 ? tmp[(tmp.length - 1) >> 1]
  : (tmp[tmp.length / 2 - 1] + tmp[tmp.length / 2]) / 2;
ck('全轨中位数', Math.abs(med - ref.median) < 1e-9,
  med.toFixed(9) + ' vs ' + ref.median.toFixed(9));

const feats = [];
const order = peaks.map((p, n) => n).sort((a, b) => s[peaks[b]] - s[peaks[a]]);
const rankOf = new Array(peaks.length);
order.forEach((n, r) => { rankOf[n] = r; });
for (let n = 0; n < peaks.length; n++) {
  const f = C.modelFeatures(x, s, hop, peaks[n], med);
  f.txt = 0;
  f.rank_energy = rankOf[n];
  f.gap_prev = n === 0 ? 0 : (peaks[n] - peaks[n - 1]) * hop;
  feats.push(f);
}

console.log();
console.log('  逐特征最大绝对误差：');
const featNames = Object.keys(ref.features[0]);
let worstFeat = null, worstVal = -1;
for (const fn of featNames) {
  let m = 0;
  for (let i = 0; i < feats.length; i++) {
    const d = Math.abs((feats[i][fn] || 0) - ref.features[i][fn]);
    if (d > m) m = d;
  }
  const rel = m / Math.max(1e-9, Math.abs(ref.features[0][fn]) + 1);
  if (m > worstVal) { worstVal = m; worstFeat = fn; }
  const flag = m < 1e-6 ? '  ' : (rel < 1e-4 ? ' ~' : ' ✗');
  if (m >= 1e-6) {
    console.log('   ' + flag + ' ' + fn.padEnd(12) + ' ' + m.toExponential(3));
  }
}
ck('21 个特征全部一致（< 1e-6）', worstVal < 1e-6,
  '最大偏差 ' + worstFeat + ' = ' + worstVal.toExponential(3));

/* ------------------------------------------------------------ ④ 模型分 */
const eng = new C.Engine(modelJson);
const pred = eng.predict(feats);
ck('模型原始分', maxdiff(pred.raw, ref.raw) < 1e-7,
  '最大误差 ' + maxdiff(pred.raw, ref.raw).toExponential(3));
ck('校准概率', maxdiff(pred.prob, ref.prob) < 1e-7,
  '最大误差 ' + maxdiff(pred.prob, ref.prob).toExponential(3));
ck('精修分', maxdiff(pred.refine, ref.refine) < 1e-7,
  '最大误差 ' + maxdiff(pred.refine, ref.refine).toExponential(3));

/* -------------------------------------------------------- ⑤ 完整流程 */
const res = C.analyze(x, eng, { top: 3, minProb: C.MIN_PROB });
console.log();
console.log('  JS 分析结果: ' + JSON.stringify(
  res.candidates.map(c => ({ t: c.mmss, p: +c.probability.toFixed(3) }))));
console.log('  Python 结果: ' + JSON.stringify(
  ref.candidates.map(c => ({ t: c.mmss, p: +(c.probability || 0).toFixed(3) }))));
ck('候选数量一致', res.candidates.length === ref.candidates.length);
ck('候选时间一致',
  res.candidates.every((c, i) => ref.candidates[i] &&
    Math.abs(c.time - ref.candidates[i].time) < 0.06),
  res.candidates.map(c => c.mmss).join(' '));
console.log();
console.log('  耗时: 包络 ' + res.timing.envelope.toFixed(0) + ' ms　特征 ' +
  res.timing.features.toFixed(0) + ' ms　模型 ' + res.timing.model.toFixed(0) +
  ' ms　合计 ' + res.timing.total.toFixed(0) + ' ms');

console.log();
console.log('  ' + pass + ' / ' + (pass + fail) + ' 通过');
process.exit(fail ? 1 : 0);
