/* FFT 单元验证 —— 用确定性信号对拍 numpy */
'use strict';
const fs = require('fs');
const path = require('path');
const DIR = __dirname;
require(path.join(DIR, 'core.js'));
const C = globalThis.ClimaxCore;

const probe = null;   // 探针文件已废弃，这里改用确定性合成信号
let pass = 0, fail = 0;
function ck(name, ok, d) {
  if (ok) { pass++; console.log('  ✓ ' + name + (d ? '　' + d : '')); }
  else { fail++; console.log('  ✗ ' + name + '　' + (d || '')); }
}

// 用一段简单信号，走 modelFeatures 里的频谱分支
// 构造：1 秒静音 + 冲击 + 1 秒静音，取中间那个峰
const N = 32000;
const x = new Float64Array(N);
for (let i = 0; i < N; i++) {
  x[i] = 0.15 * Math.sin(2 * Math.PI * 440 * i / 16000) +
         0.05 * Math.sin(2 * Math.PI * 3000 * i / 16000);
}
const db = C.energyEnvelope(x);
const s = C.smoothSame(db, Math.trunc(0.3 / C.HOP));
const tmp = Array.from(s).sort((a, b) => a - b);
const med = tmp.length % 2 ? tmp[(tmp.length - 1) >> 1]
  : (tmp[tmp.length / 2 - 1] + tmp[tmp.length / 2]) / 2;
const i0 = Math.floor(1.0 / C.HOP);
const f = C.modelFeatures(x, s, C.HOP, i0, med);
console.log('  纯音信号（440Hz + 3000Hz）的特征:');
console.log('    zcr      ' + f.zcr.toFixed(9));
console.log('    centroid ' + f.centroid.toFixed(6));
console.log('    flatness ' + f.flatness.toFixed(9));
fs.writeFileSync(path.join(DIR, 'js_feat.json'), JSON.stringify(
  { zcr: f.zcr, centroid: f.centroid, flatness: f.flatness, med: med }));
console.log();
console.log('  已写出 js_feat.json，交给 Python 对照');
process.exit(0);
