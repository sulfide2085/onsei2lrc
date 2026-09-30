/* 试不同的 top，看候选池够不够容纳两处高潮 */
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const DIR = __dirname;
require(path.join(DIR, 'core.js'));
const C = globalThis.ClimaxCore;
const E = new C.Engine(JSON.parse(
  zlib.gunzipSync(fs.readFileSync(path.join(DIR, 'model.json.gz'))).toString('utf8')));
// ref_audio.f32 是 float32（Python 写出的解码结果）
const buf = fs.readFileSync(path.join(DIR, 'ref_audio.f32'));
const x32 = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
const x = new Float64Array(x32.length);
for (let i = 0; i < x32.length; i++) x[i] = x32[i];

console.log('  官方标注: 1149 秒 (19:09) / 1576 秒 (26:16)');
console.log();
console.log('  ' + 'top'.padEnd(6) + '门槛'.padEnd(8) + '输出');
for (const top of [3, 4, 6, 8]) {
  for (const mp of [0.5, 0.0]) {
    const r = C.analyze(x, E, { top: top, minProb: mp });
    const hit = r.candidates.filter(c =>
      [1149, 1576].some(g => Math.abs(c.time - g) <= 20)).length;
    console.log('  ' + String(top).padEnd(6) + String(mp).padEnd(8) +
      r.candidates.map(c => c.mmss + '(' + (c.probability * 100).toFixed(0) + '%)')
        .join('  ') + '   命中 ' + hit + '/2');
  }
}
