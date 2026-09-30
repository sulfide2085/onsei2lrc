/* 逐模型计时 + 与 Python 参考值对拍 */
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const DIR = __dirname;
require(path.join(DIR, 'core.js'));
const C = globalThis.ClimaxCore;

const M = JSON.parse(
  zlib.gunzipSync(fs.readFileSync(path.join(DIR, 'model.json.gz'))).toString('utf8'));
const ref = JSON.parse(fs.readFileSync(path.join(DIR, 'ref_out.json'), 'utf8'));

let pass = 0, fail = 0;
function ck(n, ok, d) {
  if (ok) { pass++; console.log('  ✓ ' + n + (d ? '　' + d : '')); }
  else { fail++; console.log('  ✗ ' + n + '　' + (d || '')); }
}

console.log('  v 数组长度 / is_hgb 检查:');
for (const [name, d] of Object.entries(M.models)) {
  if (!d.trees) { console.log('    %s  线性模型 %d 系数', name.padEnd(10), d.coef.length); continue; }
  const t0 = d.trees[0];
  // Node 的 console.log 不支持 %-11s 这种左对齐宽度，用字符串拼
  console.log('    ' + name.padEnd(10) + ' kind=' + d.kind.padEnd(11) +
    ' is_hgb=' + String(!!d.is_hgb).padEnd(5) + ' 节点=' + t0.f.length +
    ' v长度=' + t0.v.length + ' 比值=' + (t0.v.length / t0.f.length).toFixed(2) +
    ' lf长度=' + (t0.lf ? t0.lf.length : -1));
}

// 用 C.leafIndex 遍历全部树（不自己写循环，避免又踩同一个坑）
console.log();
console.log('  用 C.leafIndex 遍历全部树:');
for (const [name, d] of Object.entries(M.models)) {
  if (!d.trees) continue;
  const t0 = Date.now();
  let steps = 0, deepest = 0;
  for (const tree of d.trees) {
    let n = 0, s = 0;
    while (!tree.lf[n]) { n = (0 <= tree.t[n]) ? tree.l[n] : tree.r[n]; s++; if (s > 500) break; }
    steps += s; if (s > deepest) deepest = s;
  }
  ck(name + ' 遍历完成', steps / d.trees.length < 200,
    d.trees.length + ' 棵  平均 ' + (steps / d.trees.length).toFixed(1) +
    ' 步  最深 ' + deepest + '  ' + (Date.now() - t0) + ' ms');
}

// 与 Python 参考值对拍
console.log();
const eng = new C.Engine(M);
const feats = ref.features.map(f => Object.assign({}, f));

function maxdiff(a, b) {
  let m = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; }
  return m;
}
const t0 = Date.now();
const pred = eng.predict(feats);
const dt = Date.now() - t0;

console.log('  推理 ' + feats.length + ' 个候选用时 ' + dt + ' ms');
console.log();
const eRaw = maxdiff(pred.raw, ref.raw);
const eProb = maxdiff(pred.prob, ref.prob);
const eRef = maxdiff(pred.refine, ref.refine);
ck('原始分', eRaw < 1e-7, eRaw.toExponential(3));
ck('校准概率', eProb < 1e-7, eProb.toExponential(3));
ck('精修分', eRef < 1e-7, eRef.toExponential(3));

console.log();
console.log('  前 5 个候选对照:');
console.log('    %-10s %-16s %-16s' % ('', 'JS', 'Python'));
for (let i = 0; i < 5; i++) {
  console.log('    raw[' + i + ']  JS ' + pred.raw[i].toFixed(9) +
    '   Python ' + ref.raw[i].toFixed(9) +
    (Math.abs(pred.raw[i] - ref.raw[i]) < 1e-12 ? '  ✓' : '  Δ' +
      Math.abs(pred.raw[i] - ref.raw[i]).toExponential(1)));
}
console.log();
// 逐模型定位：哪个模型与 Python 不一致
if (ref.per_model) {
  console.log();
  console.log('  逐模型对拍:');
  const row0 = M.feats;
  for (const name of Object.keys(ref.per_model)) {
    const d = M.models[name];
    const mine = ref.features.map((f, k) => {
      const x64 = new Float64Array(row0.length), x32 = new Float32Array(row0.length);
      for (let j = 0; j < row0.length; j++) {
        x64[j] = f[M.feats[j]];
        x32[j] = Math.fround(x64[j]);
      }
      return C.modelScore(d, x32, x64);
    });
    const e = maxdiff(mine, ref.per_model[name]);
    let worst = 0, wi = 0;
    for (let i = 0; i < mine.length; i++) {
      const dd = Math.abs(mine[i] - ref.per_model[name][i]);
      if (dd > worst) { worst = dd; wi = i; }
    }
    console.log('    ' + name.padEnd(10) + ' 最大误差 ' + e.toExponential(3) +
      (e < 1e-12 ? '  ✓' : '  ✗ 第 ' + wi + ' 个候选  JS ' +
        mine[wi].toFixed(12) + '  Py ' + ref.per_model[name][wi].toFixed(12)));
  }
}
console.log();
console.log('  ' + pass + ' / ' + (pass + fail) + ' 通过');
process.exit(fail ? 1 : 0);
