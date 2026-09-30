/* ============================================================================
 * 高潮检测核心 —— 与 climax_finder.py 逐行对应
 *
 * 移植时的三个坑（都是实测踩出来的，不要改）：
 *   1. **树遍历必须用 float32**。sklearn 在预测时把输入转 float32 再和阈值比较。
 *      用 float64 遍历，400 棵树里有 1 棵走向不同，误差 1.9e-03；
 *      用 float32 误差 2.8e-16。→ 一律用 Math.fround()
 *   2. **HGB 的叶子值已经乘过学习率**，不要再乘。decision_function == Σ leaf。
 *   3. **np.convolve(mode='same') 不是居中**。k 为偶数时起点是 (k-1)//2，
 *      即窗口偏左。这里按 numpy 的定义实现。
 * ========================================================================== */
(function (root) {
  'use strict';

  const SR = 16000;
  const HOP = 0.05;
  const WIN = 0.10;
  const MIN_DURATION = 60.0;
  const MERGE_GAP = 30.0;
  const SNAP_WINDOW = 12.0;
  const POOL_EXTRA = 1;
  const MIN_PROB = 0.5;
  const PROB_CAP = 0.95;

  const MODEL_FEATS = ['peak', 'pre60', 'pre30', 'pre10', 'post2', 'post10',
    'post30', 'signature', 'contrast', 'rise', 'decay', 'recover',
    'prominence', 'plateau', 'rel_time', 'rank_energy', 'gap_prev',
    'zcr', 'centroid', 'flatness', 'txt'];

  /* ---------------------------------------------------------------- 包络 */
  // 对应 energy_envelope()：逐帧 RMS(dBFS)，帧长 WIN 步长 HOP
  function energyEnvelope(x) {
    const step = Math.trunc(SR * HOP), wlen = Math.trunc(SR * WIN);
    const n = Math.max(1, Math.floor((x.length - wlen) / step));
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      const off = i * step;
      for (let j = 0; j < wlen; j++) { const v = x[off + j]; acc += v * v; }
      const m = acc / wlen;
      out[i] = 10.0 * Math.log10(Math.max(m, 1e-12));
    }
    return out;
  }

  /* ---------------------------------------------------------------- 平滑 */
  // 对应 _smooth()。numpy 的 mode='same' 取 full 卷积的中间 N 个，
  // 起点是 (k-1)//2 —— k 为偶数时窗口偏左，不是居中。
  function smoothSame(a, k) {
    k = Math.max(1, k | 0);
    const n = a.length, out = new Float64Array(n);
    const start = (k - 1) >> 1;
    const inv = 1.0 / k;
    for (let i = 0; i < n; i++) {
      let acc = 0;
      // out[i] = Σ_j a[i + start - j] * (1/k)，越界按 0（numpy 的零填充）
      for (let j = 0; j < k; j++) {
        const idx = i + start - j;
        if (idx >= 0 && idx < n) acc += a[idx];
      }
      out[i] = acc * inv;
    }
    return out;
  }

  /* ------------------------------------------------------------ 分位数 */
  // 对应 np.percentile(a, q)：线性插值
  function percentile(arr, q) {
    const n = arr.length;
    if (!n) return 0;
    const s = Float64Array.from(arr);
    s.sort();
    const idx = (n - 1) * q / 100.0;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo === hi) return s[lo];
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  }

  /* ---------------------------------------------------------------- 找峰 */
  // 对应 find_peaks()：95 分位以上的局部极大值 + 最小间隔抑制（按值降序贪心）
  function findPeaks(arr, hop, minGap, pct) {
    minGap = minGap === undefined ? 8.0 : minGap;
    pct = pct === undefined ? 95.0 : pct;
    const n = arr.length;
    if (n < 3) return [];
    const thr = percentile(arr, pct);
    const gap = Math.max(1, Math.trunc(minGap / hop));
    const idx = [];
    for (let i = 1; i < n - 1; i++) {
      if (arr[i] >= arr[i - 1] && arr[i] > arr[i + 1] && arr[i] >= thr) idx.push(i);
    }
    idx.sort((a, b) => arr[b] - arr[a]);       // 按值降序
    const kept = [];
    for (const i of idx) {
      let ok = true;
      for (const j of kept) { if (Math.abs(i - j) <= gap) { ok = false; break; } }
      if (ok) kept.push(i);
    }
    return kept;
  }

  /* ---------------------------------------------------------- 窗口均值 */
  // 对应 _winmean()
  function winMean(s, hop, i, a, b, med) {
    const n = s.length;
    let lo = Math.min(Math.max(0, i + Math.trunc(a / hop)), n - 1);
    let hi = Math.min(Math.max(lo + 1, i + Math.trunc(b / hop)), n);
    let acc = 0;
    for (let k = lo; k < hi; k++) acc += s[k];
    return acc / (hi - lo) - med;
  }

  /* ---------------------------------------------------------------- FFT */
  // numpy 的 rfft 支持任意长度，这里用混合基 Cooley-Tukey。
  // 本工具的片段长度是 16000 = 2^7 × 5^3，只用得到基 2 和基 5。
  function fftRadix(re, im) {
    const n = re.length;
    if (n <= 1) return;
    let p = 0;
    for (let f = 2; f * f <= n; f++) { if (n % f === 0) { p = f; break; } }
    if (p === 0) p = n;                        // n 是质数 → O(n²) 兜底
    const m = n / p;

    // 按 q 抽取 p 个子序列，各自做 m 点 DFT，结果**连续**存进 Y
    // ⚠️ 第一版这里写错过：存放用 j*p+q（跨步），取值却用 q*m+(k%m)（连续），
    //    两者错位 → centroid 偏 3490、flatness 偏 0.62。布局必须统一。
    const Yre = new Float64Array(n), Yim = new Float64Array(n);
    const rre = new Float64Array(m), rim = new Float64Array(m);
    for (let q = 0; q < p; q++) {
      for (let j = 0; j < m; j++) { const s = j * p + q; rre[j] = re[s]; rim[j] = im[s]; }
      fftRadix(rre, rim);
      for (let j = 0; j < m; j++) { Yre[q * m + j] = rre[j]; Yim[q * m + j] = rim[j]; }
    }

    // 组合：X[k] = Σ_q e^{-2πi·qk/n} · Y_q[k mod m]
    const kre = new Float64Array(n), kim = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      let sr = 0, si = 0;
      const km = k % m;
      for (let q = 0; q < p; q++) {
        const idx = q * m + km;
        const ang = -2 * Math.PI * q * k / n;
        const c = Math.cos(ang), s = Math.sin(ang);
        const ar = Yre[idx], ai = Yim[idx];
        sr += ar * c - ai * s;
        si += ar * s + ai * c;
      }
      kre[k] = sr; kim[k] = si;
    }
    for (let k = 0; k < n; k++) { re[k] = kre[k]; im[k] = kim[k]; }
  }

  // 对应 np.fft.rfft 的幅度谱
  function rfftAbs(seg) {
    const n = seg.length;
    const re = Float64Array.from(seg), im = new Float64Array(n);
    fftRadix(re, im);
    const half = Math.floor(n / 2) + 1;
    const out = new Float64Array(half);
    for (let k = 0; k < half; k++) out[k] = Math.hypot(re[k], im[k]);
    return out;
  }

  /* ---------------------------------------------------------- 21 个特征 */
  // 对应 model_features()。rank_energy / gap_prev / txt 需要轨内上下文，另算。
  function modelFeatures(x, s, hop, i, med) {
    const t = i * hop;
    const pk = s[i] - med;
    const f = {};
    f.peak = pk;
    f.pre60 = winMean(s, hop, i, -62, -35, med);
    f.pre30 = winMean(s, hop, i, -32, -10, med);
    f.pre10 = winMean(s, hop, i, -10, -2, med);
    f.post2 = winMean(s, hop, i, 1.5, 4, med);
    f.post10 = winMean(s, hop, i, 4, 10, med);
    f.post30 = winMean(s, hop, i, 10, 30, med);
    f.signature = f.pre30 - f.post2;
    f.contrast = f.peak - f.pre10;

    let j = i;
    const thrR = med + f.pre10 * 0.5 + (pk - f.pre10) * 0.5;
    while (j > 0 && s[j] > thrR) j--;
    f.rise = (i - j) * hop;

    let k = i;
    const thrD = med + f.post2 + (pk - f.post2) * 0.5;
    while (k < s.length - 1 && s[k] > thrD) k++;
    f.decay = (k - i) * hop;

    // recover：峰值之后多久回落到全轨中位数以下
    const afterEnd = Math.min(s.length, i + Math.trunc(60 / hop));
    let rec = 60.0;
    for (let m = i; m < afterEnd; m++) { if (s[m] < med) { rec = (m - i) * hop; break; } }
    f.recover = rec;

    // prominence：峰值 相对 ±30 秒内中位数 的突出度
    const a0 = Math.max(0, i - Math.trunc(30 / hop));
    const b0 = Math.min(s.length, i + Math.trunc(30 / hop));
    const win = Array.prototype.slice.call(s.subarray(a0, b0));
    win.sort((p, q) => p - q);
    const mdn = win.length ? (win.length % 2
      ? win[(win.length - 1) >> 1]
      : (win[win.length / 2 - 1] + win[win.length / 2]) / 2) : 0;
    f.prominence = pk - (mdn - med);

    // plateau：±3 秒内有多少比例超过「中位数 + 峰值一半」
    const w0 = Math.max(0, i - Math.trunc(3 / hop));
    const w1 = Math.min(s.length, i + Math.trunc(3 / hop) + 1);
    const thrP = med + pk * 0.5;
    let above = 0;
    for (let m = w0; m < w1; m++) if (s[m] > thrP) above++;
    f.plateau = (w1 - w0) ? above / (w1 - w0) : 0.0;

    f.rel_time = t / Math.max(1e-6, s.length * hop);

    // 频谱：峰值 ±0.5 秒的 1 秒片段
    const step = Math.trunc(hop * SR);
    const segStart = Math.max(0, i - Math.trunc(0.5 / hop)) * step;
    const segEnd = Math.min(x.length, i + Math.trunc(0.5 / hop)) * step;
    const segLen = segEnd - segStart;
    if (segLen > 512) {
      const seg = new Float64Array(segLen);
      let mean = 0;
      for (let m = 0; m < segLen; m++) { seg[m] = x[segStart + m]; mean += seg[m]; }
      mean /= segLen;
      for (let m = 0; m < segLen; m++) seg[m] -= mean;

      let changes = 0;
      let prev = Math.sign(seg[0]);
      for (let m = 1; m < segLen; m++) {
        const sg = Math.sign(seg[m]);
        if (sg !== prev) changes++;
        prev = sg;
      }
      f.zcr = changes / (segLen - 1);

      // numpy.hanning(n) = 0.5 - 0.5*cos(2πm/(n-1))
      const N1 = segLen - 1;
      const win2 = new Float64Array(segLen);
      for (let m = 0; m < segLen; m++) {
        win2[m] = seg[m] * (0.5 - 0.5 * Math.cos(2 * Math.PI * m / N1));
      }
      const sp = rfftAbs(win2);
      let sumSp = 0, sumF = 0, sumLog = 0;
      for (let b = 0; b < sp.length; b++) {
        const v = sp[b] + 1e-12;
        const fr = b * SR / segLen;
        sumSp += v; sumF += v * fr; sumLog += Math.log(v);
      }
      f.centroid = sumF / sumSp;
      f.flatness = Math.exp(sumLog / sp.length) / (sumSp / sp.length);
    } else {
      f.zcr = 0; f.centroid = 0; f.flatness = 0;
    }
    return f;
  }

  /* ------------------------------------------------------------ 树推理 */
  // 关键：x 已转 float32，阈值也是 float32 —— 与 sklearn 行为一致
  //
  // ⚠️ 叶子判定必须用导出的 is_leaf（lf），**不能靠 left === -1**：
  //    sklearn 的 Tree 叶子是 -1，但 HGB 的 TreePredictor 叶子是 **0**
  //    （实测 left==-1 的节点数为 0）。用 -1 判断会让遍历在节点间来回弹，
  //    表现成「卡住」，实际是在烧 CPU。
  //
  // ⚠️ 阈值必须**转回 float32 再比较**。导出用的是 %.9g 这种短十进制，
  //    JS 默认按 float64 解析，于是它比真实的 float32 阈值小一点点。
  //    实测有阈值正好等于某个 float32 特征值（0.1793862134218216），
  //    Python 判「x <= 阈值」成立走左，JS 判不成立走右 —— 整棵树就错了。
  //    在 Engine 构造时把 t 转成 Float32Array 即可（顺带更快）。
  function leafIndex(tree, xf) {
    const F = tree.f, T = tree.t, L = tree.l, R = tree.r, LF = tree.lf;
    let n = 0;
    while (!LF[n]) n = (xf[F[n]] <= T[n]) ? L[n] : R[n];
    return n;
  }

  // 三种森林的取值方式完全不同，必须分开写：
  //   · 随机森林分类：每个叶子存 [p0, p1] 两个值 → 取 p1，再对全部树求平均
  //   · HGB 分类：每个叶子只有 1 个原始分 → 求和后过 sigmoid，**不能取平均**
  //   · HGB 回归：同上但不过 sigmoid
  // 之前用 nCls = is_hgb ? 1 : 2 然后统一写 v[leaf*nCls + 1]，
  // 对 HGB 会取成 v[leaf+1]（错位一格）；而且 REG_HGB 的 is_hgb 没导出，
  // 会被当成随机森林去做平均。
  function forestScore(d, xf) {
    const trees = d.trees, nt = trees.length;

    if (d.kind === 'reg_forest') {
      let acc = 0;
      for (let i = 0; i < nt; i++) acc += trees[i].v[leafIndex(trees[i], xf)];
      return d.is_hgb ? (d.init + acc) : acc / nt;
    }
    if (d.is_hgb) {                       // 二分类 HGB：叶子是原始分，1 个/节点
      let z = d.init;
      for (let i = 0; i < nt; i++) z += trees[i].v[leafIndex(trees[i], xf)];
      return 1.0 / (1.0 + Math.exp(-z));
    }
    let acc = 0;                          // 随机森林：叶子是 [p0, p1]
    for (let i = 0; i < nt; i++) acc += trees[i].v[leafIndex(trees[i], xf) * 2 + 1];
    return acc / nt;
  }

  function linearScore(d, xf) {
    let z = 0;
    const c = d.coef, mu = d.mean, sd = d.scale;
    for (let i = 0; i < c.length; i++) z += c[i] * ((xf[i] - mu[i]) / sd[i]);
    z += d.intercept;
    return d.kind === 'logreg' ? 1.0 / (1.0 + Math.exp(-z)) : z;
  }

  // x32 / x64：同一个特征向量的两种精度。用哪个由模型自己决定 ——
  // sklearn 只有随机森林把输入转 float32，其余原样吃 float64。
  function modelScore(d, x32, x64) {
    const xf = d.in_f32 ? x32 : x64;
    if (d.kind === 'clf_forest' || d.kind === 'reg_forest') return forestScore(d, xf);
    return linearScore(d, xf);
  }

  /* ------------------------------------------------------------ 引擎 */
  function Engine(modelJson) {
    this.M = modelJson.models;
    this.feats = modelJson.feats;
    this.weights = modelJson.weights;
    this.iso = modelJson.iso || null;
    this.meta = modelJson.meta || {};
    this.clf = Object.keys(this.M).filter(k => k.indexOf('REG_') !== 0);
    this.reg = Object.keys(this.M).filter(k => k.indexOf('REG_') === 0);
    // 求和归一化权重（对应 predict_raw 里的 w / w.sum()）
    let sw = 0; for (const w of this.weights) sw += w;
    this.wn = this.weights.map(w => w / sw);
    // 只有随机森林的阈值降成 float32（见 leafIndex 上方注释）。
    // HGB 的阈值必须保持 float64 —— 它走分箱，是原样吃 float64 的。
    for (const name in this.M) {
      const d = this.M[name];
      if (!d.trees || !d.in_f32) continue;
      for (const t of d.trees) t.t = Float32Array.from(t.t);
    }
  }

  // feats: 21 个特征的普通对象 → 返回 {raw, prob, refine}
  Engine.prototype.predict = function (featList) {
    const n = featList.length, F = this.feats;
    const raw = new Float64Array(n), ref = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const o = featList[i];
      const x64 = new Float64Array(F.length);
      const x32 = new Float32Array(F.length);
      for (let j = 0; j < F.length; j++) {
        const v = o[F[j]];
        const x = (v === undefined || v === null || Number.isNaN(v)) ? 0 : v;
        x64[j] = x;
        x32[j] = Math.fround(x);
      }
      let acc = 0;
      for (let c = 0; c < this.clf.length; c++) {
        acc += this.wn[c] * modelScore(this.M[this.clf[c]], x32, x64);
      }
      raw[i] = acc;
      let r = 0;
      for (const k of this.reg) r += modelScore(this.M[k], x32, x64);
      ref[i] = this.reg.length ? r / this.reg.length : 0;
    }
    const prob = new Float64Array(n);
    for (let i = 0; i < n; i++) prob[i] = this.calib(raw[i]);
    return { raw: raw, prob: prob, refine: ref };
  };

  // isotonic 校准：searchsorted(x, v, 'right') - 1，两端夹住
  Engine.prototype.calib = function (v) {
    if (!this.iso) return v;
    const xs = this.iso.x, ys = this.iso.y;
    let lo = 0, hi = xs.length;
    while (lo < hi) { const md = (lo + hi) >> 1; if (xs[md] <= v) lo = md + 1; else hi = md; }
    let i = lo - 1;
    if (i < 0) i = 0; else if (i >= ys.length) i = ys.length - 1;
    return Math.min(Math.max(ys[i], 0), PROB_CAP);
  };

  /* ------------------------------------------------------------ 合并 */
  // 对应 merge_candidates()：代表点取成员时间中点，其余字段沿用最高分成员
  function mergeCandidates(cands, top, gap, pool) {
    if (!cands.length) return [];
    top = top || 3; gap = gap === undefined ? MERGE_GAP : gap;
    if (pool === undefined || pool === null) pool = top + POOL_EXTRA;
    const sorted = cands.slice().sort((a, b) => b.score - a.score).slice(0, Math.max(1, pool));
    const events = [];
    for (const c of sorted) {
      let hit = null;
      for (const e of events) { if (Math.abs(c.time - e.time) <= gap) { hit = e; break; } }
      if (hit === null) {
        const e = Object.assign({}, c);
        e._times = [c.time]; e.merged = 1;
        events.push(e);
      } else {
        hit._times.push(c.time);
        let s = 0; for (const t of hit._times) s += t;
        hit.time = s / hit._times.length;
        hit.merged += 1;
        if (c.score > hit.score) {
          const keep = Object.assign({}, c);
          keep._times = hit._times; keep.merged = hit.merged;
          events[events.indexOf(hit)] = keep; hit = keep;
        }
        hit.score = Math.max(hit.score, c.score);
        if (c.probability !== undefined && c.probability !== null) {
          hit.probability = Math.max(hit.probability || 0, c.probability);
        }
      }
    }
    for (const e of events) {
      e.mmss = mmss(e.time);
      delete e._times;
    }
    events.sort((a, b) => b.score - a.score);
    return events.slice(0, top);
  }

  // 对应 snap_to_peak_by_score()：在 ±window 内取精修分最大的峰
  function snapByScore(events, pool, scores, window) {
    window = window === undefined ? SNAP_WINDOW : window;
    if (!(window > 0)) return events;
    for (const e of events) {
      let bi = -1, bv = -Infinity;
      for (let i = 0; i < pool.length; i++) {
        if (Math.abs(pool[i].time - e.time) <= window && scores[i] > bv) { bv = scores[i]; bi = i; }
      }
      if (bi < 0) continue;
      const bt = pool[bi].time;
      if (Math.abs(bt - e.time) > 1e-6) { e.snapped_from = e.time; e.time = bt; e.mmss = mmss(bt); }
    }
    return events;
  }

  function mmss(t) {
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
  }

  function confidenceOfProb(p) {
    if (p >= 0.6) return ['高', '此区间样本外实测命中率 55~94%'];
    if (p >= 0.3) return ['中', '此区间样本外实测命中率 15~33%'];
    return ['低', '此区间样本外实测命中率 0.7~4%'];
  }

  /* ------------------------------------------------------ 完整分析流程 */
  // x: Float32Array/Float64Array（16 kHz 单声道）
  // engine: Engine
  // opts: {top, minProb, transcript: null}
  function analyze(x, engine, opts) {
    opts = opts || {};
    const top = opts.top || 3;
    const minProb = opts.minProb === undefined ? MIN_PROB : opts.minProb;
    const t0 = performance.now();

    const duration = x.length / SR;
    if (duration < MIN_DURATION) {
      return { error: '时长 ' + duration.toFixed(0) + ' 秒，短于 ' + MIN_DURATION +
        ' 秒阈值——短轨的能量特征不稳定，跳过', duration: duration };
    }

    const tEnv0 = performance.now();
    const db = energyEnvelope(x);
    const hop = HOP;
    const s = smoothSame(db, Math.trunc(0.3 / hop));
    let med = 0;
    const tmp = Array.prototype.slice.call(s);
    tmp.sort((a, b) => a - b);
    med = tmp.length % 2 ? tmp[(tmp.length - 1) >> 1]
      : (tmp[tmp.length / 2 - 1] + tmp[tmp.length / 2]) / 2;
    const peaks = findPeaks(s, hop);
    const tEnv = performance.now();

    // 21 个特征（txt 恒为 0 —— 离线版没有 ASR）
    const feats = [];
    for (const i of peaks) {
      const f = modelFeatures(x, s, hop, i, med);
      f.txt = 0;
      feats.push(f);
    }
    // 轨内上下文：能量排名 + 到上一个候选的间距
    const order = peaks.map((p, n) => n).sort((a, b) => s[peaks[b]] - s[peaks[a]]);
    const rankOf = new Array(peaks.length);
    order.forEach((n, r) => { rankOf[n] = r; });
    for (let n = 0; n < feats.length; n++) {
      feats[n].rank_energy = rankOf[n];
      feats[n].gap_prev = n === 0 ? 0 : (peaks[n] - peaks[n - 1]) * hop;
    }

    const tFeat0 = performance.now();
    const pred = engine.predict(feats);
    const tPred = performance.now();

    const cands = peaks.map((i, n) => {
      const t = i * hop;
      const conf = confidenceOfProb(pred.prob[n]);
      const c = {
        time: Math.trunc(t * 100) / 100, mmss: mmss(t),
        score: pred.raw[n], probability: pred.prob[n],
        confidence: conf[0], confidence_note: conf[1],
        peak: feats[n].peak, _ref: pred.refine[n],
      };
      return c;
    });
    // 门槛过滤（在合并之前，与 Python 一致）
    let kept = cands;
    if (minProb > 0) kept = cands.filter(c => c.probability >= minProb);

    let picked = mergeCandidates(kept.map(c => Object.assign({}, c)), top);
    if (kept.length && kept.every(c => c._ref !== undefined)) {
      picked = snapByScore(picked, kept, kept.map(c => c._ref), SNAP_WINDOW);
    }
    for (const c of picked) delete c._ref;
    picked.sort((a, b) => b.score - a.score);

    return {
      duration: duration, candidates: picked, nPeaks: peaks.length,
      transcript: false,
      timing: { total: performance.now() - t0,
                envelope: tEnv - tEnv0, features: tFeat0 - tEnv,
                model: tPred - tFeat0 },
      maxProb: picked.length ? Math.max.apply(null, picked.map(c => c.probability)) : 0,
    };
  }

  root.ClimaxCore = {
    SR, HOP, WIN, MIN_DURATION, MERGE_GAP, SNAP_WINDOW, MIN_PROB, MODEL_FEATS,
    energyEnvelope, smoothSame, percentile, findPeaks, winMean,
    modelFeatures, mergeCandidates, snapByScore, mmss, confidenceOfProb,
    analyze, Engine, leafIndex, modelScore,
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
