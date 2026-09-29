# -*- coding: utf-8 -*-
"""训练最终模型并保存 climax_model.pkl

关键纪律：
  · 特征实现从 climax_finder 导入（**唯一实现**），杜绝训练/推理不一致
  · 10 折留一作品交叉验证给出「预期成绩」，写进模型元信息
  · isotonic 校准器在**样本外预测**上拟合（不能用自己的训练预测拟合自己）
  · 最终模型用**全部数据**训练；它的成绩引用 10 折均值，不自己测自己
"""
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
sys.path.insert(0, str(ROOT))
import numpy as np

from climax_finder import (MODEL_FEATS, SR, HOP, MIN_DURATION, energy_envelope,
                           find_peaks, _smooth, model_features, load_transcript,
                           text_score_at, DEFAULT_CUES, merge_candidates,
                           snap_to_peak, snap_to_peak_by_score,
                           MERGE_GAP, POOL_EXTRA, SNAP_WINDOW, MIN_PROB)

TOL = 20.0
TOPN = 3
SEED = 0
CACHE = ROOT / "_ml" / "features_v2.json"
OUT = ROOT / "climax_model.pkl"
ASR_EXTRA = {"RJ324692": ROOT / "_climax" / "asr", "RJ362169": ROOT / "_climax2" / "asr"}
ASR_ML = ROOT / "_ml" / "asr"

MARKS = ROOT / "climax_marks.jsonl"          # 手动标注的精确高潮时刻
DONE = ROOT / "climax_tracks_done.jsonl"     # 「本轨已标完」确认


def _jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_user_gt():
    """用户手动标注 → ({rj: {track: [秒]}}, {rj: {track: 音频路径}})，只含**已标完**的音轨。

    为什么必须「标完」才能用：一轨若有 3 次高潮只标了 2 次，
    第 3 次附近的候选就会被当成负例 —— **比不标还糟**。
    """
    import re
    done = {r["audio"] for r in _jsonl(DONE)
            if r.get("done", True) and r.get("audio")}
    live = {}
    for r in _jsonl(MARKS):
        a = r.get("audio", "")
        if not a or a not in done:
            continue
        d = live.setdefault(a, {"add": set(), "del": set()})
        d["add" if r.get("action", "add") == "add" else "del"].add(
            round(float(r.get("time", 0)), 2))
    out, files = {}, {}
    for a, d in live.items():
        m = re.search(r"(RJ\d+)", a)
        tm = re.search(r"track[_\-\s]*(\d+)", Path(a).name, re.I)
        if not m or not tm:
            continue
        rj, tk = m.group(1), tm.group(1).zfill(2)
        out.setdefault(rj, {})[tk] = sorted(d["add"] - d["del"])
        files.setdefault(rj, {})[tk] = a
    return out, files


def build_dataset():
    """用 climax_finder 的规范特征函数提取全部候选"""
    from faster_whisper import decode_audio
    meta = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))
    GT, FILES = meta["gt"], meta["files"]
    # 用户手动标注的（已标完）音轨也要提特征 —— 它们可能不在原来 10 部里
    u_gt, u_files = load_user_gt()
    for rj, trks in u_gt.items():
        for tk, fp in u_files[rj].items():
            GT.setdefault(rj, {})[tk] = trks[tk]
            FILES.setdefault(rj, {})[tk] = fp
    rows, tracks = [], []
    for rj in sorted(GT):
        asrdir = ASR_EXTRA.get(rj, ASR_ML / rj)
        for tk, gts in GT[rj].items():
            fp = FILES[rj].get(tk)
            if not fp or not Path(fp).exists():
                continue
            mp3 = Path(fp)
            segs = None
            for d in (asrdir, mp3.parent, ASR_ML / rj):
                if not d.exists():
                    continue
                c = list(d.glob(f"{mp3.stem}.segments.json"))
                if c:
                    segs = json.loads(c[0].read_text(encoding="utf-8"))["segments"]
                    break
            try:
                x = decode_audio(str(mp3), sampling_rate=SR)
            except Exception as e:
                print(f"  ✗ {rj}/{tk}: {e}")
                continue
            if len(x) / SR < MIN_DURATION:
                continue
            db, hop = energy_envelope(x)
            s = _smooth(db, int(0.3 / hop))
            med = float(np.median(s))
            peaks = find_peaks(s, hop)
            if not peaks:
                continue
            feats = []
            for i in peaks:
                f = model_features(x, s, hop, i, med)
                t = i * hop
                tx, _, _ = (text_score_at(segs, t, DEFAULT_CUES) if segs else (0, [], ""))
                f["txt"] = float(tx)
                f["t"] = t
                f["TP"] = any(abs(t - g) <= TOL for g in gts)
                f["has_gt"] = bool(gts)
                f["rj"] = rj
                f["track"] = tk
                feats.append(f)
            order = sorted(range(len(peaks)), key=lambda n: -s[peaks[n]])
            rank_of = {n: r for r, n in enumerate(order)}
            for n, f in enumerate(feats):
                f["rank_energy"] = float(rank_of[n])
                f["gap_prev"] = 0.0 if n == 0 else float((peaks[n] - peaks[n - 1]) * hop)
            rows += feats
            tracks.append((rj, tk, len(peaks), len(gts)))
    return rows, tracks


# 距离加权的带宽。定义在命令行解析**之前** ——
# 之前写在解析之后，把 --dist-weight 的值又覆盖回 0 了（三个 σ 结果完全相同才发现）
DIST_SIGMA = 0.0

import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--dist-weight", type=float, default=0.0, metavar="SIGMA",
                help="按到真值的距离给正例加权 exp(-(d/σ)²)；0=关闭（默认）")
_p.add_argument("--use-feedback", action="store_true",
                help="额外使用旧的 ✓/✗ 标注。默认**不用**：实测加上去精确率反而从 "
                     "48.9%% 掉到 47.4%%（虽然只差 3 个候选，属于噪声，但没有收益）")
_a = _p.parse_args()
if _a.dist_weight > 0:
    DIST_SIGMA = _a.dist_weight

if CACHE.exists():
    print(f"复用已提取的特征：{CACHE}")
    blob = json.loads(CACHE.read_text(encoding="utf-8"))
    rows, tracks = blob["rows"], blob["tracks"]
else:
    print("提取特征（约 10 分钟）…")
    t0 = time.time()
    rows, tracks = build_dataset()
    CACHE.write_text(json.dumps({"rows": rows, "tracks": tracks}, ensure_ascii=False),
                     encoding="utf-8")
    print(f"  完成，用时 {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# 合并人工标注（播放器里标的对错）
#
# 关键：标注必须带上它所属的**作品编号**，否则交叉验证时会把测试折的标注
#       混进训练集 —— 那是泄漏。所以从音频路径里提取 RJ 编号。
# ---------------------------------------------------------------------------
# 旧 ✓/✗ 标注已弃用：标注口径不统一（用户当时不确定允许多大误差），
# 实测加进去精确率反而从 48.9% 掉到 47.4%。文件改名保留，没有删除。
# 加 --use-feedback 可以重新启用。
FEEDBACK = ROOT / "climax_feedback.DEPRECATED.jsonl"
if not _a.use_feedback:
    if FEEDBACK.exists():
        print("  旧 ✓/✗ 标注：默认不使用（加 --use-feedback 可启用）")
elif FEEDBACK.exists():
    import re as _re
    latest = {}
    for line in FEEDBACK.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[(r.get("audio"), round(r.get("time", 0), 1))] = r
    added, skipped = 0, 0
    have = {(r["rj"], r["track"], round(r["t"], 1)) for r in rows}
    for (audio, _), r in latest.items():
        if r.get("verdict") not in (0, 1) or not r.get("feats"):
            continue
        m = _re.search(r"(RJ\d+)", str(audio))
        if not m:
            skipped += 1
            continue
        rj = m.group(1)
        row = {k: float(r["feats"].get(k, 0.0)) for k in MODEL_FEATS}
        row["TP"] = bool(r["verdict"])
        row["has_gt"] = True
        row["rj"] = rj
        row["track"] = "user"
        row["src"] = "feedback"     # 来源标记：与官方标注区分开
        row["t"] = float(r["time"])
        row["txt"] = float(r["feats"].get("txt", 0.0))
        rows.append(row)
        added += 1
    print(f"  合并人工标注：{added} 条" + (f"（{skipped} 条路径里没有 RJ 编号，跳过）" if skipped else ""))

for r in rows:
    for k in MODEL_FEATS:
        v = r.get(k, 0.0)
        r[k] = float(v) if np.isfinite(v) else 0.0

# ---------------------------------------------------------------------------
# 官方标注 + 用户手动标注（已标完的），用户标注**覆盖**官方
#
# 为什么要区分来源：官方标注是出版方给的精确时刻（实测误差约 1 秒）；
# 用户手动标注是听出来的，精度取决于耳力，但**覆盖了模型漏掉的高潮**——
# 这是 ✓/✗ 式标注永远做不到的。
# ---------------------------------------------------------------------------
USER_GT, USER_FILES = load_user_gt()
_official = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
GT_FULL = {rj: {tk: list(v) for tk, v in trks.items()} for rj, trks in _official.items()}
overridden = []
for rj, trks in USER_GT.items():
    for tk, times in trks.items():
        if rj in GT_FULL and tk in GT_FULL[rj]:
            overridden.append(f"{rj}/{tk}")
        GT_FULL.setdefault(rj, {})[tk] = list(times)


def row_dist(r):
    """这一行到最近真值的距离（秒）；没有真值返回 None。"""
    g = GT_FULL.get(r["rj"], {}).get(r["track"])
    if not g:
        return None
    return min(abs(r["t"] - x) for x in g)


# 用户标注的行需要重算 TP（features 缓存里存的是按官方标注算的）
recomputed = 0
for r in rows:
    r.setdefault("src", "official")
    g = GT_FULL.get(r["rj"], {}).get(r["track"])
    if g is None:
        r["_d"] = None
        continue
    new_tp = any(abs(r["t"] - x) <= TOL for x in g)
    if new_tp != r["TP"]:
        recomputed += 1
    r["TP"] = new_tp
    r["_d"] = row_dist(r)

works = sorted({r["rj"] for r in rows})

if USER_GT:
    n_ut = sum(len(v) for v in USER_GT.values())
    n_up = sum(len(x) for v in USER_GT.values() for x in v.values())
    print(f"  用户手动标注：{len(USER_GT)} 部 / {n_ut} 轨（已标完）/ {n_up} 个点")
    if overridden:
        print(f"    覆盖了官方标注：{', '.join(overridden[:6])}"
              + (" …" if len(overridden) > 6 else ""))
    if recomputed:
        print(f"    重算了 {recomputed} 行的正负标签")
else:
    print("  用户手动标注：无（播放器里标完并勾选「本轨已标完」后才有）")

bywork = {w: [r for r in rows if r["rj"] == w] for w in works}
print(f"  数据：{len(works)} 部作品 / {len(tracks)} 轨 / {len(rows)} 候选 / "
      f"{sum(1 for r in rows if r['TP'])} 正例")

from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression

# 超参数：网格搜索 + 嵌套验证的一致选择（每折都选 depth=10）
RF_KW = dict(n_estimators=400, min_samples_leaf=5, max_depth=10,
             class_weight="balanced", random_state=SEED, n_jobs=-1)
HGB_KW = dict(max_iter=300, learning_rate=0.06, max_depth=10, min_samples_leaf=5,
              l2_regularization=1.0, class_weight="balanced", random_state=SEED)
LR_KW = dict(class_weight="balanced", max_iter=3000, C=0.3)
WEIGHTS = [1.0, 1.0, 1.0]          # 三等权（实测各加权方案都不更优）

# 时间精修用的回归模型（软标签）
REG_RF_KW = dict(n_estimators=400, min_samples_leaf=5, max_depth=10,
                 random_state=SEED, n_jobs=-1)
REG_HGB_KW = dict(max_iter=300, learning_rate=0.06, max_depth=10,
                  min_samples_leaf=5, l2_regularization=1.0, random_state=SEED)

# ---------------------------------------------------------------------------
# 时间精修模型（软标签回归）
#
# 二分类模型学的是「这附近有没有高潮」，对**时刻准不准**没有偏好。
# 所以另训一组回归模型，目标是「离真值有多近」：
#
#     y = exp(-(d / σ)²)      d = 到最近真值的距离
#
# **但只让它做局部精修，不参与全局排序。** 实测（10 折留一作品）：
#     纯用回归分排序            精确 31.2%  召回 65.9%  ≤5秒 68%   ← 排序垮了
#     分类选事件 + 回归精修       精确 49.1%  召回 72.5%  ≤5秒 55%   ← 采用
#     分类选事件 + 能量精修       精确 48.7%  召回 70.5%  ≤5秒 52%   ← 之前的做法
#     不精修                   精确 49.1%  召回 72.5%  ≤5秒 34%
#
# 也就是：**回归精修在每一项上都优于能量精修**，而且比不精修准得多。
# 窗口越大时间越准、精确率越低：
#     ±10秒 → ≤5秒 55%   精确 49.1%   召回 72.5%
#     ±15秒 → ≤5秒 62%   精确 48.8%   召回 71.4%
#     ±20秒 → ≤5秒 66%   精确 47.4%   召回 69.8%
# ---------------------------------------------------------------------------
REG_SIGMA = 2.0


def soft_target(rs, sigma=REG_SIGMA):
    """软标签：离真值越近越接近 1"""
    return np.array(
        [float(np.exp(-(r["_d"] / sigma) ** 2)) if r["_d"] is not None else 0.0
         for r in rs], float)


def fit_reg(rs):
    Xr = X(rs)
    yr = soft_target(rs)
    rf = RandomForestRegressor(**REG_RF_KW)
    hgb = HistGradientBoostingRegressor(**REG_HGB_KW)
    rid = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    rf.fit(Xr, yr)
    hgb.fit(Xr, yr)
    rid.fit(Xr, yr)
    return {"REG_RF": rf, "REG_HGB": hgb, "REG_RIDGE": rid}


def reg_score(models, rs):
    keys = [k for k in models if k.startswith("REG_")]
    if not keys:
        return np.zeros(len(rs))
    Xt = X(rs)
    return np.mean([models[k].predict(Xt) for k in keys], axis=0)


def X(rs): return np.array([[r[f] for f in MODEL_FEATS] for r in rs], float)
def Y(rs): return np.array([r["TP"] for r in rs], int)


# ---------------------------------------------------------------------------
# 时间误差得分（命令行 --dist-weight SIGMA 开启）
#
# 现状：真值 ±20 秒内的**所有**峰都算正例，模型没有动力区分哪个最准。
# 开启后正例按距离加权：w = exp(-(d/σ)²)，越准的峰权重越高。
#
# 实测（σ=5，且只有官方标注时）：≤5 秒 35% → 43%，但精确率掉 3 个点；
# 更激进的硬标注（只有最近峰算正例）把 AUC 从 0.916 打到 0.858。
# **所以默认关闭**（σ=0）。等手动精确标注足够多之后再重新评估 ——
# 那时「距离」是听出来的，比官方标注更贴近真实感知。
# ---------------------------------------------------------------------------
def make_weights(rs):
    if DIST_SIGMA <= 0:
        return None
    w = np.ones(len(rs))
    for i, r in enumerate(rs):
        if not r["TP"]:
            continue
        d = r.get("_d")
        if d is not None:
            w[i] = float(np.exp(-(d / DIST_SIGMA) ** 2))
    return w


def fit(rs):
    Xr, yr = X(rs), Y(rs)
    sw = make_weights(rs)
    rf = RandomForestClassifier(**RF_KW)
    hgb = HistGradientBoostingClassifier(**HGB_KW)
    lr = make_pipeline(StandardScaler(), LogisticRegression(**LR_KW))
    rf.fit(Xr, yr, sample_weight=sw)
    hgb.fit(Xr, yr, sample_weight=sw)
    # Pipeline 不收裸的 sample_weight，要用 stepname__参数 的形式
    lr.fit(Xr, yr, logisticregression__sample_weight=sw)
    return {"RF": rf, "HGB": hgb, "LR": lr}


def ensemble(models, rs):
    Xt = X(rs)
    keys = [k for k in models if not k.startswith("REG_")]
    ps = [models[k].predict_proba(Xt)[:, 1] for k in keys]
    return np.average(ps, axis=0, weights=WEIGHTS)


def auc(p, rs):
    p = np.asarray(p, float); lb = np.array([r["TP"] for r in rs], bool)
    if lb.sum() == 0 or (~lb).sum() == 0:
        return float("nan")
    return ((p[lb][:, None] > p[~lb][None, :]).sum()
            + 0.5 * (p[lb][:, None] == p[~lb][None, :]).sum()) / (lb.sum() * (~lb).sum())


def prec_rec(p, rs, topn=TOPN, reg=None, thr=0.0, iso=None):
    """按**工具实际的候选选取方式**评估。

    ⚠️ 这里以前是「直接取分数最高的 topn 个」，没有实现工具里的间隔抑制。
    实测过这个差别很大：不抑制时精确率 60.9%，工具实际只有 41.5% —— 报出去
    的数字比工具真实表现偏乐观了近 20 个点。现在直接调用
    climax_finder.merge_candidates，保证评估口径和工具完全一致。
    """
    tc = defaultdict(list)
    for i, (r, s) in enumerate(zip(rs, p)):
        tc[(r["rj"], r["track"])].append(
            {"time": float(r["t"]), "score": float(s), "mmss": "",
             "peak": float(r.get("peak", 0.0)),
             "_ref": float(reg[i]) if reg is not None else 0.0})
    tp = nc = 0
    cov = set(); gtot = 0
    gt = GT_FULL
    for (rj, tk), cand in tc.items():
        # 人工标注可能来自没有官方标注的作品 —— 那种没有评估基准，跳过。
        # （它们仍然参与训练，只是不能用来算精确率/召回率。）
        g = gt.get(rj, {}).get(tk)
        if g is None:
            continue
        gtot += len(g)
        base = merge_candidates(cand, top=topn)
        # ⚠️ 精修分要跟着**这一轨的候选**走，不能用整折的 reg 数组 ——
        # cand 只是 reg 的一个子集，直接传 reg 会索引错位（实测差 3 个点）
        refs = np.array([c["_ref"] for c in cand]) if cand else np.zeros(0)
        if refs.size:
            events = snap_to_peak_by_score(base, cand, refs, window=SNAP_WINDOW)
        else:
            events = snap_to_peak(base, cand)
        for c in events:
            if thr > 0 and iso is not None:
                if float(iso.predict([c["score"]])[0]) < thr:
                    continue
            nc += 1
            h = [x for x in g if abs(c["time"] - x) <= TOL]
            if h:
                tp += 1
                for x in h:
                    cov.add((rj, tk, x))
    # 一个候选都没输出时精确率是「未定义」，返回 NaN 让调用方跳过 ——
    # 算成 0%% 会冤枉地把平均值拉下来
    prec = (tp / nc) if nc else float("nan")
    return prec, len(cov) / max(1, gtot)


# 交叉验证只在**有官方标注**的作品上做。
# 人工标注的作品没有完整标注（用户只标了模型给出的候选），算不了召回率，
# 拿它当测试折没有意义 —— 所以它们始终留在训练集里。
_gt = GT_FULL
works_gt = [w for w in works if w in _gt and any(r["track"] in _gt[w] for r in bywork[w])]
extra = [w for w in works if w not in works_gt]
if extra:
    print(f"  ⚠ {len(extra)} 部作品只有人工标注、没有官方标注："
          f"{', '.join(extra)}")
    print(f"    它们参与训练，但不作为测试折（没有召回率基准）")

print("\n10 折留一作品交叉验证（给出预期成绩 + 收集样本外预测用于校准）…")
oof_raw, oof_lab = [], []
oof_parts = []                       # 存每折的 (raw, reg, rows)，供门槛评估复用
A, P, R = [], [], []
for hold in works_gt:
    tr = [r for r in rows if r["rj"] != hold]
    te = bywork[hold]
    m = fit(tr)
    m.update(fit_reg(tr))            # 时间精修模型也要在训练折上训
    raw = ensemble(m, te)
    rg = reg_score(m, te)
    a, p, rc = auc(raw, te), *prec_rec(raw, te, reg=rg)
    A.append(a); R.append(rc)
    if not np.isnan(p):
        P.append(p)
    oof_raw.append(raw)
    oof_lab.append(Y(te))
    oof_parts.append((hold, raw, rg, te))
    print(f"  {hold:<12} AUC {a:.3f}  精确 {p*100:5.1f}%  召回 {rc*100:5.1f}%")
oof_raw = np.concatenate(oof_raw)
oof_lab = np.concatenate(oof_lab).astype(float)
print(f"  {'平均':<12} AUC {np.mean(A):.3f}  精确 {np.mean(P)*100:5.1f}%  "
      f"召回 {np.mean(R)*100:5.1f}%   标准差 {np.std(P)*100:.1f}%")

print("\n在样本外预测上拟合 isotonic 校准器…")
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(oof_raw, oof_lab)
br_raw = float(np.mean((oof_raw - oof_lab) ** 2))
br_cal = float(np.mean((iso.predict(oof_raw) - oof_lab) ** 2))
print(f"  Brier 分数：未校准 {br_raw:.4f} → 校准后 {br_cal:.4f}（改善 "
      f"{(1-br_cal/br_raw)*100:.0f}%）")

# ---------------------------------------------------------------------------
# 门槛下的指标 —— **这才是用户实际看到的数字**
#
# 上面那份是「不过滤」的成绩（所有合并后的事件都算）。但 WebUI 和 CLI 默认
# 有 MIN_PROB 门槛，低分候选根本不会显示给用户。两个都报，避免误读。
# ---------------------------------------------------------------------------
print(f"\n输出门槛 ≥{MIN_PROB*100:.0f}% 时的指标（界面/命令行默认）…")
Pt, Rt = [], []
for hold, raw, rg, te in oof_parts:
    p, rc = prec_rec(raw, te, reg=rg, thr=MIN_PROB, iso=iso)
    Rt.append(rc)
    if not np.isnan(p):
        Pt.append(p)
print(f"  {'平均':<12} 精确 {np.mean(Pt)*100:5.1f}%  召回 {np.mean(Rt)*100:5.1f}%")
print(f"  （不过滤时是 精确 {np.mean(P)*100:.1f}% / 召回 {np.mean(R)*100:.1f}% ——"
      f" 门槛把精确率拉高、召回率压低，这是取舍不是退步）")

print("\n用全部数据训练最终模型…")
final = fit(rows)
final.update(fit_reg(rows))          # 时间精修模型
payload = {
    "models": final,
    "weights": WEIGHTS,
    "iso": iso,
    "feats": MODEL_FEATS,
    "meta": {
        "n_works": len(works_gt), "n_tracks": len(tracks), "n_rows": len(rows),
        "n_pos": sum(1 for r in rows if r["TP"]),
        "works": works_gt, "works_extra": extra,
        "works_official": sorted(w for w in works_gt if w in _official),
        "works_user_gt": sorted(USER_GT),
        "user_gt_points": sum(len(x) for v in USER_GT.values() for x in v.values()),
        "dist_sigma": DIST_SIGMA,
        "label_sources": {
            "official": sum(1 for r in rows if r.get("src") == "official"),
            "feedback": sum(1 for r in rows if r.get("src") == "feedback"),
        },
        "use_feedback": bool(_a.use_feedback),
        "cv_auc": float(np.nanmean(A)), "cv_auc_std": float(np.nanstd(A)),
        "cv_prec": float(np.mean(P)), "cv_prec_std": float(np.std(P)),
        "cv_rec": float(np.mean(R)),
        # 门槛下的指标（界面/命令行默认用这两项）
        "cv_prec_thr": float(np.mean(Pt)), "cv_rec_thr": float(np.mean(Rt)),
        "min_prob": MIN_PROB,
        "brier_raw": br_raw, "brier_cal": br_cal,
        "train_minutes": round(sum(1 for _ in tracks) * 0),
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "sklearn_models": {"RF": RF_KW, "HGB": HGB_KW, "LR": LR_KW},
        "ensemble": "equal-weight average of RF + HGB + LR probabilities",
        "reg_sigma": REG_SIGMA,
        "refine": "事件用分类分选，时刻用软标签回归分精修（±12 秒窗口内取最大）",
        "note": "排序用 predict_raw（原始平均分）；predict_prob 仅供显示与阈值",
        "merge_gap": MERGE_GAP, "pool_extra": POOL_EXTRA, "snap": SNAP_WINDOW,
        "select": "取分最高的 top+1 个 → 30 秒内合并取中点 → 吸附到 ±12 秒内最大峰",
    },
}
with open(OUT, "wb") as fh:
    pickle.dump(payload, fh, protocol=4)
mb = OUT.stat().st_size / 1024 / 1024
print(f"\n已保存 {OUT}（{mb:.1f} MB）")
print(f"  预期成绩（10 折留一作品）")
print(f"    门槛 ≥{MIN_PROB*100:.0f}%（实际使用）：精确率 {np.mean(Pt)*100:.1f}%　"
      f"召回率 {np.mean(Rt)*100:.1f}%（基于 {len(Pt)}/{len(works_gt)} 折）")
print(f"    不过滤（全输出）：          精确率 {np.mean(P)*100:.1f}%　"
      f"召回率 {np.mean(R)*100:.1f}%（基于 {len(P)}/{len(works_gt)} 折）")
print(f"    AUC {np.nanmean(A):.3f}")
