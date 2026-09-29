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
                           snap_to_peak, MERGE_GAP, POOL_EXTRA, SNAP_WINDOW)

TOL = 20.0
TOPN = 3
SEED = 0
CACHE = ROOT / "_ml" / "features_v2.json"
OUT = ROOT / "climax_model.pkl"
ASR_EXTRA = {"RJ324692": ROOT / "_climax" / "asr", "RJ362169": ROOT / "_climax2" / "asr"}
ASR_ML = ROOT / "_ml" / "asr"


def build_dataset():
    """用 climax_finder 的规范特征函数提取全部候选"""
    from faster_whisper import decode_audio
    meta = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))
    GT, FILES = meta["gt"], meta["files"]
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
FEEDBACK = ROOT / "climax_feedback.jsonl"
if FEEDBACK.exists():
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
        row["t"] = float(r["time"])
        row["txt"] = float(r["feats"].get("txt", 0.0))
        rows.append(row)
        added += 1
    print(f"  合并人工标注：{added} 条" + (f"（{skipped} 条路径里没有 RJ 编号，跳过）" if skipped else ""))

for r in rows:
    for k in MODEL_FEATS:
        v = r.get(k, 0.0)
        r[k] = float(v) if np.isfinite(v) else 0.0
works = sorted({r["rj"] for r in rows})
bywork = {w: [r for r in rows if r["rj"] == w] for w in works}
print(f"  数据：{len(works)} 部作品 / {len(tracks)} 轨 / {len(rows)} 候选 / "
      f"{sum(1 for r in rows if r['TP'])} 正例")

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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


def X(rs): return np.array([[r[f] for f in MODEL_FEATS] for r in rs], float)
def Y(rs): return np.array([r["TP"] for r in rs], int)


def fit(rs):
    Xr, yr = X(rs), Y(rs)
    return {
        "RF": RandomForestClassifier(**RF_KW).fit(Xr, yr),
        "HGB": HistGradientBoostingClassifier(**HGB_KW).fit(Xr, yr),
        "LR": make_pipeline(StandardScaler(), LogisticRegression(**LR_KW)).fit(Xr, yr),
    }


def ensemble(models, rs):
    Xt = X(rs)
    ps = [models[k].predict_proba(Xt)[:, 1] for k in models]
    return np.average(ps, axis=0, weights=WEIGHTS)


def auc(p, rs):
    p = np.asarray(p, float); lb = np.array([r["TP"] for r in rs], bool)
    if lb.sum() == 0 or (~lb).sum() == 0:
        return float("nan")
    return ((p[lb][:, None] > p[~lb][None, :]).sum()
            + 0.5 * (p[lb][:, None] == p[~lb][None, :]).sum()) / (lb.sum() * (~lb).sum())


def prec_rec(p, rs, topn=TOPN):
    """按**工具实际的候选选取方式**评估。

    ⚠️ 这里以前是「直接取分数最高的 topn 个」，没有实现工具里的间隔抑制。
    实测过这个差别很大：不抑制时精确率 60.9%，工具实际只有 41.5% —— 报出去
    的数字比工具真实表现偏乐观了近 20 个点。现在直接调用
    climax_finder.merge_candidates，保证评估口径和工具完全一致。
    """
    tc = defaultdict(list)
    for r, s in zip(rs, p):
        tc[(r["rj"], r["track"])].append(
            {"time": float(r["t"]), "score": float(s), "mmss": "",
             "peak": float(r.get("peak", 0.0))})
    tp = nc = 0
    cov = set(); gtot = 0
    gt = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
    for (rj, tk), cand in tc.items():
        # 人工标注可能来自没有官方标注的作品 —— 那种没有评估基准，跳过。
        # （它们仍然参与训练，只是不能用来算精确率/召回率。）
        g = gt.get(rj, {}).get(tk)
        if g is None:
            continue
        gtot += len(g)
        events = snap_to_peak(merge_candidates(cand, top=topn), cand)
        for c in events:
            nc += 1
            h = [x for x in g if abs(c["time"] - x) <= TOL]
            if h:
                tp += 1
                for x in h:
                    cov.add((rj, tk, x))
    return tp / max(1, nc), len(cov) / max(1, gtot)


# 交叉验证只在**有官方标注**的作品上做。
# 人工标注的作品没有完整标注（用户只标了模型给出的候选），算不了召回率，
# 拿它当测试折没有意义 —— 所以它们始终留在训练集里。
_gt = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))["gt"]
works_gt = [w for w in works if w in _gt and any(r["track"] in _gt[w] for r in bywork[w])]
extra = [w for w in works if w not in works_gt]
if extra:
    print(f"  ⚠ {len(extra)} 部作品只有人工标注、没有官方标注："
          f"{', '.join(extra)}")
    print(f"    它们参与训练，但不作为测试折（没有召回率基准）")

print("\n10 折留一作品交叉验证（给出预期成绩 + 收集样本外预测用于校准）…")
oof_raw, oof_lab = [], []
A, P, R = [], [], []
for hold in works_gt:
    tr = [r for r in rows if r["rj"] != hold]
    te = bywork[hold]
    m = fit(tr)
    raw = ensemble(m, te)
    a, p, rc = auc(raw, te), *prec_rec(raw, te)
    A.append(a); P.append(p); R.append(rc)
    oof_raw.append(raw)
    oof_lab.append(Y(te))
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

print("\n用全部数据训练最终模型…")
final = fit(rows)
payload = {
    "models": final,
    "weights": WEIGHTS,
    "iso": iso,
    "feats": MODEL_FEATS,
    "meta": {
        "n_works": len(works_gt), "n_tracks": len(tracks), "n_rows": len(rows),
        "n_pos": sum(1 for r in rows if r["TP"]),
        "works": works_gt, "works_extra": extra,
        "cv_auc": float(np.nanmean(A)), "cv_auc_std": float(np.nanstd(A)),
        "cv_prec": float(np.mean(P)), "cv_prec_std": float(np.std(P)),
        "cv_rec": float(np.mean(R)),
        "brier_raw": br_raw, "brier_cal": br_cal,
        "train_minutes": round(sum(1 for _ in tracks) * 0),
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "sklearn_models": {"RF": RF_KW, "HGB": HGB_KW, "LR": LR_KW},
        "ensemble": "equal-weight average of RF + HGB + LR probabilities",
        "note": "排序用 predict_raw（原始平均分）；predict_prob 仅供显示与阈值",
        "merge_gap": MERGE_GAP, "pool_extra": POOL_EXTRA, "snap": SNAP_WINDOW,
        "select": "取分最高的 top+1 个 → 30 秒内合并取中点 → 吸附到 ±12 秒内最大峰",
    },
}
with open(OUT, "wb") as fh:
    pickle.dump(payload, fh, protocol=4)
mb = OUT.stat().st_size / 1024 / 1024
print(f"\n已保存 {OUT}（{mb:.1f} MB）")
print(f"  预期成绩（10 折留一作品）：AUC {np.nanmean(A):.3f}　"
      f"精确率 {np.mean(P)*100:.1f}%　召回率 {np.mean(R)*100:.1f}%")
