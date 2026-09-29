# -*- coding: utf-8 -*-
"""解析所有官方高潮标注 → gt_all.json（修正版）

三种格式：
  A 【トラックN】X回 / ・MM:SS〜          （Finishtime.txt / 高潮时间.txt）
  B trackNN_名前 (MM:SS)→...Finish[MM:SS]... （RJ324692 readme）
  C trackNN_名前 (MM:SS)- 第N次 [MM:SS]      （RJ362169 readme）
同时把「标注轨号」映射到「音频文件路径」。
"""
import json
import re
from pathlib import Path
from collections import defaultdict

BASE = Path(r"D:\库\音声\含高潮时间")
OUT = Path(r"D:\pyitme\onsei2lrc\_ml\gt_all.json")
AUDIO = (".mp3", ".wav", ".flac", ".m4a")


def parse_finishtime(txt):
    out, cur = {}, None
    for raw in txt.splitlines():
        line = raw.strip()
        m = re.match(r"^【トラック\s*(\d+)】\s*(\d+)\s*回", line)
        if m:
            cur = int(m.group(1)); out.setdefault(cur, []); continue
        if cur is None:
            continue
        m = re.match(r"^[・·\-–]\s*(\d{1,2}):(\d{2})", line)
        if m:
            out[cur].append(int(m.group(1)) * 60 + int(m.group(2)))
    return out


def parse_readme(txt):
    """B 和 C 两种都靠「找 trackNN 开头的行，取其后的所有 [MM:SS]」"""
    out = {}
    for raw in txt.splitlines():
        line = raw.strip().strip('"')
        m = re.match(r"^track\s*(\d+)\s*[_＿]", line)
        if not m:
            continue
        n = int(m.group(1))
        # 时长是圆括号 (MM:SS)，高潮是方括号 [MM:SS] —— 只取方括号
        hits = re.findall(r"\[\s*(\d{1,2}):(\d{2})\s*\]", line)
        out[n] = [int(h[0]) * 60 + int(h[1]) for h in hits]
    return out


# ---------- 收集每个作品的音频（按文件名去重） ----------
def audio_map(d: Path):
    seen = {}
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO:
            seen.setdefault(p.name, p)          # 同名只留第一个
    return seen


GT, FILES, REPORT = {}, {}, []
for d in sorted(BASE.iterdir()):
    if not d.is_dir():
        continue
    rj = d.name
    txts = sorted(d.rglob("*.txt"))
    if not txts:
        continue
    raw = txts[0].read_text(encoding="utf-8", errors="replace")
    parsed = parse_finishtime(raw) if "トラック" in raw and "【" in raw else parse_readme(raw)
    amap = audio_map(d)

    # 轨号 → 文件
    mapping = {}
    for n in parsed:
        cands = [f for name, f in amap.items()
                 if re.match(rf"^(tr|track)0*{n}[_＿]", name)]
        if cands:
            mapping[n] = sorted(cands, key=lambda p: len(str(p)))[0]
    GT[rj] = {f"{k:02d}": v for k, v in sorted(parsed.items())}
    FILES[rj] = {f"{k:02d}": str(v) for k, v in sorted(mapping.items())}
    REPORT.append((rj, txts[0].name, len(parsed), len(mapping),
                   sum(1 for v in parsed.values() if not v),
                   sum(len(v) for v in parsed.values()), len(amap)))

print("=" * 100)
print("解析结果")
print("=" * 100)
print(f"  {'作品':<14}{'来源':<18}{'标注轨':>7}{'匹配到文件':>11}{'0回轨':>7}{'高潮点':>7}{'音频数':>7}")
print("  " + "-" * 78)
T = [0, 0, 0, 0, 0]
for rj, fmt, nt, nm, ne, np_, na in REPORT:
    for i, v in enumerate((nt, nm, ne, np_, na)):
        T[i] += v
    flag = "" if nt == nm else f"  ⚠ 差 {nt-nm}"
    print(f"  {rj:<14}{fmt:<18}{nt:>7}{nm:>11}{ne:>7}{np_:>7}{na:>7}{flag}")
print("  " + "-" * 78)
print(f"  {'合计':<14}{'':<18}{T[0]:>7}{T[1]:>11}{T[2]:>7}{T[3]:>7}{T[4]:>7}")
print(f"\n  → {T[0]} 条标注轨，{T[3]} 个高潮点，{T[2]} 条无高潮轨（天然负样本）")

# 未匹配的轨
print()
print("  未能匹配到音频文件的标注轨：")
bad = False
for rj in GT:
    for k in GT[rj]:
        if k not in FILES[rj]:
            print(f"    {rj} トラック{k}")
            bad = True
if not bad:
    print("    （无，全部匹配）")

OUT.write_text(json.dumps({"gt": GT, "files": FILES}, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(f"\n已保存: {OUT}")
