# -*- coding: utf-8 -*-
import os
"""重建 gt_all.json 里的 音频路径 与 转写路径

背景：作品目录被改名（加了中文标题后缀），音频文件名也被翻译成中文。
  旧: <库>\\RJ01586001\\MP3\\tr01_<日文原名>.mp3
  新: <库>\\RJ01586001_<中文标题后缀>\\MP3\\tr01_<中文译名>.mp3

文件名对不上了，但**轨号还在**（tr01 / track01），所以按轨号匹配：
  · 音频：在 RJ 目录下递归找文件名里轨号相同的 mp3
  · 转写：在 asr 目录下递归找文件名里轨号相同的 .segments.json

匹配不到就报出来，不瞎猜（宁可少一轨，不能错一轨）。
"""
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\pyitme\onsei2lrc")
BASE = Path(os.environ.get("ONSEI_AUDIO_BASE",
                        r"D:\库\音声\含高潮时间"))
GTF = ROOT / "_ml" / "gt_all.json"

ASR_DIRS = [ROOT / "_ml" / "asr", ROOT / "_climax2" / "asr", ROOT / "_climax" / "asr"]


def track_no(name: str):
    """从文件名里取轨号：tr01 / track01 / track_01 → 1"""
    m = re.search(r"(?:tr|track)[_\-\s]*(\d+)", name, re.I)
    return int(m.group(1)) if m else None


meta = json.loads(GTF.read_text(encoding="utf-8"))
GT, FILES = meta["gt"], meta["files"]

# RJ 号 → 实际目录
rjdirs = defaultdict(list)
for d in BASE.iterdir():
    if d.is_dir():
        m = re.match(r"(RJ\d+)", d.name)
        if m:
            rjdirs[m.group(1)].append(d)

# 每个 RJ 目录下的 mp3 与 segments.json，按轨号建索引
audio_idx, seg_idx = {}, {}
for rj, ds in rjdirs.items():
    a, g = {}, {}
    for d in ds:
        for f in d.rglob("*.mp3"):
            n = track_no(f.name)
            if n is not None:
                a.setdefault(n, []).append(f)
        for f in d.rglob("*.segments.json"):
            n = track_no(f.name)
            if n is not None:
                g.setdefault(n, []).append(f)
    audio_idx[rj], seg_idx[rj] = a, g

# 独立 asr 目录（可能不在作品目录下）
asr_extra = defaultdict(dict)
for ad in ASR_DIRS:
    if not ad.exists():
        continue
    for f in ad.rglob("*.segments.json"):
        n = track_no(f.name)
        if n is None:
            continue
        # 从路径或文件名里找 RJ 号
        m = re.search(r"(RJ\d+)", str(f))
        if m:
            asr_extra[m.group(1)].setdefault(n, []).append(f)

print(f"  作品目录 {len(rjdirs)} 个")
print()

new_files, new_segs = {}, {}
miss_a, miss_s, dup = [], [], []
for rj in sorted(FILES):
    new_files[rj], new_segs[rj] = {}, {}
    for tk in sorted(FILES[rj]):
        n = int(tk)
        cand = audio_idx.get(rj, {}).get(n, [])
        if len(cand) > 1:
            # 同一文件常有多处副本（顶层一份、嵌套的【音声】/mp3 下一份）。
            # 实测副本 md5 相同，所以按「体积 → 路径短」稳定取一份，不报歧义。
            cand = sorted(cand, key=lambda p: (p.stat().st_size, len(str(p))))
            same = len({p.stat().st_size for p in cand}) == 1
            if same:
                cand = [cand[0]]
            else:
                dup.append((rj, tk, "音频", cand))
        if len(cand) == 1:
            new_files[rj][tk] = str(cand[0])
        else:
            miss_a.append((rj, tk))
        sc = seg_idx.get(rj, {}).get(n, []) + asr_extra.get(rj, {}).get(n, [])
        sc = sorted(set(sc))
        if sc:
            new_segs[rj][tk] = str(sc[0])

print(f"  音频  匹配 {sum(len(v) for v in new_files.values())} 条　"
      f"缺失 {len(miss_a)}　歧义 {len(dup)}")
print(f"  转写  匹配 {sum(len(v) for v in new_segs.values())} 条")
if miss_a:
    print("  缺音频的:", ", ".join(f"{a}/{b}" for a, b in miss_a[:14]))
if dup:
    for rj, tk, kind, c in dup[:4]:
        print(f"  歧义 {rj}/{tk} {kind}: {[str(x) for x in c][:3]}")

if miss_a or dup:
    print()
    print("  ⚠ 未写回。请先确认这些。")
    sys.exit(1)

shutil.copy2(GTF, GTF.with_suffix(".json.bak"))
meta["files"] = new_files
meta["segs"] = new_segs
GTF.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print(f"  ✓ 已写回（备份 gt_all.json.bak）")
