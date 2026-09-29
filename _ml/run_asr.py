# -*- coding: utf-8 -*-
"""对 6 部作品跑 ASR（只转写，不翻译），输出 .segments.json + .ja.lrc

只跑还没有产出的作品。RJ324692 / RJ362169 已有产出，跳过。
"""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(r"D:\库\音声\含高潮时间")
ROOT = Path(r"D:\pyitme\onsei2lrc")
OUTROOT = ROOT / "_ml" / "asr"
OUTROOT.mkdir(parents=True, exist_ok=True)

META = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))

TODO = ["RJ01583845", "RJ01586001", "RJ01603181", "RJ01605303"]

total_mb = 0.0
for rj in TODO:
    outdir = OUTROOT / rj
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted({Path(p) for p in META["files"][rj].values()})
    mb = sum(f.stat().st_size for f in files) / 1024 / 1024
    total_mb += mb
    print(f"\n{'='*88}")
    print(f"  {rj}  {len(files)} 个文件  {mb:.0f} MB")
    print(f"{'='*88}")
    if not files:
        print("  跳过（没有文件）")
        continue
    cmd = [sys.executable, str(ROOT / "onsei2lrc.py")]
    cmd += [str(f) for f in files]
    cmd += ["--outdir", str(outdir), "--translator", "none", "--lrc-mode", "ja"]
    r = subprocess.run(cmd, cwd=str(ROOT))
    print(f"  → 退出码 {r.returncode}")

print(f"\n总音频量 {total_mb:.0f} MB")
print(f"\n产出:")
for rj in TODO:
    d = OUTROOT / rj
    n_lrc = len(list(d.glob("*.ja.lrc")))
    n_seg = len(list(d.glob("*.segments.json")))
    print(f"  {rj}: {n_lrc} 个 lrc, {n_seg} 个 segments")
