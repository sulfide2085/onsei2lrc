# -*- coding: utf-8 -*-
"""对比三种解码方式的正确性与速度"""
import gc
import io
import sys
import time
from pathlib import Path

import av
import numpy as np

SR = 16000


def decode_fw(path, sr=SR):
    """当前实现：faster_whisper.decode_audio（会连带 import ctranslate2）"""
    from faster_whisper import decode_audio
    return decode_audio(str(path), sampling_rate=sr)


def decode_av(path, sr=SR):
    """直接用 PyAV，解码即目标格式（flt32 单声道），不绕 s16"""
    res = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=sr)
    chunks = []
    with av.open(str(path), mode="r", metadata_errors="ignore") as c:
        for frame in c.decode(audio=0):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):              # 冲刷尾部
            chunks.append(f.to_ndarray().reshape(-1))
    del res
    gc.collect()                                   # 见 faster-whisper#390
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def decode_ff(path, sr=SR):
    """ffmpeg 子进程"""
    import subprocess
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


P = Path(r"D:\库\音声\含高潮时间\RJ362169\01_mp3\track05_我，想要知道.mp3")

print("=" * 88)
print("  正确性：三种方式解出来的应该几乎一样")
print("=" * 88)
a = decode_fw(P)
b = decode_av(P)
c = decode_ff(P)
print(f"  faster_whisper  {len(a):>10,} 个采样   {len(a)/SR:8.2f} 秒")
print(f"  PyAV 直接       {len(b):>10,} 个采样   {len(b)/SR:8.2f} 秒")
print(f"  ffmpeg 子进程   {len(c):>10,} 个采样   {len(c)/SR:8.2f} 秒")
n = min(len(a), len(b), len(c))
for name, x in (("PyAV", b), ("ffmpeg", c)):
    d = np.abs(a[:n] - x[:n])
    print(f"  与 faster_whisper 的差异（{name}）: 最大 {d.max():.6f}  平均 {d.mean():.8f}  "
          f"相关性 {np.corrcoef(a[:n], x[:n])[0,1]:.6f}")

print()
print("=" * 88)
print("  速度（预热后各跑 2 次）")
print("=" * 88)
dur = len(a) / SR
for name, fn in (("faster_whisper.decode_audio", decode_fw),
                 ("PyAV 直接", decode_av),
                 ("ffmpeg 子进程", decode_ff)):
    fn(P)
    ts = []
    for _ in range(2):
        t0 = time.time(); fn(P); ts.append(time.time() - t0)
    print(f"  {name:<30} {min(ts)*1000:8.0f} ms  → {dur/min(ts):6.0f}× 实时")

print()
print("=" * 88)
print("  导入开销（真正的痛点）")
print("=" * 88)
import subprocess
def imp(code):
    t0 = time.time()
    subprocess.run([sys.executable, "-c", code], capture_output=True)
    return time.time() - t0
base = imp("pass")
for label, code in (("import av", "import av"),
                    ("from faster_whisper import decode_audio",
                     "from faster_whisper import decode_audio")):
    print(f"  {label:<38} {(imp(code)-base)*1000:7.0f} ms")
