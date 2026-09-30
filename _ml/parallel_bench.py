# -*- coding: utf-8 -*-
"""干净地测量 --parallel 对显存/速度的影响

上一次测得不准：桌面本身占用约 1 GB 且在波动，两次读数差 60 MiB 说明不了问题。
这次严格对照：卸载 → 记基线 → 加载 → 记峰值 → 跑同一请求测速 → 再卸载。
"""
import subprocess
import sys
import time
from pathlib import Path

import httpx

LMS = str(Path.home() / ".lmstudio" / "bin" / "lms.exe")
MODEL = "sakura-galtransl-7b-v3.7"
IDENT = "sakura37"
CTX = 8192

BODY = {
    "model": IDENT,
    "messages": [
        {"role": "system", "content": "你是一个视觉小说翻译模型，可以通顺地使用给定的术语表"
                                      "以指定的风格将日文翻译成简体中文。"},
        {"role": "user", "content": "根据以上术语表的对应关系和备注，结合历史剧情和上下文，"
                                   "将下面的文本从日文翻译成简体中文：\n"
                                   "びゅっびゅーって…\n気持ちいい…\n"
                                   "もっとちょうだい…\nはぁ…はぁ…\n"
                                   "イク…イク…\n出る…出ちゃう…\n"
                                   "んぁ…あ…\nすごい…\n"},
    ],
    "temperature": 0.3, "top_p": 0.8, "frequency_penalty": 0.0, "max_tokens": 200,
}


def vram():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()[0]
    return int(out)


def lms(*args, timeout=300):
    return subprocess.run([LMS, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def measure(parallel, n_req=2):
    lms("unload", IDENT)
    time.sleep(6)
    base = vram()
    r = lms("load", MODEL, "--gpu", "max", "--context-length", str(CTX),
            "--identifier", IDENT, "--parallel", str(parallel), "-y")
    if "loaded successfully" not in (r.stdout + r.stderr):
        return None
    time.sleep(5)
    loaded = vram()
    # 预热一次
    try:
        httpx.post("http://127.0.0.1:1234/v1/chat/completions", json=BODY,
                   timeout=180, trust_env=False)
    except Exception:
        pass
    ts = []
    for _ in range(n_req):
        t0 = time.time()
        try:
            rr = httpx.post("http://127.0.0.1:1234/v1/chat/completions", json=BODY,
                            timeout=180, trust_env=False)
            ts.append(time.time() - t0)
            ok = rr.status_code == 200
        except Exception as e:
            ts.append(float("nan")); ok = False
    return {"parallel": parallel, "base": base, "loaded": loaded,
            "delta": loaded - base, "times": ts, "ok": ok}


print("  逐项卸载/加载会影响正在跑的任务，这里只做对照测量。\n")
res = []
for p in (1, 4):
    print(f"  --- 测量 --parallel {p} ---", flush=True)
    m = measure(p)
    if m is None:
        print("    加载失败"); continue
    res.append(m)
    print(f"    基线 {m['base']} MiB → 加载后 {m['loaded']} MiB"
          f"（模型占 {m['delta']} MiB = {m['delta']/1024:.2f} GiB）")
    print(f"    单次请求 {['%.2f' % t for t in m['times']]} 秒   成功={m['ok']}")

print()
print("  ===== 对照 =====")
print("  %-12s %12s %14s %14s" % ("parallel", "模型占用", "单次请求(首)", "单次请求(次)"))
print("  " + "-" * 58)
for m in res:
    print("  %-12s %9d MiB %13.2f秒 %13.2f秒" % (
        m["parallel"], m["delta"], m["times"][0],
        m["times"][1] if len(m["times"]) > 1 else float("nan")))
print()
if len(res) == 2:
    a, b = res[0], res[1]
    print("  parallel 4 比 1 多占 %.0f MiB，单次请求慢 %.2f 秒" % (
        b["delta"] - a["delta"], b["times"][0] - a["times"][0]))
    print()
    print("  结论：", end="")
    if abs(b["delta"] - a["delta"]) < 200:
        print("显存差异很小 —— 说明 LM Studio 没有按 parallel 倍数预分配 KV cache。")
    else:
        print("显存差异明显。")
