# -*- coding: utf-8 -*-
"""离线 HTML 的真实音声实测

用 gt_all.json 里有官方标注的**真实音轨**，通过 set_input_files 喂给页面，
比对分析结果与官方高潮时刻。

这才是有效的测试：合成音频的特征分布和真实人声差太远
（zcr / centroid / flatness / rise / decay 全对不上，概率只有 0.3），
拿它测只会让人去调测试数据，而不是发现代码问题。
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"D:\pyitme\onsei2lrc")
HTML = ROOT / "_offline" / "高潮点分析.html"
URL = HTML.as_uri()
TOL = 20.0                      # 和训练/评测一致的容差
ok = [0, 0]


def ck(name, cond, detail=""):
    if cond:
        ok[0] += 1
        print(f"  ✓ {name}" + (f"　{detail}" if detail else ""))
    else:
        ok[1] += 1
        print(f"  ✗ {name}　{detail}")


meta = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))
GT, FILES = meta["gt"], meta["files"]

CASES = []
for rj, tk in (("RJ362169", "04"), ("RJ362169", "01")):
    fp = FILES.get(rj, {}).get(tk)
    g = GT.get(rj, {}).get(tk)
    if fp and Path(fp).exists() and g is not None:
        CASES.append((rj, tk, Path(fp), sorted(g)))
print("  测试素材:")
for rj, tk, fp, g in CASES:
    print(f"    {rj}/{tk}  {fp.stat().st_size/1048576:.1f} MB  "
          f"官方标注 {len(g)} 处  {[round(x) for x in g][:6]}")
print()

with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chrome")
    for label, w, h, mob in (("桌面 1440×900", 1440, 900, False),
                             ("手机 390×844", 390, 844, True)):
        print(f"===== {label} =====")
        ctx = br.new_context(viewport={"width": w, "height": h},
                             is_mobile=mob, has_touch=mob)
        pg = ctx.new_page()
        errs, reqs = [], []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append("console:" + m.text)
              if m.type == "error" else None)
        pg.on("request", lambda r: reqs.append(r.url)
              if r.url.split(":")[0] not in ("file", "data", "blob") else None)

        pg.goto(URL, wait_until="load")
        pg.wait_for_timeout(800)
        ck("加载无报错", len(errs) == 0, "; ".join(errs[:2]))
        ck("无外部网络请求", len(reqs) == 0, ", ".join(reqs[:2]))

        for rj, tk, fp, g in CASES:
            pg.set_input_files("#file", str(fp))
            # 不能等 '.hit' 出现 —— 官方 0 处标注的轨，工具**正确地不给候选**，
            # 永远等不到 .hit。改等「分析完成」的标志：#foot 被填上。
            try:
                pg.wait_for_function(
                    "() => document.getElementById('foot').textContent" +
                    ".indexOf('分析用时') >= 0",
                    timeout=240000)
            except Exception:
                ck(f"{rj}/{tk} 出结果", False, "超时")
                continue
            pg.wait_for_timeout(400)

            got = pg.evaluate("() => CUES")
            hits = pg.evaluate("""() => [...document.querySelectorAll('#hits .hit')]
                .map(e => e.querySelector('.t').textContent + ' ' +
                          e.querySelector('.p').textContent)""")
            foot = pg.inner_text("#foot")
            print(f"    {rj}/{tk}  官方 {[round(x) for x in g]}")
            print(f"           输出 {[round(x,1) for x in got]}")
            print(f"           界面 {', '.join(hits)}")
            print(f"           {foot.split('　')[0]}")

            if g:
                matched = sum(1 for x in g if any(abs(x - t) <= TOL for t in got))
                ck(f"{rj}/{tk} 有候选命中官方标注", matched > 0,
                   f"{matched}/{len(g)} 处命中（±{TOL:.0f} 秒）")
            else:
                # 官方标注是「0 次高潮」的轨，正确行为就是不给候选（弃权）
                ck(f"{rj}/{tk} 官方 0 处标注 → 应当弃权", len(got) == 0,
                   f"却给了 {[round(t,1) for t in got]}")
            ck(f"{rj}/{tk} 时间非负", all(t >= 0 for t in got))
            errs.clear()

        sw = pg.evaluate("document.documentElement.scrollWidth")
        cw = pg.evaluate("document.documentElement.clientWidth")
        ck("无横向溢出", sw <= cw + 1, f"{sw} vs {cw}")
        pg.screenshot(path=str(HTML.parent / f"html_{w}.png"))
        ctx.close()
        print()

    br.close()

print(f"  {ok[0]} / {ok[0]+ok[1]} 通过")
sys.exit(1 if ok[1] else 0)
