# -*- coding: utf-8 -*-
"""截几张能看清成品的图：加载有结果的真实音轨"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"D:\pyitme\onsei2lrc")
HTML = ROOT / "_offline" / "高潮点分析.html"
meta = json.loads((ROOT / "_ml" / "gt_all.json").read_text(encoding="utf-8"))
FP = meta["files"]["RJ362169"]["04"]

with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chrome")
    for tag, w, h, mob in (("desktop", 1440, 900, False),
                           ("mobile", 390, 844, True)):
        ctx = br.new_context(viewport={"width": w, "height": h},
                             is_mobile=mob, has_touch=mob)
        pg = ctx.new_page()
        pg.goto(HTML.as_uri(), wait_until="load")
        pg.wait_for_timeout(600)
        pg.set_input_files("#file", FP)
        pg.wait_for_function(
            "() => document.getElementById('foot').textContent"
            ".indexOf('分析用时') >= 0", timeout=240000)
        pg.wait_for_timeout(500)
        pg.screenshot(path=str(HTML.parent / f"shot_{tag}.png"))

        # 顺便量一下「关于」那段有没有溢出容器
        m = pg.evaluate("""() => {
            const a = document.getElementById('about');
            const b = document.getElementById('body');
            const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
            return {aRight: ra.right, bRight: rb.right, aW: ra.width, bW: rb.width,
                    scrollW: a.scrollWidth, clientW: a.clientWidth,
                    over: a.scrollWidth > a.clientWidth + 1};
        }""")
        print(f"  {tag}: #about 宽 {m['aW']:.0f} / 容器 {m['bW']:.0f}　"
              f"scrollWidth {m['scrollW']} vs clientWidth {m['clientW']}　"
              f"{'✗ 文字溢出' if m['over'] else '✓ 正常换行'}")
        ctx.close()
    br.close()
print("  已写出 shot_desktop.png / shot_mobile.png")
