# -*- coding: utf-8 -*-
"""在真实浏览器里验证播放器（不是只测 API）

检查项：
  ① 页面无 JS 报错（这是之前栽过的地方）
  ② 目录浏览可用
  ③ 分析后时间轴出现标记、列表出现条目
  ④ 拖动到高潮点前并播放 → 横幅提示真的弹出
  ⑤ 导入歌词后歌词面板显示、当前行高亮
  ⑥ 播放/暂停、快进快退按钮有效
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:7861"
AUDIO_DIR = r"D:\库\音声\含高潮时间\RJ362169\01_mp3"
LRC_DIR = r"D:\pyitme\onsei2lrc\_climax2\asr"
AUDIO_NAME = "track04_没关系，我会用魔法来隐藏的…！.mp3"

errors, checks = [], []


def ck(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f"　{detail}" if detail else ""))


with sync_playwright() as pw:
    # 用系统 Chrome（Playwright 自带的 Chromium 未安装）
    br = pw.chromium.launch(
        channel="chrome",
        args=["--autoplay-policy=no-user-gesture-required"])
    pg = br.new_page(viewport={"width": 1440, "height": 900})
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
          if m.type == "error" else None)

    print("=" * 84)
    print("① 加载页面")
    print("=" * 84)
    pg.goto(BASE, wait_until="networkidle")
    pg.wait_for_timeout(1200)
    ck("窗口标题为空（不取名）", pg.title() == "", repr(pg.title()))
    ck("路径输入框存在", pg.locator("#pinput").count() == 1)
    ck("无 JS 报错", not errors, "; ".join(errors[:2]))

    print()
    print("=" * 84)
    print("② 目录浏览")
    print("=" * 84)
    ck("根级列出驱动器", pg.locator("#listing .row.dir").count() >= 2,
       f"{pg.locator('#listing .row.dir').count()} 个")
    ck("根级有分组标题", pg.locator("#listing .secn").count() >= 1)
    # 直接跳到音频目录
    pg.evaluate("p => browse(p)", AUDIO_DIR)
    pg.wait_for_timeout(900)
    n_audio = pg.locator("#listing .row:not(.dir):not(.lrc)").count()
    ck("列出音频文件", n_audio >= 7, f"{n_audio} 个")

    print()
    print("=" * 84)
    print("③ 选择文件 + 分析")
    print("=" * 84)
    row = pg.locator(f'#listing .row[data-path$="{AUDIO_NAME}"]')
    if row.count() == 0:
        row = pg.locator("#listing .row:not(.dir):not(.lrc)").nth(3)
    row.click()
    pg.wait_for_timeout(1500)
    ck("标题显示文件名", AUDIO_NAME[:10] in pg.inner_text("#title"))
    ck("窗口标题变成文件名", pg.title() == AUDIO_NAME, pg.title()[:30])
    ck("播放按钮已启用", pg.locator("#pp").is_enabled())
    dur = pg.evaluate("au.duration")
    ck("音频已加载（时长>0）", dur and dur > 60, f"{dur:.0f}s" if dur else "无")

    pg.click("#go")
    pg.wait_for_function("() => !document.getElementById('go').disabled",
                         timeout=180_000)
    pg.wait_for_timeout(700)
    n_hit = pg.locator("#hits .hit").count()
    n_mk = pg.locator("#tl .mk").count()
    ck("高潮点列表有内容", n_hit > 0, f"{n_hit} 条")
    ck("时间轴有标记", n_mk == n_hit, f"标记 {n_mk} / 列表 {n_hit}")
    ck("分析无报错", not pg.inner_text("#err"), pg.inner_text("#err")[:50])

    print()
    print("=" * 84)
    print("④ 高潮点提示（核心功能）")
    print("=" * 84)
    first = pg.evaluate("cands[0].time")
    print(f"     第一个高潮点在 {first:.1f}s，先退到它前面 10 秒再播放")
    pg.evaluate(f"seek({max(0, first - 10)})")
    pg.wait_for_timeout(400)
    pg.evaluate("au.play()")
    # 等「即将」提示（用 wait_for_function，避免缓冲导致的固定等待不可靠）
    try:
        pg.wait_for_function(
            "() => document.getElementById('banner').classList.contains('on')",
            timeout=25_000)
        pre_on, pre_txt = True, pg.inner_text("#bt")
    except Exception:
        pre_on, pre_txt = False, "未出现"
    ck("临近时出现提示", pre_on, pre_txt)
    # 等真正触发
    try:
        pg.wait_for_function("() => cands.filter(c => c.fired).length >= 1",
                             timeout=25_000)
        fired = pg.evaluate("cands.filter(c => c.fired).length")
    except Exception:
        fired = 0
    ck("到达高潮点时触发", fired >= 1, f"{fired} 个已触发")
    ck("时间轴标记有脉冲动画",
       pg.locator("#tl .mk.pulse").count() >= 1 or pg.locator("#tl .mk.done").count() >= 1,
       f"pulse {pg.locator('#tl .mk.pulse').count()} / done {pg.locator('#tl .mk.done').count()}")
    ck("提示音开关存在", pg.locator("#beep").is_checked())
    ck("到点暂停开关存在", pg.locator("#pauseat").count() == 1)

    print()
    print("=" * 84)
    print("⑤ 播放控制")
    print("=" * 84)
    pg.evaluate("au.pause()")
    t0 = pg.evaluate("au.currentTime")
    pg.click("#f10")
    pg.wait_for_timeout(250)
    t1 = pg.evaluate("au.currentTime")
    ck("+10s 按钮", abs((t1 - t0) - 10) < 1.5, f"{t0:.1f} → {t1:.1f}")
    pg.click("#b10")
    pg.wait_for_timeout(250)
    t2 = pg.evaluate("au.currentTime")
    ck("−10s 按钮", abs((t2 - t1) + 10) < 1.5, f"{t1:.1f} → {t2:.1f}")

    pg.click("#pp")
    pg.wait_for_timeout(500)
    ck("播放按钮切到暂停态", "暂停" in pg.inner_text("#pp"), pg.inner_text("#pp"))
    pg.click("#pp")
    pg.wait_for_timeout(400)
    ck("暂停按钮切回播放态", "播放" in pg.inner_text("#pp"), pg.inner_text("#pp"))

    # 点击时间轴跳转
    box = pg.locator("#tl").bounding_box()
    pg.mouse.click(box["x"] + box["width"] * 0.15, box["y"] + box["height"] / 2)
    pg.wait_for_timeout(300)
    t3 = pg.evaluate("au.currentTime")
    ck("点击时间轴可跳转", abs(t3 / pg.evaluate("au.duration") - 0.15) < 0.05,
       f"跳到 {t3:.0f}s（目标 ~{pg.evaluate('au.duration')*0.15:.0f}s）")

    # 列表项点击
    pg.locator("#hits .hit").first.click()
    pg.wait_for_timeout(400)
    ck("点击列表项可跳转", pg.evaluate("Math.abs(au.currentTime - cands[0].time) < 4"),
       f"{pg.evaluate('au.currentTime'):.0f}s")
    pg.evaluate("au.pause()")

    print()
    print("=" * 84)
    print("⑥ 导入歌词")
    print("=" * 84)
    pg.evaluate("p => browse(p)", LRC_DIR)
    pg.wait_for_timeout(900)
    n_lrc = pg.locator("#listing .row.lrc").count()
    ck("侧栏列出歌词文件", n_lrc > 0, f"{n_lrc} 个")
    if n_lrc:
        # 选一个与当前音频对应的
        target = pg.locator('#listing .row.lrc').filter(has_text="track04").first
        (target if target.count() else pg.locator("#listing .row.lrc").first).click()
        pg.wait_for_timeout(1000)
        vis = pg.locator("#lyrcard").is_visible()
        nln = pg.locator("#lyr div").count()
        ck("歌词面板显示", vis and nln > 10, f"{nln} 行")
        # 跳到歌词中间看高亮
        mid = pg.evaluate("lyr[Math.floor(lyr.length/2)].time")
        pg.evaluate(f"seek({mid})")
        pg.wait_for_timeout(600)
        ck("当前行高亮", pg.locator("#lyr div.on").count() == 1,
           f"{pg.locator('#lyr div.on').count()} 行高亮")
        pg.click("#clr")
        pg.wait_for_timeout(400)
        ck("清除歌词有效", not pg.locator("#lyrcard").is_visible())

    print()
    print("=" * 84)
    print("⑦ 无高潮的轨（弃权提示）")
    print("=" * 84)
    pg.evaluate("p => browse(p)", AUDIO_DIR)
    pg.wait_for_timeout(800)
    rows = pg.locator("#listing .row:not(.dir):not(.lrc)")
    idx = -1
    for i in range(rows.count()):
        if "track05" in (rows.nth(i).get_attribute("data-path") or ""):
            idx = i
            break
    if idx >= 0:
        rows.nth(idx).click()
        pg.wait_for_timeout(1200)
        pg.click("#go")
        pg.wait_for_function("() => !document.getElementById('go').disabled",
                             timeout=180_000)
        pg.wait_for_timeout(600)
        meta = pg.inner_text("#meta")
        ck("低分轨标出「可能没有高潮」", "可能没有高潮" in meta, meta[-30:])
    else:
        ck("找到 track05", False, "未找到")

    print()
    print("=" * 84)
    print("⑧ 全程 JS 报错")
    print("=" * 84)
    ck("无 JS 报错", not errors, "; ".join(errors[:3]) if errors else "")

    pg.screenshot(path=r"D:\pyitme\onsei2lrc\_shot_player.png", full_page=False)
    br.close()

print()
print("=" * 84)
ok = sum(1 for _, o, _ in checks if o)
print(f"  结果：{ok} / {len(checks)} 项通过")
bad = [n for n, o, _ in checks if not o]
if bad:
    print("  未通过：" + "、".join(bad))
print("=" * 84)
sys.exit(0 if not bad else 1)
