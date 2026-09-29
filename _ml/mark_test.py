# -*- coding: utf-8 -*-
"""手动标注精确高潮点 —— 浏览器端到端测试

关键点：标注是独立于模型候选的，所以**模型漏掉的高潮也能标出来**。
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:7861"
AU = r"D:\库\音声\含高潮时间\RJ362169\01_mp3\track04_没关系，我会用魔法来隐藏的…！.mp3"
MARKS = Path(r"D:\pyitme\onsei2lrc\climax_marks.jsonl")
DONE = Path(r"D:\pyitme\onsei2lrc\climax_tracks_done.jsonl")
checks, errors = [], []


def ck(n, ok, d=""):
    checks.append((n, ok, d))
    print(f"  {'✓' if ok else '✗'} {n}" + (f"　{d}" if d else ""))


# 备份并在测试期间清空，保证可重复
bm = MARKS.read_text(encoding="utf-8") if MARKS.exists() else None
bd = DONE.read_text(encoding="utf-8") if DONE.exists() else None
MARKS.write_text("", encoding="utf-8")
DONE.write_text("", encoding="utf-8")

with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chrome",
                            args=["--autoplay-policy=no-user-gesture-required"])
    pg = br.new_page(viewport={"width": 1400, "height": 950})
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

    pg.goto(BASE, wait_until="networkidle")
    pg.wait_for_timeout(800)
    pg.evaluate("p => pick(p, p.split('\\\\').pop())", AU)
    pg.wait_for_timeout(1500)

    print("=" * 78)
    print("① 标记按钮与快捷键")
    print("=" * 78)
    ck("标记按钮已启用", pg.locator("#mark").is_enabled())
    ck("标注卡片初始隐藏", not pg.locator("#mkcard").is_visible())

    pg.evaluate("au.currentTime = 1149.0")
    pg.click("#mark")
    pg.wait_for_timeout(700)
    ck("点击后出现 1 个标注", pg.evaluate("marks.length") == 1,
       f"{pg.evaluate('marks.length')} 个")
    ck("标注卡片显示", pg.locator("#mkcard").is_visible())
    ck("标注值正确", abs(pg.evaluate("marks[0]") - 1149.0) < 1.5,
       f"{pg.evaluate('marks[0]'):.1f}")
    ck("列表出现条目", pg.locator("#mklist .hit").count() == 1)
    ck("顶部统计正确", "共 1 个" in pg.inner_text("#mkstat"), pg.inner_text("#mkstat"))

    print()
    print("=" * 78)
    print("② 时间轴上的绿色标记")
    print("=" * 78)
    n_user = pg.locator("#tl .mk.user").count()
    ck("时间轴有手动标记", n_user == 1, f"{n_user} 个")
    pos = pg.evaluate("document.querySelector('#tl .mk.user').style.left")
    ck("标记位置约 66%", abs(float(pos.rstrip('%')) - 1149 / 1742 * 100) < 3, pos)

    print()
    print("=" * 78)
    print("③ 快捷键 M")
    print("=" * 78)
    pg.evaluate("au.currentTime = 1576.0")
    pg.locator("body").press("KeyM")
    pg.wait_for_timeout(700)
    ck("按 M 也能标记", pg.evaluate("marks.length") == 2,
       f"{pg.evaluate('marks.length')} 个")
    ck("时间轴有 2 个标记", pg.locator("#tl .mk.user").count() == 2)

    print()
    print("=" * 78)
    print("④ 重复标记会被拒绝")
    print("=" * 78)
    pg.evaluate("au.currentTime = 1576.2")
    pg.click("#mark")
    pg.wait_for_timeout(600)
    ck("同一位置不重复加", pg.evaluate("marks.length") == 2,
       f"{pg.evaluate('marks.length')} 个")

    print()
    print("=" * 78)
    print("⑤ 标注独立于模型候选（这是关键）")
    print("=" * 78)
    pg.click("#go")
    pg.wait_for_function("() => !document.getElementById('go').disabled", timeout=180_000)
    pg.wait_for_timeout(800)
    n_cand = pg.locator("#hits .hit").count()
    # 标一个模型肯定没给的时刻（比如 300 秒）
    pg.evaluate("au.currentTime = 300.0")
    pg.click("#mark")
    pg.wait_for_timeout(700)
    ck("能标出模型没给的时刻", pg.evaluate("marks.length") == 3,
       f"候选 {n_cand} 个，标注 {pg.evaluate('marks.length')} 个")
    ck("分析后标注没被清掉", pg.locator("#tl .mk.user").count() == 3)

    print()
    print("=" * 78)
    print("⑥ 点击标注跳转 + 删除")
    print("=" * 78)
    pg.locator("#mklist .hit").first.locator(".t").click()
    pg.wait_for_timeout(600)
    ck("点标注可跳转", abs(pg.evaluate("au.currentTime") - 300.0) < 3,
       f"{pg.evaluate('au.currentTime'):.0f}s")
    pg.evaluate("au.pause()")
    pg.locator("#mklist .hit").first.locator(".vb button").click()
    pg.wait_for_timeout(700)
    ck("可删除标注", pg.evaluate("marks.length") == 2,
       f"{pg.evaluate('marks.length')} 个")
    ck("删除后时间轴同步", pg.locator("#tl .mk.user").count() == 2)

    print()
    print("=" * 78)
    print("⑦ 「本轨已标完」开关")
    print("=" * 78)
    ck("默认未勾选", not pg.locator("#mkdone").is_checked())
    ck("提示文字说明未标完不用于训练",
       "不会" in pg.inner_text("#mkhint") or "才会用于训练" in pg.inner_text("#mkhint"))
    pg.check("#mkdone")
    pg.wait_for_timeout(700)
    ck("可勾选标完", pg.locator("#mkdone").is_checked())
    ck("提示文字变为已标完", "已标完" in pg.inner_text("#mkhint"),
       pg.inner_text("#mkhint")[:40])

    print()
    print("=" * 78)
    print("⑧ 刷新后恢复")
    print("=" * 78)
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(700)
    pg.evaluate("p => pick(p, p.split('\\\\').pop())", AU)
    pg.wait_for_timeout(2000)
    ck("刷新后标注恢复", pg.evaluate("marks.length") == 2,
       f"{pg.evaluate('marks.length')} 个")
    ck("刷新后标完状态恢复", pg.locator("#mkdone").is_checked())
    ck("刷新后时间轴恢复", pg.locator("#tl .mk.user").count() == 2)

    print()
    print("=" * 78)
    print("⑨ 导出")
    print("=" * 78)
    with pg.expect_download(timeout=15000) as dl:
        pg.click("#exp")
    d = json.loads(Path(dl.value.path()).read_text(encoding="utf-8"))
    ck("导出成功", bool(d), json.dumps(d, ensure_ascii=False)[:60])
    ck("格式与 gt_all.json 同构", "RJ362169" in d and "04" in d.get("RJ362169", {}))
    ck("点数正确", len(d["RJ362169"]["04"]) == 2, f"{len(d['RJ362169']['04'])} 个")

    ck("无 JS 报错", not errors, "; ".join(errors[:2]))
    pg.screenshot(path=r"D:\pyitme\onsei2lrc\_shot_player.png")
    br.close()

# 恢复
if bm is not None:
    MARKS.write_text(bm, encoding="utf-8")
else:
    MARKS.write_text("", encoding="utf-8")
if bd is not None:
    DONE.write_text(bd, encoding="utf-8")
else:
    DONE.write_text("", encoding="utf-8")
print("\n  （已恢复测试前的标注文件）")

print()
print("=" * 78)
ok = sum(1 for _, o, _ in checks if o)
print(f"  结果：{ok} / {len(checks)} 项通过")
bad = [n for n, o, _ in checks if not o]
if bad:
    print("  未通过：" + "、".join(bad))
print("=" * 78)
sys.exit(0 if not bad else 1)
