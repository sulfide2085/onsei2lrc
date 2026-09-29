# -*- coding: utf-8 -*-
"""验证人工标注功能（浏览器端到端 + 落盘检查）"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:7861"
AUDIO_DIR = r"D:\库\音声\含高潮时间\RJ362169\01_mp3"
AUDIO_NAME = "track04_没关系，我会用魔法来隐藏的…！.mp3"
FB = Path(r"D:\pyitme\onsei2lrc\climax_feedback.jsonl")

checks, errors = [], []


def ck(n, ok, d=""):
    checks.append((n, ok, d))
    print(f"  {'✓' if ok else '✗'} {n}" + (f"　{d}" if d else ""))


# 先备份已有标注，测试后恢复。
# 必须**清空**再测：✓/✗ 是切换语义，如果同一时刻已有标注，
# 点一下反而变成撤销，断言就会假失败（之前踩过）。
backup = FB.read_text(encoding="utf-8") if FB.exists() else None
FB.write_text("", encoding="utf-8")
before = 0

with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chrome",
                            args=["--autoplay-policy=no-user-gesture-required"])
    pg = br.new_page(viewport={"width": 1400, "height": 900})
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

    pg.goto(BASE, wait_until="networkidle")
    pg.wait_for_timeout(800)
    pg.evaluate("p => pick(p, p.split('\\\\').pop())", AUDIO_DIR + "\\" + AUDIO_NAME)
    pg.wait_for_timeout(1200)
    pg.click("#go")
    pg.wait_for_function("() => !document.getElementById('go').disabled", timeout=180_000)
    pg.wait_for_timeout(800)

    print("=" * 80)
    print("① 按钮存在")
    print("=" * 80)
    n = pg.locator("#hits .hit").count()
    ck("每个候选都有 ✓ 按钮", pg.locator("#hits .hit .vb button.y").count() == n, f"{n} 个")
    ck("每个候选都有 ✗ 按钮", pg.locator("#hits .hit .vb button.n").count() == n, f"{n} 个")

    print()
    print("=" * 80)
    print("② 标注 ✓")
    print("=" * 80)
    first = pg.locator("#hits .hit").first
    first.locator(".vb button.y").click()
    pg.wait_for_timeout(700)
    ck("行变绿边", "marked-y" in (first.get_attribute("class") or ""))
    ck("✓ 按钮高亮", "on" in (first.locator(".vb button.y").get_attribute("class") or ""))
    ck("顶部计数更新", "已标" in pg.inner_text("#fbstats"), pg.inner_text("#fbstats"))

    print()
    print("=" * 80)
    print("③ 标注 ✗ 并检查不会误跳转")
    print("=" * 80)
    pg.evaluate("au.pause(); au.currentTime = 0")
    pg.wait_for_timeout(300)
    t0 = pg.evaluate("au.currentTime")
    pg.locator("#hits .hit").nth(1).locator(".vb button.n").click()
    pg.wait_for_timeout(700)
    ck("行变红边且淡化",
       "marked-n" in (pg.locator("#hits .hit").nth(1).get_attribute("class") or ""))
    t1 = pg.evaluate("au.currentTime")
    ck("点按钮不会触发跳转", abs(t1 - t0) < 0.6, f"{t0:.1f} → {t1:.1f}")

    print()
    print("=" * 80)
    print("④ 点行主体仍能跳转")
    print("=" * 80)
    # 合并后候选可能只有 2 个，按实际数量取
    n_hit = pg.locator("#hits .hit").count()
    idx = min(2, n_hit - 1)
    pg.locator("#hits .hit").nth(idx).locator(".t").click()
    pg.wait_for_timeout(600)
    tgt = pg.evaluate(f"cands[{idx}].time")
    ck("点行跳转到候选点前 2 秒",
       abs(pg.evaluate("au.currentTime") - max(0, tgt - 2)) < 2.5,
       f"{pg.evaluate('au.currentTime'):.0f}s / 目标 {max(0,tgt-2):.0f}s")
    pg.evaluate("au.pause()")

    print()
    print("=" * 80)
    print("⑤ 撤销")
    print("=" * 80)
    first.locator(".vb button.y").click()          # 再点一次 = 撤销
    pg.wait_for_timeout(700)
    ck("撤销后绿边消失", "marked-y" not in (first.get_attribute("class") or ""))
    first.locator(".vb button.y").click()          # 再标回来
    pg.wait_for_timeout(700)
    ck("可重新标注", "marked-y" in (first.get_attribute("class") or ""))

    print()
    print("=" * 80)
    print("⑥ 刷新后恢复")
    print("=" * 80)
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(700)
    pg.evaluate("p => pick(p, p.split('\\\\').pop())", AUDIO_DIR + "\\" + AUDIO_NAME)
    pg.wait_for_timeout(1000)
    pg.click("#go")
    pg.wait_for_function("() => !document.getElementById('go').disabled", timeout=180_000)
    pg.wait_for_timeout(900)
    ck("刷新后恢复 ✓ 标注",
       "marked-y" in (pg.locator("#hits .hit").first.get_attribute("class") or ""))
    ck("刷新后恢复 ✗ 标注",
       "marked-n" in (pg.locator("#hits .hit").nth(1).get_attribute("class") or ""))
    ck("刷新后计数仍在", "已标" in pg.inner_text("#fbstats"), pg.inner_text("#fbstats"))

    print()
    print("=" * 80)
    print("⑦ 导出")
    print("=" * 80)
    with pg.expect_download(timeout=15000) as dl:
        pg.click("#exp")
    d = dl.value
    p = d.path()
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    ck("导出成功", data.get("count", 0) >= 2, f"{data.get('count')} 条")
    ck("含正负样本", data.get("pos", 0) >= 1 and data.get("count", 0) - data.get("pos", 0) >= 1,
       f"{data.get('pos')} 真 / {data.get('count',0)-data.get('pos',0)} 假")
    feats_ok = all(len(r.get("feats", {})) >= 20 for r in data["rows"])
    ck("每条都带 21 个特征", feats_ok)

    pg.screenshot(path=r"D:\pyitme\onsei2lrc\_shot_player.png")
    ck("无 JS 报错", not errors, "; ".join(errors[:2]))
    br.close()

print()
print("=" * 80)
print("⑧ 落盘文件检查")
print("=" * 80)
lines = FB.read_text(encoding="utf-8").splitlines() if FB.exists() else []
new = [json.loads(l) for l in lines[before:] if l.strip()]
ck("写入了 JSONL", len(new) >= 3, f"新增 {len(new)} 行")
if new:
    r = [x for x in new if x["verdict"] == 1]
    ck("记录含 feats", r and len(r[0].get("feats", {})) == 21,
       f"{len(r[0].get('feats', {})) if r else 0} 个特征")
    ck("记录含原始分与概率", r and "raw" in r[0] and "prob" in r[0])

# 恢复测试前的状态
if backup is not None:
    FB.write_text(backup, encoding="utf-8")
    print("  （已恢复测试前的标注文件）")
elif FB.exists():
    FB.unlink()
    print("  （已删除测试产生的标注文件）")

print()
print("=" * 80)
ok = sum(1 for _, o, _ in checks if o)
print(f"  结果：{ok} / {len(checks)} 项通过")
bad = [n for n, o, _ in checks if not o]
if bad:
    print("  未通过：" + "、".join(bad))
print("=" * 80)
sys.exit(0 if not bad else 1)
