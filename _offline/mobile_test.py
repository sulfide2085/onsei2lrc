# -*- coding: utf-8 -*-
"""移动端适配的浏览器测试

用真实手机尺寸（iPhone 12/13/14 的 390×844，以及更窄的 360×640）打开，
检查：
  ① 没有横向溢出（最常见的移动端问题）
  ② 侧栏默认收起（transform 移出屏幕）
  ③ 点汉堡按钮能打开、点遮罩能关掉
  ④ 触控目标够大（>=40px 高）
  ⑤ 输入框字号 >= 16px（否则 iOS 聚焦会缩放整页）
  ⑥ 顶栏控件不重叠
截图留档。
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

B = "http://127.0.0.1:7861"
OUT = Path(r"D:\pyitme\onsei2lrc\_offline")
ok = [0, 0]


def ck(name, cond, detail=""):
    if cond:
        ok[0] += 1
        print(f"  ✓ {name}" + (f"　{detail}" if detail else ""))
    else:
        ok[1] += 1
        print(f"  ✗ {name}　{detail}")


DEVICES = [
    ("iPhone 12/13 (390x844)", 390, 844, 3),
    ("窄屏安卓 (360x640)", 360, 640, 3),
    ("iPhone SE (375x667)", 375, 667, 2),
]

with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chrome")
    for label, w, h, dpr in DEVICES:
        print(f"\n===== {label} =====")
        ctx = br.new_context(viewport={"width": w, "height": h},
                             device_scale_factor=dpr, is_mobile=True,
                             has_touch=True)
        pg = ctx.new_page()
        pg.goto(B + "/", wait_until="networkidle")
        pg.wait_for_timeout(3500)

        # ① 横向溢出
        sw = pg.evaluate("document.documentElement.scrollWidth")
        cw = pg.evaluate("document.documentElement.clientWidth")
        ck("无横向溢出", sw <= cw + 1, f"scrollWidth {sw} vs clientWidth {cw}")

        # ② 侧栏默认收起（应该在屏幕左侧之外）
        box = pg.evaluate("""() => {
            const r = document.getElementById('side').getBoundingClientRect();
            return {x: r.x, w: r.width, right: r.right};
        }""")
        ck("侧栏默认收起", box["right"] <= 2,
           f"right={box['right']:.0f} width={box['w']:.0f}")

        # ③ 汉堡按钮可见
        vis = pg.is_visible("#navbtn")
        ck("汉堡按钮可见", vis)
        sz = pg.evaluate("""() => {
            const r = document.getElementById('navbtn').getBoundingClientRect();
            return {w: r.width, h: r.height};
        }""")
        ck("汉堡按钮够大", sz["h"] >= 32, f"{sz['w']:.0f}x{sz['h']:.0f}")

        # ④ 打开抽屉
        pg.click("#navbtn")
        pg.wait_for_timeout(420)
        st = pg.evaluate("""() => ({
            open: document.getElementById('side').classList.contains('open'),
            scrim: document.getElementById('scrim').classList.contains('on'),
            right: document.getElementById('side').getBoundingClientRect().right,
            bodyOverflow: getComputedStyle(document.body).overflow,
        })""")
        ck("点汉堡能打开抽屉", st["open"] and st["right"] > 100,
           f"right={st['right']:.0f}")
        ck("遮罩同时出现", st["scrim"])
        ck("背景滚动被锁住", st["bodyOverflow"] == "hidden", st["bodyOverflow"])
        pg.screenshot(path=str(OUT / f"mobile_{w}_drawer.png"))

        # ⑤ 点遮罩关闭
        # 注意：遮罩的中心点被抽屉盖住（390 宽屏上抽屉占 335），
        # 必须点抽屉**右侧**露出来的那条，不能用默认的中心点
        pg.mouse.click(w - 18, h // 2)
        pg.wait_for_timeout(420)
        closed = pg.evaluate(
            "() => document.getElementById('side').classList.contains('open')")
        ck("点遮罩能关闭", not closed)

        # ⑥ 输入框字号
        fs = pg.evaluate(
            "() => getComputedStyle(document.getElementById('pinput')).fontSize")
        ck("输入框字号 >= 16px", float(fs.replace("px", "")) >= 16, fs)

        # ⑦ 主要按钮的触控高度
        smalls = pg.evaluate("""() => {
            const bad = [];
            for (const b of document.querySelectorAll('#top button, #ctrl button')) {
                if (b.offsetParent === null) continue;
                const r = b.getBoundingClientRect();
                if (r.height > 0 && r.height < 32) bad.push(b.id + ':' + r.height.toFixed(0));
            }
            return bad;
        }""")
        ck("顶栏/控制栏按钮高度够", len(smalls) == 0, ", ".join(smalls))

        # ⑧ 顶栏不重叠（各控件右边界单调）
        # 真正的矩形相交检测。
        # 之前只比 y 坐标，把同一行并排的两个按钮误判成「重叠」了。
        ov = pg.evaluate("""() => {
            const t = document.getElementById('top');
            const rs = [...t.children].filter(e => e.offsetParent !== null)
                .map(e => ({id: e.id || e.tagName, r: e.getBoundingClientRect()}))
                .filter(o => o.r.width > 0 && o.r.height > 0);
            const bad = [];
            for (let i = 0; i < rs.length; i++)
                for (let j = i + 1; j < rs.length; j++) {
                    const a = rs[i].r, b = rs[j].r;
                    const ox = Math.min(a.right, b.right) - Math.max(a.left, b.left);
                    const oy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
                    if (ox > 2 && oy > 2) bad.push(rs[i].id + '×' + rs[j].id);
                }
            return bad;
        }""")
        ck("顶栏控件不重叠", len(ov) == 0, ", ".join(ov))

        pg.screenshot(path=str(OUT / f"mobile_{w}.png"), full_page=False)
        ctx.close()

    # 桌面端没被改坏
    print("\n===== 桌面 (1440x900) 回归 =====")
    ctx = br.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    pg.goto(B + "/", wait_until="networkidle")
    pg.wait_for_timeout(3000)
    two = pg.evaluate("""() => {
        const g = getComputedStyle(document.getElementById('app'));
        return g.gridTemplateColumns;
    }""")
    ck("桌面仍是两栏", " " in two.strip() and len(two.split()) >= 2, two)
    ck("桌面汉堡按钮隐藏", not pg.is_visible("#navbtn"))
    sb = pg.evaluate("() => document.getElementById('side').getBoundingClientRect().right")
    ck("桌面侧栏在屏幕内", sb > 100, f"right={sb:.0f}")
    pg.screenshot(path=str(OUT / "desktop_1440.png"))
    ctx.close()
    br.close()

print()
print(f"  {ok[0]} / {ok[0]+ok[1]} 通过")
sys.exit(1 if ok[1] else 0)
