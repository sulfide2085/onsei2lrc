# -*- coding: utf-8 -*-
"""对照评测：把本工具产出的 LRC 与参考译文 LRC 做客观比对

指标：
  * 行数 / 时间跨度
  * 时间轴一致性：每条参考行能否在 ±tol 秒内找到对应行（覆盖率），以及匹配行的平均/中位时间差
  * 若两边都有译文：译文长度比、完全相同的行占比

用法：
    python eval_compare.py 我的.lrc 参考.lrc [--tol 3.0]
"""
import argparse
import re
import sys
import unicodedata

TS = re.compile(r"^\[(\d+):(\d+(?:[.:]\d+)?)\](.*)$")


def parse_lrc(path):
    rows = []
    for line in open(path, encoding="utf-8-sig", errors="replace"):
        m = TS.match(line.strip())
        if not m:
            continue
        mm, ss, text = m.group(1), m.group(2), m.group(3).strip()
        ss = ss.replace(":", ".")
        t = int(mm) * 60 + float(ss)
        if text:
            rows.append((t, text))
    return rows


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[\s、。，．！？!?…‥・「」『』（）()【】\[\]〜~ー―—\-]+", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mine")
    ap.add_argument("ref")
    ap.add_argument("--tol", type=float, default=3.0, help="时间匹配容差（秒）")
    a = ap.parse_args()

    mine, ref = parse_lrc(a.mine), parse_lrc(a.ref)
    print(f"我的输出 : {len(mine):4d} 行   跨度 {mine[0][0]:.1f}s - {mine[-1][0]:.1f}s" if mine else "我的输出: 空")
    print(f"参考译文 : {len(ref):4d} 行   跨度 {ref[0][0]:.1f}s - {ref[-1][0]:.1f}s" if ref else "参考译文: 空")
    if not mine or not ref:
        return 1

    # 参考行 → 最近的我方行
    deltas, matched = [], 0
    for t, _ in ref:
        best = min(mine, key=lambda r: abs(r[0] - t))
        d = best[0] - t
        if abs(d) <= a.tol:
            matched += 1
            deltas.append(d)
    deltas.sort()
    print(f"\n时间轴一致性（容差 ±{a.tol}s）")
    print(f"  参考行命中率 : {matched}/{len(ref)} = {matched/len(ref)*100:.1f}%")
    if deltas:
        n = len(deltas)
        mean = sum(deltas) / n
        med = deltas[n // 2]
        p90 = deltas[int(n * 0.9)]
        print(f"  时间差 平均 {mean:+.2f}s  中位 {med:+.2f}s  P90 {p90:+.2f}s")
        print(f"  绝对差 平均 {sum(abs(d) for d in deltas)/n:.2f}s  最大 {max(abs(d) for d in deltas):.2f}s")

    # 覆盖率：我方行里有多少条能对上参考行（防止我方碎成太多行）
    rev = 0
    for t, _ in mine:
        if abs(min(ref, key=lambda r: abs(r[0] - t))[0] - t) <= a.tol:
            rev += 1
    print(f"  我方行被参考覆盖 : {rev}/{len(mine)} = {rev/len(mine)*100:.1f}%")

    # 译文对比（两边都是中文时才有意义）
    if ref and mine and re.search(r"[\u4e00-\u9fff]", ref[0][1]):
        pairs = []
        for t, txt in ref:
            cand = min(mine, key=lambda r: abs(r[0] - t))
            if abs(cand[0] - t) <= a.tol:
                pairs.append((txt, cand[1]))
        if pairs:
            ratio = [len(norm(m)) / max(1, len(norm(r))) for r, m in pairs]
            same = sum(1 for r, m in pairs if norm(r) == norm(m))
            print(f"\n译文对比（{len(pairs)} 对）")
            print(f"  长度比 平均 {sum(ratio)/len(ratio):.2f}（1.0=长度相当，<1 说明我的译文更短）")
            print(f"  完全相同行 : {same} ({same/len(pairs)*100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
