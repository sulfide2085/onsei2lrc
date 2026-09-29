# _ml/ 下的脚本说明

## 正式流程（用这两个）

| 脚本 | 作用 |
|---|---|
| 	rain_model.py | 训练 + 10 折验证 + 保存 climax_model.pkl |
| parse_gt.py / 
un_asr.py | 准备标注与转写 |

**评估口径以 	rain_model.py 为准** —— 它直接调用 climax_finder 的
merge_candidates + snap_to_peak，且保留所有官方标注音轨（含「0 次」的空轨）。

## 调研脚本（结论已写进 README，脚本保留作证据）

| 脚本 | 结论 |
|---|---|
| landmark_test.py | 换声学地标全部更差，能量局部极大值已是最优 |
| dist_weight.py | 按距离加权训练不划算 |
| peak_pick.py | 发现「最近峰更突出/更响/排名更高」→ 时间吸附的依据 |
| merge_test.py / greedy_merge_test.py | 选出合并窗口 30 秒、池大小 top+1 |
| kde_test.py / kde_hybrid.py | 概率云（KDE）测了 20+ 组参数，打不过合并+吸附 |
| conf_time.py / 	ime_error.py | 误差无系统性偏移；特征预测不了误差 |
| pair_analysis.py | 60% 的相邻候选是「一次高潮被拆成两个」 |
| lgo_rerun.py | 各算法在统一口径下重跑 |
| why_diff.py / 
epro_578.py | 查清 57.8% 与 48.9% 的差异来源 |

## ⚠️ 这些脚本里的一个已知偏差

merge_test.py 的 evaluate()、以及从它 exec 出来的若干脚本，用的是：

    if not gts: continue      # ← 错

官方标注里有 13 轨标着「0 次」（[] 空列表），这些音轨被整个跳过了。
它们是最难的负样本（模型照样输出 3 个候选，全是误报），
跳过它们会把精确率虚高约 11 个点。

**	rain_model.py 用的是 if g is None: continue，是对的。**
引用这些脚本的精确率时要留意这一点。
