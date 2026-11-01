# 支撑材料文件列表

> 运行环境：Python 3.10+，需 numpy / scipy / matplotlib / pandas / openpyxl
> 完整复现：进入 `程序代码/`后执行 `python run.py`，总耗时约 **50~70 分钟**
>
> （首次运行前建议清空 `results/*_cache.json`）
>
> 运行后自动生成 `程序代码/figures/`（20 张）与 `程序代码/results/`（`result1~4.xlsx` 及各项汇总数据）。

---

## 一、程序代码

### 主流程

| 文件               | 说明                                                         | 产出                                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `common.py`        | 公共模块：附件读取、`Chamber` 烘房工况、`Props` 物性公式、非均匀网格 `make_grid`、顶点型有限体积求解器 `FVSolver`（Kirchhoff 变换 + 变步长 BDF2 + Newton/Picard 迭代）、`simulate()` 推进驱动、温度解析解 `analytic_T_q1`、结果写出 `write_result_workbook` | —                                                            |
| `data_analysis.py` | 附件1 两阶段统计划分（趋势检验 + 均值检验 + ADF 平稳性）     | `data_summary.json`、`innov_stationarity.json`、`data_chamber.png` |
| `q1.py`            | 问题1：温度 Bessel 级数解析解 + 水分数值解（不耦合）         | `result1.xlsx`、`q1_summary.json`、`q1_fields.npz`、`q1_curves.png`、`q1_profiles.png` |
| `q2.py`            | 问题2：变物性热质耦合模型（0~3 h 每秒输出）                  | `result2.xlsx`、`q2_summary.json`、`q2_fields.npz`、`q2_curves.png`、`q2_maps.png` |
| `q3.py`            | 问题3：烘干判据 + 60 s 粗搜 / 0.05 s 细回溯定位 t\*          | `result3.xlsx`、`q3_summary.json`、`q3_fields.npz`、`q3_curves.png`、`q3_profiles.png` |
| `q4.py`            | 问题4：物质坐标 ξ=r/R(t) 收缩模型（PCHIP 半径插值）          | `result4.xlsx`、`q4_summary.json`、`q4_fields.npz`、`q4_curves.png`、`q4_profiles.png` |

### 模型检验

| 文件           | 说明                                                         | 产出                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `checks.py`    | 数值格式检验、网格与步长收敛性、Weibull 光滑拟合噪声检验、六参数 ±10% 灵敏度分析 | `verify_q1.json`、`conv_q3.json`、`verify_noise.json`、`sensitivity.json`、`verify_weibull.png`、`sens_tornado.png` |
| `verify_2d.py` | 二维有限长圆柱模型端部效应检验（scipy 稀疏 LU）              | `verify_2d.json`、`verify_2d.npz`、`v_2d_field.png`          |

### 创新拓展

| 文件             | 说明                                                         | 产出                                                         |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `innov_model.py` | 收缩本构 (R/R₀)²=1−b(C₀−C_s) 反演标定与自洽预测              | `innov_q4_fields.npz`、`innov_q4noshrink.npz`、`innov_const_cache.json`、`innov_constitutive.json` |
| `plots.py`       | 收缩本构三联图、问题3/4 三维时空曲面、横截面圆盘序列、四问成果看板 | `innov_3d_q3.png`、`innov_disks_q3.png`、`innov_3d_q4.png`、`innov_disks_q4.png`、`innov_dashboard.png`、`innov_figures.json`、`innov_constitutive.png` |
| `innov_study.py` | 蒙特卡洛 60 次模拟；扩散系数与边界系数的扩展灵敏度           | `innov_mc.json`、`innov_sens_ext.json`、`innov_mc.png`、`innov_tornado_ext.png` |

### 附件1 附件2

- `附件1.xlsx`：烘房温度与水分浓度实测数据（0~14400 s，241 组）
- `附件2.xlsx`：烘干过程药材半径实测数据（0~72 h，每 0.5 h 一组）

### 总控

| 文件     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| `run.py` | 按依赖顺序依次运行上述所有模块。**顺序不可调换**：`innov_model.py` 必须先于 `plots.py`（后者读取前者生成的 `innov_q4_fields.npz`、`innov_const_cache.json`、`innov_constitutive.json`） |

---

## 二、图像合集（20 张，300 dpi）

| 程序文件                 | 生成脚本           | 论文图号                    |
| ------------------------ | ------------------ | --------------------------- |
| `innov_dashboard.png`    | `plots.py`         | 图2 四问结果总览            |
| `data_chamber.png`       | `data_analysis.py` | 图3 附件1工况与两阶段划分   |
| `q1_curves.png`          | `q1.py`            | 图4 问题1各位值随时间       |
| `q1_profiles.png`        | `q1.py`            | 图5 问题1径向分布           |
| `q2_curves.png`          | `q2.py`            | 图6 问题2各位值随时间       |
| `q2_maps.png`            | `q2.py`            | 图7 问题2时空分布           |
| `q3_curves.png`          | `q3.py`            | 图8 问题3水分变化+干燥速率  |
| `q3_profiles.png`        | `q3.py`            | 图9 问题3径向分布           |
| `innov_3d_q3.png`        | `plots.py`         | 图10 问题3三维时空曲面      |
| `innov_disks_q3.png`     | `plots.py`         | 图11 问题3横截面演化        |
| `innov_mc.png`           | `innov_study.py`   | 图12 蒙特卡洛               |
| `q4_curves.png`          | `q4.py`            | 图13 问题4半径变化+收缩影响 |
| `q4_profiles.png`        | `q4.py`            | 图14 问题4径向分布          |
| `innov_3d_q4.png`        | `plots.py`         | 图15 问题4三维时空曲面      |
| `innov_disks_q4.png`     | `plots.py`         | 图16 问题4收缩圆盘          |
| `innov_constitutive.png` | `plots.py`         | 图17 收缩本构标定与自洽预测 |
| `v_2d_field.png`         | `verify_2d.py`     | 图18 二维端部效应验证       |
| `verify_weibull.png`     | `checks.py`        | 图19 Weibull 拟合           |
| `sens_tornado.png`       | `checks.py`        | 图20 参数灵敏度             |
| `innov_tornado_ext.png`  | `innov_study.py`   | 图21 扩散结构参数灵敏度     |

> 说明：论文图1（技术路线）为单独绘制，不在此 20 张内。

---

## 三、表格合集

### 数值结果

- `result1.xlsx` ~ `result4.xlsx`：四问数值结果，对应论文表1~表4、表5、表6
  - **注意**：`result1.xlsx`、`result2.xlsx` 内含三个 sheet（`Sheet`、`温度`、`水分浓度`），数据在**后两个**标签页；`Sheet` 为新建工作簿时残留的空白页，Excel 默认激活它，看起来像空表。`result3.xlsx`、`result4.xlsx` 单 sheet，打开即见数据。

---

## 四、题目数据文件

- `附件1.xlsx`：烘房温度与水分浓度实测数据（0~14400 s，241 组）
- `附件2.xlsx`：烘干过程药材半径实测数据（0~72 h，每 0.5 h 一组）

---

## 五、AI工具使用详情

* AI工具使用详情.pdf

## 六、复现方式

```bash
# 1) 进入程序代码目录（该目录下已含 附件1.xlsx、附件2.xlsx）
cd 程序代码

# 2) 首次运行或修改参数后，清空缓存
rm -f results/*_cache.json

# 3) 运行全部模块（顺序不可调换）
python run.py
```

### 注意事项

1. **图数**：`run.py` 默认输出 **20 张** `figures/*.png`，与论文图2~图21 一一对应。
3. **`result1/2.xlsx` 空白页**：数据在"温度""水分浓度"标签页，`Sheet` 页为残留空白，不影响结果。
5. **运行时间**：完整运行约 50~70 分钟；中断后重跑会利用缓存续算。

