# B题四问代码与结果

项目代码已合并为 **q1.py、q2.py、q3.py、q4.py 四个可读文件**。公共模型位于 q1.py，其他文件使用普通 Python 模块导入共享函数；运行时不需要解压旧源码。第三方环境 `.venv` 不计入项目代码。

## GitHub 版本与首次安装

本仓库按用户选择发布轻量版本：包含四个代码文件、依赖清单、原始题目数据、参数、中文图表及计算汇总。全精度状态轨迹 `work/runs`、虚拟环境、逐候选缓存与旧压缩备份由 `.gitignore` 排除，完整文件继续保留在原工作目录。文中“全部历史记录”和 `archive` 的说明对应本地工作目录，不代表这些文件已上传。

```powershell
git clone https://github.com/syczk301/jianmoB.git
cd jianmoB
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment.lock
.\.venv\Scripts\python.exe q1.py check
```

不带参数运行四个文件可整理随库结果，`check` 和 `smoke` 可直接运行。分析报告中的运行编号用于追溯本地轨迹；`figures`、`tables`、`ice-analysis` 等涉及读取状态轨迹的命令，需要先恢复本地 `work/runs`，或依次重新运行对应校准、搜索和复算流程。已有中文图表和汇总表可直接查看，无需重新求解。

## 文件与结果

| 代码 | 内容 | 按题号整理的结果 |
| --- | --- | --- |
| [q1.py](q1.py) | 物性、有限体积守恒方程、积分器、实验校准、电压优化、冰量分析、表格与中文绘图 | [Q1 结果说明](results/q1/结果说明.md) |
| [q2.py](q2.py) | 恒流、线性与阶梯加载搜索，最低初温扫描和边界复算 | [Q2 结果说明](results/q2/结果说明.md) |
| [q3.py](q3.py) | 纯预热、协同加热、延长时限、能耗与负载资格核查 | [Q3 结果说明](results/q3/结果说明.md) |
| [q4.py](q4.py) | 预冷、离线参考电压表、动态控制、鲁棒性与离散化检查 | [Q4 结果说明](results/q4/结果说明.md) |

目录用途：

- `results/q1` 至 `results/q4`：供查阅的表格、中文图、分析说明及来源哈希清单。
- `work/outputs`：原有全部计算报告、参数、搜索汇总和中间结果。结果文件正文中的旧路径 `outputs/...` 现在对应这里。
- `work/runs`：全部历史工况的全精度状态轨迹、配置、指标；运行编号保持不变。
- `work/evaluation_ledger`：候选评估记录。
- `work/整理核查`：新旧数值等价报告、数据迁移哈希核查、原函数到四个文件的映射。
- `data`、`configs`：原始/处理后数据及参数登记。`data/original/B题.zip` 是原题的同哈希副本，可用于迁出本目录后的读取。
- `archive`：整理前源码压缩备份及此前交付压缩包。备份已逐文件校验，保留原源码注释与旧命令。

完整表格汇编保存在 [B题完整表格](work/outputs/完整表格/B题完整表格_当前候选与网格核查.md)。题设格式表是既有定稿快照；后续重新搜索时，以 `计算记录` 和 `计算表格` 中的运行编号为准，需据新记录更新题设表及其注释。

## 快速运行

在本目录打开 PowerShell，现有 Python 3.12 环境可直接运行：

```powershell
.\.venv\Scripts\python.exe q1.py
.\.venv\Scripts\python.exe q2.py
.\.venv\Scripts\python.exe q3.py
.\.venv\Scripts\python.exe q4.py
```

不带参数时仅按题号整理已有结果，避免意外启动长时间优化。`--help` 可查看每问命令。无需原 `src/fcstart` 或 `pip install -e .`。

```powershell
# 原有物理、守恒、约束和控制测试，已嵌入四个代码文件
.\.venv\Scripts\python.exe q1.py check
# 单片、五片模型短时真实积分及 Windows 多进程检查
.\.venv\Scripts\python.exe q1.py smoke --workers 2

# Q1 当前电压候选：拟合、复算、192格冰量、分析与图表
.\.venv\Scripts\python.exe q1.py optimize --budget 12 --workers 3
.\.venv\Scripts\python.exe q1.py replay --candidate shape_joint
.\.venv\Scripts\python.exe q1.py ice-refine --workers 2
.\.venv\Scripts\python.exe q1.py ice-analysis
.\.venv\Scripts\python.exe q1.py figures
.\.venv\Scripts\python.exe q1.py tables

# Q2 搜索和初温边界细化
.\.venv\Scripts\python.exe q2.py run --budget 80 --workers 4
.\.venv\Scripts\python.exe q2.py refine --workers 4
.\.venv\Scripts\python.exe q2.py replay

# Q3 搜索、时限扩展与网格复算
.\.venv\Scripts\python.exe q3.py run --budget 80 --workers 4
.\.venv\Scripts\python.exe q3.py extend --workers 4
.\.venv\Scripts\python.exe q3.py convergence
.\.venv\Scripts\python.exe q3.py qualify

# Q4 预冷、参考表、控制策略搜索与核查
.\.venv\Scripts\python.exe q4.py precool
.\.venv\Scripts\python.exe q4.py reference
.\.venv\Scripts\python.exe q4.py run --budget 6 --workers 4
.\.venv\Scripts\python.exe q4.py convergence
.\.venv\Scripts\python.exe q4.py extended-robustness --workers 4
```

以上计算命令会更新相应工作结果，按需执行。`q3.py extend` 保留原扩展流程，同时更新 Q4 的零功率对照。Q1 的旧模型校准入口为 `q1.py calibrate`；会重新选择旧模型参数，若使用，应重新运行依赖它的 Q2—Q4。

迁到新机器时创建 Python 3.12 环境后，用 `python -m pip install -r environment.lock` 安装依赖。四个文件应放在同一目录。绘图继续使用中文；默认 CPU 多进程，不声称已实现 GPU 求解。

## 本次整理的验证与结果边界

- 原 23 项测试通过。Q1 两组实验全时段、Q2 阶梯加载、Q3 加热及 Q4 动态控制代表性短时工况，与整理前代码的状态、电压、温度、冰量、守恒量逐元素完全相等；仅运行耗时不参与比较。
- 单片、五片及 Q2 批量 Windows 多进程检查通过。整理前全部数据另存哈希清单，移动后逐文件核对。
- Q1 沿用当前 `shape_joint` 电压候选，Q2—Q4 继续使用 `work/outputs/selected_model.json` 中旧冻结模型。本次没有改变物理方程、参数和已得结论。
- Q1 是联合校准回代，尚不能声称独立预测验证通过；96/192格冰峰值尚未收敛。Q2 未找到可行候选不等于不存在可行解；Q3、Q4 成功为当前模型和约束下的条件性结果。
- 合并后缓存的代码哈希由四个文件共同生成，因此新计算使用新的运行编号。旧轨迹保持原编号和原配置，不被标记为新代码生成。

旧源码、测试、整理辅助脚本只留在压缩备份中，不再作为散落的项目代码保留。
