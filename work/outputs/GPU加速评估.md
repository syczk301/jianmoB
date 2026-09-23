# GPU加速评估

可以迁移到GPU，但现有SciPy BDF程序尚未使用GPU。本机实测硬件为NVIDIA GeForce RTX 5070 Ti，显存16303 MiB，驱动595.97。本文件记录2026-09-23的检查，不承诺尚未测量的加速倍数。

## 实测瓶颈

`profile_cpu.py`绕过仿真缓存，对−20℃实验、14格MEA、单片模型进行一次完整CPU求解。带性能探针耗时11.99 s，同时有其他计算任务运行，因此此耗时不作为稳定的吞吐基准。

| 函数 | 调用次数 | 累计时间/s |
|---|---:|---:|
| RHS物理右端项 | 7046 | 6.918 |
| Jacobian有限差分 | 78 | 4.759 |
| 相态恢复 | 14243 | 4.330 |
| 物理事件判据 | 6233 | 3.787 |
| 稀疏LU分解 | 366 | 0.055 |

这些累计时间相互包含，不能求和。原始记录见 `cpu_profile.json`。在这个样本中，单独替换线性分解库不会解决主要耗时；需同时减少重复物性与事件计算、优化Jacobian求导。不能直接将这一单片结果推广为所有五片细网格工况的耗时分布。

## 适合的实现路径

1. 保留当前CPU求解器和已保存轨迹作为独立校验基准。先对14/48/96格、开环与反馈各取代表工况做不带性能探针的重复计时。
2. 收紧当前整片连接的保守Jacobian稀疏结构，保留电压反应热产生的非局部依赖；同时测试解析/自动微分Jacobian。现有方向导数测试应覆盖每次稀疏结构修改。
3. 在WSL2/Linux独立环境实现JAX双精度物性、通量和状态恢复，再试Diffrax的Kvaerno隐式刚性求解器。JAX官方不支持原生Windows CUDA；本轮未验证本机WSL中的JAX/CUDA运行环境。[JAX安装支持](https://docs.jax.dev/en/latest/installation.html)、[JAX构建说明](https://docs.jax.dev/en/latest/developer.html)、[Diffrax求解器选择](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/)
4. 优先尝试相同网格的批量参数工况。分别测batch 1、8、16的峰值显存、预编译耗时和稳态吞吐；按事件终止分歧分组，防止一个困难工况拖住整批。控制反馈的因果采样、欠压锁存、分段输入和成功/实际关热双时刻必须原样保留。
5. 对CPU与GPU逐工况比较状态、守恒残差、最低电压、启动时间、辅助能耗及物理失败类型。原有1 mV、0.02 K等门槛及局部冰峰值检查应继续使用；近可行边界的候选须独立重放。单条轨迹是否更快必须实测。

不建议将现有NumPy数组直接换成CuPy后宣称整个BDF求解已在GPU上运行；应按SciPy的求解器后端支持逐项确认。[SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

GPU加速不会修复当前跨温度电压波形失配。模型预测资格与计算速度是两个独立验收项。
