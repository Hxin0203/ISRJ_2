# ReCoVP 正式 GPU 双模型与全维度评测

本轮依据 2026-11-24 用户明确放行执行。旧 CPU `NO_GO_EXP1_PREFLIGHT`、源码与阴性结果保持原样；新运行位于 `outputs/recovp_formal_dev/`，不从 TRAIN16 过拟合检查点热启动。

## 已锁定训练协议

- RAW_IQ，E0_rect_k12；Base 与 NoEquiv 均 58,943 有效实参数。
- 原 512 TRAIN、256 DEV，哈希与原生 snr_db/jsr_db 校验；micro-batch 8、256 epochs、48 更新/epoch、seeds 17/29/41。
- 原 Adam、学习率调度与 `Lnew = endpoint + 0.1 pc + damage + 0.05 tail`；确定性 cuDNN、关闭 TF32。
- 实数特征/CNN 使用 float32，FFT/模板为 complex64；Gram 与 12×12 解析求解升至 complex128，写回 float32。恢复仍为 y-j_hat，参考仍为同噪声 S+E。
- 所有 257 个每轮检查点、优化器/RNG 与负结果保留；仅固定第 256 轮评分。

## 初次启动或进程退出后的接续

从 thoughtful-tesla 工作树执行；已有活动控制器时不要启动第二个控制器。锁会拒绝重复运行。

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:ISRJ2_CORRECTION_ARM='tail'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
python -B tools/vp_1103_recovp_gpu_quickcheck.py
python -B -u tools/vp_1103_recovp_gpu_run.py
```

快检的复缩放/置换门槛为输出相对误差 1e-5；NoEquiv 不要求相位等变，但参数预算、梯度、有限性与列置换要求相同。第一篇完整 RSDR 组门控不同于单脉冲前端。

当前 GPU 运行的实时进度以本目录 `STATUS.json` 为准。启动时已修复训练模块的 `ISRJ2_CORRECTION_ARM` 导入环境选择，首次导入失败的收据与日志保存在私有目录；该次没有开始任何参数更新或产生训练检查点。

## 自动执行顺序

1. Base/17 与 NoEquiv/17 都完整训练 256 轮，不受旧 CPU 性能门槛阻止。
2. 运行 `vp_1103_recovp_formal_score.py`，对两模型及 P0-Tail、UNet-Tail、RSDR-Net 进行 DEV 新指标评分、原生 SNR/JSR（含交叉档）统计、复缩放压力测试和 ReCoVP 三层漂移分解。
3. seed17 Winner 事先锁定排序：无扰安全优先，然后 Pfa 升序、Pd 降序、RD 残峰升序、JCR 降序、absCPL 升序、子空间角升序。只用 seed17 DEV，绝不扫描后重选。
4. Winner 若 `Pfa<=0.115 AND (mean_abs_CPL<=0.8 OR sin2_angle<=0.05) AND clean_NMSE<=5e-4`，自动补齐 Winner/29、Winner/41，并更新逐种子与种子均值。未完成或未解锁种子不当作 0 或排除后冒称三种子。
5. 无论竞争力门槛是否通过，冻结 Winner/17 完整运行第一篇对齐新扫描、绘图与统一计时。

不重训传统方法及历史基线。RSDR 通过既有适配器只读加载第一篇冻结权重，标注原训练 seed40312102、epoch32、每组件 16384 updates，与当前轻量模型不构成同训练预算实验。

数学快检与性能门槛分离。数值发散/执行错误仍会保存异常并停止；不会用坏模型伪造结果。新目录的 STOP/PAUSE 文件可在安全边界控制执行。旧目录的暂停标记不适用于这次明确授权的新运行。

完整扫描及复现说明见 [扫描套件](../recovp_isrj1_suite/REPRODUCE.md)。本轮不执行 SoftShare、CVaR、CONFIRM 或任何历史 TEST/Gate 数组。
