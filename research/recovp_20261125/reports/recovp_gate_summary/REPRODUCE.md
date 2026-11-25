# ReCoVP 当前复现与门禁

在独立工作树/输出归档中复现，保留已存在的负结果、源码和权重。只接受已核验的 TRAIN512、DEV256；这些入口不支持 CONFIRM/TEST。

## 原生指标与参照

规范分档与帧描述池化沿用 `snr_jsr_protocol/CONFIG.json`。直接读取生成器的原生 `snr_db/jsr_db`；SNR `<0 / [0,10) / >=10`、JSR `<0 / [0,15) / [15,30) / >=30` dB。有效威胁集合为 JSR≥15，另作汇总，不作为第五个互斥直方档。

Exp0 和 Exp2a 不读取 DEV；仅独立参照脚本使用 DEV 元数据与已存、校验通过的固定末轮 parent 指标。

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
python -B tools/vp_1103_recovp_references.py
python -B tools/vp_1103_exp0_anchor_audit.py
python -B tools/vp_1103_recovp_review.py --anchor-only
python -B tools/vp_1103_exp2a_softshare.py
python -B tools/vp_1103_recovp_train_noise.py
python -B tools/vp_1103_recovp_checks.py
python -B tools/vp_1103_exp1_preflight.py
python -B tools/vp_1103_recovp_review.py
python -B tools/vp_1103_recovp_verify.py
```

Exp1 固定同一 TRAIN16、seed17，共 3840 更新，不挑 epoch。源码中锁定初始/末步严格数学门禁、梯度、无扰保持与性能门槛。具体算术、数值保护和参数量见 `exp1_recovp_base_preflight/DESIGN.md`。

## 自动接续

在另一进程/会话启动接续器。等待期间不分配 CUDA 模型；只有 Exp1 全 PASS 且严格随机输入数学检查通过后才启动正式 GPU：

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
python -B tools/vp_1103_recovp_run.py
```

依序执行 Base seed17 DEV 止损、条件三种子、NoEquiv 单组和复缩放 DEV 派生对照；Exp2a 支持且 Base 验证通过后才执行 SoftShare 条件种子。任何预检失败保留 PAUSE，并把未启动的种子明确归档。独立确认不随自动接续解锁。

## CVaR 与计时

CVaR 使用真正的 parent 级 RU 目标，alpha=0.9、softplus 温度0.05、额外权重0.05。这些值事先固定，不使用 DEV 调参。每个 parent 的三类记录完整聚合。四格均从同种子新初始化、用相同 parent 批次重训 Lnew 控制及风险单元，避免把旧记录随机批次与新 parent 批次混成损失归因。

每轮512个独立 parent分成48次优化更新，每次实际前向 micro-batch最多8条 parent×类型记录。无梯度第一遍测完整 parent 损失，第二遍以解析 RU 导数回传；两遍之间不更新参数。这与当前确定性、无 BatchNorm/Dropout 的前向相容，梯度恒等单测在 `CVAR_UNIT_TEST.json`。CVaR 的 t 是另记的1个辅助实参数，初值0、同 Adam/调度、无权重衰减，不计作额外网络层。

风险配对 seed17 若至少一个骨干在预定 parent 区间显示明确收益、且两组风险单元均未触发安全线，则补齐整个2×2的共同seed29/41，保留其中负交互结果。否则保留完整seed17矩阵，明确不启动其余种子。

```powershell
python -B tools/benchmark_latency_unified.py --suite recovp --warmup 50 --repeats 200
```

统一入口完整计入 RMS、齐次提峰、float64 实数 CNN/特征/头、亚采样 FFT、相干特征、complex128 求解与 float32 写回。仅已实际训练的模型参与计时，未训练 SoftShare 明示为 `NOT_TRAINED_GATE`，不以随机权重冒充。原统一脚本版本另存私有源码归档。

最后在 Gate-ReCoVP 暂停。若 Exp1 CPU 未过关，后续 GPU、CVaR 与全模型计时均不执行；此时交付是带明确阻断状态的阶段性阴性收据，而非不存在的三种子 DEV 成绩。
