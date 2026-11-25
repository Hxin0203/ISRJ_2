# 第一篇对齐冻结扫描复现

本任务明确授权独立新合成扫描，用于冻结模型能力表征；不解锁 CONFIRM 或读取任何旧 TEST/Gate 数组，不把这些扫描称作最终独立确认。

## 先冻结计划与核验适配器

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -B tools/vp_1103_recovp_radar_metrics.py
python -B tools/vp_1103_recovp_suite_checks.py
python -B tools/vp_1103_recovp_isrj1_suite.py --prepare
```

`PLAN.json` 在正式模型结果出现之前锁定生成器、方法、条件、样本量及源码。新的随机命名空间为 `scan`、role code 8、seed 2026112407。模型只读取观测；S/J/E 只用于评测、明确标注的 Reference 控制与诊断。

## 全部 579 工况、2811 方法工况

| 模块 | 条件数 | 每条件 parent | 方法 |
|---|---:|---:|---|
| 固定典型 ISRRJ | 1 | 1 | 原始、UNet、P0、RSDR、Winner |
| 无模糊相参 | 3 | 16 | 同上 |
| SNR×JSR | 310 | 512 | 原始、含噪 Reference、P0、Winner |
| 切片宽度×SNR | 55 | 128 | 原始、P0、Winner |
| 跨方法 14×4×3 类 | 168 | 256 | 全部 7 种方法 |
| 无干扰 14×3 组 | 42 | 512 | 原始、UNet、P0、RSDR、Winner |

第一篇正文写 310 格，生成脚本采用 SNR=-30…0、步长1 dB 共31档；JSR=-10…35、步长5 dB 共10档。任务书列出的全范围步长5和过渡区步长1只是两个绘图子集，不能相加当成310个不同工况。本轮运行完整310格并额外导出两种子集视图。

跨方法 SNR 档固定为 `-30,-28,-26,-25,-24,-23,-22,-20,-15,-10,-5,0,5,10`；JSR 为 `0,10,20,35`。三类均覆盖完整56格。宽度取 `6,18,36,54,72` 点（0.1/0.3/0.6/0.9/1.2 us），M=2、周期=3w+1；该实验明确使用组总能量 JSR_E=20 dB，不能混写成 active-power JSR。

目标/噪声/干扰几何在同组工况与所有方法间共用；记录配对 invariant hash。每个工况按4个 parent分块恢复；传统方法4个 CPU worker，各 parent独立拟合。八脉冲、四状态、类型与工况不能扩张独立样本数。跨SNR的累计3584为14×256描述性试验次数，配对区间先在同一个256-parent内平均14个工况再 bootstrap。

## 训练完成后的自动或手工接续

```powershell
python -B tools/vp_1103_recovp_formal_score.py
python -B tools/vp_1103_recovp_isrj1_suite.py
```

必须先有两个正式 seed17 的第256轮检查点与 `recovp_formal_dev/WINNER.json`。控制器已经编排以上接续；活动控制器运行时不要重复启动。每个方法工况和每个 parent块均有身份、源码/输入绑定及哈希，重复命令校验后复用，不重选赢家。

## 指标与统计解释

主指标严格依任务公式：JCR、有效9032个RD检测单元的残余峰值过冲、signed/abs CPL及sin²目标角、signed/abs BVR00/BVR01。兼容 eU/eS/e0/cS/SJRIF 和实际 CA-CFAR 指标保留在 JSON/CSV；无干扰 JCR 因J=0记为未定义，不冒充0 dB。

新指标的数学解释同时注明边界：匹配滤波通常不是酉变换；相关 CUT 不能用独立事件乘积给出严格整帧虚警公式；RD残峰是风险代理而非普适CFAR上界；BVR是能量比，不证明背景白性或恒虚警性。CPL/角度对全局相位旋转可能不敏感，因此复增益实虚部与脉间相位/多普勒指标同时保留。反例与公式单测见 `METRIC_UNIT_TESTS.json`/`HARNESS_CHECKS.json`。

每个工况给 parent bootstrap95% CI、帧事件二项区间及实际 CUT 计数。CUT不视为独立伯努利；零事件不当作真实概率为零，附基于独立 parent 的保守上界。跨大量工况区间为描述性，未作多重比较校正；不据此声称普遍优越或创新确认。

## 图与计时

扫描完成后自动渲染高清PNG（300 dpi）与PDF，保留CSV源表。热力图不插值，同指标跨方法/子图使用相同色标；曲线不平滑，展示点位parent区间。典型场景固定parent、首个受扰脉冲，局部时域窗为目标支撑加两侧24点，禁止按结果挑例。多普勒4096点零填充只用于谱插值，原生仍为8脉冲。

```powershell
python -B tools/recovp_gpu/benchmark_latency_unified.py
```

该版本化入口保持旧计时源码收据不变，复用 `tools/benchmark_latency_unified.py::measure` 的同一同步内核。CPU线程2、同设备、TF32关闭、预热50、重复200、batch=1组、报告原生单脉冲与整8脉冲P50/P95和显存。输入已在设备上，传输成本统一排除；归一、提峰、FFT、相干特征、实CNN、complex128求解、写回全部计入。

RSDR原生门控要求8脉冲，单脉冲栏为不适用，另报组时延/8的摊销值，绝不把它冒充真实单脉冲测量。张量适配与原RSDR恢复器先做相对误差1e-5以内的核验。MACs只覆盖Conv1d/Linear，明确排除FFT/求解等；端到端计时覆盖全部算子。

完成579工况、图与计时后写最终 VERIFICATION.json，在 Gate-ReCoVP 暂停人工审阅。之前的 `WAITING`/`NOT_EVALUATED` 状态不代表已获得成绩。
