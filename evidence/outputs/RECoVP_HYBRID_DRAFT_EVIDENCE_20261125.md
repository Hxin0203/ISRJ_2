# ReCoVP-Hybrid 论文草稿证据摘要

- 文档：`manuscript/ReCoVP_Hybrid_论文草稿_20261125.md`
- 草稿状态：研究草稿；seed 17 尚未完成 256 epochs 和固定 DEV 评测。
- 生成日期：2026-11-25
- 处理边界：本摘要和草稿均在本地处理；未调用 HTTP 水印服务、外部模型或文件写钩子。

## 1. 证据索引

| ID | 支持内容 | 本地证据 | 状态 |
|---|---|---|---|
| E-HYB-SOURCE | Hybrid 结构、Stage 1/2 前向、锚点、窗口、变量投影、复值门控 | `tools/vp_1103_recovp_hybrid.py`；SHA 见下表 | 已核对 |
| E-HYB-OBJECTIVE | B4 端点、脉压、damage、Stage 1 辅助项和总目标 | `tools/vp_1103_recovp_hybrid_objective.py`；`paper2/network.py`；SHA 见下表 | 已核对 |
| E-HYB-TAIL | 匹配滤波尾峰代理损失、有效距离单元、top-8、理想化 CA-CFAR 参考 | `tools/unet_tail_loss/objective.py` | 已核对 |
| E-HYB-PREFLIGHT | 参数量、有限样本等变误差、列置换、梯度覆盖、数据绑定 | `evidence/outputs/recovp_hybrid_seed17/PREFLIGHT.json` | PASS |
| E-HYB-CONFIG | 训练规模、优化器、学习率、恢复策略、设备和数据范围 | `evidence/outputs/recovp_hybrid_seed17/CONFIG.json` | 已核对 |
| E-HYB-STATUS | 当前 epoch、最近完整恢复点、训练集过程指标 | `evidence/outputs/recovp_hybrid_seed17/STATUS.json` | TRAINING |
| E-HYB-PROGRESS | epoch 20 完整记录、checkpoint SHA、恢复点和待办边界 | `evidence/outputs/recovp_hybrid_seed17/PROGRESS_20261125.md` | 已核对 |
| E-HYB-TEXT | 离线 clean-user-facing-text 审计规范和状态 | `evidence/outputs/recovp_hybrid_seed17/TEXT_AUDIT.json`；新草稿审计记录见下文 | 低密度、仅审计 |
| E-BASE-RSDR | Base/RSDR 的 DEV 描述性背景比较 | `evidence/outputs/RSDR_UNIFIED_20261115` 及既有稿件证据 | 仅背景 |

## 2. 源码与数据 SHA 绑定

以下 SHA 直接取自 Hybrid preflight/config 收据，供复核源码和数据身份使用：

| 文件 | SHA-256 |
|---|---|
| `tools/vp_1103_recovp_hybrid.py` | `8fa2a6625a08fb00ed1e33c472199fd6b1f89c4f80cf5ee71b07e25c62314c30` |
| `tools/vp_1103_recovp_hybrid_objective.py` | `caf19f45af39afbca9c6c1ebbc47bfb54144d6a0a27c1f70974488a28b4fd074` |
| `tools/unet_tail_loss/objective.py` | `69430d6acb597312770e997f1f4cafb39ad7e7bfe96f76eddcb25c29c6079c3d` |
| `paper2/network.py` | `121c709ce117e4eb000835c87fe6c16ceeab1142dabdd6d296ead5593ee87b30` |
| `data/train.npz`（预检收据名称） | `6148d097fecb0381e56227eb80cf1379e5f2750816ba65ce12f5457ee678f0fd` |
| `data/train.receipt.json` | `93567c81489b6e78b4120b6b4e1e5aedf56e08338d29c96b6236d992a269aed3` |
| `data/dev.npz`（预检收据名称） | `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1` |
| `data/dev.receipt.json` | `61e439503223258b969fcb8dd53a709bc5f60b1a128cde232ef645f98c1e2092` |

> 注：表中 `tools/vp_1103_recovp_hybrid_objective.py`、train/dev 收据的短路径/摘要字段按原收据记录；复核时以 `PREFLIGHT.json`、`CONFIG.json` 中的完整绑定为准。训练数据不随本任务提交到 GitHub。

## 3. 已核对的结构事实

- 输入：单向量 RAW_IQ，形状 `[B,2,1800]`；内部转为复数长度 1800。
- RMS：`scale = sqrt(mean(|y|^2))`，特征在归一化输入上计算，输出再乘回 scale。
- Stage 1：实值特征为幅度、固定匹配滤波幅度、一阶滞后乘积实部和虚部；宽度 `(13,26,44)`；观察值驱动 `K=16` 锚点；最小锚点间隔 6；有限时延修正；5/15/31 点平滑窗；复数变量投影与自适应正对角精度。
- Stage 2：输入为 Stage 1 残差和校正量；六类不变特征；无偏置复卷积；径向门控；宽度 `(9,13,17,21)`；瓶颈空洞率 `1,2,4`；残差/跳跃连接；全局统计量和两个物理统计量形成 scalar gate。
- 合成：`final = stage1_residual - stage2_delta`。
- 训练目标：`L_new(final)+0.05 L_tail(final)+0.25 [L_new(stage1)+0.05 L_tail(stage1)]`。
- 尾峰项：只在目标缺失状态；8 脉冲匹配滤波和慢时间 FFT；有效距离单元 `[36,1165)`；top-8；`N_CFAR_REFERENCE=436`；名义 `PFA=1e-6`；温度 3 dB softplus。

## 4. 轻量性与预检数值

| 项目 | 数值 | 解释 |
|---|---:|---|
| Stage 1 有效实参数 | 28,948 | 当前前向可训练参数 |
| Stage 2 有效实参数 | 28,577 | 当前前向可训练参数 |
| Hybrid 总有效实参数 | 57,525 | 低于 59,700 上限 |
| ReCoVP-Base | 58,943 | 同口径背景 |
| 参数匹配 U-Net | 59,714 | 同口径参数背景 |
| RSDR-Net | 3,017,877 | 同口径背景 |
| 最大受测等变相对误差 | `2.5422161797905574e-07` | 有限样本实现预检 |
| Stage 1 列置换误差 | `0.0` | 有限样本实现预检 |
| 有限非零梯度张量 | `100/100` | 真实训练 microbatch 预检 |

参数量不能直接推出延迟、显存、吞吐或能耗；当前没有 Hybrid 最终 DEV 延迟和最终性能结果。

## 5. 训练状态

- 服务器：B；目录：`D:/code/Xin/ISRJ_222`。
- physical GPU：1；UUID：`GPU-bb60dd3a-738e-2398-908f-7f77ac05549b`。
- 计划：256 epochs；seed 17；512 training parents；direct/repeat/cyclic；1536 records/epoch；batch 32；48 updates/epoch。
- 当前状态：`STATUS.json` 记录 epoch 21/256、update 24/48；最近完整提交 epoch 20。
- epoch 20 训练集记录：联合损失 `0.5955245466708826`，Final `0.1348079847327123`，Stage1 `1.8428662545823802`，单轮 `96.12274089991115 s`。
- epoch 20 checkpoint SHA-256：`8e831179df94f08dfb08676f419c42453d72633a8323f6387e9177c056b8fa75`。
- 训练恢复策略：每个完整 epoch 恢复点；每 32 epochs 里程碑；保存 optimizer/RNG 状态和 SHA 收据。
- 训练与 DEV 仅使用 TRAIN/DEV 绑定；不读取 CONFIRM 或历史 TEST。

训练集过程值不等于最终 DEV 检测率、虚警率或 Stage 2 泛化收益。

## 6. 背景比较的证据边界

已有 ReCoVP-Base 和 RSDR-Net 数值只能表述为同一 DEV 协议下的描述性比较：

| 模型 | JCR/dB | Pd/% | 帧 Pfa/% | absCPL | damage | 有效实参数 |
|---|---:|---:|---:|---:|---:|---:|
| ReCoVP-Base | 12.2564 | 84.8958 | 6.90104 | 0.175693 | 0.0716197 | 58,943 |
| RSDR-Net | 17.7605 | 86.9792 | 5.07812 | — | 0.0947035 | 3,017,877 |

训练 seed、预算、结构和数据协议不同；这些数值不能替代 Hybrid 的最终 DEV 结果，也不能支持全面优于或结构因果结论。

## 7. 待补结果清单

1. 固定 DEV256 的 Hybrid Final/Stage1Only 配对指标和 bootstrap 区间。
2. 复缩放、无干扰改变量、门控分布、SNR/JSR 分层和 CUDA 延迟。
3. Base、参数匹配 U-Net、RSDR 的统一对照与 Equal-PFA 比较。
4. 失败 parent 回查、CPU 延迟和完整 SUMMARY/REPORT/VERIFICATION。

在上述结果完成前，不得在稿件中写“Hybrid 已完成最终 DEV”“Hybrid 已优于任何模型”“已完成独立泛化验证”或“已完成最终性能排名”。

## 8. clean-user-facing-text 离线审计记录

- 技能：`C:/Users/admin/.codex/skills/clean-user-facing-text/SKILL.md`
- 处理模式：offline detect-only；no rewrite。
- 保护范围：公式、数学 Unicode、数值、单位、变量、代码、路径、URL、引用编号和限制均保留。
- 新草稿审计：在草稿创建并备份后运行 `inspect_text.py --audit --json`；结果写入 `evidence/outputs/RECoVP_HYBRID_DRAFT_TEXT_AUDIT_20261125.json`。
- 若结果为 low/medium，按技能要求只做核对，不为分数强行改写；中文文本的英文 stylometry 只作为未校准提示。
- 审计不声称通过 Turnitin/GPTZero，不证明人工写作，也不证明移除秘密水印。

## 9. 证据边界声明

本文草稿将 Stage 1/Stage 2 的结构、目标函数和预检结果写成已完成证据；将最终 DEV 性能、独立泛化、CFAR 等虚警优势和硬件效率写成待补或未建立。任何后续修改都应同步更新本摘要、来源 SHA 和审计记录。
