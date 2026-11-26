# ReCoVP-Hybrid seed 17：DEV256 阶段性复核

**复核状态：部分完成；保持人工审阅暂停。** 本包整理了已封存的 Hybrid 核心 DEV 结果和本地可读的同集基线摘要。跨模型配对 bootstrap、等虚警、Hybrid 分类型/高 JSR 分层、Top-10 失败样本、CPU 延迟及共享雷达物理指标仍未完成，不能将本轮称为完整的 full review。

## 训练与资产身份

Hybrid seed 17 已完成 256/256 epochs、12,288 次更新，训练完成收据记录 257 个恢复点（epoch 000 至 256）。模型有效实参数为 57,525，低于 59,700 上限。最终 checkpoint SHA-256 为 `e0b6cb1be9559ed5e75a9ce3c4e178e85aba22f0cec753f52d3f523cbaf05212`。

Hybrid 固定评测使用 256 个 DEV parent，每个 parent 含 3 个干扰变体。DEV SHA-256 为 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`，与 Base、P0、UNet 和 RSDR 冻结归档中的 DEV 身份一致。Hybrid 核心结果 `dev_epoch_256_v1.json` 的 SHA-256 为 `877cfe7f6baf4fb3a59bff58da4a344e5437e43f000b3b3a9808a47acbda7df1`。

## Hybrid 核心结果

| 指标 | Hybrid Final | parent bootstrap 95% CI |
|---|---:|---:|
| `u_nmse` | 0.04312 | [0.03706, 0.04950] |
| `frame_pfa_jammer` | 5.60% | [3.91%, 7.42%] |
| `Pd` | 87.11% | [83.20%, 90.89%] |
| `damage` | 0.05967 | [0.05173, 0.06871] |
| `no_jammer_pooled_nmse` | 4.57×10⁻⁶ | [2.22×10⁻⁶, 8.26×10⁻⁶] |
| `SJRI-F` | 11.097 dB | [10.288, 11.853] dB |
| false-peak suppression | 13.460 dB | [12.498, 14.414] dB |
| amplitude ratio | 0.9152 | [0.9082, 0.9219] |

相对同一 checkpoint 的 Stage1Only，Hybrid Final 的 `u_nmse` 差值为 −0.35873，`damage` 为 −0.06591，`SJRI-F` 为 +0.6801 dB，jammer frame Pfa 为 −1.823 个百分点，false-peak suppression 为 +1.669 dB。这五项的 256-parent 配对 95% CI 均不跨零。`Pd` 点估计增加 1.69 个百分点，但差值 CI 为 [−0.13, +3.65] 个百分点，包含零，因此当前不能说检测率显著提升。

## 同 DEV 基线的描述性对照

下表给出 DEV256 同集、单种子点估计。每个模型单独的 parent bootstrap 区间已保存在来源收据中；Hybrid 对其它模型的**配对差值区间未取得**，所以这些点差不能视为显著性检验。

| 模型 | 有效实参数 | `u_nmse` ↓ | jammer frame Pfa ↓ | `Pd` ↑ | `damage` ↓ | `SJRI-F` ↑ | 无干扰合并 NMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReCoVP-Hybrid / 17 | 57,525 | **0.04312** | **5.60%** | 87.11% | 0.05967 | **11.097 dB** | **4.57×10⁻⁶** |
| ReCoVP-Base / 17 | 58,943 | 0.49665 | 6.90% | 84.90% | 0.07162 | 10.244 dB | 1.94×10⁻⁶ |
| P0-NoDechirp-Tail / 17 | 59,714 | 0.04866 | 10.29% | **87.63%** | **0.04479** | 10.807 dB | 9.19×10⁻⁵ |
| UNet-Tail / 17 | 59,714 | 0.05449 | 10.81% | 86.46% | 0.05078 | 10.657 dB | 1.17×10⁻⁴ |
| RSDR-Net / native seed 40312102 | 3,017,877 | **0.03343** | **5.08%** | 86.98% | 0.09470 | **11.338 dB** | 1.35×10⁻⁵ |

Hybrid 对 Base 的恢复误差、jammer frame Pfa、Pd、damage 和 SJRI-F 点估计均较好；其无干扰合并 NMSE 则略高于 Base。相对 P0 和 UNet，Hybrid 的 `u_nmse`、jammer frame Pfa 和 SJRI-F 点估计更好，但 P0/UNet 的目标 `damage` 更低，且 P0 的 Pd 点估计更高。相对 RSDR，Hybrid 参数约少 52.5 倍，目标损伤较低，Pd 点估计略高；RSDR 的 `u_nmse`、Pfa 和 SJRI-F 点估计更好。RSDR 是不同模型、不同训练预算的参照，不属于匹配预算对照。

因此，现有数据支持“Hybrid 在该 DEV256 单种子上显著改善了 Stage1Only 的恢复和干扰抑制表现，并在若干点估计指标上优于同量级基线”。它不支持“全面压制所有轻量基线或大模型”的结论，也不足以确定跨模型差异显著。

## Repeat 高 JSR 参考

冻结基线归档中 `repeat / JSR ≥ 30 dB` 有 133 个 parent。Base 的 `Pd` 为 58.65%、frame Pfa 为 15.04%、JCR 为 11.11 dB、BVR01 为 3.20 dB；P0 对应为 79.70%、12.78%、21.18 dB、0.026 dB；UNet 为 77.44%、13.53%、20.59 dB、0.027 dB；RSDR 为 82.71%、7.52%、22.07 dB、−0.153 dB。Hybrid 的对应分层行没有出现在本机公开 JSON 中，不能据此声称 repeat 重灾区已经翻盘。

## 等变性、安全门和推理延迟

预检收据中的有限复缩放测试通过：Stage 1/Stage 2/最终输出的最大相对误差为 `2.54×10⁻⁷`，锚点列置换误差为 0。另一个覆盖 DEV parent 的随机复缩放汇总，其输出相对 L2 误差均值为 `2.33×10⁻⁵`、P95 为 `8.72×10⁻⁵`，高于严格 `1×10⁻⁵` 界限；同一测试中的 Pd 和 Pfa 变化为 0。应区分预检有限样本通过与 DEV 聚合误差偏高这两项证据，不能将后者表述为通过严格误差门槛。

安全门 α 的均值在 Y00/Y10 为 0.00045/0.00053，在 Y01/Y11 为 0.02481/0.02301。P90 分别为 0.00048、0.00076、0.09787 和 0.09273。这与无干扰时门控较低、受扰状态中部分样本门控增大的现象一致，但仅是分布描述。

现有 CUDA event 结果为 1 pulse P50/P95 27.86/29.26 ms、8 pulse 29.28/30.66 ms。尚无 CPU 数据、其它模型同机延迟或 Stage 1/Stage 2 分项耗时，故不能得出实时性或瓶颈归因结论。

## 未完成审计与决定

Hybrid 的 parent 汇总数组现已在服务器和本机只读副本中核验，SHA-256 均为 `b81774bf19c0ab8f833fc916637450ed43fbba42aa29466ccd2c5dba6b537bda`。数组含 parent 级汇总指标和门控/等变性量，但不含原始 IQ 波形、逐干扰类型标签对应输出或可供基线配对的逐 parent 分数。因此跨模型配对 bootstrap、Equal-PFA、Hybrid 分类型结果、Top-10 状态回查和新物理指标仍缺必要输入；目前不能从边际统计恢复这些量。

ReCoVP-Hybrid 多 seed runner 将 seed 17 设为已冻结基准，并允许 seed 29、41；截至本次检查，已启动 2 个 seed（17、29），其中 seed17 已完成 256 轮，seed29 正在训练，seed41 尚未启动。

本轮复核没有启动训练、推理评测、扫描，也没有访问 CONFIRM/历史 TEST。seed17 仍为单种子 DEV 证据，不能冻结为论文终版或宣称全面优于基线。seed29 是先前已启动的独立训练，2026-11-26T06:14:58Z 状态显示 epoch 54/256、36/48 update，最近提交 epoch 53，latest checkpoint SHA-256 为 `862ca8334e5e395faa91ad6ca17a1e4e4a42a2726bcd027a767465cd406b87dc`；本轮未启动或停止该作业。GPU0/GPU1 最新利用率为 98%/87%，均不满足低于 85% 的新作业准入条件。配对、等虚警、分层、失败 parent 和延迟审计仍待足够输入与资源条件满足后再做。


## 最新进度核验（2026-11-26T06:14:58Z）

- 服务器 `rsdr-b` 可读；seed17 checkpoint 与 DEV 身份不变。服务器 parent 数组 SHA-256 已远程校验为 `b81774bf19c0ab8f833fc916637450ed43fbba42aa29466ccd2c5dba6b537bda`，本机副本同哈希。
- 已有数组可支持当前核心摘要及少量 parent 级汇总复核，但不包含完成任务书所需的原始信号、模型间逐 parent 配对分数和延迟测量输入。
- seed29 当前在训练：状态文件显示 epoch 54/256、update 36/48，最近已提交 epoch 53；latest checkpoint SHA-256 为 `862ca8334e5e395faa91ad6ca17a1e4e4a42a2726bcd027a767465cd406b87dc`。这是既有作业，本轮未改动。
- GPU 利用率快照：GPU0 98%，GPU1 87%；两卡都高于新作业启动阈值 85%。
- **结论：**Hybrid Stage-2 对 Stage1Only 的配对改善有原始 DEV 收据支持；相对各基线目前只有同集描述性点估计。新任务书列出的跨模型统计与分层审计仍未完成，暂不应冻结架构或扩展到 CONFIRM/TEST。
