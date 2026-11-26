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

Hybrid 的 parent 级数组只在服务器 `D:/code/Xin/ISRJ_222/outputs/recovp_hybrid_seed17/seed_17/dev_epoch_256_parent_values.npz`，其 SHA-256 收据为 `b81774bf19c0ab8f833fc916637450ed43fbba42aa29466ccd2c5dba6b537bda`。本机设备清单当前将服务器 `rsdr-b` 标为不可读，本地该数组不存在。因此本轮无法按 parent 重采样并重校准，也无法复算 Hybrid 的物理雷达指标、Top-10 或 CPU 延迟。

当前不应冻结 Hybrid 为论文终版，不应放行 seed 29/41、579 扫描或 CONFIRM/TEST。建议保持队列暂停；待恢复对服务器 B 的只读访问并核验 parent 数组 SHA 后，再完成同一指标代码下的配对、等虚警、分层、失败 parent 和效率审计。此复核轮次自身没有启动训练、改动架构、运行扫描或访问 CONFIRM/历史 TEST。
