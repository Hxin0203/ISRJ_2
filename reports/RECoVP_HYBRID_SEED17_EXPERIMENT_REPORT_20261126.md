# ReCoVP-Hybrid seed17 实验结果报告

**证据时间：2026-11-25 17:27:45 UTC（服务器 B）**

## 1. 报告范围

本报告汇总 `ReCoVP-Hybrid` seed 17 的 256 轮训练、固定 DEV256 核心评测和当前可支持的实验结论。评测使用服务器 B 的 `D:/code/Xin/ISRJ_222`，DEV 集包含 256 个 parent，每个 parent 包含 3 种干扰变体。结果文件记录了 Hybrid Final 与同一检查点的 Stage1Only 配对结果。

本报告不把 DEV 结果写成独立 TEST 结论，也不把不同校准协议下的既有 U-Net、Base 或 RSDR 结果直接混排为最终排名。

## 2. 训练与流水线状态

- 训练：`seed=17`，`epoch=256/256`，状态为 `TRAINING_COMPLETE`。
- 恢复点：`epoch_000` 至 `epoch_256`，共 257 个恢复收据，最新快照校验为 `PASS`。
- 身份绑定：训练配置、训练源码、预检收据、checkpoint 和 DEV 数据收据均已绑定；DEV 数据 SHA-256 为 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`。
- 固定 DEV256：已完成，评测状态为 `CORE_DEV_COMPLETE_REVIEW_REQUIRED`。
- 评测设备：物理 GPU 1；启动时 GPU 利用率为 `0%`，符合“低于 85% 才准入”的并行资源规则。
- 评测耗时：约 `157.56 s`。

## 3. 核心 DEV 结果

下表的差值定义为 `Hybrid Final - Stage1Only`。区间是 256 个 parent 的配对 bootstrap 95% 区间。

| 指标 | Hybrid Final | Stage1Only | 配对差值 | 差值 95% CI |
|---|---:|---:|---:|---:|
| `u_nmse` | 0.04312 | 0.40185 | -0.35873 | [-0.46821, -0.26552] |
| `damage` | 0.05967 | 0.12559 | -0.06591 | [-0.13931, -0.01359] |
| `SJRI-F` (dB) | 11.0968 | 10.4168 | +0.6801 | [+0.5301, +0.8342] |
| `Pd` | 0.8711 | 0.8542 | +0.01693 | [-0.00130, +0.03646] |
| jammer frame `Pfa` | 0.05599 | 0.07422 | -0.01823 | [-0.03125, -0.00521] |

其他 Hybrid Final 指标：

- false-peak suppression：均值 `13.4596 dB`，bootstrap 95% CI `[12.4979, 14.4141] dB`。
- 无干扰 `Pd`：`0.9219`，bootstrap 95% CI `[0.8867, 0.9531]`。
- noise frame `Pfa`：`0.003906`；jammer cell `Pfa`：`6.0260×10^-5`。
- 幅度比：`0.9152`；相位误差：`-0.0332°`。

## 4. 等变性与效率

复缩放测试覆盖幅度范围 `[0.2, 5.0]` 和相位范围 `[-π, π]`，同一 parent 内使用配对复数因子。Hybrid 的复缩放等变相对 L2 误差为：

- 随机复缩放：均值 `2.3328×10^-5`，P95 `8.7177×10^-5`。
- 固定幅度 `0.1`：均值 `3.0509×10^-7`，P95 `9.7903×10^-7`。
- 固定幅度 `10.0`：均值 `3.0361×10^-5`，P95 `9.4349×10^-5`。

CUDA event 延迟为：

- 1 pulse：P50 `27.8648 ms`，P95 `29.2633 ms`。
- 8 pulse：P50 `29.2768 ms`，P95 `30.6639 ms`。

上述延迟是模型评测调用的 CUDA 计时，不代表包含脉压、多普勒、CFAR、数据搬运在内的完整雷达接收链延迟。

## 5. 阶段性结论

1. **Stage 2 残差精修在当前固定 DEV 分布上产生了明确收益。** 相比同检查点 Stage1Only，`u_nmse` 和 `damage` 均下降，SJRI-F 上升，jammer frame `Pfa` 下降；这些配对差值的 bootstrap 区间没有跨过零。
2. **检测概率的提升幅度较小，当前证据不足以宣称检测优势。** `Pd` 的点估计从 `0.8542` 上升到 `0.8711`，但配对 95% CI 包含零。
3. **复数缩放等变性保持在较低误差水平。** 该结果支持当前实现对受测复缩放输入的数值稳定性，属于有限样本验证，不是对所有输入的数学证明。
4. **当前结果支持“Hybrid 的残差精修改善固定 DEV 上的恢复和干扰抑制指标”这一结论。** 它不支持“已经完成所有基线公平排名”“已经完成独立泛化验证”或“已经证明现场实时性”。

## 6. 仍需完成的任务书项目

- Base、P0、U-Net、RSDR 与 Hybrid 的统一配对比较和 equal-PFA 表格。
- Top-10 失败 parent 回查。
- CPU 延迟比较。
- 完整的 `SUMMARY`、`REPORT`、`VERIFICATION` 和 `REPRODUCE` 交付包。
- 独立确认集或 TEST 级别的最终泛化结论。

既有 U-Net/消融报告使用独立的三 seed 等虚警校准协议，不能直接替代上述统一比较；相关结果应作为背景证据单独引用。

## 7. 文件位置

本地工作区：`G:/Project Code/ISRJ_222/`

- 本报告：`reports/RECoVP_HYBRID_SEED17_EXPERIMENT_REPORT_20261126.md`
- 证据摘要：`evidence/outputs/RECoVP_HYBRID_FINAL_EVIDENCE_20261126.md`
- 训练完成收据：`evidence/outputs/recovp_hybrid_seed17/TRAINING_COMPLETE.json`
- 固定 DEV256 结果：`evidence/outputs/recovp_hybrid_seed17/dev_epoch_256_v1.json`
- 后台流水线状态：`evidence/outputs/recovp_hybrid_seed17/POSTTRAINING_STATUS.json`
- 评测快照：`evidence/outputs/recovp_hybrid_seed17/LIVE_SNAPSHOT.json`
- 复现实验说明：`evidence/outputs/recovp_hybrid_seed17/REPRODUCE.md`

服务器 B 的原始 checkpoint、`dev_epoch_256_parent_values.npz` 和完整日志仍保留在 `D:/code/Xin/ISRJ_222/`，不上传到 GitHub。

## 8. 证据哈希

- `dev_epoch_256_v1.json`：`877cfe7f6baf4fb3a59bff58da4a344e5437e43f000b3b3a9808a47acbda7df1`
- DEV checkpoint：`e0b6cb1be9559ed5e75a9ce3c4e178e85aba22f0cec753f52d3f523cbaf05212`
- DEV 数据：`0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`
- DEV 数据收据：`61e439503223258b969fcb8dd53a709bc5f60b1a128cde232ef645f98c1e2092`

本报告只使用离线本地文件和服务器回传证据编写，没有调用外部模型、水印服务或文件写钩子。
