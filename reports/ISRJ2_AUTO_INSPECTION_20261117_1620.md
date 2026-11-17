# ISRJ_2 首次两小时巡检基线

巡检时间：2026-11-17 16:21 UTC  
自动化：`isrj-2`，当前对话 Thread Automation，周期为两小时。  
分支：`codex/1103-experiment`。

## 总体状态

当前状态为 **RUNNING_HEALTHY + WAIT_DEPENDENCY**。服务器 B 的任务身份、源码绑定和恢复点检查未发现漂移。CorrectionNet 原损失控制组已开始训练；CorrectionNet 尾部损失组仍在等待控制组。U-Net 尾部损失训练已完成三 seed 的 256 轮和配对 DEV 筛查，筛查规则通过；这只是 DEV 开发结论，尚不能作为独立确认。

## 实时任务状态

| 任务 | 设备 | Seed / 已保存轮次 | 状态 | 下一动作 |
|---|---:|---|---|---|
| T：U-Net + tail | GPU 1 | 17/256，29/256，41/256 | 训练完成，DEV summary 已有 | 保留结果，作为 CN0 依赖 |
| CN0：CorrectionNet 原损失 | GPU 1 | 17/11，29/0，41/0 | RUNNING_HEALTHY | 继续三 seed × 256；每轮恢复点、每32轮里程碑 |
| CN1：CorrectionNet + 0.05 tail | GPU 1 | 17/0，29/0，41/0 | WAIT_DEPENDENCY | 等 CN0 训练和配对 DEV 完成 |
| L：U-Net dilation | GPU 1 | 17/0，29/0，41/0 | PAUSED_AUTHORIZED | 仅 CN1、配对 DEV、四格汇总完成后释放自有暂停 |
| N：U-Net normalization | GPU 0 | 17/256，29/175，41/0 | RUNNING_HEALTHY | 继续 seed 29，再完成 seed 41 和 DEV |
| F：U-Net frequency | GPU 0 | 17/0，29/0，41/0 | WAIT_DEPENDENCY | 等 N 完成训练与配对 DEV |

CN0 seed 17 已保存 11 轮，最新 checkpoint SHA 已核对匹配；CN0 数值恢复预检已通过。当前 T 的三 seed checkpoint SHA 均匹配，T 的 DEV summary SHA 与完成收据一致。CN0/CN1 的冻结源绑定无漂移，L 的暂停令牌仍归本研究所有。

## 相比前一状态

- T 从训练未完成推进到三 seed 均 256 轮，并生成 DEV 筛查摘要。
- CN0 从等待依赖转为实际训练，seed 17 从第 3 轮推进至已保存第 11 轮。
- CN1 仍等待 CN0，L 仍按协议暂停。
- N 继续运行，seed 29 从 169 轮推进至已保存 175 轮；F 仍等待 N。

## 当前 DEV 证据

T 相对其普通 U-Net 对照的 seed-level 差值为：

- jammer-frame Pfa：`-0.0065104167`、`-0.0013020833`、`-0.0065104167`；
- Pd：`-0.0026041667`、`-0.0026041667`、约 `0`；
- damage：`+0.0011426364`、`+0.0004933829`、`+0.0004045942`；
- SJRIF：`+0.0726314624 dB`、`+0.0859707794 dB`、`+0.0365611847 dB`。

三项预设筛查条件均满足：Pfa 中位数改善、每个 seed 的 Pd 下降不超过 `0.02`、平均 damage 增量不超过 `0.005`。结论限定为 **DEV 筛查通过**，不能写成统计显著、非劣效或独立确认。CN0/CN1 必须用本模型同环境控制验证 tail 是否有效。

## 与论文目标的差距

1. **波形恢复：部分已验证。** T 有 DEV 端点指标，但 CorrectionNet 新控制尚未完成，尚无本模型新损失结论。
2. **目标保护：仍需本模型配对评价。** 目前只能引用 T 的 damage 筛查值，不能外推到 CN1。
3. **虚假峰和实际 Pfa：机制线索已有，独立结果不足。** tail 的 DEV 指标支持继续验证，但不能替代新确认集。
4. **公平优于 U-Net：未证实。** 需要四格架构×损失结果，并报告逐 seed 差值。
5. **创新机制必要性：未证实。** 需要本模型原损失控制、tail 对照以及后续最小组合。
6. **部署成本与泛化：未完成。** MAC、完整八脉冲 P50/P95、显存和新独立确认仍待安排。

## 本轮实际操作与边界

已读取并核验两份思路文件的 SHA；检查项目规则、handoff、冻结计划、GPU 队列和服务器现场；创建两小时 Thread Automation `isrj-2`。没有创建重复控制器、没有释放 L 的暂停、没有启动未授权新训练，也没有上传数据、权重、原始数组、缓存或完整日志。

## 下一轮优先事项

先核对 CN0 是否继续产生有效恢复点；若完成，核对其三 seed 完成收据和配对 DEV，再由既有控制器进入 CN1。GPU 0 继续 N→F。只有 CN1 的四格 DEV 汇总完成后，才评估是否释放 L 或提出下一步本模型组合实验。

证据来源：服务器实时 `tools.snapshot_idea_pipeline_20261117` 与 `tools.snapshot_correction_priority_20261117` 输出；T 的 `DEV_SCREEN_SUMMARY.json`；本地冻结 priority overlay 和 tail study plan。