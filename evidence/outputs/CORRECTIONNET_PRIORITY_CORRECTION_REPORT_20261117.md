# 本模型实验优先级纠偏与实际部署记录

记录日期：2026-11-17。新队列快照：14:59:52 UTC（洛杉矶 06:59:52）；原运行任务快照：15:00:05 UTC（洛杉矶 07:00:05）。两份快照不是同一时刻。

## 对用户问题的直接回答

近期新增优化实验确实全部落在 U-Net 上。这是我把“先用简单骨干筛查因素”扩展得过长造成的排期偏差。最初用 U-Net 检查尾部损失是否有通用收益是合理的，后来连续把归一化、长程卷积和频率处理也排给 U-Net，而没有同步安排本模型验证，已经不能满足第二篇论文的推进目标。

CorrectionNet 此前已有八 seed 的 256 轮训练，以及去斜、RMS、PC 损失和目标保护等消融。本轮缺的是新增优化在本模型上的受控验证。历史训练及消融不应被反复包装成新优化进展；U-Net 筛查结果也不能直接算成本模型的创新或性能提升。

## 已落实的调整

在服务器 B 的专用第二篇目录注册了两个新的持久控制器，源码已冻结。当前正在计算的任务继续运行；尚未开训的 U-Net dilation 通过其已有 PAUSE 接口暂停。没有停止训练、修改冻结依赖、删除旧模型或占用第一篇目录。

| 顺序 | GPU 1 任务 | 实际状态 | 训练计划 |
| --- | --- | --- | --- |
| 1 | 当前 U-Net + tail | 训练中；seed 17/29 已到 256，seed 41 已保存 55 轮 | 完成第三 seed 与原定配对 DEV |
| 2 | CorrectionNet 原损失新控制 | 已部署，WAIT_DEPENDENCY；控制器 PID 14952 | seed 17/29/41，各 256 轮 |
| 3 | CorrectionNet + 相同 tail | 已部署，WAIT_DEPENDENCY；控制器 PID 31212 | seed 17/29/41，各 256 轮 |
| 4 | 原 U-Net dilation | 已确认 PAUSED，未训练 | 本模型两组及配对 DEV 完成后自动释放本实验拥有的 PAUSE |

GPU 0 原归一化与后续频率筛查继续作为辅助证据。它们不是本模型的替代研究路线。以上两组本模型训练还没有保存第一个 epoch，GPU 数值恢复预检尚未执行。CPU 检查通过不代表训练已经开始，也不代表性能提升。

新原损失控制不是覆盖原八 seed 结果。历史训练存在续训和运行设置差异，严格检验新增 tail 时需要同一执行条件下的控制。两组保持原始去斜、RMS 和 CorrectionNet 骨干不变，只把 tail 权重从 0 改为 0.05；控制组也记录未加权 tail，便于分量比较。

## 这轮必须回答的问题

首先检验 CorrectionNet 自身在相同初始化、数据、训练预算与精度下，新增 tail 是否降低目标缺席时的含干扰帧虚警，同时保持 Pd、SJRIF 与目标增量损伤。所有三个 seed 和所有损失分量均报告，不以新旧 total loss 绝对数值排序，不挑最佳 seed 或最佳轮次。

其次形成骨干×损失四格：U-Net 原损失、U-Net + tail、CorrectionNet 原损失、CorrectionNet + tail。控制器在本模型 tail 三 seed 评价完成后自动生成 `FOUR_CELL_DEV_SUMMARY.json`，包含各格指标、两骨干的损失效应及二者差异。此差异是 DEV 描述性检查，不是三 seed 已足以证明交互显著或独立泛化的结论。

若 tail 只改善 U-Net，必须如实报告该骨干依赖，不能继续声称问题已在本模型上解决。若 tail 在本模型有效，再评估是否与此前表现较好的去斜移除组合；组合必须补充本骨干控制，不能拿 U-Net 结果替代。若以显著目标损伤换取虚警下降，则不接受该因素。长程、频率和归一化的可保留因素随后也须转移到本骨干逐项验证。

此前两份创新思路中的相位／时延一致性、选择性迁移、分母及难样本诊断仍按证据推进。它们目前是诊断或条件研究项；本轮没有把这些方向伪称成已部署的训练。当前修正仅提前本模型的新损失验证，不凭少量 DEV 数据临时改模型或权重。

## 检查证据与边界

两组均通过服务器 CPU 前向、反向、损失及梯度有限性、初始构造一致性、有效参数数与尺度测试。53 个冻结源码／协议 SHA 在本机和服务器一致；训练／DEV 数据与 receipt SHA 一致，历史三个参考权重与其完成记录及源码身份已验证。12 个 Python 文件 AST 及两个 PowerShell 入口的语法检查通过。

两个队列控制器存活，绑定无源码漂移；dilation 的 PAUSE 内容属于本实验，CONTROL_STATUS 已确认 PAUSED。GPU 精确数值恢复检查必须等待前项训练与 DEV 完成、设备准入通过后执行。每个完成 epoch 保存完整 model／optimizer／RNG 恢复点，每 32 轮另存里程碑，保留原模型和失败记录。新控制组与 tail 组均有累计 96 小时预算；服务器重启后仍需检查并恢复控制器，不承诺无人干预的无限自动重试。

冻结 operations 模块内的旧 snapshot 子命令会尝试读取 Windows 进程锁的第一个字节，可能报 PermissionError；这不影响训练控制器。为保持冻结源码不变，本轮使用独立只读入口 `tools.snapshot_correction_priority_20261117`，跳过锁字节后读取 PID，并成功生成状态证据。后续查看队列使用该入口。

## 最后复核

15:03:52 UTC 的独立轻量快照确认：两个本模型控制器仍存活并等待依赖，均未保存首个 epoch，源码无漂移；dilation 仍为本实验拥有的 PAUSED。当前 U-Net tail 的依赖状态记录 seed 41 已保存 64 轮，此信息来自前项状态文件，并非重新执行完整恢复归档审计。原 14:59:52／15:00:05 快照保持原样，不覆盖历史 SHA 绑定。

## 文件索引

- 冻结方案：`CORRECTIONNET_TAIL_STUDY_PLAN_20261117.md`。
- 本轮覆盖排期：`CORRECTIONNET_PRIORITY_PIPELINE_20261117.json`，在原 18 项任务清单上覆盖 GPU 1 的执行顺序，未部署项仍明确标为计划。
- 队列证据：`CORRECTIONNET_PRIORITY_STATUS_20261117.json`。
- 原任务快照：`CORRECTIONNET_PRIORITY_EXISTING_JOBS_20261117.json`；其中 dilation 原 STATUS 的 WAIT_DEPENDENCY 需结合本轮 PAUSED 控制状态理解。
- 工程检查：`CORRECTIONNET_CPU_CHECK_ORIGINAL_20261117.json`、`CORRECTIONNET_CPU_CHECK_TAIL_20261117.json`。
- 本地与服务器身份复核：`CORRECTIONNET_PRIORITY_VERIFICATION_20261117.json`。
- 启动入口：`tools/start_correction_tail_study.ps1`；初次部署入口会拒绝重复注册，已有队列恢复应使用对应启动入口并保留原绑定。

关键源码和汇总同步 `paper2`。训练数据、权重、逐 parent 数组、缓存、凭据和完整日志继续只保存在专用本地／服务器目录。
