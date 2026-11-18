# ISRJ_2 权威任务台账

ID 唯一。P2 的 ID 与本地 run 目录同名，training/ 相对 outputs/structured_20261118/。P0 的完成单位为历史筛查及配对汇总，P1/P4 的完成单位为论证审查，不代表新模型训练完成。D01/D02 是决定，不重复计入 25 组。总数 57 项含管理和研究任务。

| Task ID | 阶段 | 来源章节 | 目标 | 证据 | 状态 | 依赖 | 自动执行范围 | 预算 | 结果文件 | 完成判据 | 下一行动 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H01 | P0 | 11-18 实验总报告 §二、三 | U-Net 尾部 | DEV 通过 | COMPLETED | 历史收据 | 仅核验 | 0 GPU | ISRJ2_STATUS_RECONCILIATION_20261118.md | 核对种子和 DEV 摘要 | 保留正负结果 |
| H02 | P0 | 11-18 实验总报告 §二、三 | CorrectionNet 尾部 | DEV 通过 | COMPLETED | 历史收据 | 仅核验 | 0 GPU | ISRJ2_STATUS_RECONCILIATION_20261118.md | 核对种子和 DEV 摘要 | 保留正负结果 |
| H03 | P0 | 11-18 实验总报告 §二、三 | 噪声归一化 | DEV 未通过 | COMPLETED | 历史收据 | 仅核验 | 0 GPU | ISRJ2_STATUS_RECONCILIATION_20261118.md | 核对种子和 DEV 摘要 | 保留正负结果 |
| H04 | P0 | 11-18 实验总报告 §二、三 | 频率处理 | DEV 未通过 | COMPLETED | 历史收据 | 仅核验 | 0 GPU | ISRJ2_STATUS_RECONCILIATION_20261118.md | 核对种子和 DEV 摘要 | 保留正负结果 |
| H05 | P0 | 11-18 实验总报告 §二、三 | 扩张卷积 | DEV 未通过 | COMPLETED | 历史收据 | 仅核验 | 0 GPU | ISRJ2_STATUS_RECONCILIATION_20261118.md | 核对种子和 DEV 摘要 | 保留正负结果 |
| H06 | P0 | 11-18 实验总报告 §二、三 | 骨干 × 损失四格 | 三种子配对 DEV | COMPLETED | 历史收据 | 仅核验 | 0 GPU | ISRJ2_STATUS_RECONCILIATION_20261118.md | 核对种子和 DEV 摘要 | 保留正负结果 |
| M01 | P1 | 模型思路·复相位与坐标 | 复相位与坐标的前提审查 | 16 场景诊断 | COMPLETED | 数学或历史报告 | 仅核验 | 0 GPU | ISRJ2_CLAIM_EVIDENCE_MATRIX.md | 记录条件或反例 | 遇新证据再审 |
| M02 | P1 | 模型思路·固定去斜表示 | 固定去斜表示的前提审查 | 旧微批不同 | COMPLETED | 数学或历史报告 | 仅核验 | 0 GPU | ISRJ2_CLAIM_EVIDENCE_MATRIX.md | 记录条件或反例 | 遇新证据再审 |
| M03 | P1 | 模型思路·长程影响范围 | 长程影响范围的前提审查 | 有效感受野非理论范围 | COMPLETED | 数学或历史报告 | 仅核验 | 0 GPU | ISRJ2_CLAIM_EVIDENCE_MATRIX.md | 记录条件或反例 | 遇新证据再审 |
| M04 | P1 | 模型思路·四状态代数 | 四状态代数的前提审查 | 恒零反例 | COMPLETED | 数学或历史报告 | 仅核验 | 0 GPU | ISRJ2_CLAIM_EVIDENCE_MATRIX.md | 记录条件或反例 | 遇新证据再审 |
| M05 | P1 | 模型思路·近碰撞与稳定性 | 近碰撞与稳定性的前提审查 | 单射近碰撞反例 | COMPLETED | 数学或历史报告 | 仅核验 | 0 GPU | ISRJ2_CLAIM_EVIDENCE_MATRIX.md | 记录条件或反例 | 遇新证据再审 |
| M06 | P1 | 模型思路·结构分解条件 | 结构分解条件的前提审查 | 子空间交集反例 | COMPLETED | 数学或历史报告 | 仅核验 | 0 GPU | ISRJ2_CLAIM_EVIDENCE_MATRIX.md | 记录条件或反例 | 遇新证据再审 |
| M07 | P1 | 模型思路·帧内与场景尾部 | 帧内与场景尾部的前提审查 | 聚合次序反例 | COMPLETED | 数学或历史报告 | 仅核验 | 0 GPU | ISRJ2_CLAIM_EVIDENCE_MATRIX.md | 记录条件或反例 | 遇新证据再审 |
| full_base_17 | P2 | 结构规划 P3／CONFIG | full_base_17 | 本地持久轮次 340 | RUNNING | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_base_17 | 最终收据及身份相符 | 读检查点和总收据 |
| full_base_29 | P2 | 结构规划 P3／CONFIG | full_base_29 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_base_29 | 最终收据及身份相符 | 读检查点和总收据 |
| full_base_41 | P2 | 结构规划 P3／CONFIG | full_base_41 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_base_41 | 最终收据及身份相符 | 读检查点和总收据 |
| full_base_53 | P2 | 结构规划 P3／CONFIG | full_base_53 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_base_53 | 最终收据及身份相符 | 读检查点和总收据 |
| full_base_67 | P2 | 结构规划 P3／CONFIG | full_base_67 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_base_67 | 最终收据及身份相符 | 读检查点和总收据 |
| full_tail_17 | P2 | 结构规划 P3／CONFIG | full_tail_17 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_tail_17 | 最终收据及身份相符 | 读检查点和总收据 |
| full_tail_29 | P2 | 结构规划 P3／CONFIG | full_tail_29 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_tail_29 | 最终收据及身份相符 | 读检查点和总收据 |
| full_tail_41 | P2 | 结构规划 P3／CONFIG | full_tail_41 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_tail_41 | 最终收据及身份相符 | 读检查点和总收据 |
| full_tail_53 | P2 | 结构规划 P3／CONFIG | full_tail_53 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_tail_53 | 最终收据及身份相符 | 读检查点和总收据 |
| full_tail_67 | P2 | 结构规划 P3／CONFIG | full_tail_67 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/full_tail_67 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_base_17 | P2 | 结构规划 P3／CONFIG | no_dechirp_base_17 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_base_17 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_base_29 | P2 | 结构规划 P3／CONFIG | no_dechirp_base_29 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_base_29 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_base_41 | P2 | 结构规划 P3／CONFIG | no_dechirp_base_41 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_base_41 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_base_53 | P2 | 结构规划 P3／CONFIG | no_dechirp_base_53 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_base_53 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_base_67 | P2 | 结构规划 P3／CONFIG | no_dechirp_base_67 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_base_67 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_tail_17 | P2 | 结构规划 P3／CONFIG | no_dechirp_tail_17 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_tail_17 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_tail_29 | P2 | 结构规划 P3／CONFIG | no_dechirp_tail_29 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_tail_29 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_tail_41 | P2 | 结构规划 P3／CONFIG | no_dechirp_tail_41 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_tail_41 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_tail_53 | P2 | 结构规划 P3／CONFIG | no_dechirp_tail_53 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_tail_53 | 最终收据及身份相符 | 读检查点和总收据 |
| no_dechirp_tail_67 | P2 | 结构规划 P3／CONFIG | no_dechirp_tail_67 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/no_dechirp_tail_67 | 最终收据及身份相符 | 读检查点和总收据 |
| unet_tail_17 | P2 | 结构规划 P3／CONFIG | unet_tail_17 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/unet_tail_17 | 最终收据及身份相符 | 读检查点和总收据 |
| unet_tail_29 | P2 | 结构规划 P3／CONFIG | unet_tail_29 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/unet_tail_29 | 最终收据及身份相符 | 读检查点和总收据 |
| unet_tail_41 | P2 | 结构规划 P3／CONFIG | unet_tail_41 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/unet_tail_41 | 最终收据及身份相符 | 读检查点和总收据 |
| unet_tail_53 | P2 | 结构规划 P3／CONFIG | unet_tail_53 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/unet_tail_53 | 最终收据及身份相符 | 读检查点和总收据 |
| unet_tail_67 | P2 | 结构规划 P3／CONFIG | unet_tail_67 | 仅计划身份 | PLANNED | 冻结身份及主机 | 只读 | 512 计划；本代理 0 GPU | training/unet_tail_67 | 最终收据及身份相符 | 读检查点和总收据 |
| Q01 | P3 | 任务书 P3；对账及差距表 | 种子与模型/数据绑定 | 15 份历史完成收据 | COMPLETED | 已有轻量证据 | 可复核 | CPU | ISRJ2_RESEARCH_GAP_TRACKER.md | 注明可复核范围 | 仅随证据更新 |
| Q02 | P3 | 任务书 P3；对账及差距表 | 父场景统计独立性 | 160 父场景／92160 行 | COMPLETED | 已有轻量证据 | 可复核 | CPU | ISRJ2_RESEARCH_GAP_TRACKER.md | 注明可复核范围 | 仅随证据更新 |
| Q03 | P3 | 任务书 P3；对账及差距表 | 通用队列隔离测试 | 五个 dummy 测试含反例 | COMPLETED | 已有轻量证据 | 可复核 | CPU | ISRJ2_RESEARCH_GAP_TRACKER.md | 注明可复核范围 | 仅随证据更新 |
| Q04 | P3 | 任务书 P3；对账及差距表 | 效率口径 | 两进程八轮 | COMPLETED | 已有轻量证据 | 可复核 | CPU | ISRJ2_RESEARCH_GAP_TRACKER.md | 注明可复核范围 | 仅随证据更新 |
| Q05 | P3 | 任务书 P3；对账及差距表 | DEV 与独立确认隔离 | 旧模型独立证据有适用对象 | COMPLETED | 已有轻量证据 | 可复核 | CPU | ISRJ2_RESEARCH_GAP_TRACKER.md | 注明可复核范围 | 仅随证据更新 |
| I01 | P4 | 两份资料的方向一、相位与去斜章节 | 物理变换与坐标一致性初审 | 已归档卡片，模型未获确认 | COMPLETED | 已有报告 | 文献审查 | CPU | IDEA_REVIEW_01.md | 写明依据、缺口和权限 | 对新增证据复核 |
| I02 | P4 | 模型思路·四状态一致性；项目代数复核 | 四状态干预初审 | 已归档卡片，模型未获确认 | COMPLETED | 已有报告 | 文献审查 | CPU | IDEA_REVIEW_02.md | 写明依据、缺口和权限 | 对新增证据复核 |
| I03 | P4 | 模型思路·长程重复；创新性研究·近邻 | 长程关系初审 | 已归档卡片，模型未获确认 | COMPLETED | 已有报告 | 文献审查 | CPU | IDEA_REVIEW_03.md | 写明依据、缺口和权限 | 对新增证据复核 |
| I04 | P4 | 创新性研究 §8；模型思路·CVaR | 局部与场景风险尾部初审 | 已归档卡片，模型未获确认 | COMPLETED | 已有报告 | 文献审查 | CPU | IDEA_REVIEW_04.md | 写明依据、缺口和权限 | 对新增证据复核 |
| I05 | P4 | 创新性研究 §6；模型思路·教师 | 选择性知识迁移初审 | 已归档卡片，模型未获确认 | COMPLETED | 已有报告 | 文献审查 | CPU | IDEA_REVIEW_05.md | 写明依据、缺口和权限 | 对新增证据复核 |
| I06 | P4 | 创新性研究 §7；模型思路·多脉冲 | 多脉冲联合结构初审 | 已归档卡片，模型未获确认 | COMPLETED | 已有报告 | 文献审查 | CPU | IDEA_REVIEW_06.md | 写明依据、缺口和权限 | 对新增证据复核 |
| I07 | P4 | 模型思路·低秩与可辨识性；追加规划 P1/P2 | Hankel 与结构分解初审 | 已归档卡片，模型未获确认 | COMPLETED | 已有报告 | 文献审查 | CPU | IDEA_REVIEW_07.md | 写明依据、缺口和权限 | 对新增证据复核 |
| I08 | P4 | 11 月 18 日追加规划 P5/P6 | 条件后验与慢时间初审 | 已归档卡片，模型未获确认 | COMPLETED | 已有报告 | 文献审查 | CPU | IDEA_REVIEW_08.md | 写明依据、缺口和权限 | 对新增证据复核 |
| I09 | P4 | 创新性研究 §9；模型思路·轻量化 | 轻量部署与重复测量初审 | 已归档卡片，模型未获确认 | COMPLETED | 已有报告 | 文献审查 | CPU | IDEA_REVIEW_09.md | 写明依据、缺口和权限 | 对新增证据复核 |
| W01 | P5 | 原始资料 §1、9 | 贡献与证据对应 | 首篇可靠度/Transformer 不归入现骨干；第二篇的固定变换/校正器与研究设想分列 | COMPLETED | 已归档结果 | 可维护 | CPU | ISRJ2_IDEA_BACKLOG.md | 指明引用范围 | 按新收据更新 |
| W02 | P5 | 原始资料 §1、9 | 论文表图索引 | 四格见阶段总报告 §三，五项筛查见 §二；效率见 FORWARD_RETIMING；无新独立比较图 | COMPLETED | 已归档结果 | 可维护 | CPU | ISRJ2_ABSENCE_SUMMARY.md | 指明引用范围 | 按新收据更新 |
| S01 | 管理 | 无人值守任务书 §三、四 | 每小时管理 | 按运行回执核验 | RUNNING | Cindy/SSH/本机在线 | 是 | 每小时 CPU；0 GPU | ISRJ2_HOURLY_INSPECTION_LOG.md | 停用时终结 | 核查下一 UTC 整点 |
| D01 | 管理 | 任务书 §六 | 协议与资源决策 | 本地主机冲突 | AWAITING_AUTHORIZATION | 负责人决定 | 否 | 0 GPU | ISRJ2_PENDING_DECISIONS.md | 有效决策和记录 | 保持现场 |
| D02 | 管理 | 任务书 §六 | 协议与资源决策 | 缺部署及逐组许可 | AWAITING_AUTHORIZATION | 负责人决定 | 否 | 0 GPU | ISRJ2_PENDING_DECISIONS.md | 有效决策和记录 | 保持现场 |

## 版本与归档

源文件全文按字节指纹登记于小时巡检报告。2026-11-17 的总方向文件、11 月 18 日已含历史对话的模型思路、11 月 18 日追加结构规划分别引用。历史工作树基点 fcf9245、远端报告基点 8b9849c，新 25 组另有配置身份。相同主题不合并为同一冻结实验。

## 初始化来源登记

逐文件读取并保存的版本身份。本地时间列为 UTC 修改时间，不作为批准时间。Git 记录使用各文件最后提交时间（含时区）。

| 来源 | 版本或修改时间 | 行数 | SHA-256 |
|---|---|---:|---|
| gptpro创新性研究.md | 本地 2026-11-17 09:02 | 293 | 368ca894f2a575cc53e0032b5fc8cc90496e4584a2ad59fa73512d5f79cc0fd9 |
| ISRJ_2模型思路.md | 本地 2026-11-18 13:07 | 2509 | d0df96cb1994fcf4a48429b98e1c87c3c2fc1126feb2b923678d3add79700d99 |
| AGENTS.md | 本地 2026-11-18 13:33 | 5 | 71686aa3cb467ad2488797d4aae535f17f5387d0c0ac998daa6ba6909efca100 |
| STRUCTURED_RESEARCH_PLAN_20261118.md | 本地 2026-11-18 13:39 | 39 | 28bfdfdbc69dd8328684ff6a7e588e7ddfd6753eb712377173c67978cff289d6 |
| AGENTS.md | 54b3b98 2026-11-17T05:47:41-08:00 | 10 | 17f20ecdd9119fd3f787b51270d856cf5f5828f49faf05db65cbdf86bcd5b28f |
| handoff/CURRENT_STATE.md | a8a710a 2026-11-17T07:05:32-08:00 | 71 | b1cf95cf077c1a134209b8789baaa4aa6caeb087d14380eeea8ec24f8ca04203 |
| reports/ISRJ2_EXPERIMENT_SUMMARY_20261118.md | 5c1b0a5 2026-11-18T05:18:24-08:00 | 77 | 6217159eef4da04006773f59aee8b29bf56b40c89342eb40251fa81d037d16f2 |
| reports/ISRJ2_STATUS_RECONCILIATION_20261118.md | 8b9849c 2026-11-18T07:30:49-08:00 | 134 | 9f22dadd7eb5590a5a126d1b0c019cf1e482a826e608d5dddf412d98375a86fd |
| reports/ISRJ2_AUTO_RESEARCH_LOG.md | 8b9849c 2026-11-18T07:30:49-08:00 | 40 | d32f45067bbe7eadb857d4a1d77fac26f1dd6af7244a592e8c9f5a8120637402 |
| reports/ISRJ2_RESEARCH_GAP_TRACKER.md | 8b9849c 2026-11-18T07:30:49-08:00 | 28 | 7beb7e5249bf96a23d7d837d5c07206719efd97dcdfebae53023402190bd0f6c |
| reports/STRUCTURED_EVIDENCE_STATUS_20261118.md | dd338de 2026-11-18T07:02:50-08:00 | 26 | 91760f9d3287cef6e16c9316391ebf818d56d109889812629c6717d5e0e5169b |
