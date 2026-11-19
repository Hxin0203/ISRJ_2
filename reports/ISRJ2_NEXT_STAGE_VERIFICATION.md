## 2026-11-19 03:08 UTC supervisor recovery

The 02:20 timer and 03:00 agent-mode hourly run both FAILED with queued heartbeat Provider identity mismatch. The isolated CWC roundtrip remains verified, but AUTO_CWC_ENABLED is false. The sole hourly schedule is restored to script mode; manual scheduler test 2ccf592d-a386-48d7-835b-30d177890aca succeeded at 03:07:11 UTC. This is not a natural-trigger pass. Next natural script run: 04:00 UTC. Do not automatically switch back to agent mode or resend c2c_a719.

The missing 03-hour archive was recovered at 03:06 UTC, SHA 2ce4e7a716a1f7ca14c4f0168141c9819892b4d5. Training snapshot: 5 starts, 4 fixed milestones, 0 complete/evaluated, unet_tail_17 saved 362. B access remains unavailable. A failed same-hour B recheck now records its error without losing the original archive. See reports/ISRJ2_SUPERVISOR_RECOVERY_20261119.md for actual receipts, the passing regression check and remaining limits.

Earlier dated evidence follows.

## 2026-11-19 02:24 UTC CWC acceptance update

CWC_ROUNDTRIP_VERIFIED: c2c_a719 completed INIT / PLAN / EXECUTED / DONE in the codex-with-chatgpt Project with the happy-liskov connector. Six isolated tests passed and the original fixture hash was unchanged. DONE received 02:24:39.549 UTC; subsequent management inspection succeeded at 02:24:40.083 UTC. Do not resend this task. Official checkpoint cleared.

AUTO_CWC_ENABLED remains pending: the 02:20 natural timer run is still running, and it fired during an active turn. Delivery/completion and the first natural recurring agent-mode run at 03:00 UTC need verification; historical script-mode success is separate. No actual outage-continuation test has been claimed. See reports/CWC_END_TO_END_ACCEPTANCE.md for event times and original local receipt paths.

Current training snapshot: started 5/25, byte-verified fixed 512 milestones 4/25, training-plus-evaluation completion 0/25. unet_tail_17 saved 230, STATUS 231. No training change. D01/D02 (host and extension budgets) remain unresolved; D03 (CWC authorization) resolved by R1.

Historical snapshots below retain their original dates.

# ISRJ_2 下一阶段核验

2026-11-19 01:49:15 UTC 独立只读核验：启动 5/25，固定 512 轮检查点字节及身份核验 4/25，训练结束与统一评价收据齐全 0/25。unet_tail_17 已保存 122 轮，STATUS 第 123 轮。 旧版台账的小时行保留了各自采样时刻。下表时间为 UTC。

## 本地训练资产与交接

| 方法（seed 17） | 启动 | 最后持久保存 | 轮次 | 固定 512 文件实算 SHA-256 |
|---|---|---|---:|---|
| full_base_17 | 11-18T13:54:42 | 11-18T17:00:22 | 512 | 65cca45cf24244b0636f6975eb8d627c318033462884d384feed6da4570149c4 |
| full_tail_17 | 11-18T17:00:22 | 11-18T19:50:11 | 512 | e2e6e8de2a7c867b18582dfa0af05516e7a6457e8c39864876bf41df6bdfcbe7 |
| no_dechirp_base_17 | 11-18T19:50:11 | 11-18T22:30:01 | 512 | 93bc001a9b78cd6f191f7c31cf7403d25e19404db4207be84eb6651efc58b25d |
| no_dechirp_tail_17 | 11-18T22:30:01 | 11-19T01:11:21 | 512 | f4afb145fcf8bc3cdefb275ed5f7c332ffcb9f059bf6b23fe3b0f39b16484aca |
| unet_tail_17 | 11-19T01:11:21 | 11-19T01:49:08 | 122 | 尚无固定 512 文件 |

五份 STARTED 的 PID 均为 35200，当前进程的工作目录与入口相符。四个固定文件及各自 latest 文件逐字节重算，匹配索引；第五组 latest 也已重算。所有索引的配置、方法、种子、训练数据及源码绑定匹配 STARTED。当前轮次来自 2026-11-19T01:49:15.188232+00:00 的 STATUS；活动组持久收据 SHA-256 为 `39d18e2f099c33ba74b62a730726f0713f6088bc4de698940ca452371785a5bf`。种子 29、41、53、67 的五种配置均无启动收据。

控制器源码按预算、种子、配置遍历。实测顺序即上表；四个前组的最后保存时间均早于下一组启动。未发现独立的逐组退出收据、全队列训练结束或评价收据，也未发现本队列的 FAILURE 收据。正常转换有运行证据。硬退出恢复及其余任务转换仍待观察。

- CONFIG 字节 SHA-256：`68d402746f85d5414b405997d38785dde6c48971c5a933978afee0b44a896619`。配置列出的九份源码全部实算一致，控制器 SHA-256 为 `d6e0aea84c93dfec2e57c8952c397ee6260e865db3e42907c0d2a80169a2674b`。
- 本轮只读哈希对应新训练文件，数据 `111af12dbe3b051e73aa3f852d7ea35ed2a08344adb38e7213b9b02d981cbeed`，数据收据 `dff53ed786d5dde11a16219fef4c0c9f56b0eb03e2cdecd1d08614a148970818`。收据角色 train、生成种子 2026111821、512 个父场景，不涉及旧确认数组。
- 六项临时目录测试通过：正常返回、常规异常、重复调度、硬退出及残留锁、原循环缺逐组批准反例、隔离许可检查拒绝未批准项。最后一项验证拒绝回执、执行账本仅含许可项且没有全组完成收据。许可检查只存在于虚拟任务入口，生产控制器仍有原缺口。

## 整点巡检

全设备查询找到一个有效 ISRJ 整点任务，UTC `0 * * * *`。按 Cindy 实际触发收据与相同 UTC 时槽的 JSON 归档逐条关联。下列十次各时槽一次；所有调度结果 success，B 读取均未成功，本地启动/结束身份检查无报错，新增正式结束及评价收据均为零。

| 实际触发 UTC | 触发收据 | 已启动 | 持久轮次之和（本小时增量） |
|---|---|---:|---:|
| 11-18T16:00:00.551 | f2685381-e35c-43fe-a091-7a2e25cd6bec | 1 | 340（区间起点） |
| 11-18T17:00:00.918 | dbab0a45-972f-4ea6-9baa-7b75fd965ac4 | 1 | 510（+170） |
| 11-18T18:00:00.778 | 78c06db1-6b3a-43f4-bdcc-188d6dcbd8d3 | 2 | 680（+170） |
| 11-18T19:00:00.729 | f4793816-ff30-4669-b366-826bfb5bc7d5 | 2 | 867（+187） |
| 11-18T20:00:00.216 | d306b619-2781-4ec3-b528-42a080022ac1 | 3 | 1055（+188） |
| 11-18T21:00:00.011 | f34c33c4-185d-41b9-ab16-ae2e0abf1f40 | 3 | 1247（+192） |
| 11-18T22:00:00.955 | d2cde726-68e8-40db-9834-f61a57ac7e61 | 3 | 1439（+192） |
| 11-18T23:00:00.721 | ec401d2d-9966-4c9b-b3ca-22871db41f66 | 4 | 1631（+192） |
| 11-19T00:00:00.684 | f5268d3f-33a6-455f-ad09-6e3462c40ccd | 4 | 1821（+190） |
| 11-19T01:00:00.333 | bd247d43-3c1e-4272-8003-6bd17d969681 | 4 | 2011（+190） |

原小时记录仅保存 RuntimeError，无法逐次判定根因。11-18 16:05:59 的独立查询证明当时 B 的指定目录缺入口和配置。11-19 本轮两次独立查询均在 SSH 握手超时；无法认定 B 当前目录仍与此前相同。今后小时脚本保留可用的错误类型、返回码或已分类原因；自然触发回执以调度器为准。整点之间允许训练领先于台账。小时脚本已增加固定里程碑与评价计数，并同步四份报告。

## 预算、执行环境与协作决定

冻结来源是 STRUCTURED_RESEARCH_PLAN_20261118 第 25 行、CONFIG 及其绑定的入口。五方法乘五种子，512 基础预算共 12,800 组轮次；两个条件延长段各另加 6,400，上限共 25,600。配置描述两个连续 32 轮窗口与 1% 条件。没有逐组资源批准、本地主机例外或额外 GPU 时额度的回执。本管理任务新增计算预算为零，延长与主机归属列入 D01/D02。冻结控制器会自行判断延长；审计程序不能阻止它，无主动停止或改写。

本地 AGENTS 的 Git blob 为 6900ad744b9080298d24dbb82d2e8b960cf25269；远端 8d43492 的 blob 为 27766138f27aed9350aafb8f01c7652d2fe5e04b。前者规定新资产在当前工作区，后者指定 B 上长训练，且明确关闭 ChatGPT 协作。服务器现场版本以先前查验为历史依据，本轮握手失败。待决定是否准许当前本地资产作为正式结果以及各延长段的执行主机、额度和逐组授权。

MANUAL_READ_ONLY_REVIEW_VERIFIED。指定项目内既有对话验证 happy-liskov 并收到 DONE。自动协作测试状态 BLOCKED_POLICY_CONFLICT；本轮没有新派发、回包时间或完成收据。待决策的隔离用例：以单独的温度单位转换示例文件提出普通代码审查问题，要求返回工作区名称、文件内唯一标记及边界条件。按任务 ID 先查派发表，记录发送和收到 UTC、工作区和标记比对；失败保留原回执，同一任务禁止自动重发。该用例未用于修改训练代码。

## 科研证据范围

历史五项三种子 DEV 筛查已归档，两个尾部损失方案通过；其余三个配置保留负结果。历史虚警变化为 U-Net −0.477、CorrectionNet −0.694 个百分点，均有目标增量损伤代价。新队列尚缺统一 DEV 评价及独立确认，性能排名、效率与泛化均尚无结论。当前四个里程碑均为同一训练种子，不能代表跨种子稳定性。

九张 IDEA_REVIEW_01 至 09 卡片仍按既有证据归档，涵盖坐标变换、四状态、长程关系、尾部、迁移、多脉冲、结构分解、条件后验与部署。既有数学条件或反例保留原适用范围，模型收益设想保持待验证，本轮没有新增支持其有效性的模型证据。

本轮交付包含四份状态台账、本报告及待决策文件。GitHub 仅同步上述管理摘要；本地隔离测试、巡检实现及实验资产保留在工作区。阶段核验提交可从本文件的 Git 历史按提交标题 “Verify four fixed milestones and reconcile hourly evidence” 定位。

## 发布与归档验证

六份管理报告已同步到 paper2 的 codex/1103-experiment 并回读远端 SHA：[7bb6d3b4a909326414b0533dfc060c9290924ec3](https://github.com/Hxin0203/ISRJ_2/commit/7bb6d3b4a909326414b0533dfc060c9290924ec3)。

另用临时 Git 仓库验证更新后的小时归档：四份报告与发布版本逐字节对齐（统一换行），重复执行同一时槽保留同一提交。该测试与截至 01:00 UTC 的十份自然调度回执分别记录。新归档版本首次自然触发尚待下一整点。普通软件协作测试已准备独立温度换算文件和任务记录，实际派发次数为零。
