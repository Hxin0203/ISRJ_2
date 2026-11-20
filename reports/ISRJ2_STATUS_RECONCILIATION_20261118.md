# ISRJ_2 实验状态对账 · 2026-11-18

历史五项 DEV 筛查均已完成，两个尾部损失配置通过，另三项未通过。新协议 25 组中有 1 组留下启动、检查点及继续执行的记录，达到 512 轮的有 0 组。两批任务分属不同协议。执行状态取自 14:58:28 UTC 的快照。

## 分支和资料版本

本地工作分支 `cindy/happy-liskov` 的 HEAD 为 `fcf9245c0bd9d2a85d3261138908b9852df2a750`，无上游。只读核对时，GitHub 与 paper2 抓取结果均指向 `c1d3195c47c5f1665eeb3335e724c8c9cade20b5`。远端有七份指定交接及历史报告，本地工作树有尚未提交的结构诊断报告、输入、源码及测试。发布基于远端历史，不合并两棵工作树。原始快照见本地 `reports/STRUCTURED_EVIDENCE_STATUS_20261118.md`。

两份附件按字节计算 SHA-256。《gptpro创新性研究》为 `368ca894f2a575cc53e0032b5fc8cc90496e4584a2ad59fa73512d5f79cc0fd9`，《ISRJ_2模型思路》为 `d0df96cb1994fcf4a48429b98e1c87c3c2fc1126feb2b923678d3add79700d99`，均与本轮 SOURCES.json 相符。早期两小时任务记载第二份为 `a153014739878663df88457771c67559f8008e84ae3aae153c46dfcb7798af9e`，应按两个版本引用。仅凭哈希无法确定改动段落。

## 历史筛查证据索引

服务器 B 上逐份核对 15 份种子完成收据，每个 run 的三种子均完成 256 轮。重新计算最终权重、绑定源码、CONFIG、train/dev 数据和 DEV 摘要的哈希；核对 run、seed 和完成收据。五组 DEV 摘要及管线收据的原始字节均与上述远端提交一致。源身份列为收据中路径至 SHA-256 映射经键排序、紧凑 JSON 序列化后的 SHA-256。每行链接指向组完成收据，同目录 DEV_SCREEN_SUMMARY.json 给出三个种子的评价。

| run ID 与组收据 | DEV | 冻结配置 SHA-256 | 源身份 SHA-256 |
|---|---|---|---|
| [correction_mf_tail_20261117_v1](../evidence/outputs/correction_mf_tail_20261117_v1/PIPELINE_STATUS.json) | 通过 | `c837e5503335cbd027f78a2516960d5f91cfc674f8ed146af92d46c35dcb848e` | `51bffd45f90ca5e01eb98cf0ae42cfff05304060c789b214151b6769773ea28c` |
| [unet_dilation_screen_20261117_v1](../evidence/outputs/unet_dilation_screen_20261117_v1/PIPELINE_STATUS.json) | 未通过 | `2e522ab1736dec226420a4ff6d251afc9d72042adb035f06dba6356bd59a703f` | `7d8bec5dd4138613d2ed665d2cbfb05427bc6841c2825ddee14015048dec348a` |
| [unet_frequency_screen_20261117_v1](../evidence/outputs/unet_frequency_screen_20261117_v1/PIPELINE_STATUS.json) | 未通过 | `202a35c1538363f28b85a45abb4dc58c85e3dc3e06d967521f0b8471baeb83e5` | `7d8bec5dd4138613d2ed665d2cbfb05427bc6841c2825ddee14015048dec348a` |
| [unet_mf_tail_screen_20261117_v2](../evidence/outputs/unet_mf_tail_screen_20261117_v2/PIPELINE_STATUS.json) | 通过 | `cef0a79c03a022e09bce91237f909179a692952c5c44f34c19fd8315a404c1af` | `efcc5a2392975a5e8b45f03090a4d23eea06ed4a1cf782a9a7e28c11fac0b15d` |
| [unet_noise_norm_screen_20261117_v1](../evidence/outputs/unet_noise_norm_screen_20261117_v1/PIPELINE_STATUS.json) | 未通过 | `e5b01d1c77363be8b6cfd389604e61bf2e5c4ad89c081fba177d1e93f9ca7be0` | `2527ee4ed3a71291a43059c71b43c6c32360236b0c463fa563060f123ae3c3f1` |

逐种子路径相对服务器 `outputs/<run ID>/`。末两列为 SHA-256 的前 16 位，核对使用全长指纹。

| run ID | seed | 完成轮次 | 完成收据 | 收据指纹 | 最终权重指纹 |
|---|---:|---:|---|---|---|
| correction_mf_tail_20261117_v1 | 17 | 256 | `arm/seed_17/TRAINING_COMPLETE.json` | `d303dac7c991077c` | `7ff216c12dc0e7cb` |
| correction_mf_tail_20261117_v1 | 29 | 256 | `arm/seed_29/TRAINING_COMPLETE.json` | `e321b06355a8a43f` | `7580acb740903c07` |
| correction_mf_tail_20261117_v1 | 41 | 256 | `arm/seed_41/TRAINING_COMPLETE.json` | `129e8296412b4dc0` | `67fef54ef8071f79` |
| unet_dilation_screen_20261117_v1 | 17 | 256 | `factor/seed_17/TRAINING_COMPLETE.json` | `5efa6ea66e497b50` | `37af9dcb4d6e6c1b` |
| unet_dilation_screen_20261117_v1 | 29 | 256 | `factor/seed_29/TRAINING_COMPLETE.json` | `0ce47ea3d4b0b35a` | `18673feb5a1e8489` |
| unet_dilation_screen_20261117_v1 | 41 | 256 | `factor/seed_41/TRAINING_COMPLETE.json` | `18c409ca44bb4d08` | `cde5cd6835074c44` |
| unet_frequency_screen_20261117_v1 | 17 | 256 | `factor/seed_17/TRAINING_COMPLETE.json` | `2f8a10a6d51a11cd` | `35da93e8bd694298` |
| unet_frequency_screen_20261117_v1 | 29 | 256 | `factor/seed_29/TRAINING_COMPLETE.json` | `e49f619605d6aa2e` | `4cd25189c0ca23d5` |
| unet_frequency_screen_20261117_v1 | 41 | 256 | `factor/seed_41/TRAINING_COMPLETE.json` | `c2562081c0d45d87` | `7dad9bfbe3aea1be` |
| unet_mf_tail_screen_20261117_v2 | 17 | 256 | `tail/seed_17/TRAINING_COMPLETE.json` | `b2e683efa27920e4` | `ba7ce8f34d59ad7c` |
| unet_mf_tail_screen_20261117_v2 | 29 | 256 | `tail/seed_29/TRAINING_COMPLETE.json` | `b81866277b48b664` | `6f313d278927b8d1` |
| unet_mf_tail_screen_20261117_v2 | 41 | 256 | `tail/seed_41/TRAINING_COMPLETE.json` | `e54ed31385de9fd8` | `2454a60c19c04d7b` |
| unet_noise_norm_screen_20261117_v1 | 17 | 256 | `norm/seed_17/TRAINING_COMPLETE.json` | `6246ea1cacd4b191` | `d38a3c86bfeabbda` |
| unet_noise_norm_screen_20261117_v1 | 29 | 256 | `norm/seed_29/TRAINING_COMPLETE.json` | `4f3e2666efe70945` | `e2dc212a42e2a4bf` |
| unet_noise_norm_screen_20261117_v1 | 41 | 256 | `norm/seed_41/TRAINING_COMPLETE.json` | `03184a9c5f61b8ab` | `d2a59c36bd51c646` |

按逐种子记录重算三项冻结条件，五项结论与组收据一致。条件为虚警差中位数小于零、每个种子的 Pd 变化至少 −0.02、平均损伤变化至多 +0.005。三种子共用 DEV 数据指纹 `0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`。完整指标见[实验总报告](ISRJ2_EXPERIMENT_SUMMARY_20261118.md)和[历史归档说明](ISRJ2_PROGRESS_RECEIPTS_20261118.md)。

## 新任务逐组状态

用户提出增加轮数、重复次数和收敛诊断。本地 11 月 18 日 STRUCTURED_RESEARCH_PLAN 与 training/CONFIG.json 将其落实为五方法 × 五种子，512 基础轮次，条件延长列为 768/1024。未找到逐组单独批准记录。本次指令只授权审计。每行均使用同一冻结配置 `68d402746f85d5414b405997d38785dde6c48971c5a933978afee0b44a896619`，均无独立评价收据。核对基点 c1d3195 未归档这批训练状态；此次只发布进度摘要。

| 计划 run ID（方法_种子） | 状态 | 已核保存轮次 | 状态时间（11 月 18 日） | 完成收据 |
|---|---|---|---|---|
| `full_base_17` | 执行第 175 轮 | 174（固定里程碑 128） | 14:58:28 UTC | 无 |
| `full_base_29` | 仅计划 | — | — | 无 |
| `full_base_41` | 仅计划 | — | — | 无 |
| `full_base_53` | 仅计划 | — | — | 无 |
| `full_base_67` | 仅计划 | — | — | 无 |
| `full_tail_17` | 仅计划 | — | — | 无 |
| `full_tail_29` | 仅计划 | — | — | 无 |
| `full_tail_41` | 仅计划 | — | — | 无 |
| `full_tail_53` | 仅计划 | — | — | 无 |
| `full_tail_67` | 仅计划 | — | — | 无 |
| `no_dechirp_base_17` | 仅计划 | — | — | 无 |
| `no_dechirp_base_29` | 仅计划 | — | — | 无 |
| `no_dechirp_base_41` | 仅计划 | — | — | 无 |
| `no_dechirp_base_53` | 仅计划 | — | — | 无 |
| `no_dechirp_base_67` | 仅计划 | — | — | 无 |
| `no_dechirp_tail_17` | 仅计划 | — | — | 无 |
| `no_dechirp_tail_29` | 仅计划 | — | — | 无 |
| `no_dechirp_tail_41` | 仅计划 | — | — | 无 |
| `no_dechirp_tail_53` | 仅计划 | — | — | 无 |
| `no_dechirp_tail_67` | 仅计划 | — | — | 无 |
| `unet_tail_17` | 仅计划 | — | — | 无 |
| `unet_tail_29` | 仅计划 | — | — | 无 |
| `unet_tail_41` | 仅计划 | — | — | 无 |
| `unet_tail_53` | 仅计划 | — | — | 无 |
| `unet_tail_67` | 仅计划 | — | — | 无 |

从早期快照的第 144 轮，到 14:56 UTC 第 168 轮，再到第 175 轮，只有同一组持续留下新记录。早期确认的第 128 轮固定检查点今天复核一致，SHA-256 为 `2aa842fd7cd2ab81ef399288aef6ad5a2735233999069883c0bcfcf6e9f59c87`。本次还验证了可轮换 latest.pt 的第 174 轮检查点 `89520bc83d58746e636a4811b5f3a0c3f455686634a99dc4bf7ba115bae84b33`。两者的配置、训练数据、数据收据和源码绑定均匹配 STARTED.json。之前的截图不能还原第 144 轮时 latest 的确切内容。此次哈希检查没有反序列化模型。下次巡检需以本地 `outputs/structured_20261118/training/{CONFIG,STATUS}.json` 及各组目录的 latest.json、对应文件和完成收据分别核实。未达到预算，尚无收敛、性能排名或失败结论。

## 五批完整性

检查范围为本地 `outputs/structured_20261118/diagnostic_v2/COMPLETE.json` 列出的五份 parents_*.csv。逐表重算哈希，逐行检查全部字段及数值有限性，以生成种子、父场景、方法、正则、脉冲、类型组成复合键。五表哈希均匹配，重复键、空字段、非有限值均为零。每批 32 个父场景，每个父场景均具有同一套 576 个条件，总计 92,160 行。统计独立单位为 160 个父场景。另一个局部歧义诊断含 80 个父场景，不能并入这 160 个扩大样本量。

## 任务书逐项核销与证据等级

| 项目 | 已完成与仍缺证据 |
|---|---|
| 资料、源码、历史服务器巡检 | 两份资料指纹、历史 15 份收据与绑定源已核；新队列的单组恢复点已核。 |
| 半小时巡检与前后对比 | 找到两个执行器，已纠正重叠；历史截图至本次形成带时间的进度链。新的自动执行成功需运行收据，见[巡检日志](ISRJ2_AUTO_RESEARCH_LOG.md)。 |
| 五单因素及四格 | B：两项尾部在 DEV 降虚警且平均损伤略升。F：其余三个具体配置未通过。骨干 × 损失四格已有 DEV。固定骨干的去斜 × 损失同协议四格仍缺。 |
| 四状态一致性 | A：按定义交互项等于两种背景的增量误差之差。F：取恒零恢复器，交互为零同时丢失真实增量；严格保真的宽泛主张有反例。 |
| 近碰撞与可辨识性 | A/F：可用单射构造任意接近的输出，有限精度近碰撞不能证明严格多解。E：项目分布的不可辨识比例未核实。 |
| 教师、结构与新候选 | 历史主网格边际计数限定教师与基线分歧范围。D：分歧可为蒸馏提供收益；E：缺 ISRJ 蒸馏学生正式对照。已完成结构与歧义诊断，未取得完整新模型对照。 |
| 效率 | 已有模型两个进程八轮前向实测，七轮 U-Net 中位延迟较低，一轮反向。E：整链耗时、峰值显存及跨设备结论缺证。[重测口径](ISRJ2_FORWARD_RETIMING_20261118.md) |
| 独立校准和泛化 | C 仅适用于历史报告注明的旧模型独立校准比较。E：新候选没有独立结果，历史数值不能转移。 |
| 新 25 组 | 一组部分执行，24 组仅计划，整组性能证据等级 E。没有重启、延长或重复执行。 |
| 进度文件与远端同步 | 重写当前差距表，保留历史巡检日志并追加本次记录；只发布管理报告。 |

等级 A 为有前提的数学结论，B 为历史 DEV，C 为指定对象的独立证据，D 为假设，E 为证据不足，F 为反例或负结果。数学与历史对照的原始说明见[主张矩阵](ISRJ2_CLAIM_EVIDENCE_MATRIX.md)。窗口范围、协议及数据身份不同的结果分别引用。

## 执行边界

本轮只读核验已有研究产物并维护管理记录。定向改进对抗性干扰下雷达探测的模型优化与性能验证不在本轮执行范围。新队列缺乏完成证据；既有训练资源耗尽和训练软件故障均无证据。巡检会话路由另有独立失败记录。可持续工作包括收据与哈希核对、父场景统计、进度差分、版本对齐、数学反例及论文结论范围审查。

## 发布前的巡检增量

15:01:40 UTC，持久检查点已从 174 推进到 184，状态记录第 185 轮。启动数和完成数仍为 1/25、0/25。服务器 B 的既定项目目录下没有与本地同路径的新队列训练目录。旧服务器协议与本地新计划分别登记，未发现能把两者视为同一轮的收据。下次计划时间 15:30 UTC。

## 执行身份核查 · 11 月 18 日 15:16 UTC

| 核查对象 | 本地工作站 | 服务器 B 指定项目 |
|---|---|---|
| 主机、账户 | DESKTOP-RHKS2SI，admin | DESKTOP-RFFERM3，rsdr_runner |
| 工作目录 | G:/Project Code/ISRJ_222/.cindy-worktrees/happy-liskov | D:/code/Xin/ISRJ_222 |
| 版本 | cindy/happy-liskov，fcf9245c0bd9d2a85d3261138908b9852df2a750，无上游。当前研究源码、测试、资料和报告含未跟踪文件，运行身份以字节哈希核验。 | 项目及上级无 .git，SSH 环境 PATH 无 git。部署副本的分支、HEAD、未提交差异无法现场核验。 |
| 进程与环境 | PID 35200，13:54:23.994 UTC 创建，进程所有者、实际 cwd、入口 python -u -m tools.structured_train 与启动及运行收据相符。Python 3.13.3、PyTorch 2.9.1+cu126、CUDA RTX 4060；驱动进程表也有该 PID。 | 只读检查环境为 Python 3.10.9，主机列出两块 RTX 5060 Ti。可见 Python 进程中无本队列训练入口，未把其他项目进程算入本队列。 |
| 新协议 | STARTED、CONFIG、STATUS、latest、epoch_128 可核。15:15:56 UTC 保存第 225 轮，状态记录第 226 轮，相对 15:06 的第 196 轮增加 29 轮。 | 15:16:13 UTC 检查 outputs 下 141 个目录，25 个 run ID、目录及协议身份匹配均为零，读取错误为零。tools/structured_train.py、训练 CONFIG 和专用数据收据均不在预期路径。排除旧 Gate 与链接目录。 |

服务器及远端实验分支的 AGENTS.md 要求长训练使用服务器 B；本地工作区的版本只写新训练与数据留在当前工作区。当前执行位置与服务器版明确规定冲突。两端使用了不同的交接版本，本地主机直接启动了本地入口。未见服务器例外批准或新协议部署回执；缺少记录解释当初为何选择本地。项目早期服务器队列与本地新队列不能认定为同一个冻结协议。

**25 组状态。** 列为 seed，行与列唯一确定 run ID。R 为 RUNNING（本地），P 为 PLANNED（无启动收据）。P 的保存轮次均不适用，所有格子均无整组完成证据。服务器指定目录中的所有格子均无启动证据。

| 方法 | 17 | 29 | 41 | 53 | 67 |
|---|---|---|---|---|---|
| full_base | R | P | P | P | P |
| full_tail | P | P | P | P | P |
| no_dechirp_base | P | P | P | P | P |
| no_dechirp_tail | P | P | P | P | P |
| unet_tail | P | P | P | P | P |

绑定的本地训练入口按预算、种子、方法串行调用；首组尚未返回，所以其余 24 组尚未轮到。无逐组 WAIT_DEPENDENCY、资源拒绝、崩溃或恢复收据。第 128 轮固定文件重新计算的 SHA-256 与早期记录一致；九份绑定源码、训练数据、配置、数据收据均复核一致。配置指纹 68d402746f85d5414b405997d38785dde6c48971c5a933978afee0b44a896619，训练入口 d6e0aea84c93dfec2e57c8952c397ee6260e865db3e42907c0d2a80169a2674b，数据 111af12dbe3b051e73aa3f852d7ea35ed2a08344adb38e7213b9b02d981cbeed，数据收据 dff53ed786d5dde11a16219fef4c0c9f56b0eb03e2cdecd1d08614a148970818。B 缺对应对象，无法完成两端的逐字节同一性认定。

服务器与本地按九个绑定文件单独比较，五份公共源码逐字节一致，新训练入口与三个 research 模块均缺。远端任务身份仍无法配对。

15:30 UTC 的自然巡检已核实本地 R 格仍为 full_base_17：已保存第 262 轮，状态文件记录第 263 轮。对照 15:06 的第 196 轮增加 66 轮。保存时间 15:29:49.494 UTC，滚动检查点 SHA-256 为 2649b4f7883ae9ec227fbb56e84a1ac447832261192c0d6b3842c8390391a379。其余 24 格仍为 P。任务启动数、完成数分别为 1/25 和 0/25。

## 串行控制器与通用队列测试

15:35 UTC 仍为原进程、同一入口与同一工作区；最新核实的持久轮次为 276，控制器状态第 277 轮。与 15:30 的保存轮次比较增加 14。服务器 B 的协议文件与模块仍缺。本地冻结入口按预算、种子、方法调用。单组正常返回时循环直接进入下一个方法；所有组返回后才写预算及全组完成收据。

可重复的本地检查命令为 `python tests/test_queue_evidence.py -v`。测试校验入口 SHA-256，提取原文件的循环与锁/异常处理语句，仅在临时目录调用 dummy 函数，使用独立子进程模拟异常退出；无模型、数据和 GPU。五项断言均通过：

| 场景 | 已观察到的结果 |
|---|---|
| 首组正常返回 | 同种子下一配置被调用，随后下一个种子。 |
| 抛出异常 | 保留 FAILURE 收据，停止本次循环，无全组 COMPLETE。 |
| 有活动锁时再次触发 | 拒绝进入任务体。此项不证明正常退出后重放具备逐组幂等性。 |
| 硬退出 | 已写的虚拟任务账本和锁保留，无常规失败收据；再触发受残留锁阻挡。 |
| 枚举项缺逐组批准 | **发现反例**。未列入独立许可集合的 dummy 项仍被调用。原循环不执行逐组许可核查，巡检器不提供此批准。 |

本轮确认正常返回的通用转移语义，未验证 512 轮实算完成、真实下一组启动或端到端恢复。一般串行转移未复现故障。原训练控制器及活跃锁保持原状。

## 16:06 UTC 巡检与交接证据复核

| 范围 | 时间（UTC）与来源 | 观察结果 |
|---|---|---|
| 整点自动巡检（部分验证） | 计划 16:00；Cindy 收据 f2685381-e35c-43fe-a091-7a2e25cd6bec，16:00:00.551 触发，16:00:11.181 结束；同一小时的 JSON 快照 | 调度状态 success。本地已保存 340 轮。脚本记录 B 读取失败，未提供单独的失败时间。 |
| B 独立只读复核（直接观察） | 16:05:59.861 服务器时钟；独立 SSH 查询，退出码 0 | 指定项目目录存在，缺少训练入口和本协议配置；指定训练目录下找到 0 份逐组启动收据。仅覆盖该目录及查询时刻。 |
| 串行交接（隔离验证） | 16:06 UTC 复跑 tests/test_queue_evidence.py；5 个虚拟任务测试全部通过，耗时 0.562 秒 | 正常返回后可进入下一项；硬退出残留锁，逐组授权检查存在反例。真实交接尚待首组结束和下一组同控制器启动收据。 |
| 本地训练（持久收据） | full_base_17/latest.json，16:06:01.401 | 已保存 356/512；较整点增加 16 轮。此记录不提供模型性能结论。 |

## 2026-11-19 01:49 UTC 独立复核

2026-11-19 01:49:15 UTC 独立只读核验：启动 5/25，固定 512 轮检查点字节及身份核验 4/25，训练结束与统一评价收据齐全 0/25。unet_tail_17 已保存 122 轮，STATUS 第 123 轮。 [阶段核验及收据](ISRJ2_NEXT_STAGE_VERIFICATION.md)

<!-- UTC slot 2026-11-19T02 -->
## 2026-11-19 02:00:01 UTC

本地有启动收据 5/25，训练结束收据核验 0/25，无启动收据 20。持久轮次 full_base_17=512、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=155；控制器状态 TRAINING，记录轮次 156。B 状态未核实。 512 固定检查点核验 4/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 37、unet_tail_17 None。本地 HEAD fcf9245c0bd9，检查前远端 c4bdcbf8cdf2。

临时目录六项队列测试通过，含原循环授权反例与隔离许可检查。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T03 -->
## 2026-11-19 03:06:13 UTC

本地有启动收据 5/25，训练结束收据核验 0/25，无启动收据 20。持久轮次 full_base_17=512、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=362；控制器状态 TRAINING，记录轮次 363。B 状态未核实。 512 固定检查点核验 4/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 207。本地 HEAD fcf9245c0bd9，检查前远端 36f4ae6ab982。

独立抽象二点算例复算：均值平方误差 1/16，独立同分布样本互比 1/8；不外推到模型性能。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T04 -->
## 2026-11-19 04:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=23、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 TRAINING，记录轮次 24。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 None、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 150。本地 HEAD fcf9245c0bd9，检查前远端 4c605aa04d63。

五份父场景表字节指纹与原完成收据一致；独立单位仍为父场景。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T05 -->
## 2026-11-19 05:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=213、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 TRAINING，记录轮次 214。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 190、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 0255165700cc。

出版方书目核验 [Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html)；仅核题名及原文可达性。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T06 -->
## 2026-11-19 06:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=403、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 TRAINING，记录轮次 404。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 190、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 dab476344e6f。

五项历史摘要哈希、15 条逐种子记录与门槛复算一致；2 项通过 DEV。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T07 -->
## 2026-11-19 07:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 32、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 fdbf58bdb54e。

临时目录六项队列测试通过，含原循环授权反例与隔离许可检查。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T08 -->
## 2026-11-19 08:00:01 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 a037e37e8f29。

独立抽象二点算例复算：均值平方误差 1/16，独立同分布样本互比 1/8；不外推到模型性能。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T09 -->
## 2026-11-19 09:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 44e38dfdef92。

五份父场景表字节指纹与原完成收据一致；独立单位仍为父场景。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T10 -->
## 2026-11-19 10:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 556d4ccc8da6。

出版方书目核验 [Composing Normalizing Flows for Inverse Problems](https://proceedings.mlr.press/v139/whang21b.html)；仅核题名及原文可达性。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T11 -->
## 2026-11-19 11:00:01 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 2231a2e40819。

五项历史摘要哈希、15 条逐种子记录与门槛复算一致；2 项通过 DEV。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T12 -->
## 2026-11-19 12:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 80f949e31833。

临时目录六项队列测试通过，含原循环授权反例与隔离许可检查。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T13 -->
## 2026-11-19 13:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 95a7a90daf8e。

独立抽象二点算例复算：均值平方误差 1/16，独立同分布样本互比 1/8；不外推到模型性能。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T14 -->
## 2026-11-19 14:00:01 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 77e5f51c01a8。

五份父场景表字节指纹与原完成收据一致；独立单位仍为父场景。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T15 -->
## 2026-11-19 15:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 38eb94c21000。

出版方书目核验 [Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces](https://proceedings.mlr.press/v97/becker19a.html)；仅核题名及原文可达性。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T16 -->
## 2026-11-19 16:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 ca039bc8e6da。

五项历史摘要哈希、15 条逐种子记录与门槛复算一致；2 项通过 DEV。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T17 -->
## 2026-11-19 17:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 293c89e36603。

临时目录六项队列测试通过，含原循环授权反例与隔离许可检查。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T18 -->
## 2026-11-19 18:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 d61360952c7f。

独立抽象二点算例复算：均值平方误差 1/16，独立同分布样本互比 1/8；不外推到模型性能。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T19 -->
## 2026-11-19 19:00:01 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 e490dfd440e9。

五份父场景表字节指纹与原完成收据一致；独立单位仍为父场景。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T20 -->
## 2026-11-19 20:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 a6a560d9624f。

出版方书目核验 [Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html)；仅核题名及原文可达性。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T21 -->
## 2026-11-19 21:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 02c28dc5b7e4。

五项历史摘要哈希、15 条逐种子记录与门槛复算一致；2 项通过 DEV。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T22 -->
## 2026-11-19 22:00:01 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 b6cf7b0cc3fb。

临时目录六项队列测试通过，含原循环授权反例与隔离许可检查。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-19T23 -->
## 2026-11-19 23:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 555618c5dd11。

独立抽象二点算例复算：均值平方误差 1/16，独立同分布样本互比 1/8；不外推到模型性能。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。

<!-- UTC slot 2026-11-20T00 -->
## 2026-11-20 00:00:00 UTC

本地有启动收据 6/25，训练结束收据核验 0/25，无启动收据 19。持久轮次 full_base_17=512、full_base_29=435、full_tail_17=512、no_dechirp_base_17=512、no_dechirp_tail_17=512、unet_tail_17=512；控制器状态 MIGRATED_TO_SERVER，记录轮次 436。B 状态未核实。 512 固定检查点核验 5/25，训练及 DEV 收据齐全 0/25。 相对前次快照的保存轮次变化：full_base_17 0、full_base_29 0、full_tail_17 0、no_dechirp_base_17 0、no_dechirp_tail_17 0、unet_tail_17 0。本地 HEAD fcf9245c0bd9，检查前远端 2c39f8127987。

五份父场景表字节指纹与原完成收据一致；独立单位仍为父场景。 B 只读访问未完成 server read failed: 255 []。下一次复核进程、持久收据和环境冲突 D01/D02。
