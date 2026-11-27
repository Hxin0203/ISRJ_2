# Hybrid 剩余三种子暂停回执

更新时间：2026-11-26 20:59 PST。按用户要求暂停训练，后续由用户另行安排补足，不在资源空闲时自动续接。

| seed | 状态 | 已保存 epoch / 原计划 | Adam 更新次数 | 原物理 GPU |
| --- | --- | --- | --- | --- |
| 79 | 已暂停 | 75 / 256 | 3,600 | 1 |
| 97 | 已暂停 | 139 / 256 | 6,672 | 0 |
| 109 | 尚未启动，排队取消 | 0 / 256 | 0 | 未分配 |

seed 17、29、41、53、67 已完成的五种子报告保持原样。以上两个中途结果不能当作 epoch 256 的完整结果并入五种子统计。

## 停止与保存

先停止 8-seed、split、queue recovery 三套控制器及等待启动器，再等待两个 worker 各自完成一轮完整检查点发布。seed 79 在 2026-11-27 04:56:45 UTC 停止，seed 97 在 04:57:36 UTC 停止。训练程序没有优雅暂停接口，因此在核验完整 epoch 检查点后终止指定 worker；下一轮可能已执行的少量更新未保存，恢复时从下一轮起重算。

没有改动冻结训练源码，没有删除 PROCESS.lock 元数据，没有终止无关作业。暂停完成及完整性审核结束时，两次进程检查均未发现这些控制器、启动器或训练 worker 残留。审核时 GPU0 利用率为 4%、显存 885 MiB，GPU1 利用率为 0%、显存 0 MiB；这是瞬时快照，GPU0 残留用量未作为本训练仍在运行的证据，也未清理其他占用。

## 数据存放与完整性

原始训练目录（服务器 B）：

- `D:/code/Xin/ISRJ_222/outputs/recovp_hybrid_seed79/seed_79/`
- `D:/code/Xin/ISRJ_222/outputs/recovp_hybrid_seed97/seed_97/`

完整额外备份（同服务器独立目录，非异地容灾备份）：

- `D:/code/Xin/ISRJ_222/backups/recovp_hybrid_user_pause_20261127/recovp_hybrid_seed79/`
- `D:/code/Xin/ISRJ_222/backups/recovp_hybrid_user_pause_20261127/recovp_hybrid_seed97/`
- 上述备份根目录下 `sources/` 保留冻结绑定源码副本，另有暂停与完整性审核 JSON。

seed 79 保留 recovery epoch 000–075 共 76 个、32/64 里程碑和 latest，共核验 79 个权重文件；seed 97 保留 recovery epoch 000–139 共 140 个、32/64/96/128 里程碑和 latest，共核验 145 个权重文件。全部文件 SHA256 与 sidecar 一致，恢复点连续，没有遗漏已完成轮次。

两份 latest 均已在 CPU 载入，确认模型 106 个状态项、Adam 100 个参数状态，以及 Python、NumPy、torch CPU、torch CUDA 随机数状态存在；global_step 与 epoch×48 一致。源码、train/dev 数据、CONFIG、PREFLIGHT 和检查点身份绑定均核验一致。额外备份的 237 / 433 个文件逐文件 SHA256 与原目录一致。

| seed | 最新可恢复权重 | SHA256 |
| --- | --- | --- |
| 79 | `recovery/epoch_075.pt` | `1c215e815c5cc114cfe33e68f2e09fde29b99c988e45ebc9d73ce6d1eae067f8` |
| 97 | `recovery/epoch_139.pt` | `9c71cc2b1ecf1b9121d493ed85beb452713643ff58eefff3472c0dae66ce11f9` |

训练数据本体保持原地，未复制或上传 GitHub；训练权重及原始完整日志也未上传。审核证据存放在项目 `evidence/outputs/recovp_hybrid_user_pause_20261127/`；本回执及审核 JSON 同步备份到 `G:/Files/Markdown/ISRJ_2/opus对话/02_回执/Hybrid_pause_20261127/`。

## 后续续训交接

必须先得到用户新的明确续训指令。单凭 GPU 低于 85% 不构成续训授权。未来在服务器 B 执行，继续保留全部已完成恢复点；79 从 epoch 76 起、97 从 epoch 140 起、109 从头训练，原计划各 256 轮。

恢复前再次验证来源、数据、配置、预检和检查点 SHA，按保存的模型、Adam、随机数和 global_step 恢复，并核对主机、根目录和 GPU UUID。现有 extended 包装入口会重新执行预检并写出预检文件；不能未经核查直接重跑导致冻结绑定变化。本次仅停止保存，没有执行续训或验证新的跨 GPU 恢复流程。

现有流水线不识别暂停标记。停止其进程才是本次阻止自动续接的实际措施，`USER_PAUSED.json` 是交接状态证据；项目 AGENTS.md 已持久记录不得自动重启。不要把旧 STATUS.json 的历史 TRAINING 内容解读成当前状态。
