# ISRJ_2 ReCoVP-Hybrid 256轮原稿参照实验进度

本次核查快照为UTC 2026年11月28日11时25分。总协议见MASTER_PLAN.md v1.2，任务状态见TASKS.json。当前实验对象明确为 ISRJ_2 的 ReCoVP-Hybrid。RSDR-Net 只作为前期 ISRJ1 的历史冻结参照或协议来源，不启动 ISRJ1 新实验。512暂停审计、8seed统一DEV评价和P2数据/源码核验已完成，R3主条件网格正在服务器B恢复运行。

## 2026-11-28 10:28 UTC 流水线现场

服务器入口为 `rsdr-b`，工程根目录为 `D:/code/Xin/ISRJ_222`。持久控制器为 `tools/original_replication_pipeline.py`，由 `tools/start_original_replication_pipeline.ps1` 通过WMI启动，状态文件为 `outputs/original_paper_replication_256_v1/pipeline/PIPELINE_STATUS.json`。P2已生成并核验 `P2_COMPLETE.json`，R3主网格已派发五个冻结Hybrid seed、CVEDNet、Max-TF和GMM-OMP，输出和日志均使用独立目录。缺失的97、109等帧校准已另行以WMI持久进程启动，等待GPU0资源后继续。

最近一次资源快照记录GPU0利用率1%、显存570/16311 MiB，GPU1利用率100%、显存16037/16311 MiB，主机物理内存占用24.29%。当前运行任务沿用已登记设备，后续派发严格重新查询资源并使用利用率低于85%的设备，主机物理内存低于90%时继续。512轮续训仍保持暂停。

R3原控制器在UTC 11:04:20因恢复代码将日志路径当作文件句柄调用`close()`而退出，已保存89个条件文件，现有计算worker未被停止。修复后的`tools/original_replication_suite.py`已部署，`recover_original_replication_r3.ps1`等待现有worker自然完成后从已保存条件继续，`recover_original_replication_postprocessing.ps1`随后执行R3汇总。该故障属于调度器错误，未发现模型、数据或checkpoint身份错误。

## 512轮暂停与恢复资料

服务器B的512续训控制器和活动worker已于UTC 2026年11月28日07时51分暂停，对应PST 11月27日23时51分。此次快照中的相关活动进程均属于256轮筛选。保存情况如下。

|seed|原训练端点|续训最后完整epoch|逐轮恢复点核验|
|---|---:|---:|---|
|17|256|326|原0至256，续256至326|
|29|256|355|原0至256，续256至355|
|41|256|325|原0至256，续256至325|
|53|256|326|原0至256，续256至326|
|67|256|280|原0至256，续256至280|
|79|256|280|原0至256，续256至280|
|97|256|279|原0至256，续256至279|
|109|256|256|原0至256，续训待启动|

暂停审计逐一检查8seed共2056个原训练恢复点的SHA、回执身份与序列，检查既有续训恢复点的SHA及序列，并将每个最新完整端点、配置、冻结源码、预检和完成记录复制到独立备份。最新payload已在CPU加载，Adam状态非空且包含Python、NumPy、CPU和CUDA RNG字段。后续实际续训入口继续执行优化器和随机状态恢复核验。

服务器备份目录为 `D:/code/Xin/ISRJ_222/backups/recovp_hybrid_512_priority_pause_20261128T075130Z/`。原256端点另有 `backups/recovp_hybrid_256_before512_20261127/seed_{seed}/` 独立归档。暂停审计收据为 `evidence/outputs/Original_paper_replication_256/PRIORITY_PAUSE_RECOVERY_AUDIT.json`。

seed109原preflight字节缺失，配置中记录SHA为 `ff5dda96ca4278eaa4f6e8465d99f50cc69351348441c9745c632153849fbaa1`，可用恢复桥接文件SHA为 `9e124798345223d591d2401670d98a4090e77c77229af27d15baca5d40be1346`。此历史追溯缺口随该seed的评价和报告保留。

## ReCoVP-Hybrid 8seed筛选

候选seed为17、29、41、53、67、79、97、109，全部固定epoch256。统一DEV有256个物理父场景，独立阈值校准有1024个物理父场景。按预登记的5%等帧FA工作点Pd排序，依次以D_S、E_U、seed打破同分，全部8个完成后冻结前5名。

服务器已在UTC 08时08分33秒通过WMI启动持久控制器。8个 ReCoVP-Hybrid seed 的统一 DEV 收据和 SHA 核验已完成。冻结的前5名为 **67、97、53、41、109**，详见服务器 `outputs/original_paper_replication_256_v1/seed_selection/SELECTED_SEEDS.json`。seed109 的原始 preflight 字节缺失，保留恢复桥接限制。

首个Start-Process启动尝试在SSH会话退出后未存活，已依据项目既有WMI模式修复。新入口为 `tools/start_hybrid_256_seed_selection.ps1`，评价入口为 `tools/hybrid_256_seed_selection.py`。进程锁用于约束同目录并发写入。现场快照和启动收据保存在 `evidence/outputs/Original_paper_replication_256/`，完整运行日志与父场景数组保留服务器。

## 四方法与后续实验

|工作包|当前完成数|计划范围|状态|
|---|---:|---|---|
|P0 暂停与恢复审计|8/8 seed|全部既有恢复点及端点备份|已核验|
|P1 统一DEV筛选|0/8 seed|完整排序后选5seed|seed17校准中|
|P2 数据和指标一致性|1|新版本、配对、阈值、四方法小样本|已核验，收据已保存|
|R1 全视野时域/脉压/多普勒|0|ReCoVP-Hybrid前5seed及三种对照方法同场景图|待R3后按依赖派发|
|R2 切片宽度|0/1100条件格|M2/M4、5强度、5宽度、11SNR、2种JSR口径|待评价|
|R3 主条件表|恢复中/105条件格|7SNR×5JSR×3机制，另补原稿网格|89个方法-条件文件已保存，等待现有worker完成后续跑|
|R4 四专题|0/4专题|背景FA、机制泛化、无噪声目标、无干扰保持|待评价|
|R5 原稿热图|0/930机制条件格|优先450格，补齐930格|待评价|
|R6 初轮消融|0/13变体|3配对seed，最多39个新训练实例|待训练与复用核验|

现有 CVED-Net 适配器实际绑定第一篇数据训练、epoch320封存端点。计划 v1.1 已据此更正来源。它用于历史参照与适配核验，正式主比较安排本工程数据上的3seed、epoch256 CVED-Net。新训练和扩展实验的性能结果按实际完成收据填报。前期 RSDR-Net 仅保留为论文中已有的历史冻结参照，不纳入本轮新实验任务。

已核验原稿e_S与现稿D_S公式相同，统一展示D_S（原e_S），底层表保留别名。极端SNR组为[-30,-20)，普通组为[-20,0]。区间估计按物理parent ID联合重采样，逐seed与跨seed离散分别保存。

## 持续复查与存储

已有周期任务 `isrj-2` 已更新为每两小时在本聊天复查总计划，状态ACTIVE，目标聊天为 `01a220ef-c556-74b1-877e-158ab6cbdd79`。复查内容包括实际完成格、训练恢复epoch、图表、SHA、失败诊断及后续授权工作。512续训按新优先级与剩余算力安排。

总计划、TASKS.json及本报告保存在本地 `G:/Project Code/ISRJ_222/reports/Original_paper_replication_256/`，并镜像到服务器同名reports目录和用户文档的 `02_回执/Original_paper_replication_256/`。数据与复数组保留服务器专用目录，R2保存全部请求条件的Y11恢复输出，R1保存示例全四状态波形。逐父场景四状态指标、输入组件、变换和模型身份支持其余条件复算。

本轮默认实施选择包括5%等帧FA的DEV排序、双JSR切片口径、完整310格/机制补充、初轮3seed消融及最多4项重点变体补到5seed。完整科学实验仍在执行，后续阶段依TASKS.json核对。
