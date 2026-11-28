# ISRJ_2 ReCoVP-Hybrid 256轮原稿参照实验进度

## 2026年11月28日05时04分PST核验与分机制补算

本次服务器快照对应UTC 13时04分。GPU0利用率85%，显存15937/16311 MiB，GPU1利用率100%，显存16005/16311 MiB，物理内存占用43.43%。运行负载随推理和CPU评价变化，后续GPU任务在利用率低于85%、物理内存低于90%时自动派发。

GPU0的R5五个Hybrid seed持续并行。seed67、97、53、41、109分别保存85、86、77、57、50个分块，合计355块，每块16个父场景、三种机制。五seed完整条件格分别为2、2、2、1、1。已逐块核验本快照全部355块的parent ID、绑定SHA、三机制形状、有限值及指标别名。独立物理样本量仍为共享的512个父场景。

核查发现，旧R3入口在保存前对三类机制求均值，落盘指标形状为父场景一维。旧文件按三机制平均的聚合参照登记。三机制主表由新补算入口生成。Max-TF已核验64块并完成4个条件格，GMM-OMP已核验5块。两个CPU worker与双卡任务并行，资源和进度证据见`DUAL_GPU_MECHANISM_RECOVERY_20261128.json`。

新后处理`tools/original_replication_mechanism_summary.py`已通过WMI持久启动。它检查R3/R5组件与幅度变换完全一致，在重叠条件直接引用R5前256个物理父场景，逐块验证绑定并生成分机制、逐seed表。该路径复用已有Hybrid推理。传统方法使用独立补算分块。旧后处理监督脚本的自身进程匹配已修复，旧汇总调用已转接新入口。

P2收据覆盖资产和工程检查，完整科学一致性验收正在补做。正式CVED-Net同数据、256轮、3seed训练仍待实现和派发。R1已修正典型JSR至20 dB、四状态/脉冲索引和设备准入，工程维度检查通过，服务器运行及完整图件验收待资源与后续扩展。R1预览收据单列，参考曲线、三机制、16父场景相干诊断及图件QA列为后续验收项。

本次持久执行已覆盖R5 Hybrid队列、R3传统方法分机制补算和分机制结果衔接。R2、R4、R6及正式CVED-Net的完整执行入口继续列在任务书中待补齐。seed109沿用原始preflight缺失及恢复桥接的追溯限制。

## 2026年11月28日04时50分PST双卡并行快照

根据用户要求，R5检测网格已提前与R3并行。GPU0正在运行ReCoVP-Hybrid的67、97、53、41、109五个256轮seed，GPU1继续运行R3已有任务。R5已写入100个父场景分块，各块包含16个物理父场景与三种干扰机制，seed67已完成首个512父场景条件格。运行期间已观测到GPU0利用率100%、GPU1利用率99%；随后快照的服务器物理内存占用为44.48%。利用率随推理和CPU指标计算阶段变化。

R5入口为`tools/original_replication_detection.py`及`tools/start_original_replication_detection.ps1`。持久队列先计算150个优先SNR/JSR格，再补齐310格，每个seed使用相同512个物理父场景。每次派发重新读取两卡利用率与物理内存，完成CUDA初始化后再查询下一次资源。队列登记进程和设备身份，并按已保存父场景分块恢复。

已抽查五个seed的落盘分块，验证parent ID、结果绑定、有限值、三机制维度及E_U、D_S/e_S、E_0、Hits别名。该快照覆盖R5中的Hybrid评价，R5四方法完整验收继续依总计划执行。资源、队列、进程和分块证据保存在`evidence/outputs/Original_paper_replication_256/GPU0_PARALLEL_R5_20261128.json`。

## 前次R3恢复快照

本次核查快照为UTC 2026年11月28日11时25分。总协议见MASTER_PLAN.md v1.2，任务状态见TASKS.json。当前实验对象明确为 ISRJ_2 的 ReCoVP-Hybrid。RSDR-Net 只作为前期 ISRJ1 的历史冻结参照或协议来源，不启动 ISRJ1 新实验。512暂停审计、8seed统一DEV评价和P2数据/源码核验已完成，R3主条件网格正在服务器B恢复运行。

## 2026-11-28 10:28 UTC 流水线现场

服务器入口为 `rsdr-b`，工程根目录为 `D:/code/Xin/ISRJ_222`。持久控制器为 `tools/original_replication_pipeline.py`，由 `tools/start_original_replication_pipeline.ps1` 通过WMI启动，状态文件为 `outputs/original_paper_replication_256_v1/pipeline/PIPELINE_STATUS.json`。P2已生成并核验 `P2_COMPLETE.json`，R3主网格已派发五个冻结Hybrid seed、CVEDNet、Max-TF和GMM-OMP，输出和日志均使用独立目录。缺失的97、109等帧校准已另行以WMI持久进程启动，等待GPU0资源后继续。

最近一次资源快照记录GPU0利用率1%、显存570/16311 MiB，GPU1利用率100%、显存16037/16311 MiB，主机物理内存占用24.29%。当前运行任务沿用已登记设备，后续派发严格重新查询资源并使用利用率低于85%的设备，主机物理内存低于90%时继续。512轮续训仍保持暂停。

R3原控制器在UTC 11:04:20因恢复代码将日志路径当作文件句柄调用`close()`而退出，已保存89个条件文件，现有计算worker未被停止。修复后的`tools/original_replication_suite.py`已部署，`recover_original_replication_r3.ps1`等待现有worker自然完成后从已保存条件继续，`recover_original_replication_postprocessing.ps1`现已衔接分机制增量核验。该故障属于调度器错误，未发现模型、数据或checkpoint身份错误。

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

首个Start-Process启动尝试在SSH会话退出后未存活，已依据项目既有WMI模式修复。新入口为 `tools/start_hybrid_256_seed_selection.ps1`，评价入口为 `tools/hybrid_256_seed_selection.py`。后续并行队列使用独立目录、任务身份登记及原子写入。现场快照和启动收据保存在 `evidence/outputs/Original_paper_replication_256/`，完整运行日志与父场景数组保留服务器。

## 四方法与后续实验

|工作包|当前完成数|计划范围|状态|
|---|---:|---|---|
|P0 暂停与恢复审计|8/8 seed|全部既有恢复点及端点备份|已核验|
|P1 统一DEV筛选|8/8 seed|完整排序后选5seed|已冻结67、97、53、41、109|
|P2 数据和指标一致性|工程核验完成|新版本、配对、阈值、四方法小样本|逐机制科学一致性补验中|
|R1 全视野时域/脉压/多普勒|0|ReCoVP-Hybrid前5seed及三种对照方法同场景图|典型预览等待派发，完整协议待扩展|
|R2 切片宽度|0/1100条件格|M2/M4、5强度、5宽度、11SNR、2种JSR口径|待评价|
|R3 主条件表|分机制补算中/105条件格|7SNR×5JSR×3机制，另补原稿网格|传统方法补算及Hybrid重叠条件复用已接通|
|R4 四专题|0/4专题|背景FA、机制泛化、无噪声目标、无干扰保持|待评价|
|R5 原稿热图|五seed合计8个SNR/JSR实例格|优先450格，补齐930格|GPU0五Hybrid seed并行，完整四方法继续推进|
|R6 初轮消融|0/13变体|3配对seed，最多39个新训练实例|待训练与复用核验|

现有 CVED-Net 适配器实际绑定第一篇数据训练、epoch320封存端点。计划 v1.1 已据此更正来源。它用于历史参照与适配核验，正式主比较安排本工程数据上的3seed、epoch256 CVED-Net。新训练和扩展实验的性能结果按实际完成收据填报。前期 RSDR-Net 仅保留为论文中已有的历史冻结参照，不纳入本轮新实验任务。

已核验原稿e_S与现稿D_S公式相同，统一展示D_S（原e_S），底层表保留别名。极端SNR组为[-30,-20)，普通组为[-20,0]。区间估计按物理parent ID联合重采样，逐seed与跨seed离散分别保存。

## 持续复查与存储

已有周期任务 `isrj-2` 已更新为每两小时在本聊天复查总计划，状态ACTIVE，目标聊天为 `01a220ef-c556-74b1-877e-158ab6cbdd79`。复查内容包括实际完成格、训练恢复epoch、图表、SHA、失败诊断及后续授权工作。512续训按新优先级与剩余算力安排。

总计划、TASKS.json及本报告保存在本地 `G:/Project Code/ISRJ_222/reports/Original_paper_replication_256/`，并镜像到服务器同名reports目录和用户文档的 `02_回执/Original_paper_replication_256/`。数据与复数组保留服务器专用目录，R2保存全部请求条件的Y11恢复输出，R1保存示例全四状态波形。逐父场景四状态指标、输入组件、变换和模型身份支持其余条件复算。

本轮默认实施选择包括5%等帧FA的DEV排序、双JSR切片口径、完整310格/机制补充、初轮3seed消融及最多4项重点变体补到5seed。完整科学实验仍在执行，后续阶段依TASKS.json核对。
