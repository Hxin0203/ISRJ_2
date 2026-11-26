# ISRJ_2 对比实验总计划（除本模型外）

版本：2026-11-26（UTC）。本文件规定本轮及后续对比实验的顺序、资源、输入身份、产出与对应的论文表图。ReCoVP-Hybrid 的8个种子仍在服务器B上训练，本计划不改动该任务，也不新增 Hybrid 种子。所有长任务只在服务器B（`rsdr-b`，`D:/code/Xin/ISRJ_222`）运行；GPU 准入按利用率低于85%判定，不停止、不迁移任何在跑任务。不读取 CONFIRM 或历史 TEST 作为选型依据。

## 1 可复用资源清点（E0，已完成）

| 资源 | 位置 | 身份核验 | 用途 |
|---|---|---|---|
| Hybrid seed17 逐场景数组 | `outputs/recovp_hybrid_seed17/seed_17/dev_epoch_256_parent_values.npz` | SHA `b81774bf…7bda`，256个场景ID | 配对参照 |
| CorrectionNet（原版，8种子中的17/29/41） | `outputs/paper2_256_epoch_8seed_20261114/seed_*/dev_epoch_256_v4.npz` | 场景ID与Hybrid逐一相同 | 基线 |
| CorrectionNet 原损失复训对照、+尾峰 | `outputs/correction_original_control_20261117_v1`、`correction_mf_tail_20261117_v1` | 同上 | 基线 |
| CorrectionNet 消融（去去斜/去RMS/去脉压损失/去目标损伤损失） | `outputs/unet_ablation_3seed_256_20261115/no_*` | 同上 | 基线、损失项证据 |
| 普通U-Net及+尾峰/+噪声归一/+频域坐标/+膨胀 | `unet_ablation…/unet`、`unet_mf_tail_screen_20261117_v2`、`unet_noise_norm_screen…`、`unet_frequency_screen…`、`unet_dilation_screen…` | 同上 | 基线 |
| ISRJ_1 RSDR-Net 及其 1103 协议代码 | `D:/Project Code/ISRJ`、`tools/rsdr_adapter.py`、`tools/rsdr_vendor/` | 原生种子40312102，32轮 | 大模型参照 |
| ISRJ_1 切片宽度与未受扰比例协议 | `D:/Project Code/ISRJ_publication_width_20261113_v4/evidence/outputs/{width_dense_m2,eta_m2}_20261114/code/` | PROTOCOL_LOCK 与 SOURCE_MANIFEST | E5 的条件生成与雷达指标 |
| 1103 TEST 网格（CorrectionNet、传统方法、RSDR） | `evidence/outputs/paper2_1103_test_20261114`、`paper2_1103_test_classical_20261114` | 已封存 | 仅作历史背景，不与 Hybrid 混排 |

## 2 实验清单与状态

| 编号 | 实验 | 输入与规模 | 资源 | 产出→论文位置 | 状态 |
|---|---|---|---|---|---|
| E1 | DEV256 逐场景配对比较：Hybrid 对12个基线族 | 现有数组，12族×3种子×256场景，4000次场景bootstrap | 服务器CPU，<1分钟 | `E1/PAIRED_DEV_STATS.json` → 表4扩展、5.5节 | **已完成** |
| E2 | 基线训练随机性：12族的3种子均值与标准差 | 同上 | 同上 | 同一文件 → 表7基线部分 | **已完成** |
| E3 | 相同实测虚警下的检测：复用 2026-11-16 已登记的等虚警集（校准1024、评测2048个场景，种子2026111601/02，与TRAIN/DEV/确认集无重叠），统一评测器导出逐帧CFAR最大比值与目标单元比值；每个方法在自己的校准输出上取1%帧分位作门限；扫门限得到Pfa–Pd曲线 | 12族×3种子 + Hybrid 已完成种子（17、29）冻结检查点（只评测） | 每个任务启动前选利用率<85%的GPU，约3分钟/模型 | 表5、图7 | **运行中**（2026-11-26）；验收通过：CorrectionNet seed17 门限、Pd、Pfa 与11-16结果逐位一致，已完成的18个旧模型行全部复现。原计划新建CAL256取消，不再生成新数据 |
| E4 | ReCoVP-Base、P0-NoDechirp-Tail、RSDR-Net 的逐场景数组 | 定位`recovp_formal_dev/scoring`来源检查点后重评 | GPU1 | 补齐表4前后对照的配对区间 | 待定位检查点 |
| E5 | 受控条件扫描：SNR×JSR×三种转发方式；切片宽度（A/B组）；未受扰脉冲比例 | 复用ISRJ_1 v4协议的条件定义，新种子独立生成SWEEP集（不用TEST种子），每格64个场景 | GPU1 + CPU | 图9、图10 | 待E3评测器完成后复用同一评测接口 |
| E6 | 传统方法 Max-TF、GMM-OMP 在 DEV256 与等虚警集 | Max-TF 采用第一篇论文同款配置（同一份 max_tf.py，`MaxTFConfig` 默认值：窗长65、步长16、平滑5、分位0.7）；1103 配置（51/12/3）保留为敏感性对照；GMM-OMP 与第一篇逐行一致（仅导入路径不同，默认参数）。BCS 按用户要求不进论文表格，已算结果只保留 | 服务器CPU | 5.3节对比方法表 | **运行中**：1103配置 Max-TF、GMM-OMP 已完成；第一篇配置 Max-TF 的 DEV 已完成，等虚警运行中 |
| E7 | 外部学习方法 CVEDNet：直接使用用户在第一篇已训练并封存的第320轮权重（官方代码 @1f1dbf3，11.02 M 参数，SHA `8cbdad02…663a`），不在 ISRJ_2 数据上再训练 | DEV256 + 等虚警集，只评测 | GPU（<85%准入） | 5.3节对比方法表 | **运行中**：DEV256 已完成，等虚警运行中。注意：该权重在第一篇 TRAIN512 上训练、监督无噪目标 S，属跨数据分布的零样本比较 |
| E8 | 计算开销：全部模型同机单脉冲/8脉冲批量 GPU 与 CPU 计时，Hybrid两阶段分项 | 冻结检查点 | 需GPU空闲窗口；CPU可先做 | 表8 | 待GPU空闲 |
| E9 | 典型场景复数输出：按Hybrid目标损伤配对差的第10/50/90百分位预先选3个场景，导出各模型实部波形、脉压与RD图 | 3场景×全部模型 | GPU1，分钟级 | 图5、图6 | 待E3评测器 |
| E10 | 相干残余诊断：受扰无目标状态下残余的脉压峰值/残余能量比，检验“结构化撤销留下的残余相干性更低”的解释 | 与E3同一次推理导出 | 同E3 | 6.1节机理段 | 待E3 |
| —— | 本模型相关（Hybrid 8种子、匹配消融） | —— | —— | 表6、表7 | 不在本轮；Hybrid完成后接入同一评测器 |

## 3 执行顺序与闸门

1. E1/E2 已完成，结果只读保存。
2. E3 先写统一评测器并做两项验收：（a）对任一已有基线在DEV上重算的 Pd、帧虚警与旧数组逐场景完全一致；（b）门限缩放因子为1时CAL上的帧虚警与名义CFAR一致。验收通过后再批量运行。
3. CAL256 生成前写登记文件（父版本、用途、种子、规模、SHA），生成后核对与TRAIN/DEV/TEST无场景ID重叠。
4. E9、E10 与 E3 共用一次推理。
5. E5 的SWEEP集同样追加登记；先跑学习类基线，传统方法按CPU余量排在后面。
6. E7 改为直接复用用户已训练的 CVEDNet（2026-11-26 用户指定：对比方法为 Max-TF、OMP、CVEDNet，BCS 暂不考虑）；MREDNet 复现与在 ISRJ_2 上重训 CVEDNet 均暂不进行，除非之后另行决定。
7. Hybrid 8种子完成后，用同一评测器接入，无需重写任何流程。

## 4 统计口径

物理场景为独立单元；三种转发变体先在场景内平均。配对差值为 Hybrid 减基线，基线取3个种子的逐场景平均；同时报告逐种子差值。区间为4000次场景bootstrap的2.5%/97.5%分位。种子间标准差单独报告，不与场景区间合并。单种子 Hybrid 与3种子平均基线的比较只说明该训练实例的位置，Hybrid 的训练随机性待8种子结果补齐。

## 5 2026-11-26 17:08 UTC 更新：服务器总控接管剩余步骤

- 总控：`tools/paper2_baseline_suite/controller.py`，服务器 B 上以 WMI 托管进程运行，状态写入 `outputs/paper2_baseline_suite_20261126/CONTROLLER_STATUS.json`（含心跳），日志在 `controller_logs/`。本机断线不影响执行。
- 顺序：等 E3 队列完成 → E7b RSDR-Net（第一篇冻结权重，原生种子 40312102、32 轮，DEV256 + 等虚警）→ 汇总 `SUITE_SUMMARY.json`/`SUITE_TABLES.md` → E5 登记、生成、评测、汇总 → 补入 E3 队列启动后才完成的 Hybrid 种子（含 seed41）→ 之后每 30 分钟检查新完成的 Hybrid 种子并接入 E3/E5，至 8 个种子或 96 小时为止。
- E4 决定：RSDR-Net 接入同一评测（E7b）。ReCoVP-Base/P0 的检查点所在工作树已不存在、权重未上传 GitHub，不重新训练；论文中只引用已封存的 DEV 点估计与各自区间。
- E5 决定：SNR {-25,…,10} dB × JSR {0,10,20,30} dB，共 32 格，每格 64 个物理场景，第 k 格种子 2026112700+k；先写 `E5/REGISTRATION.json` 再生成，生成后核对与 TRAIN/DEV/确认集/两套等虚警集无场景重叠；只用于评测。方法：Hybrid（已完成种子）、CorrectionNet、U-Net（各3种子）、CVEDNet、RSDR-Net、Max-TF（第一篇配置）、GMM-OMP。切片宽度与未受扰比例扫描需要生成器新接口，本轮不做。
- BCS 按用户要求不进论文表格，汇总时排除。
