# 固定主数据集使用说明

## 存放位置

- 本机数据位于 `G:/Project Code/ISRJ_222/data/hybrid_condition_master_20261127/`。
- 服务器归档位于 `D:/code/Xin/ISRJ_222/data/hybrid_condition_master_20261127/`。
- 实际复数分量保存在 `components/test_0_0.npz`，结构标签保存在 `STRUCTURAL_METADATA.json.gz`。
- `REGISTRATION.json` 固定网格、种子、场景选择和统计规则。`DATA_MANIFEST.json` 记录数据 SHA、生成器校验和集合重合检查。

生成种子为 2026112901，共 512 个物理 parent。每个 parent 有 direct、repeat、cyclic 三种干扰变体、8 个脉冲、每脉冲 1800 点，并可组成 E、S+E、J+E、S+J+E 四种状态。三类干扰共享目标和噪声。

## 从已有分量选择工况

在项目根目录运行 Python，选择器返回内存中的复数分量和身份信息。

```python
from tools.hybrid_condition_benchmark import select
from paper2.evaluation_1103_v4 import states_from_components

subset = select(snr=-28, jsr=20, start=0, stop=128)
states = states_from_components(subset["s"], subset["j"], subset["epsilon"])
print(states.shape)  # (128, 4, 3, 8, 1800)
```

主网格的 SNR 为 `−30、−28、−26、−25、−24、−22、−20、−18、−16、−15、−14、−12、−10、−8、−6、−5、−4、−2、0、5、10 dB`，JSR 为 `−10、0、10、20、30、35、40 dB`。选择器校验范围与分量 SHA，选择操作复用已保存数据。

基准分量在 SNR=JSR=0 dB 生成。幅度缩放使用 `S=S0`、`E=E0*10**(-SNR/20)`、`J=J0*10**(JSR/20)`。两个远端工况与原生成器直接生成结果核对，最大相对误差低于 5×10⁻⁸。

本轮登记使用索引 `[0,128)` 的 parent 和 48 个工况。索引 `[128,512)` 的 384 个 parent 留作后续登记子集。新增实验应在读取结果前保存选择清单、用途、比较指标、统计方法和门限策略。

## 结构标签

```python
import gzip
import json
from tools.hybrid_condition_benchmark import DATA

with gzip.open(DATA / "STRUCTURAL_METADATA.json.gz", "rt", encoding="utf-8") as handle:
    metadata = json.load(handle)

target_rows = metadata["targets"]
jammed_pulse_rows = metadata["jammed_pulses"]
```

标签包含 parent 索引、干扰类型、脉冲索引、采样窗宽度、采样周期、回放次数、采样相位、转发延迟、回放间隔及频偏。标签通过相同种子的完整重放恢复，全部保存数组逐元素一致。当前共 6201 条受扰脉冲记录。

当前样本覆盖窗宽 6–72 点、周期 36–260 点、回放次数 1–10、采样相位 0–250 点、延迟 1–96 点、回放间隔 0–64 点、频偏约 ±200 kHz。宽度 6 点也出现在当前冻结生成器的训练范围内。分析边界时应同时记录实际回放片段数和字典容量。

## 汇总与不确定性

极端工况使用 `−30 ≤ SNR < −20 dB`，普通工况使用 `−20 ≤ SNR ≤ 0 dB`。本轮将公共端点 −20 dB 归入普通组。高于 0 dB 的主网格点作为扩展组单列。

组内先在每个 parent 内平均干扰类型，再在登记的 SNR×JSR 格间等权平均。每格同时保留逐种子指标和均值、SD、最小值、最大值。场景 bootstrap 联动抽取一个 parent 的全部工况和类型。

147 个幅度工况对应 75,264 个逻辑 condition-parent 组合，独立场景数为 512。本轮已评测的独立场景数为 128。后续使用相同 parent 扩展幅度网格时，场景独立样本量仍按所选 parent 计数。

本数据用于开发诊断。独立最终确认集继续按冻结协议单独验收。新增波形结构、更多物理 parent 或超出注册范围的幅度工况，应登记新的数据版本。

## 结果入口

`REPORT.md` 提供结论和限制，`CONDITION_TABLES.md` 提供逐格完整数值，`CONDITION_METRICS.csv` 保留种子离散程度，`SEED_REGIME_METRICS.csv` 保留逐训练实例的分组结果，`figures/` 保存 PNG 与 PDF 图。
