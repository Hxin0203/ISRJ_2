# CorrectionNet-Tail 参数量独立核验

核验时间为 2026-11-27 03:31 PST。本次在 CPU 上从当前模型定义实例化模型，使用两种输入批量运行前向与反向传播。

| 统计口径 | CorrectionNet-Tail | 标准 U-Net |
| --- | ---: | ---: |
| 登记参数总量 | 112,756 | 59,714 |
| requires_grad=True 的参数量 | 112,756 | 59,714 |
| 当前计算图有效参数量 | 59,714 | 59,714 |
| 当前配置闲置参数量 | 53,042 | 0 |

CorrectionNet-Tail 的三个门控模块合计 21,840 个闲置参数，两个可靠度传播模块合计 31,202 个闲置参数。当前配置采用 `gate_mode='none'` 和 `propagation_mode='static'`。门控分支直接返回特征，可靠度分支使用插值。这些固定分支解释了两次反向传播中相同的闲置参数集合。

实测输入形状为 `(2, 2, 1800)` 和 `(1, 2, 1800)`。每次先运行 `zero_grad(set_to_none=True)`，再对输出平方均值反向传播，以 `p.grad is not None` 统计当前计算图中的参数张量元素数。两模型输出及所产生梯度均为有限值。有效参数按整张量计数，梯度元素是否恰好为零不会改变该口径。

`wp7_local_timing.py` 第 118 行的 `sum(p.numel() for p in model.parameters())` 正确统计登记参数，得到 112,756。将这个字段与论文中的有效参数直接比较造成了此前的口径混用。只用 `requires_grad=True` 过滤仍会得到 112,756。

结论是另一智能体更正后的解释成立。论文中的 59,714 有实测及代码依据，应明确写作“有效参数量”。当前实现的登记参数总量为 112,756。历史预检已同时断言这两个数字，当前核心源码的 SHA256 与 2026-11-17 预检记录一致。

本次直接实测对象为 CorrectionNet-Tail 与标准 U-Net。十二类变体的逐项断言需结合各自模型定义及已有预检记录核对。

项目证据位于 `G:/Project Code/ISRJ_222/evidence/outputs/CORRECTIONNET_PARAMETER_AUDIT_20261127.json`，其中保存逐模块计数、环境、输入形状和源码 SHA256。交接说明补充至 `G:/Files/Markdown/ISRJ_2/opus对话/03_分析/17_说明_CorrectionNet参数计数口径_20261127.md`。
