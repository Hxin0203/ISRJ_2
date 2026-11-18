# 18 篇文献按轻量网络创新来源分类

这张表逐篇对应研究笔记中的 18 篇文献。“可直接评估”表示能用来审计现有 CorrectionNet 的方法主张或成本。“可迁移假设”表示论文提出了值得检验的通用恢复原理，其在 CorrectionNet 上的收益未知。表中新增训练方案以民用图像为对象。部分文献贡献的是评价方法、数据或近邻证据，本身没有提出轻量网络。

| 创新来源与原文 | 对 CorrectionNet 的判断 | 实验或核验方案，以及什么结果算有信息 |
|---|---|---|
| **训练信息 · 1** [Dynamic Contrastive Knowledge Distillation，AAAI 2025](https://arxiv.org/abs/2412.08939) | 可迁移假设。恢复教师能提供干净标签以外的训练偏置；论文中的动态对比蒸馏与普通输出模仿、按数字类别挑选教师是三种方法。 | 在手写图像上比较无教师、普通输出蒸馏、选择性蒸馏。选择性还要胜过总权重匹配和随机选择。这次实际检验了后两类对照，未复现论文的动态对比方法。 |
| **训练信息 · 2** [Knowledge Transfer with Jacobian Matching，ICML 2018](https://proceedings.mlr.press/v80/srinivas18a.html) | 可迁移假设。学生可以学习教师对输入变化的响应；这一大类思想已有先例，旧教师在部分指标占优不等于其所有响应都值得模仿。 | 在有成对曝光参考的照片上，把仅匹配教师输出与额外匹配亮度响应分开比较；同时测参考误差和未见曝光下的响应误差。输出模仿能解释全部收益时，不认领响应迁移贡献。 |
| **训练信息 · 3** [Sobolev Training，NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html) | 直接用于审计新颖性。函数值、局部导数与两次有限变化是不同的监督对象，已有的配对误差不自动等同于导数监督。 | 在具有已知亮度变换的图像上比较端点监督和端点加响应监督；用多个变化幅度检验局部关系的适用范围，两组共享原始图像及标签权限。 |
| **训练信息 · 4** [Noise2Noise，ICML 2018](https://proceedings.mlr.press/v80/lehtinen18a.html) | 有条件可迁移。重复观测本身不足以保证监督无偏，需要数据符合相应的条件期望假设。 | 固定图像和小 U-Net，比较干净参考、独立噪声参考、相关噪声参考。若假设失效时优势消失，可解释收益来自何种数据关系。 |
| **训练信息 · 5** [DeepInterpolation，Nature Methods 2021](https://www.nature.com/articles/s41592-021-01285-2) | 有条件可迁移。它解释了哪些内容在相邻观测中可预测；平滑后的低均方误差不足以说明稀少细节被保留。 | 在民用图像序列中分别评价静态纹理和只出现一帧的细节；用相同上下文的监督基线，记录细节消失率和平均恢复误差。 |
| **骨干简化 · 6** [NAFNet，ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf) | 可迁移的是“逐项证明组件必要性”。同容量 U-Net 已是强对手，仅借用一个块不足以支撑新颖性。 | 在照片去噪中每次只删改一个块，锁定通道、数据、训练和计时方法；若简单版本在预设非劣界内且更快，保留简单版本。 |
| **骨干简化 · 7** [nnU-Net，Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z) | 可直接评估配置公平性。其主要证据来自医学分割，一个被充分配置的 U-Net 对照同样适用于评估恢复论文的架构主张。 | 普通 U-Net 和候选用等额开发搜索资源选择宽度与训练设置，锁定后比较多个成本档位。看到自己的最佳点后临时挑弱基线，不构成胜出证据。 |
| **结构性质 · 8** [Normalization-Equivariant Neural Networks，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/12143893d9d37c3569dda800b95cabd9-Paper-Conference.pdf) | 可迁移假设。该论文约束的是图像强度的平移和缩放；CorrectionNet 的其他变换主张需要各自的假设与证据。 | 在图像上分别测“变换前后输出是否满足关系”和“未见亮度下恢复是否改善”；加入相同数据增强的 U-Net。前一项变好不能替代后一项。 |
| **结构性质 · 9** [Normalization Equivariance for Arbitrary Backbones，ICML 2026](https://proceedings.mlr.press/v306/saied26a.html) | 可作简单替代解释。某些性质能由参数无关的输入输出包装获得，未必需要专用骨干。 | 对该论文覆盖的图像强度变换，比较原 U-Net、带包装的 U-Net、专用结构；若两者性质与效果相当，专用结构只有在成本等方面另有收益才有保留理由。 |
| **部署效率 · 10** [ECBSR，ACM Multimedia 2021](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf) | 可迁移假设。可精确折叠的训练分支与蒸馏不同：前者可验证计算等价，后者的知识保留需要实测。 | 在图像超分辨率上核验折叠前后输出误差，再用相同的最终推理图比较不同训练方式的质量。分别报告数值等价、质量与设备计时。 |
| **部署效率 · 11** [FasterNet，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.pdf) | 可直接评估。相同参数数目或相似卷积运算量不保证相同延迟，现有 CorrectionNet 与 U-Net 的计时正能说明这个区别。 | 做同设备、同精度、同批量的热身后计时，记录中位数、P95、峰值内存和前后处理范围。给基线选择相同延迟档位的机会。 |
| **风险与工作点 · 12** [Learn then Test，Annals of Applied Statistics 2025](https://doi.org/10.1214/24-AOAS1998) | 可直接用于设计独立校准。开发集满足损伤门槛，不能代替新数据上的风险声明。 | 对预先定义的有界超限事件，用独立校准样本筛选工作点；报告没有可接受工作点的情况。未验证有界性的连续指标不能直接套用有界风险结论。 |
| **风险与工作点 · 13** [Pareto Testing，ICLR 2023](https://arxiv.org/abs/2210.07913) | 可直接用于比较质量、保真和成本。完整实施涉及风险控制，单画散点图还不足以称为 Pareto Testing。 | 在开发集找候选，在未参与选择的样本检验多个约束；和只做全局强度调节的基线比较。区分选到较好工作点与改变可达到的性能边界。 |
| **风险与工作点 · 14** [Conformal Bounds on Full-Reference Image Quality，TMLR 2025](https://arxiv.org/abs/2505.09528) | 可迁移评价想法。单张未来图像的质量界、总体平均损伤和罕见失败概率有不同含义。 | 在有参考照片上用独立数据校准质量界；新图上同时报告覆盖率和区间宽度。即使覆盖达标，区间过宽也可能没有实用价值。 |
| **证据与归因 · 15** [nnU-Net Revisited，MICCAI 2024](https://papers.miccai.org/miccai-2024/562-Paper2847.html) | 可直接评估骨干归因。训练改进也应给普通 U-Net；仅与旧训练的 U-Net 比较不能识别架构收益。 | 在通用恢复数据上做骨干 × 训练方法的四组比较，分别报告各骨干相对自身的收益、相同训练下的绝对差距，以及两者训练收益之差。 |
| **证据与归因 · 16** [Accounting for Variance in ML Benchmarks，MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html) | 可直接评估不确定性。一次训练下的大量重复观测，不能替代独立训练种子。 | 两次图像试验均保持种子配对、按原图重采样、保留逐种子负结果；区间明确限定于已训练的三个模型对。 |
| **外部数据 · 17** [SIDD，CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Abdelhamed_A_High-Quality_Denoising_CVPR_2018_paper.pdf) | 可作通用恢复主张的外部检验。SIDD 提供手机实拍数据；它不是轻量架构，也不能充当雷达任务确认集。 | 如有新假设进入照片研究，按完整场景、设备划分，防止同场景裁块跨集合。分别报告细节保真、恢复质量、延迟和设备间排序。 |
| **领域近邻 · 18** [ISRJ-FNet，IEEE Signal Processing Letters](https://ieeexplore.ieee.org/abstract/document/11663417/) | 用于界定新颖性。发表摘要已涉及扰动下一致性，不能认领这项宽泛概念；现有阅读深度不足以判断两个方法等价。 | 做全文证据对照表，逐项记录任务对象、训练和推理可用信息、约束对象、对照和作者报告的局限。若关键差异只来自未核实的推断，暂缓新颖性声明。 |

## 如何使用这张表

若研究主张是“选择性教师有效”，首先面对第 1、2、3 项的训练近邻，并通过第 15、16 项的归因与不确定性检验。若主张是“模型更轻”，第 10、11 项决定要测的成本。第 12 至 14 项只支持各自定义下的风险声明。数据集覆盖、统计控制或已知数学性质，都不能单独证明新模型跑赢普通 U-Net。
