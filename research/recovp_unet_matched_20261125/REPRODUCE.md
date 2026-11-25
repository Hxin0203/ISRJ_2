# 精确同参数量 U-Net 结构对照

目的：直接检验 ReCoVP-NoEquiv 是否比同容量普通 U-Net 更有优势，而不是把与3.02M模型的差异当作结构贡献。

模型：标准三尺度实数1D U-Net，通道为(21,40,74)，卷积+GroupNorm+ReLU，最大池化与线性上采样；RAW_IQ输入、RMS归一，预测修正量，恢复输出始终为y-j_hat。全部 **58,943** 实参数都在前向和梯度图中，没有补零参数或闲置参数，恰好匹配ReCoVP-NoEquiv。

原UNet-Tail为59,714参数、仅多1.31%，其历史结果保留；此次另行重训精确容量版本。宽度只按计数选择，不看DEV选型。本对照属于容量匹配的完整架构对照，不能声称只改变了单个VP算子。

协议：同512 TRAIN、256 DEV，seeds17/29/41，256轮、48updates/epoch、micro-batch8；相同Adam、学习率、确定性cuDNN、TF32关闭及Lnew=endpoint+0.1PC+damage+0.05tail。训练循环经AST核验：除模型工厂、运行身份外，与ReCoVP正式循环一致。每轮权重、优化器与RNG全部保留，仅固定256轮评分，不筛seed或挑epoch。

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:ISRJ2_CORRECTION_ARM='tail'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
python -B tools/vp_1103_unet_matched_preflight.py
python -B -u tools/vp_1103_unet_matched_run.py
```

已有活动补充控制器时不重复启动。新任务先等原NoEquiv三种子训练/评分完成，再寻找GPU空闲时段；不停止、重启或修改原训练/扫描进程。本进程PyTorch显存限额25%以保留原扫描空间，首次准入需连续两次空闲>=4GiB、利用率<70%。若补充任务出现显存不足，保留尝试收据与最后完整epoch/RNG，保持micro8不变，等原扫描完整结束后再恢复同seed；不把资源异常隐瞒为成功，也不丢弃失败seed。

正式结果与ReCoVP-NoEquiv相同三个seed逐一配对；先在同parent内跨类型平均、跨三个seed平均，再对256独立parent聚类bootstrap95%CI。训练seed变异与parent不确定性不混作一种样本数。

报告Pd/Pfa、JCR、RD残峰、signed/abs CPL、目标角、signed/abs BVR、兼容eU/eS/e0与SJRIF，完整原生SNR/JSR分层。明确列出显著收益和显著损害，出现权衡时报告TRADEOFF，不凭单项虚警降低宣称“跑赢”。不修改已冻结ReCoVP Winner或原579工况计划。

STATUS.json是运行状态；PREFLIGHT.json是参数量、梯度与训练循环口径核验。正式SUMMARY/REPORT/VERIFICATION只在三种子完成评分后写入。当前不读取CONFIRM、历史TEST或旧Gate数组。
