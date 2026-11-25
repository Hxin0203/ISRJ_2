# ReCoVP-Hybrid seed 17 后台流水线

本次范围是现有 seed 17 训练至 256 epochs，随后评测固定 256 DEV parent，再进入审阅节点。训练已在服务器 B 运行；交接时不要再启动一份训练。

## 读取当前状态

服务器为 `ssh rsdr-b`，实验目录为 `D:/code/Xin/ISRJ_222`。服务器 PowerShell 中执行：

```powershell
Set-Location 'D:/code/Xin/ISRJ_222'
Get-Content 'evidence/outputs/recovp_hybrid_seed17/STATUS.json'
Get-Content 'evidence/outputs/recovp_hybrid_seed17/POSTTRAINING_STATUS.json'
& '.venv/Scripts/python.exe' -B -m tools.snapshot_recovp_hybrid_seed17
```

快照只校验 TRAIN/DEV 文件 SHA、源码、检查点及其收据，并在 CPU 上读取恢复点元数据。它不会启动训练或模型推理。`LIVE_SNAPSHOT.json` 给出已完整保存的轮次、恢复点校验结果与带时间戳的 ETA。

## 独立启动评测衔接器

本次已启动衔接器，无需重复执行。今后确认旧控制器已退出且需要重新衔接时，从本机运行：

```powershell
ssh rsdr-b powershell -NoProfile -ExecutionPolicy Bypass -File D:/code/Xin/ISRJ_222/tools/start_recovp_hybrid_posttraining.ps1
```

启动器通过隐藏的 Windows WMI 独立进程运行，避免 SSH 断开终止会话子进程。控制器持有独立的 OS 进程锁，并只读取训练锁元数据。重复启动会被拒绝，不修改当前训练。

控制器等待真正的 `TRAINING_COMPLETE.json`，而非仅凭 epoch 256 已保存就开始评测。评测先核对最终 checkpoint、收据、CONFIG/PREFLIGHT、源码和固定 DEV 的 SHA 绑定。新增任务同时考察物理 GPU 0 与 GPU 1，选择利用率最低且严格低于 85% 的卡；两卡均不满足时等待。允许同卡并行，不添加显存或 CPU 准入门槛。

自动执行的评测命令为：

```powershell
& '.venv/Scripts/python.exe' -m tools.vp_1103_recovp_hybrid_evaluate --batch-parents 2
```

由控制器在子进程环境中设置所选 `CUDA_VISIBLE_DEVICES`。评测器支持物理 GPU 0 或 GPU 1，并校验服务器、目录及显卡 UUID。

## 产物与结束条件

主要状态为 `WAITING_FOR_SEED17_TRAINING`、`WAITING_FOR_GPU_BELOW_85_PERCENT`、`RUNNING_FIXED_DEV256` 和 `CORE_DEV_COMPLETE_REVIEW_REQUIRED`。训练死亡、源码变化、身份错误或数值错误会记录失败状态及错误内容并停止。只有评测启动前明确的利用率准入竞争失败会等待重试。

核心结果位于服务器 `outputs/recovp_hybrid_seed17/seed_17/dev_epoch_256_v1.json`，轻量镜像位于 `evidence/outputs/recovp_hybrid_seed17/seed_17/dev_epoch_256_v1.json`。逐 parent 数组保留在服务器，不上传 GitHub。

自动评测包括 Hybrid Final 与同检查点 Stage1Only、parent 配对 bootstrap、复缩放、无干扰改变量、门控分布、原生 SNR/JSR 分层和 CUDA 延迟。`CORE_DEV_COMPLETE_REVIEW_REQUIRED` 只表示这些核心评测完成；任务书中的 Base/P0/UNet/RSDR 全对照、Equal-PFA、Top-10 回查、CPU 延迟及完整科学报告仍需继续完成。

控制器不启动 seed 29/41、579 扫描或其他新训练，也不读取 CONFIRM 与历史 TEST。服务器断电或训练异常后的自动重启不在本次验证范围内；每个完整 epoch 的恢复点和每 32 epochs 里程碑继续保留，供后续校验恢复。

## 本机离线验证

```powershell
python -m pytest -q tests/test_finish_recovp_hybrid_seed17.py tests/test_server_gpu_admission.py tests/test_recovp_hybrid_evaluate.py
```

本次结果为 46 项通过。测试不需要 CUDA、训练数据或 checkpoint。服务器部署验证另见 `PIPELINE_VERIFICATION.json`；该验证不等于最终 DEV 性能结论。
