# ReCoVP-Hybrid 512-epoch continuation handoff

更新时间 2026-11-27 UTC

## 当前状态

- 目标为 8 个 seed（17、29、41、53、67、79、97、109）从 epoch 256 续训至 epoch 512，学习率固定为 `2e-6`。
- seed 17、29、41、53、67、79、97 已完成 epoch 256，并已写入 `backups/recovp_hybrid_256_before512_20261127/seed_{seed}/`。
- seed 109 当前仅提交到 epoch 131，状态文件显示正在训练阶段但没有发现对应训练进程。其最新 checkpoint 为 `epoch_131.pt`，SHA-256 为 `6d5f974cf0fe0da649d0798e753a3af68bb263176fdc082f67bd254144e15f21`。
- 续训 controller 已恢复为唯一实例并处于 `WAITING_FOR_ALL_256`。它会在全部父代通过审计后才启动 257–512，当前不会错误地启动 7 个续训实例。

## 已修复

`paper2/io.py` 的大文件哈希读取会触发 `MemoryError`。续训控制器现在使用 `tools/stable_hash.py` 的 64 KiB 有界、失败即停哈希读取，并将该模块纳入 source binding。三份同步到服务器的 Python 文件与本地 SHA 已核对一致。

## 阻塞项

seed 109 的 epoch 131 checkpoint 和配置绑定的 preflight SHA 为 `ff5dda96ca4278eaa4f6e8465d99f50cc69351348441c9745c632153849fbaa1`。服务器当前 preflight 文件 SHA 为 `9e124798345223d591d2401670d98a4090e77c77229af27d15baca5d40be1346`，在服务器的 outputs、evidence、backups、handoff 和 deployment_revisions 范围内未找到旧文件。没有旧 preflight 之前不能通过父代审计，也不能安全续训 seed 109，不能修改 checkpoint identity、配置 SHA 或伪造回执。

## 恢复条件

找回与旧 SHA 精确匹配的 seed 109 preflight 后，先运行 `python -m tools.recovp_hybrid_continue512 --seed 109 --audit-only`，确认完整 0–256 恢复序列、配置、数据、源文件、优化器、RNG 和 GPU UUID，再让唯一 controller 自动启动 8 个续训。原始 0–256 目录不会覆盖，续训写入 `outputs/recovp_hybrid_512_seed{seed}/seed_{seed}/`。
