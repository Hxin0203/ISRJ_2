# ReCoVP-Hybrid seed17 最终 DEV 证据摘要

- 证据时间：2026-11-25T17:27:45Z
- 训练状态：`TRAINING_COMPLETE`, epoch 256/256
- 恢复点：257 个，epoch 000--256
- DEV 状态：`CORE_DEV_COMPLETE_REVIEW_REQUIRED`
- DEV 文件：`evidence/outputs/recovp_hybrid_seed17/dev_epoch_256_v1.json`
- DEV 文件 SHA-256：`877cfe7f6baf4fb3a59bff58da4a344e5437e43f000b3b3a9808a47acbda7df1`
- checkpoint SHA-256：`e0b6cb1be9559ed5e75a9ce3c4e178e85aba22f0cec753f52d3f523cbaf05212`
- DEV 数据 SHA-256：`0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1`
- DEV 数据收据 SHA-256：`61e439503223258b969fcb8dd53a709bc5f60b1a128cde232ef645f98c1e2092`
- 评测 admission：physical GPU 1，启动利用率 0%，门槛为 `<85%`

## 关键数值

| 指标 | Hybrid Final | Stage1Only | 配对差值 | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| u_nmse | 0.043122743968324 | 0.401853451892423 | -0.358730707924099 | [-0.468212771356333, -0.265524107588905] |
| damage | 0.059674371786719 | 0.125586420224611 | -0.065912048437892 | [-0.139306544599346, -0.013592540589800] |
| SJRI-F dB | 11.0968160765015 | 10.4167602008493 | +0.680055875652179 | [0.530062768727595, 0.834189659435819] |
| Pd | 0.87109375 | 0.854166666666667 | +0.016927083333333 | [-0.001302083333333, 0.036458333333333] |
| jammer frame Pfa | 0.0559895833333333 | 0.07421875 | -0.018229166666667 | [-0.03125, -0.005208333333333] |
| equivariance relative L2 mean | 2.33284792151567e-05 | — | — | — |
| equivariance relative L2 P95 | 8.71772681421135e-05 | — | — | — |
| CUDA latency 1 pulse P50/P95 ms | 27.8648167 / 29.2633038 | — | — | — |
| CUDA latency 8 pulse P50/P95 ms | 29.2768002 / 30.6639268 | — | — | — |

## 结论边界

当前证据支持 Hybrid 在固定 DEV256 分布上改善恢复误差、目标损伤、SJRI-F 和 jammer frame PFA；Pd 点估计上升，但配对区间包含零。完整跨模型公平排名、equal-PFA、Top-10 失败审计、CPU 延迟和独立泛化仍待完成。

原始 checkpoint、raw parent values、训练数据和完整日志不进入 GitHub。
