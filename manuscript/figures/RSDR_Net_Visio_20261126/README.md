# RSDR-Net model architecture figure

本图展示给定终稿中的 RSDR-Net 结构，包含总体推理链、第二可靠度估计器与第二专家的展开结构，以及可靠度调制单元。模型结构仅依据本目录列出的英文版和中文版终稿；第一篇论文的 Visio 插图只用于参考配色、线型和版式。

主文件为可编辑的原生 Visio 文档 `RSDR_Net_Architecture.vsdx`。同时提供矢量 PDF、SVG 和 300 dpi PNG。`figure_geometry.json` 与 `make_geometry.py` 保存图形对象的几何布局，`build_visio.ps1` 可在 Windows/Visio 环境重新生成 Visio 与矢量导出，`verify_figure.py` 检查源稿哈希、文字、面板边界和导出完整性。

## Figure caption

图 2 RSDR-Net 的结构。（a）双专家与有界融合推理链。原始观测 $Y$ 同时保留为候选输入，并与两专家输出 $U_1,U_2$ 按固定凸权重构成基准 $A$；三个局部路由器产生候选 $L$，脉冲组采用门根据 $Y,A,L$ 的十项帧级统计量输出 $\rho\in[0,0.25]$，再将 $A$ 与 $L$ 有界混合得到 $F$。虚线表示停止梯度条件或组级统计量。（b）第二可靠度估计器 $\mathcal E_2$ 和第二专家提案网络 $\mathcal D_2$。估计器由三尺度卷积编码器、粗尺度四层四头 Pre-LN Transformer（RoPE）及跳接解码器组成；$\mathcal D_2$ 在三尺度残差 U-Net 中加入可靠度调制，并在输出动作门处限制采用比例。$\mathcal D_1$ 使用相同残差 U-Net 主干，但不含可靠度条件输入与特征调制。（c）可靠度调制单元及第二专家输出动作门。各模块均逐脉冲处理快时间序列；仅脉冲组采用门汇总完整 $P\times N$ 帧统计量。边界图 $b_2$ 为训练期辅助输出。

## Source documents

| Source | SHA-256 |
|---|---|
| `G:/Files/Markdown/ISRJ/草稿/终稿1124/修订稿_CVEDNet_20261126_第二轮/论文终稿_英文版.md` | `80FAD871CA3257165A2E8A2873EC187B82834CB494E11E914B478A43909DCF3A` |
| `G:/Files/Markdown/ISRJ/草稿/终稿1124/修订稿_CVEDNet_20261126_第二轮/论文终稿_中文版 - 副本.md` | `EC76F6CA779B883F2FA7D98065CD1C327B53791A62C4973F1E4BAF9FAFD65CE6` |

`BUILD_RECEIPT.json` records the Visio export, and `QA_FIGURE.json` records the subsequent vector and native-object checks. `RSDR_Net_review.png` is a screen-size preview; `RSDR_Net_paper_size_check.png` is a reduced-size legibility check.
