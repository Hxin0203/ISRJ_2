"""Assemble the three transparent architecture insets at a common display width."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent


def main(dictionary_png):
    canvas = Image.new("RGB", (680, 720), "white")
    pen = ImageDraw.Draw(canvas)
    title = ImageFont.truetype("msyhbd.ttc", 25)
    caption = ImageFont.truetype("msyh.ttc", 18)
    panels = [
        ("纯净 LFM 回波", "同相分量 I  ·  脉冲中心局部", "01_lfm_clean_detail.png"),
        ("ISRRJ 受扰观测", "同相分量 I  ·  SNR +10 dB  ·  JSR +30 dB", "02_isrrj_observation_detail.png"),
        ("模型实际使用的局部字典", "与受扰观测同一时窗  ·  16 行原子  ·  蓝 −48 dB 至黄 0 dB", dictionary_png),
    ]
    for i, (label, desc, file) in enumerate(panels):
        y = 16 + 235 * i
        pen.text((32, y), label, fill="#182d3b", font=title)
        pen.text((32, y + 40), desc, fill="#465e69", font=caption)
        with Image.open(HERE / file) as original:
            picture = original.convert("RGBA").resize((616, 139), Image.Resampling.LANCZOS)
        canvas.paste(picture, (32, y + 75), picture)
        if i < 2:
            pen.line((32, y + 225, 648, y + 225), fill="#d7e1e3")
    canvas.save(HERE / "preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dictionary_png", help="Name of the unlabelled dictionary panel PNG")
    main(parser.parse_args().dictionary_png)
