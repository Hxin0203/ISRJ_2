"""Export a top-left-origin drawing scene to an editable Microsoft Visio page."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from win32com.client import DispatchEx


ROOT = Path(__file__).resolve().parent
PPI = 96


def rgb(value):
    return "RGB(%d,%d,%d)" % tuple(
        int(value[start:start + 2], 16) for start in (1, 3, 5)
    )


def cell(shape, name, formula):
    shape.CellsU(name).FormulaU = str(formula)


def box(page, height, x, y, w, h):
    return page.DrawRectangle(x / PPI, (height - y - h) / PPI,
                              (x + w) / PPI, (height - y) / PPI)


def style(shape, item):
    fill = item.get("fill", "none")
    stroke = item.get("stroke", "none")
    cell(shape, "FillPattern", int(fill != "none"))
    if fill != "none":
        cell(shape, "FillForegnd", rgb(fill))
    if stroke == "none":
        cell(shape, "LinePattern", 0)
    else:
        cell(shape, "LinePattern", 2 if item.get("dash") else 1)
        cell(shape, "LineColor", rgb(stroke))
        cell(shape, "LineWeight", f"{item.get('sw', 1) / PPI} in")
    cell(shape, "ShdwPattern", 0)


def path_shape(page, height, item):
    points = list(item["points"])
    if item["type"] == "poly" and points[0] != points[-1]:
        points.append(points[0])
    coords = [c for x, y in points for c in (x / PPI, (height - y) / PPI)]
    if len(coords) == 4:
        return page.DrawLine(*coords)
    return page.DrawPolyline(coords, 0)


def text_shape(page, height, item):
    size = item["size"]
    text = item["text"]
    anchor = item.get("anchor", "start")
    # A loose invisible text block prevents wrap when Visio substitutes a font.
    width = max(3 * size, size * (len(text) + 1))
    offset = {"start": 0, "middle": width / 2, "end": width}[anchor]
    shape = box(page, height, item["x"] - offset, item["y"] - size,
                width, 1.31 * size)
    style(shape, {})
    shape.Text = text
    cell(shape, "Char.Font", f'FONT("{item.get("font", "Arial")}")')
    cell(shape, "Char.Size", f"{size / PPI} in")
    cell(shape, "Char.Style", int(item.get("bold", False)) + 2 * int(item.get("italic", False)))
    cell(shape, "Char.Color", rgb(item["color"]))
    cell(shape, "Para.HorzAlign", {"start": 0, "middle": 1, "end": 2}[anchor])
    cell(shape, "VerticalAlign", 1)
    cell(shape, "Para.SpLine", "-100%")
    for name in ("LeftMargin", "RightMargin", "TopMargin", "BottomMargin"):
        cell(shape, name, "0 in")
    return shape


def draw_item(page, height, item, location):
    kind = item["type"]
    if kind == "text":
        return text_shape(page, height, item)
    if kind == "image":
        shape = page.Import(str((location / item["path"]).resolve()))
        cell(shape, "LockAspect", 0)
        for name, value in (("Width", item["w"]), ("Height", item["h"]),
                            ("PinX", item["x"] + item["w"] / 2),
                            ("PinY", height - item["y"] - item["h"] / 2)):
            cell(shape, name, f"{value / PPI} in")
        return shape
    if kind in ("line", "poly"):
        shape = path_shape(page, height, item)
    elif kind == "rect":
        shape = box(page, height, item["x"], item["y"], item["w"], item["h"])
    else:
        raise ValueError(f"Unknown drawing element {kind}")
    style(shape, item)
    return shape


def export(scene_path, out, proof):
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    app = DispatchEx("Visio.InvisibleApp")
    doc = None
    try:
        app.AlertResponse = 7
        app.ScreenUpdating = 0
        doc = app.Documents.Add("")
        page = doc.Pages.Item(1)
        page.Name = "ReCoVP-Hybrid"
        cell(page.PageSheet, "PageWidth", f"{scene['width'] / PPI} in")
        cell(page.PageSheet, "PageHeight", f"{scene['height'] / PPI} in")
        cell(page.PageSheet, "DrawingResizeType", 0)
        for i, item in enumerate(scene["items"], 1):
            shape = draw_item(page, scene["height"], item, scene_path.parent)
            shape.NameU = f"{item['type']}-{i}"
        doc.SaveAs(str(out))
        if proof:
            doc.ExportAsFixedFormat(1, str(proof), 1, 0)
    finally:
        if doc is not None:
            doc.Saved = True
            doc.Close()
        app.Quit()
    with ZipFile(out) as package:
        if package.testzip() is not None:
            raise ValueError("Corrupt Visio package")
        page_xml = ET.fromstring(package.read("visio/pages/page1.xml"))
        children = page_xml.findall("./{*}Shapes/{*}Shape")
        if len(children) != len(scene["items"]):
            raise ValueError(f"Found {len(children)} shapes for {len(scene['items'])} items")
        print(f"{len(children)} native Visio objects, including "
              f"{sum(s.find('{*}Text') is not None for s in children)} editable labels. {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", type=Path, default=ROOT / "layout.json")
    parser.add_argument("--out", type=Path, default=ROOT / "ReCoVP_Hybrid_plate.vsdx")
    parser.add_argument("--proof", type=Path, help="Optional PDF proof rendered with Visio")
    args = parser.parse_args()
    export(args.scene.resolve(), args.out.resolve(), args.proof and args.proof.resolve())

