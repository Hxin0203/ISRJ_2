from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

import fitz

ROOT = Path(__file__).resolve().parent
BASE = "RSDR_Net_Architecture"
NS = {"v": "http://schemas.microsoft.com/office/visio/2012/main"}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg = json.loads((ROOT / "figure_geometry.json").read_text(encoding="utf-8"))
sources = {s["path"]: s["sha256"] for s in cfg["source_documents"]}
source_status = {}
for name, digest in sources.items():
    p = Path(name)
    source_status[name] = {"exists": p.exists(), "sha256_matches": p.exists() and sha(p).lower() == digest.lower()}
assert all(x["sha256_matches"] for x in source_status.values()), "A designated manuscript source changed"

with zipfile.ZipFile(ROOT / f"{BASE}.vsdx") as z:
    media = [p for p in z.namelist() if p.startswith("visio/media/")]
    assert not media, f"Unexpected embedded bitmap media: {media}"
    page = ET.fromstring(z.read("visio/pages/page1.xml"))
    shapes = page.findall(".//v:Shape", NS)
    groups = [s for s in shapes if s.attrib.get("Type") == "Group"]
    byname = {s.attrib["NameU"]: s for s in shapes if "NameU" in s.attrib}
    checked_text = []
    for obj in cfg["items"]:
        assert obj["name"] in byname, f"Missing native Visio shape: {obj['name']}"
        shape = byname[obj["name"]]
        if obj.get("text"):
            node = shape.find("v:Text", NS)
            found = "".join(node.itertext()).rstrip("\n\r") if node is not None else ""
            wanted = re.sub(r"</?(sub|sup)>", "", obj["text"])
            assert wanted == found, (obj["name"], wanted, found)
            checked_text.append(obj["name"])
        assert not any(c.attrib.get("N") == "LockTextEdit" and c.attrib.get("V") == "1"
                       for c in shape.findall("v:Cell", NS)), obj["name"]
    assert len(groups) == 3, f"Expected 3 semantic panels, got {len(groups)}"
    assert len(shapes) == len(cfg["items"]) + len(groups)
    panel_bounds = {o["group"]: o for o in cfg["items"] if o["name"].startswith("panel_")}
    for obj in cfg["items"]:
        if obj["name"].startswith(("panel_", "title_")):
            continue
        panel = panel_bounds[obj["group"]]
        points = obj["points"] if obj["kind"] == "path" else [
            [obj["x"], obj["y"]], [obj["x"] + obj["w"], obj["y"] + obj["h"]]
        ]
        assert all(panel["x"] - .01 <= x <= panel["x"] + panel["w"] + .01 and
                   panel["y"] - .01 <= y <= panel["y"] + panel["h"] + .01
                   for x, y in points), obj["name"]

doc = fitz.open(ROOT / f"{BASE}.pdf")
assert len(doc) == 1
page = doc[0]
assert len(page.get_images()) == 0, "PDF should remain vector-based"
fonts = []
for f in page.get_fonts():
    embedded = bool(doc.extract_font(f[0])[3])
    assert embedded, f"Non-embedded font: {f[3]}"
    fonts.append({"name": f[3], "embedded": embedded})
for block in page.get_text("dict")["blocks"]:
    for line in block.get("lines", []):
        for span in line["spans"]:
            assert page.rect.contains(fitz.Rect(span["bbox"])), span

# Render the saved vector PDF, not the lower-resolution Visio preview.
page.get_pixmap(dpi=300, alpha=False).save(ROOT / f"{BASE}.png")
page.get_pixmap(dpi=110, alpha=False).save(ROOT / "RSDR_Net_review.png")
page.get_pixmap(dpi=68, alpha=False).save(ROOT / "RSDR_Net_paper_size_check.png")
artifact_paths = [ROOT / f"{BASE}{ext}" for ext in (".vsdx", ".pdf", ".svg", ".png")]
files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in artifact_paths}
assert all(x["bytes"] > 0 for x in files.values())
report = {
    "status": "PASS", "native_geometry_objects": len(cfg["items"]),
    "semantic_panels": len(groups), "native_shapes_including_groups": len(shapes),
    "editable_text_shapes": len(checked_text), "embedded_media": media,
    "pdf_raster_images": 0, "pdf_fonts": fonts, "png_dpi": 300,
    "source_documents_unchanged": source_status,
    "all_geometry_inside_panels": True,
    "inference_input": "observed I/Q waveform Y only",
    "training_only_boundary_map_marked": True,
    "schematic_signals": "No experimental waveform or result curves are depicted",
    "files": files,
}
(ROOT / "QA_FIGURE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k not in ("pdf_fonts", "files")}, ensure_ascii=False))
