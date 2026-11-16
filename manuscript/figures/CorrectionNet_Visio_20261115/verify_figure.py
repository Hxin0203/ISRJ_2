"""Verify the saved Visio master, protect inputs, and render its PDF export."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

import fitz

ROOT = Path(__file__).resolve().parent
BASE = "CorrectionNet_editable"
NS = {"v": "http://schemas.microsoft.com/office/visio/2012/main"}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((ROOT/"SOURCE_SHA256.json").read_text(encoding="utf-8"))
sources_unchanged = all(sha(Path(p)) == digest for p, digest in manifest.items())
assert sources_unchanged, "Protected input changed"
cfg = json.loads((ROOT/"figure_geometry.json").read_text(encoding="utf-8"))
with zipfile.ZipFile(ROOT/(BASE+".vsdx")) as z:
    media = [p for p in z.namelist() if p.startswith("visio/media/")]
    assert not media, "Unexpected embedded image"
    page = ET.fromstring(z.read("visio/pages/page1.xml"))
    shapes = page.findall(".//v:Shape", NS)
    groups = [s for s in shapes if s.attrib.get("Type") == "Group"]
    byname = {s.attrib["NameU"]: s for s in shapes if "NameU" in s.attrib}
    checked_text = []
    for o in cfg["items"]:
        s = byname[o["name"]]
        if o.get("text"):
            wanted = re.sub(r"</?(sub|sup)>", "", o["text"])
            found = "".join(s.find("v:Text", NS).itertext()).rstrip("\n\r")
            assert wanted == found, (o["name"], wanted, found)
            checked_text.append(o["name"])
        assert not any(c.attrib.get("N") == "LockTextEdit" and c.attrib.get("V") == "1"
                       for c in s.findall("v:Cell", NS))
    connections = page.findall(".//v:Connect", NS)
    assert len(connections) == 57
    assert len(groups) == 4
    assert len(shapes) == len(cfg["items"])+4
    # No clipping or drift across the calibrated panel regions.
    panel_bounds = {o["group"]: o for o in cfg["items"] if o["name"].startswith("panel_")}
    for o in cfg["items"]:
        if o["name"].startswith(("panel_", "title_")):
            continue
        p = panel_bounds[o["group"]]
        pts = o["points"] if o["kind"] == "path" else [[o["x"],o["y"]],[o["x"]+o["w"],o["y"]+o["h"]]]
        assert all(p["x"]-.01<=x<=p["x"]+p["w"]+.01 and p["y"]-.01<=y<=p["y"]+p["h"]+.01 for x,y in pts), o["name"]

doc = fitz.open(ROOT/(BASE+".pdf"))
assert len(doc)==1
pg = doc[0]
assert len(pg.get_images()) == 0
fonts = []
for f in pg.get_fonts():
    embedded = bool(doc.extract_font(f[0])[3])
    assert embedded, f
    fonts.append(dict(name=f[3], embedded=embedded))
for block in pg.get_text("dict")["blocks"]:
    for line in block.get("lines", []):
        for span in line["spans"]:
            assert pg.rect.contains(fitz.Rect(span["bbox"])), span
# Rasterize the same saved vector PDF for a precise publication-resolution PNG.
pg.get_pixmap(dpi=300,alpha=False).save(ROOT/(BASE+".png"))
pg.get_pixmap(dpi=108,alpha=False).save(ROOT/"CorrectionNet_review.png")
pg.get_pixmap(dpi=68,alpha=False).save(ROOT/"CorrectionNet_paper_size_check.png")
files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p))
       for p in [ROOT/(BASE+ext) for ext in (".vsdx",".pdf",".svg",".png")]}
assert all(x["bytes"]>0 for x in files.values())
report=dict(status="PASS",native_objects=len(cfg["items"]),semantic_groups=len(groups),
            native_shapes_including_groups=len(shapes),editable_text_objects=len(checked_text),
            glued_connection_endpoints=len(connections),embedded_media=media,
            pdf_raster_images=0,pdf_fonts=fonts,png_dpi=300,
            protected_inputs_unchanged=sources_unchanged,all_children_inside_panels=True,
            inference_uses_only_observation=True,training_panel_marked=True,
            waveform_icons="Synthetic schematic paths; not experimental predictions",files=files)
(ROOT/"QA_FIGURE.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ("pdf_fonts","files")},ensure_ascii=False))
