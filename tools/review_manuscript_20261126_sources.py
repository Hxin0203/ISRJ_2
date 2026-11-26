"""Read-only extraction of author-supplied reference PDFs for manuscript revision."""
from pathlib import Path
import hashlib
import json
import fitz

DEST = Path('G:/Files/Markdown/ISRJ_2/草稿/草稿1125/审阅与修订_20261126')
DEST.mkdir(parents=True, exist_ok=True)
root = Path('G:/Files/文件_2026/FIles/论文/2026.10论文')
manifest = []
for prefix, label in [('MRED-net', 'MREDNet'), ('1-s2.0-S105120042500154X', 'GMM'), ('IET Radar Sonar Navi - 2024 - Meng', 'Meng')]:
    matches = list(root.glob(prefix + '*.pdf'))
    if len(matches) != 1:
        raise RuntimeError((label, matches))
    source = matches[0]
    doc = fitz.open(source)
    body = '\n'.join(f'\n--- PAGE {i+1} ---\n' + page.get_text() for i, page in enumerate(doc))
    (DEST / f'{label}_local_reading.txt').write_text(body, encoding='utf-8')
    manifest.append(dict(label=label, path=str(source), pages=len(doc), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
(DEST / 'local_reference_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=True, indent=2))
