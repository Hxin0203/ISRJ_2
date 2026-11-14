"""Small atomic records and content identity."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def source_bindings():
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
            for p in sorted((ROOT/'paper2').rglob('*.py'))}
