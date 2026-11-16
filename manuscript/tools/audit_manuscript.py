"""Offline prose audit with immutable backup; no manuscript replacement."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'ISRJ_2_初稿.md'
SKILL=Path('C:/Users/admin/.codex/skills/clean-user-facing-text/scripts/inspect_text.py')

def protected(text):
    spans=[]
    pattern=r'\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|```[\s\S]*?```|`[^`]+`|https?://[^\s)]+|(?m:^\[[0-9]+\][^\n]*$)|\[[0-9][0-9,–— \-]*\]|(?m:^>[^\n]*$)|(?m:^\|[^\n]*$)'
    for m in re.finditer(pattern,text):
        spans.append((m.start(),m.end(),m.group()))
    # Preserve numerical values and units by leaving them outside cleanup scope.
    for m in re.finditer(r'[−+]?\d[\d.,]*(?:%|\s*(?:dB|MHz|kHz|Hz|μs))?',text):
        if not any(a<=m.start()<b for a,b,_ in spans):spans.append((m.start(),m.end(),m.group()))
    return sorted(spans)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    tag=sys.argv[1] if len(sys.argv)>1 else 'review'
    if len(sys.argv)>2:
        SOURCE=Path(sys.argv[2]).resolve()
    data=SOURCE.read_bytes(); text=data.decode('utf-8');sha=hashlib.sha256(data).hexdigest()
    (ROOT/'backups').mkdir(exist_ok=True)
    backup=ROOT/'backups'/f'{tag}_{sha[:12]}.md'
    if not backup.exists(): shutil.copy2(SOURCE,backup)
    assert hashlib.sha256(backup.read_bytes()).hexdigest()==sha
    spans=protected(text)
    chars=list(text)
    for a,b,_ in spans:
        for i in range(a,b):
            if chars[i]!='\n': chars[i]=' '
    prose=''.join(chars)
    work=ROOT/'work';work.mkdir(exist_ok=True)
    p=work/f'{tag}_prose_only.txt';p.write_text(prose,encoding='utf-8')
    run=subprocess.run([sys.executable,str(SKILL),'--audit','--json',str(p)],capture_output=True,text=True,encoding='utf-8')
    (work/f'{tag}_skill_audit.json').write_text(run.stdout,encoding='utf-8')
    suspicious=[{'offset':i,'codepoint':f'U+{ord(c):04X}','name':unicodedata.name(c,'?'),
                 'protected':any(a<=i<b for a,b,_ in spans)}
                for i,c in enumerate(text) if unicodedata.category(c)=='Cf' or c in '\ufffd']
    try: result=json.loads(run.stdout)
    except ValueError: result={'raw':run.stdout,'stderr':run.stderr}
    report={'stage':tag,'utc':datetime.now(timezone.utc).isoformat(),'manuscript_sha256':sha,
            'backup':backup.relative_to(ROOT).as_posix(),'characters':len(text),
            'han_characters':len(re.findall(r'[\u4e00-\u9fff]',text)),
            'protected_span_count':len(spans),'prose_nonspace_characters':len(re.sub(r'\s','',prose)),
            'protected_sha256':hashlib.sha256('\n'.join(s for _,_,s in spans).encode()).hexdigest(),
            'invisible_or_replacement_characters':suspicious,'skill_exit_code':run.returncode,
            'skill_audit':result,'interpretation':'Offline inspection only. English stylistic heuristics are not calibrated for Chinese scientific prose; no authorship or detector-evasion claim. No whole-document cleanup performed.'}
    (ROOT/f'LANGUAGE_AUDIT_{tag}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('stage','characters','han_characters','manuscript_sha256','protected_span_count','invisible_or_replacement_characters','skill_exit_code')},ensure_ascii=False))
