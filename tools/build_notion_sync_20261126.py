"""Prepare bounded Notion report archive. Does not connect, train or evaluate."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'reports/notion_sync_20261126'
DEST.mkdir(exist_ok=True)
def inventory():
    paths=set()
    for pattern in ('PAPER2*.md','FAMILY*.md','reports/*.md','evidence/outputs/*.md','evidence/outputs/recovp_research_stage_review_20261125_r2/*.md','evidence/outputs/recovp_hybrid_seed17/*.md','evidence/outputs/recovp_hybrid_seed17/full_review/*.md','manuscript/recovp_paper_20261126/*.md'):
        paths.update(ROOT.glob(pattern))
    paths={p for p in paths if not p.name.startswith(('REVIEW_INPUT','REVIEW_PACKET')) and p.name!='ISRJ2_AUTO_RESEARCH_LOG.md'}
    extra=['evidence/outputs/correction_mf_tail_20261117_v1/DEV_SCREEN_SUMMARY.json','evidence/outputs/correction_mf_tail_20261117_v1/FOUR_CELL_DEV_SUMMARY.json','evidence/outputs/correction_mf_tail_20261117_v1/PIPELINE_STATUS.json','evidence/outputs/recovp_hybrid_seed17/dev_epoch_256_v1.json','evidence/outputs/recovp_hybrid_seed17/full_review/REVIEW_PROGRESS.json','manuscript/ReCoVP_Hybrid_论文草稿_20261125.md','manuscript/RECOVP_PAIRED_DEV_DRAFT_1125.md','outputs/cvednet_literature_claim_audit_20261124/source_manifest.json','manuscript/recovp_paper_20261126/doi_registry.json','manuscript/recovp_paper_20261126/FINAL_AUDIT.json','manuscript/recovp_paper_20261126/TARGETED_CHANGELOG.json']
    paths.update(ROOT/p for p in extra)
    existing=DEST/'SOURCE_INVENTORY.json'
    rows=json.loads(existing.read_text(encoding='utf-8')) if existing.exists() else []
    known={r['path'] for r in rows}
    for p in sorted(paths,key=lambda p:p.relative_to(ROOT).as_posix()):
        if p.relative_to(ROOT).as_posix() in known:continue
        i=len(rows)
        raw=p.read_bytes()
        # No datasets, arrays, weights, caches, credentials or full process logs.
        assert p.suffix in ('.md','.json')
        category='history'
        name=p.as_posix().lower()
        if 'recovp' in name:category='recovp'
        elif 'unet' in name or 'classical' in name or 'correction' in name or 'residual' in name:category='experiments'
        elif 'innovation' in name or 'idea' in name:category='decisions'
        rows.append(dict(index=i,path=p.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),category=category))
    (DEST/'SOURCE_INVENTORY.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows

def payload(index, volume=0):
    rows=json.loads((DEST/'SOURCE_INVENTORY.json').read_text(encoding='utf-8'))
    row=rows[index]
    raw=(ROOT/row['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Source changed since inventory')
    original=raw.decode('utf-8')
    parts=[]
    pending=''
    # Bound JSON literal size below Notion rich-text's 2000-character boundary.
    for char in original:
        candidate=pending+char
        if len(json.dumps(candidate,ensure_ascii=False))>1500:
            parts.append(pending);pending=char
        else:pending=candidate
    if pending:parts.append(pending)
    markdown=f"# 来源快照 S{index+1:03d}\n\n路径：`{row['path']}`\n\n原文件字节数：{row['bytes']}\n\nSHA-256：`{row['sha256']}`\n\n这是来源文件的完整文本保全，不以历史状态冒充当前结果。阅读摘要见上级专题页；以下按顺序为JSON字符串片段，逐片解码拼接并用UTF-8编码，可恢复原文件，含原有换行。不是摘要或截取。\n\n"
    volumes=(len(parts)+29)//30
    chosen=parts[volume*30:(volume+1)*30]
    if not chosen:raise RuntimeError('Invalid volume')
    for i,part in enumerate(chosen, start=volume*30):
        markdown+=f'## 原文片段 {i+1}/{len(parts)}\n\n```json\n'+json.dumps(part,ensure_ascii=False)+'\n```\n\n'
    title=f"S{index+1:03d}｜{Path(row['path']).name}"
    if volumes>1:title+=f'｜原文{volume+1}/{volumes}'
    return dict(**row,title=title,markdown=markdown,parts=len(parts),volumes=volumes,volume=volume)

if __name__=='__main__':
    if len(sys.argv)>1:print(json.dumps(payload(int(sys.argv[1]),int(sys.argv[2]) if len(sys.argv)>2 else 0),ensure_ascii=False))
    else:
        rows=inventory();print(json.dumps(dict(count=len(rows),bytes=sum(r['bytes'] for r in rows),files=[dict(index=r['index'],path=r['path']) for r in rows]),ensure_ascii=False))
