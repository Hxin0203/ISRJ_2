"""Index final figures and audit manuscript identities against saved evidence.

Read-only with respect to manuscripts and scientific evidence. Writes the figure
manifest, human-readable index and a scoped audit; does not run model inference.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / 'manuscript'
FIG = MAN / 'figures'
SPEC = [
    (1, 'CorrectionNet_Visio_20261115/CorrectionNet_editable', ['vsdx','pdf','svg','png'],
     ['paper2/network.py','paper2/vendor/legacy_model.py'],
     '网络及四状态监督；波形图标是示意，不是预测数据。VSDX 为原生可编辑对象。'),
    (2, 'fig02_convergence', ['pdf','svg','png'],
     ['evidence/outputs/CONVERGENCE_256_8SEED_20261114/TRAINING_LOSS_BY_EPOCH.csv',
      'evidence/outputs/CONVERGENCE_256_8SEED_20261114/DEV_METRICS_BY_CHECKPOINT.csv'],
     '8 seed、256 轮；2048 条逐轮记录、64 个 DEV 里程碑；阴影为 seed 样本标准差。'),
    (3, 'fig03_repeat_jsr20', ['pdf','svg','png'],
     ['evidence/outputs/TEST_CONDITION_TABLE_1103.csv'],
     '重复转发、JSR=20 dB，256 parent/条件；oracle 与不可观测参考单列。'),
    (4, 'fig04_dense_jsr20', ['pdf','svg','png'],
     ['evidence/outputs/RSDR_UNIFIED_20261115/CONDITION_COMPARISONS.csv'],
     'JSR=20 dB，SNR 步长 1 dB，512 parent/条件；仅为完整网格切片。'),
    (5, 'fig08_full_detection_grid', ['pdf','svg','png'],
     ['evidence/outputs/RSDR_UNIFIED_20261115/CONDITION_COMPARISONS.csv'],
     '全部 310 稠密条件，512 parent/条件；SNR/JSR 步长 1/5 dB，无插值；共同名义 CFAR。'),
    (6, 'fig05_fixed_parent_pc_doppler', ['png'],
     ['evidence/outputs/RSDR_UNIFIED_20261115/figures/fixed_parent0_coherent_001.png'],
     '固定 parent 0，SNR=0 dB、JSR=20 dB；冻结 PNG，非矢量，不估读重画。'),
    (7, 'fig06_fixed_parent_rd', ['png'],
     ['evidence/outputs/RSDR_UNIFIED_20261115/figures/fixed_parent0_rd_001.png'],
     '同图 6 场景；仅冻结 PNG。横向四面板缩放后可能难读，需源数据重排或拆图。'),
    (8, 'fig07_width_snr_minus20_jsre35', ['pdf','svg','png'],
     ['evidence/outputs/RSDR_UNIFIED_20261115/CONDITION_COMPARISONS.csv'],
     '扫描 B，SNR=-20 dB、JSR_E=35 dB、M=2/4，128 parent/条件；宽度与周期一起变化。'),
]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write_json(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')

def make_manifest():
    figures, sources, mapping = {}, {}, []
    for n, stem, formats, source_paths, note in SPEC:
        files = []
        for ext in formats:
            p = FIG / (stem + '.' + ext)
            key = p.relative_to(ROOT).as_posix()
            figures[key] = sha(p)
            files.append(key)
        for key in source_paths:
            sources[key] = sha(ROOT / key)
        mapping.append({'number':n, 'files':files, 'sources':source_paths, 'scope_and_limits':note})
    manifest = {'purpose':'Final manuscript figures; frozen evidence and schematic architecture; no new experiments',
                'sources':sources, 'figures':figures, 'manuscript_mapping':mapping}
    write_json(FIG/'MANIFEST.json', manifest)
    lines = ['# 图件与证据索引', '',
        '对应 2026-11-16 中英文完整修订稿。正文图号与文件名尾数并不总相同，以下映射为准。所有图件和源文件的完整 SHA-256 见 `figures/MANIFEST.json`。', '',
        '| 正文图号 | 文件 | 格式 | 条件与解释边界 |', '|---|---|---|---|']
    for n, stem, formats, _, note in SPEC:
        links = '、'.join(f'[{ext}](figures/{stem}.{ext})' for ext in formats)
        lines.append(f'| 图 {n} | {links} | {" / ".join(formats)} | {note} |')
    lines += ['', '## 源证据身份', '', '| 文件 | SHA-256 |', '|---|---|']
    lines += [f'| `{k}` | `{v}` |' for k,v in sources.items()]
    lines += ['', '## 结果表及复核入口', '',
        '- 表 4：8 seed 的 DEV 里程碑 CSV；表 5：`TEST_CONDITION_TABLE_1103.csv`；表 6：`RSDR_UNIFIED_20261115/PAIRED_RESULTS.json`；表 7：条件汇总及相参性结果；表 8：`UNET_DEV_3SEED_20261116.json`。路径前缀见上述表格。',
        '- `tools/index_and_audit_delivery.py` 核验公式、引用、图表编号、中英数字、主要结果及图件 SHA，输出 `CONSISTENCY_AUDIT_20261116.json`。其通过状态仅覆盖已编码检查，不代表所有数学论证已形式化证明。',
        '- 模型图 `QA_FIGURE.json` 记录原生对象、可编辑文字及粘连端点数量。旧 `fig01_architecture` 为历史产物，不属于本次正文图件。',
        '- 图 6/7 的两份源 PNG 与正文 PNG 字节相同。后续期刊排版需从原始绘图数据改进图 7，不能将位图装入 PDF 后宣称矢量。',
        '- 缺少完整证据的消融、等虚警 ROC、统一硬件效率和外部失配图暂不绘制。所需实验和图件见 `投稿前实验与模型改进路线_20261116.md`。',
        '- 不含训练数据、模型权重或逐样本 raw arrays；这里只索引轻量汇总和已导出的图件。', '']
    (MAN/'图件与证据索引.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    return manifest

def table_rows(t, label, n):
    section = re.split(r'^\*\*'+re.escape(label)+rf' {n}\.',t,flags=re.M)[1]
    block = re.search(r'(?m)^\|[^\n]*\n(?:\|[^\n]*\n?)+',section)[0]
    return [[c.strip() for c in line.strip().strip('|').split('|')] for line in block.strip().splitlines()[2:]]

def numbers(s):
    return re.findall(r'[-+]?\d+(?:\.\d+)?',s.replace('−','-').replace(',',''))

def citation_numbers(body):
    # Numerical intervals inside TeX are not bibliographic citations.
    prose = re.sub(r'```[\s\S]*?```|`[^`]+`|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$\n]+\$', '', body)
    cited = set()
    for raw in re.findall(r'\[((?:\d+\s*(?:[–-]\s*\d+)?)(?:\s*,\s*\d+\s*(?:[–-]\s*\d+)?)*)\]', prose):
        for part in raw.split(','):
            ends = [int(n.strip()) for n in re.split('[–-]', part)]
            if len(ends) == 1:
                cited.add(ends[0])
            else:
                cited.update(range(ends[0], ends[1] + 1))
    return cited

def main():
    manifest = make_manifest()
    checks = []
    def check(name, ok, detail):
        checks.append({'check':name,'pass':bool(ok),'detail':detail})
    texts = {lang:(MAN/name).read_text(encoding='utf-8') for lang,name in
             [('zh','ISRJ_2_论文修订稿_20261116.md'),('en','ISRJ_2_Manuscript_EN_20261116.md')]}
    eqs = {}
    for lang,t in texts.items():
        tags = list(map(int,re.findall(r'\\tag\{(\d+)\}',t)))
        check(f'{lang}: numbered equations',tags==list(range(1,44)),tags)
        eqs[lang] = {int(re.search(r'\\tag\{(\d+)\}',m)[1]):m for m in re.findall(r'\\\[[\s\S]*?\\\]',t) if '\\tag{' in m}
        figs=list(map(int,re.findall(r'^\*\*(?:图|Fig\.) (\d+)\.',t,re.M)))
        tabs=list(map(int,re.findall(r'^\*\*(?:表|Table) (\d+)\.',t,re.M)))
        check(f'{lang}: figure and table numbering',figs==list(range(1,9)) and tabs==list(range(1,9)),{'figures':figs,'tables':tabs})
        refs=list(map(int,re.findall(r'^\[(\d+)\] ',t,re.M)))
        check(f'{lang}: bibliography',refs==list(range(1,23)),refs)
        body=t.split('## 参考文献' if lang=='zh' else '## References')[0]
        cited=citation_numbers(body)
        check(f'{lang}: citations resolve and all references cited',cited==set(range(1,23)),sorted(cited))
        paths=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',t)
        expected=['figures/'+stem+'.png' for _,stem,_,_,_ in SPEC]
        check(f'{lang}: figure paths',paths==expected and all((MAN/p).exists() for p in paths),paths)
        unresolved=re.findall(r'\{\{[A-Z_][A-Z_0-9]*\}\}|\b(?:TODO|TBD|FIXME)\b',t)
        check(f'{lang}: no unresolved template tokens',not unresolved,unresolved)
    mismatches=[k for k in range(1,44) if k!=34 and eqs['zh'].get(k)!=eqs['en'].get(k)]
    check('bilingual numbered equation identity',not mismatches,{'mismatches':mismatches,'exception':'34: algorithm prose translated'})
    for n in range(1,9):
        z=table_rows(texts['zh'],'表',n); e=table_rows(texts['en'],'Table',n)
        zn=[numbers(' | '.join(r)) for r in z];en=[numbers(' | '.join(r)) for r in e]
        check(f'bilingual table {n} numeric identity',zn==en,{'rows_zh':len(z),'rows_en':len(e),'different_rows':[i for i,(a,b) in enumerate(zip(zn,en)) if a!=b]})
    unet=read_json(MAN/'UNET_DEV_3SEED_20261116.json')
    keys=['u_nmse','damage','sjrif_db','pd','frame_pfa_jammer']
    precisions=[6,6,4,3,3]
    for lang,label in [('zh','表'),('en','Table')]:
        for row in table_rows(texts[lang],label,8):
            for cell,key,dp in zip(row[2:],keys,precisions):
                factor=100 if key in ('pd','frame_pfa_jammer') else 1
                expected=[f"{unet['summary'][row[0]][key][stat]*factor:.{dp}f}" for stat in ('mean','sample_sd')]
                check(f'{lang}: table8 {row[0]} {key}',numbers(cell)==expected,{'actual':numbers(cell),'expected':expected})
    for binding in unet['source_bindings']:
        check('U-Net source SHA '+binding['path'],sha(ROOT/binding['path'])==binding['sha256'],binding['sha256'])
    pair=read_json(ROOT/'evidence/outputs/RSDR_UNIFIED_20261115/PAIRED_RESULTS.json')
    for lang,label in [('zh','表'),('en','Table')]:
        for row,key in zip(table_rows(texts[lang],label,6),keys):
            factor=100 if key in ('pd','frame_pfa_jammer') else 1
            dp=4 if factor==100 else 6
            expected=[f"{pair['groups']['main_all']['metrics'][m][key]['mean']*factor:.{dp}f}" for m in ('model','rsdr')]
            check(f'{lang}: table6 {key}',[row[1],row[2]]==expected,{'actual':[row[1],row[2]],'expected':expected})
    with (ROOT/'evidence/outputs/CONVERGENCE_256_8SEED_20261114/TRAINING_LOSS_BY_EPOCH.csv').open(newline='',encoding='utf-8') as f: train=list(csv.DictReader(f))
    with (ROOT/'evidence/outputs/CONVERGENCE_256_8SEED_20261114/DEV_METRICS_BY_CHECKPOINT.csv').open(newline='',encoding='utf-8') as f: dev=list(csv.DictReader(f))
    expected_seeds={17,29,41,53,67,79,97,109}
    check('training trajectory complete',len(train)==2048 and {(int(r['seed']),int(r['epoch'])) for r in train}=={(s,e) for s in expected_seeds for e in range(1,257)},len(train))
    check('DEV milestones complete',len(dev)==64 and {(int(r['seed']),int(r['epoch'])) for r in dev}=={(s,e) for s in expected_seeds for e in range(32,257,32)},len(dev))
    for lang,label in [('zh','表'),('en','Table')]:
        for row in table_rows(texts[lang],label,4):
            epoch=int(row[0]);subset=[r for r in dev if int(r['epoch'])==epoch]
            values=[statistics.mean(float(r[f'model__{k}__mean']) for r in subset) for k in keys]
            expected=[f'{v*(100 if k in ("pd","frame_pfa_jammer") else 1):.{dp}f}' for k,v,dp in zip(keys,values,[6,6,4,3,3])]
            check(f'{lang}: table4 epoch {epoch}',row[1:]==expected,{'actual':row[1:],'expected':expected})
    qa=read_json(FIG/'CorrectionNet_Visio_20261115/QA_FIGURE.json')
    check('Visio native package',qa['status']=='PASS' and not qa['embedded_media'] and qa['editable_text_objects']>0,{'native_objects':qa['native_objects'],'editable_text_objects':qa['editable_text_objects'],'glued_endpoints':qa['glued_connection_endpoints']})
    for name,entry in qa['files'].items():
        check('Visio QA file SHA '+name,sha(FIG/'CorrectionNet_Visio_20261115'/name)==entry['sha256'],entry['sha256'])
    for n,stem,exts,srcs,note in SPEC:
        if n in (6,7):check(f'figure {n} frozen copy',sha(FIG/(stem+'.png'))==sha(ROOT/srcs[0]),srcs[0])
    metadata=read_json(MAN/'REFERENCE_METADATA.json')
    check('recorded Crossref query status',all(r.get('http_status')==200 for r in metadata),{'records':len(metadata),'scope':'Retrieval status, not author identity verification'})
    report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL',
            'scope':'Structural, numeric and provenance checks only; not a proof of scientific correctness or submission readiness',
            'manuscript_sha256':{n:sha(MAN/n) for n in ['ISRJ_2_论文修订稿_20261116.md','ISRJ_2_Manuscript_EN_20261116.md']},
            'checks':checks,'remaining_limits':['Incomplete ablations and independent confirmation','Unmatched empirical false alarms','Figure 7 remains a frozen raster with small panels','Author/venue-specific finalization required']}
    write_json(MAN/'CONSISTENCY_AUDIT_20261116.json',report)
    failures=[c for c in checks if not c['pass']]
    print(json.dumps({'status':report['status'],'checks':len(checks),'failures':failures},ensure_ascii=False,indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
