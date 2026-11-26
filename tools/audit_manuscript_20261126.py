"""Offline editorial checks of the revised manuscript and existing aggregate receipts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import math

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'manuscript/recovp_paper_20261126'
P=D/'ISRJ_2论文草稿.md'
s=P.read_text(encoding='utf-8')
frozen=(D/'REVIEW_INPUT_FROZEN.md').read_text(encoding='utf-8')
j=json.loads((ROOT/'evidence/outputs/recovp_hybrid_seed17/dev_epoch_256_v1.json').read_text())
checks=[]
def check(name,passed,detail):
    checks.append(dict(name=name,status='passed' if passed else 'confirmed_internal_error',detail=detail))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
check('35 display equations unchanged',re.findall(r'\$\$.*?\$\$',s,re.S)==re.findall(r'\$\$.*?\$\$',frozen,re.S),'Targeted prose and table edits only; proof validity additionally reviewed in REVIEW_1_MATH.md')
check('equation numbering',list(map(int,re.findall(r'\\tag\{(\d+)\}',s)))==list(range(1,36)),'1..35 exactly once')
abstract=s.split('## 摘要')[1].split('**关键词')[0]
check('no detailed numeric abstract',not re.search(r'\d|%|百分点',abstract),'Abstract describes problem, method, qualitative tradeoffs and boundary')
check('complete main sections',all(f'## {i} ' in s for i in range(1,8)),'Introduction, problem, method, training, experiments, discussion, conclusion; references and appendix present')
check('no empty numbered result figures',not re.search(r'图 [234] 待补',s),'Needed figures are listed in companion gap report, not represented as completed results')
ref=s.split('## 参考文献')[1].split('## 附录')[0]
numbers=list(map(int,re.findall(r'^\[(\d+)\]',ref,re.M)))
check('29 reference entries',numbers==list(range(1,30)),numbers)
body=s.split('## 参考文献')[0]
cited=set()
for token in re.findall(r'\[([0-9,–\- ]+)\]',body):
    for part in token.split(','):
        mm=re.fullmatch(r'\s*(\d+)[–-](\d+)\s*',part)
        if mm:cited.update(range(int(mm[1]),int(mm[2])+1))
        elif part.strip().isdigit():cited.add(int(part))
check('all reference entries used',cited==set(range(1,30)),sorted(cited))
for metric in ('u_nmse','damage','sjrif_db','frame_pfa_jammer','pd','amplitude','no_jammer_pooled_nmse','cell_pfa_jammer'):
    a=j['metrics']['stage1__'+metric]['mean']; b=j['metrics']['hybrid__'+metric]['mean']
    delta=j['paired_parent_differences']['hybrid_minus_stage1__'+metric]
    check('paired mean identity '+metric,math.isclose(b-a,delta['mean'],rel_tol=1e-10,abs_tol=1e-15),dict(stage1=a,final=b,delta=delta['mean'],ci95=delta['parent_bootstrap_ci95']))
check('adverse findings retained',all(x in s for x in ('+0.00790','+0.04957','0.98582','0.91518','2.016×10⁻⁶','6.026×10⁻⁵')),'Damage median/P95, amplitude reduction, no-jammer increase, cell-Pfa included')
check('CFAR geometry',73*7-25*3==436 and 8*(1201-72)==9032,dict(reference_cells=436,valid_cells=9032))
check('training updates',256*48==12288 and '12,288' in s,'256 epochs *48 updates')
check('probability intervals bounded',all(0<=v<=1 for k,x in j['metrics'].items() if k.startswith(('hybrid__','stage1__')) and ('pfa' in k or k.endswith(('__pd','__pd_no_jammer'))) for v in [x['mean'],*x['parent_bootstrap_ci95']]),'Both mean and archived interval endpoints')
check('table column integrity',all(len(set(len(line.split('|')) for line in block.splitlines()))==1 for block in re.findall(r'(?m)^\|.+(?:\n\|.+)+',s)),'Avoid unescaped absolute-value bars breaking Markdown tables')
check('no unintended invisible controls',not re.search('[\u200b\u200c\u200d\ufeff\u202a-\u202e\u2066-\u2069]',s),'Offline character check, not authorship detection')
check('core result SHA',sha(ROOT/'evidence/outputs/recovp_hybrid_seed17/dev_epoch_256_v1.json')=='877cfe7f6baf4fb3a59bff58da4a344e5437e43f000b3b3a9808a47acbda7df1','Existing aggregate JSON unchanged')
for path,expected in j['checkpoint_identity']['source_sha256'].items():
    check('frozen source '+path,sha(ROOT/path)==expected,'Current source compared against checkpoint receipt')
review_hashes={'REVIEW_1_MATH.md':'cc7460875c87a727fa36746996d040cb676e4fd90bdc6546f614fecf5bad7bd5','REVIEW_2_EVIDENCE.md':'67f2e978c302444aada247256044f7ba14317b1da18edcf8649a18cb199f1a74','REVIEW_3_NARRATIVE.md':'abfcdecbdec6d3eb7d8ff6ede6d12e42fba505ab143ae027a67882e53714eb4e'}
for path,h in review_hashes.items():check('review frozen '+path,sha(D/path)==h,h)
record=dict(checked_utc=datetime.now(timezone.utc).isoformat(),scope='offline manuscript, aggregate JSON and source identity only; no inference/training/raw arrays',manuscript_sha256=sha(P),frozen_input_sha256=sha(D/'REVIEW_INPUT_FROZEN.md'),checks=checks,unresolved=['Matched Hybrid ablations, cross-model paired/equal-PFA, repeated seeds, independent confirmation, real output/condition figures','Original runtime versions and dependency closure not fully captured','Structure proof does not establish floating-point universal guarantee','Original-output-normalized scaling error is not expected-scaled-output-relative error'])
record['status']='PASS' if all(x['status']=='passed' for x in checks) else 'FAIL'
(D/'FINAL_AUDIT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=record['status'],checks=len(checks),failed=[x for x in checks if x['status']!='passed'],manuscript_sha256=record['manuscript_sha256']),ensure_ascii=False))
raise SystemExit(0 if record['status']=='PASS' else 1)
