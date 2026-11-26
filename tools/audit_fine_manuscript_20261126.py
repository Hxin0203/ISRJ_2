from pathlib import Path
import re,json,hashlib,collections
D=Path('manuscript/recovp_fine_revision_20261126');p=D/'ISRJ_2论文精修稿.md';s=p.read_text(encoding='utf-8');old=(D/'INPUT_FROZEN.md').read_text(encoding='utf-8')
checks=[]
def check(name,passed,detail=''): checks.append(dict(name=name,passed=bool(passed),detail=detail))
def formulas(t):return [re.sub(r'\\tag\{[^}]+\}','',b).strip() for b in re.findall(r'\$\$([\s\S]*?)\$\$',t)]
check('all original display mathematics conserved',collections.Counter(formulas(s))==collections.Counter(formulas(old)),'35 expressions unchanged except tags and relocation')
tags=re.findall(r'\\tag\{([^}]+)\}',s)
check('main equation sequence',tags[:24]==[str(n) for n in range(1,25)])
check('appendix sequence',tags[24:]==['A'+str(n) for n in range(1,12)])
check('no dangling equation references',all(x in tags for x in re.findall(r'式（([A]?\d+)）',s)))
oldrows=[r for r in old.splitlines() if r.startswith('|') and (r.startswith('|ReCoVP') or r.startswith('|P0-') or r.startswith('|UNet-') or r.startswith('|RSDR-') or r.startswith('|$') or r.startswith('|目标复增益') or r.startswith('|受扰') or r.startswith('|检测') or r.startswith('|无干扰') or r.startswith('|SJRI'))]
check('all original numeric result rows unchanged',all(r in s for r in oldrows),len(oldrows))
check('formal sections',all(f'## {n} ' in s for n in range(1,8)))
check('abstract without detailed numeric results',not re.search(r'\d|百分点|%',s.split('## 摘要')[1].split('**关键词')[0]))
refs=s.split('## 参考文献')[1].split('## 附录')[0];nums=list(map(int,re.findall(r'^\[(\d+)\]',refs,re.M)))
check('37 references',nums==list(range(1,38)))
cited=set()
for token in re.findall(r'\[([\d,–\- ]+)\]',s.split('## 参考文献')[0]):
 for part in token.split(','):
  m=re.fullmatch(r'\s*(\d+)[–-](\d+)\s*',part)
  if m:cited.update(range(int(m[1]),int(m[2])+1))
  elif part.strip().isdigit():cited.add(int(part))
check('all references have uses',cited==set(nums),dict(unused=sorted(set(nums)-cited)))
check('all Markdown tables have stable columns',all(len(set(len(r.split('|')) for r in b.splitlines()))==1 for b in re.findall(r'(?m)^\|.+(?:\n\|.+)+',s)))
check('necessary figure slots',all(f'图{n}位置' in s for n in range(2,6)))
check('model detail and two gates',all(x in s for x in ('59→64→1','61→16→1','22→12→1','20→4→1','0.431','第一阶段干扰估计能量')))
check('adverse findings retained',all(x in s for x in ('+0.00790','+0.04957','0.98582','0.91518','2.016×10⁻⁶','6.026×10⁻⁵')))
check('three comparison supports differentiated',all(x in s for x in ('同检查点','并非共享基础场景','不同聚合权重','同DEV')))
check('no stale fulltext deficit',not re.search('尚未取得.*全文|完整方法差异仍需逐式核查|摘要已提出',s))
check('no internal execution prose',not any(x in s for x in ('本次写作','流水线','CORE_DEV','待办','是否继续开发','提交SHA','尚未完成的训练')))
check('no invisible formatting controls',not re.search('[\u200b\u200c\u200d\ufeff\u202a-\u202e\u2066-\u2069]',s))
record=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',checks=checks,manuscript_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((D/'INPUT_FROZEN.md').read_bytes()).hexdigest(),scope='static editorial audit; no training, inference, raw array recomputation or independent test',review_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('REVIEW_*.md')})
(D/'FINAL_AUDIT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(record,ensure_ascii=False))
raise SystemExit(record['status']!='PASS')
