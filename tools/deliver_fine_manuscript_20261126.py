from pathlib import Path
import hashlib,json,re,shutil
D=Path('manuscript/recovp_fine_revision_20261126')
def parse(t):
 return {re.search(r'\\tag\{([^}]+)\}',x)[1]:re.sub(r'\\tag\{[^}]+\}','',x).strip() for x in re.findall(r'\$\$([\s\S]*?)\$\$',t)}
old=parse((D/'INPUT_FROZEN.md').read_text(encoding='utf-8'));new=parse((D/'ISRJ_2论文精修稿.md').read_text(encoding='utf-8'))
mp={k:next(q for q,v in new.items() if v==x) for k,x in old.items()}
r=D/'EDIT_OPERATIONS.json';j=json.loads(r.read_text(encoding='utf-8'))
if 'equation_map' in j:j['initial_equation_map']=j.pop('equation_map')
j['final_equation_map']=mp;j['output_sha256']=hashlib.sha256((D/'ISRJ_2论文精修稿.md').read_bytes()).hexdigest();r.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
(D/'05_公式迁移对照.md').write_text('# 公式迁移对照\n\n原显示表达式全部保持，改变编号和编排；正文24组，附录11组。\n\n|原式|精修位置|\n|---|---|\n'+'\n'.join('|'+k+'|'+v+'|' for k,v in mp.items()),encoding='utf-8')
root=Path('G:/Files/Markdown/ISRJ_2/草稿/草稿1125');dest=root/'精修_20261126';dest.mkdir(exist_ok=True)
src=root/'ISRJ_2论文草稿.md';backup=dest/'修改前_ISRJ_2论文草稿.md'
if not backup.exists():
 assert hashlib.sha256(src.read_bytes()).hexdigest()==hashlib.sha256((D/'INPUT_FROZEN.md').read_bytes()).hexdigest(),'User draft changed; do not overwrite'
 shutil.copyfile(src,backup)
 if (root/'全文检查记录.md').exists():shutil.copyfile(root/'全文检查记录.md',dest/'修改前_全文检查记录.md')
for p in D.iterdir():
 if p.is_file():shutil.copyfile(p,dest/p.name)
shutil.copyfile(D/'ISRJ_2论文精修稿.md',root/'ISRJ_2论文精修稿.md')
shutil.copyfile(D/'ISRJ_2论文精修稿.md',root/'ISRJ_2论文草稿.md')
shutil.copyfile(D/'02_修改说明与全文审查.md',root/'全文检查记录.md')
assert (root/'ISRJ_2论文草稿.md').read_bytes()==(D/'ISRJ_2论文精修稿.md').read_bytes()
print(json.dumps(dict(delivered=str(root/'ISRJ_2论文精修稿.md'),sha256=j['output_sha256'],backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest()),ensure_ascii=False))
