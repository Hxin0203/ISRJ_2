"""Assemble the full English manuscript with shared equation identities."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
zh=(ROOT/'ISRJ_2_论文修订稿_20261116.md').read_text(encoding='utf-8')
t=(ROOT/'tools/manuscript_en_template.md').read_text(encoding='utf-8')
t=t.replace('{{REST}}',(ROOT/'tools/manuscript_en_results.md').read_text(encoding='utf-8'))
equations={}
for m in re.finditer(r'\\\[[\s\S]*?\\\]',zh):
    tag=re.search(r'\\tag\{(\d+)\}',m[0])
    if tag:
        assert int(tag[1]) not in equations
        equations[int(tag[1])]=m[0]
assert sorted(equations)==list(range(1,44))
equations[34]=r'''\[
\begin{array}{ll}
1.&\text{Read shared }(S,J,E)\text{ and construct }Y_{00},Y_{10},Y_{01},Y_{11};\\
2.&\text{For each state, compute }z=DY/a(y),\quad\delta_z''=F_\theta(z);\\
3.&\text{Compute }c_{ab}=a(y)D^{\mathrm H}\delta_z''\text{ and losses (30)--(33)};\\
4.&\text{Backpropagate, clip the gradient norm, and update }\theta;\\
5.&\text{Save recovery and identity records each epoch; retain milestones every 32 epochs.}
\end{array}
\tag{34}
\]'''
for k,v in equations.items():
    token='{{EQ'+str(k)+'}}'
    assert t.count(token)==1,(k,t.count(token))
    t=t.replace(token,v)
abstract=zh.split('## Abstract\n\n',1)[1].split('\n\n## 1 ',1)[0]
t=t.replace('{{ABSTRACT}}',abstract)
refs=zh.split('## 参考文献\n\n',1)[1].split('\n\n## 附录',1)[0]
refs=refs.replace('在线发表条目，未补入尚未确认的卷页。','Online publication; unverified volume and page details are not supplied.')
refs=re.sub(r'(?m)^\[15\].*$', '[15] Authors. Reliability-guided selective waveform restoration for interrupted-sampling repeater jamming. Unpublished research manuscript, versions of 3 and 13 November 2026. Cited for inherited methodology and the frozen comparator; not a peer-reviewed publication. Author details and publication status require updating before submission.', refs)
t=t.replace('{{REFERENCES}}',refs)
assert '{{' not in t
assert not re.search(r'[\u4e00-\u9fff]',t)
out=ROOT/'ISRJ_2_Manuscript_EN_20261116.md'
out.write_text(t,encoding='utf-8')
print(out.name, len(t), len(t.split()))
