from pathlib import Path
import re,json
D=Path('manuscript/recovp_fine_revision_20261126');p=D/'ISRJ_2论文精修稿.md';s=p.read_text(encoding='utf-8')
s=s.replace('再结合修正能量比与残差脉压峰均功率比','再结合第一阶段干扰估计能量与原观测能量之比，以及初步恢复脉压峰均功率比')
s=s.replace('实数路径估计调制强度','实数路径估计调制强度')
needle='两者都只由观测派生量产生，不是由真值选择的处理开关。'
s=s.replace(needle,needle+'层内门采用sigmoid与SiLU复合，因SiLU有有限下界，该门约落在[0.431,1)内；它不是可将特征完全关闭的零一掩膜。该结构范围不等于实测门控分布。')
s=s.replace('候选及窗|16时延；16×1800实窗；每候选22路窗输入','候选及窗|16时延；59→64→1时延头；16×1800实窗；22→12→1窗头')
s=s.replace('正则系数求解|复字典1800×16；16×16正定系统','正则系数求解|61→16→1正则头；复字典1800×16；16×16正定系统')
s=s.replace('两次修正均从观测中减去。','两次修正均从观测中减去。')
s=s.replace('ISRJ-FNet[26]将距离结构先验和目标扰动下的掩码一致性用于HRRP滤波。','ISRJ-FNet[26]在多脉冲HRRP上利用距离分布对称性和目标稀疏性，并对原目标及距离翻转目标叠加相同干扰，约束两分支的掩码一致性。')
s=s.replace('MCA-ADMM-Net[27]则利用目标与干扰在不同字典下的结构差异建立联合分解，并展开ADMM求解。','MCA-ADMM-Net[27]分别以时延LFM字典和局部Fourier/STFT字典表示目标与干扰，再展开ADMM执行双分量稀疏求解。')
s=s.replace('该区别说明计算与信息接口，尚不单独证明性能优势。','ISRJ-FNet约束HRRP掩码在特定目标扰动下的一致性，本文则对固定干扰和噪声下的复数输出增量直接评价；MCA-ADMM-Net联合表示两种分量，本文的二次子问题只求当前干扰字典的系数。Lei等[31]的去噪网络也已采用全局门控，故门控残差本身不作为独立创新。上述区别界定计算与信息接口，尚不单独证明性能优势。')
# Keep source, version and workflow details in a separate evidence document.
a=s.index('## 附录 A 结果来源与复现说明');b=s.index('## 附录 B 实现关系与目标响应分解')
(D/'SOURCE_AND_REPRODUCIBILITY.md').write_text(s[a:b],encoding='utf-8')
s=s[:a]+s[b:]
s=s.replace('## 附录 B 实现关系与目标响应分解','## 附录 A 实现关系与目标响应分解').replace('见附录B','见附录A')
# Put appendix formulas in their original dependency order and number A1..A11.
blocks=re.findall(r'\$\$[\s\S]*?\$\$',s[s.index('## 附录 A'):]);order=['A7','A1','A2','A8','A3','A9','A10','A4','A5','A6','A11'];newmap={tag:'A'+str(i+1) for i,tag in enumerate(order)}
appendix=s[s.index('## 附录 A'):]
parts={re.search(r'\\tag\{(A\d+)\}',block)[1]:block for block in blocks}
s=s[:s.index('## 附录 A')]+'''## 附录 A 实现关系与目标响应分解

本附录依次保留共享背景残差与目标增量的关系、复增益分解、模板及正则头的数值定义、正规方程与自适应误差界、复值卷积展开、精修误差变化和尾峰代理。它们用于复现与解释，不构成额外的性能保证。目标增量分解要求目标非零；正规方程依赖正则系数严格为正；误差变化恒等式不要求独立性或线性近似。

'''+ '\n\n'.join(parts[tag] for tag in order)+'\n'
s=re.sub(r'(\\tag\{|式（)(A\d+)(\}|）)',lambda m:m[1]+newmap[m[2]]+m[3],s)
# Remove dangling connective phrases created by moving displays.
s=s.replace('则\n\n两恢复残差之差','。\n\n两恢复残差之差').replace('由正交分解得到\n\n损伤由','由正交分解可知，损伤由')
s=s.replace('在长度 $N=4096$ 的频域中对单位能量模板施加相位斜坡，再截取前 $T=1800$ 点，得到\n\n模板通过4096点DFT中的时延相位斜坡生成，再截取前1800点，精确表达见','在长度 $N=4096$ 的频域中对单位能量模板施加相位斜坡，再截取前 $T=1800$ 点，精确表达见')
s=s.replace('对 $\\boldsymbol c^*$ 求 Wirtinger 导数，驻点满足\n\n对应的复数正规方程见','对 $\\boldsymbol c^*$ 求Wirtinger导数得到正规方程，见')
s=s.replace('若 $\\boldsymbol e_1=\\widehat{\\boldsymbol u}_1-\\boldsymbol u$，则','记第一阶段误差为 $\\boldsymbol e_1=\\widehat{\\boldsymbol u}_1-\\boldsymbol u$。')
s=s.replace('由三角不等式得\n\n其三角不等式界见','由三角不等式得到的误差界见')
s=s.replace('采用\n\n尾峰代理采用','相应的尾峰代理采用')
# Retain a publication-facing availability statement, not an execution log.
s+='\n\n## 数据与代码说明\n\n研究源码、配置及可公开的评测摘要归档于 https://github.com/Hxin0203/ISRJ_2 。数据生成与模型评价条件见正文，版本、来源哈希和复现范围见伴随材料。训练数据、检查点及原始数组未随本文公开。\n'
p.write_text(s,encoding='utf-8')
print('Final manuscript characters',len(s))
