from pathlib import Path
import fitz, json, hashlib
root=Path('G:/Files/文件_2026/论文编写/2026.10论文')
dest=Path('G:/Files/Markdown/ISRJ_2/草稿/草稿1125/精修_20261126/本地文献')
dest.mkdir(parents=True,exist_ok=True)
names=['1809.07454v3.pdf','26012303.pdf','Trans_Interrupted_Sampling_Repeater_Jamming_Suppression_Using_Complex-Valued_FISTA-Net.pdf','2307.03368v2.pdf','JNLEST.pdf','Interrupted_Sampling_Repeater_Jamming_Suppression_for_Wideband_Radar_With_MCA-ADMM-Net.pdf','Structure-Informed_Deep_Filtering_for_Radar_HRRP_ISRJ_Mitigation.pdf','End-to-End Interrupted Sampling Repeater Jamming Countermeasure Network Under Low Signal-to-Noise Ratio.pdf','Gu_Interpreting_Super-Resolution_Networks_With_Local_Attribution_Maps_CVPR_2021_paper.pdf','kornblith19a.pdf','chen18a.pdf','1311.2901v3.pdf','Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.pdf','Wang_Recovering_Realistic_Texture_CVPR_2018_paper.pdf']
names += [p.name for p in root.glob('IET*2026*Lei*')]
rows=[]
for i,name in enumerate(names,1):
 p=root/name; doc=fitz.open(p); text='\n'.join(f'\n--- PAGE {j+1} ---\n'+page.get_text() for j,page in enumerate(doc));out=dest/f'R{i:02d}.txt';out.write_text(text,encoding='utf-8');rows.append(dict(id=f'R{i:02d}',filename=name,path=str(p),pages=len(doc),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),extraction=str(out),opening=text[:1800]))
(dest/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False))
