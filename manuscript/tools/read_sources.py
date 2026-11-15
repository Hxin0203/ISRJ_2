"""Extract user-supplied papers locally; never uploads document contents."""
from pathlib import Path
import hashlib
import json
import re
import fitz

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('G:/Files/文件_2026/论文编写/2026.10论文')
OUT = ROOT / 'work' / 'literature'
OUT.mkdir(parents=True, exist_ok=True)
names = [
    'Interrupted_Sampling_Repeater_Jamming_Suppression_for_Wideband_Radar_With_MCA-ADMM-Net.pdf',
    'Structure-Informed_Deep_Filtering_for_Radar_HRRP_ISRJ_Mitigation.pdf',
    'Trans_Interrupted_Sampling_Repeater_Jamming_Suppression_Using_Complex-Valued_FISTA-Net.pdf',
    'Trans_ISRJ_Suppression_Using_Joint_Morphological_Component_Analysis_and_Signal_Reconstruction_Based_on_Jamming_Parameters_Estimation1.pdf',
    'End-to-End Interrupted Sampling Repeater Jamming Countermeasure Network Under Low Signal-to-Noise Ratio.pdf',
    'MRED-net.pdf',
    'Gaussian Interrupted sampling repeater jamming suppression based on Gaussian mixture model and sparse reconstruction.pdf',
    'IET Radar Sonar   Navi - 2022 - Zhang - A new scheme of target detection for pulse Doppler radar in interrupted sampling.pdf',
    'IET Radar Sonar   Navi - 2024 - Meng - Interrupted sampling repeater jamming suppression based on multiple extended.pdf',
    'IET Radar Sonar   Navi - 2026 - Lei - Interrupted Sampling Repeater Jamming Suppression Based on Point‐Wise Temporal.pdf',
    'E-基于多级多域深度联合抗干扰网络的中断采样中继器干扰抑制方法.pdf',
]
records = []
for i, name in enumerate(names, 1):
    path = SOURCE / name
    if not path.exists():
        records.append({'id': f'L{i:02}', 'path': str(path), 'status': 'missing'})
        continue
    doc = fitz.open(path)
    text = '\n'.join(f'\n=== PAGE {j+1} ===\n{page.get_text(sort=True)}' for j, page in enumerate(doc))
    target = OUT / f'L{i:02}.txt'
    target.write_text(text, encoding='utf-8')
    records.append({'id': f'L{i:02}', 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'pages': len(doc), 'metadata': doc.metadata, 'dois': sorted(set(re.findall(r'10\.\d{4,9}/[^\s<>]+', text))),
                    'extract': str(target), 'status': 'extracted_not_yet_verified'})
    print(f'L{i:02}: {len(doc)} pages, {len(text)} chars')
(ROOT / 'SOURCE_DOCUMENTS.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
