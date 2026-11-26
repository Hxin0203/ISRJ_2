"""Retrieve DOI registry metadata; no unpublished text is transmitted."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import urllib.request

DOIS = [
 '10.1016/j.sigpro.2023.109329', '10.1016/j.dsp.2025.105132',
 '10.1049/rsn2.12568', '10.3390/rs14143445', '10.3390/rs16111989',
 '10.3390/s25133925', '10.1016/j.sigpro.2025.110230',
 '10.1016/j.ast.2025.111218', '10.1016/j.sigpro.2026.110998',
 '10.1016/j.sigpro.2026.110574', '10.1137/0710036',
 '10.1088/0266-5611/19/2/201', '10.1007/s10589-012-9492-9',
 '10.1007/978-3-319-24574-4_28', '10.1109/CVPR.2016.90',
 '10.1214/aos/1176344552', '10.1109/LGRS.2024.3448611',
]
DEST = Path('manuscript/recovp_paper_20261126')
DEST.mkdir(parents=True, exist_ok=True)

def fetch(doi):
    url='https://api.crossref.org/works/'+doi
    try:
        with urllib.request.urlopen(url, timeout=25) as response:
            m=json.load(response)['message']
        return dict(doi=doi, title=m.get('title'), author=m.get('author'),
                    venue=m.get('container-title'), volume=m.get('volume'),
                    issue=m.get('issue'), page=m.get('page'), article_number=m.get('article-number'),
                    published=m.get('published'), published_online=m.get('published-online'),
                    published_print=m.get('published-print'), type=m.get('type'),
                    update_to=m.get('update-to'), relation=m.get('relation'),
                    source=url, status='registry_metadata_retrieved')
    except Exception as exc:
        return dict(doi=doi,status='unresolved',error=str(exc))

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        data=list(pool.map(fetch,DOIS))
    (DEST/'doi_registry.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    for x in data:
        print(json.dumps({k:x.get(k) for k in ('doi','title','venue','published','status')},ensure_ascii=True))
