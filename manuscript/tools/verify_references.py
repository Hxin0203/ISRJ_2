"""Retrieve public bibliographic metadata only; never send manuscript text."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import requests

ROOT = Path(__file__).resolve().parents[1]
DOIS = [
    '10.1007/s11432-007-2017-y',
    '10.1049/iet-rsn.2018.5658',
    '10.1016/j.dsp.2025.105132',
    '10.1049/rsn2.12300',
    '10.1016/j.sigpro.2023.109329',
    '10.3390/rs14143445',
    '10.1049/rsn2.12568',
    '10.3390/s25133925',
    '10.1016/j.dsp.2024.104850',
    '10.1109/TAES.2025.3646182',
    '10.1109/TAES.2025.3648973',
    '10.1109/TNNLS.2026.3736439',
    '10.1109/LSP.2026.3726844',
    '10.1049/rsn2.70251',
    '10.1007/978-3-319-24574-4_28',
    '10.1109/CVPR.2016.90',
    '10.1109/TIP.2017.2662206',
    '10.1007/978-3-030-01261-8_1',
    '10.1214/aos/1176344552',
    '10.1109/TAES.1986.310745',
]

def retrieve(doi):
    url = 'https://api.crossref.org/works/' + doi
    result = {'doi': doi, 'queried_utc': datetime.now(timezone.utc).isoformat(), 'source': url}
    try:
        r = requests.get(url, timeout=24, headers={'User-Agent': 'ISRJ-bibliographic-audit/1.0'})
        result['http_status'] = r.status_code
        r.raise_for_status()
        m = r.json()['message']
        fields = ['title', 'author', 'container-title', 'volume', 'issue', 'page', 'article-number',
                  'published', 'published-print', 'published-online', 'issued', 'DOI', 'URL', 'type']
        result['metadata'] = {k: m[k] for k in fields if k in m}
        result['status'] = 'metadata_retrieved_not_author_verified'
    except Exception as exc:
        result['status'] = 'retrieval_failed_use_local_pdf_if_available'
        result['error'] = str(exc)
    return result

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(retrieve, DOIS))
    (ROOT / 'REFERENCE_METADATA.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    for r in records:
        m = r.get('metadata', {})
        print(r['doi'], r['status'], m.get('title'), m.get('volume'), m.get('page'), m.get('article-number'))
