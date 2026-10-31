"""Use ICAI's attempt-specific material notices for resources without local labels."""
import hashlib, json, pathlib, requests, re, urllib.parse
from bs4 import BeautifulSoup
import pymupdf
ROOT = pathlib.Path(__file__).resolve().parents[1]
file = ROOT / 'knowledge' / 'manifest.json'
m = json.loads(file.read_text(encoding='utf-8-sig'))
sources = [
 ('May 2026', 'https://boslive.icai.org/assets/BOS_Study_Material_applicable_for_May_2026_Examination.pdf'),
 ('Nov 2026', 'https://boslive.icai.org/assets/BOS_Study_Material_applicable_for_Final_Course_November_2026_Examination.pdf')
]
verified = []
snapshots = json.loads((ROOT / 'knowledge' / 'applicability-evidence.json').read_text(encoding='utf-8-sig')) if (ROOT / 'knowledge' / 'applicability-evidence.json').exists() else []
m.setdefault('warnings', [])
for attempt, url in sources:
    r = requests.get(url, timeout=45, headers={'User-Agent': 'CA-Bhaiya-Source-Checker/5.0 (+https://ca.arpitagarwala.com)'})
    if r.status_code in (403, 429, 500, 502, 503, 504):
        prior = next((s for s in snapshots if s['url'] == url and s['attempt'] == attempt and s.get('verifiedFacts', {}).get('ibsDigests') == ['July 2024', 'July 2025 Volume II']), None)
        if not prior: r.raise_for_status()
        verified.append({**prior, 'recheck': 'blocked; retained exact-attempt verified facts'})
        m['warnings'].append({'url': url, 'attempt': attempt, 'status': r.status_code, 'retainedVerificationDate': prior['verifiedAt'], 'message': 'Historical applicability notice could not be re-fetched; exact previously verified mappings retained. No new applicability inferred.'})
        print('Retained verified fixed-attempt notice: ' + attempt + ' (HTTP ' + str(r.status_code) + ')', flush=True)
        continue
    r.raise_for_status()
    pdf = pymupdf.open(stream=r.content, filetype='pdf')
    text = ' '.join(' '.join(p.get_text() for p in pdf).split())
    if 'IBS Case Study Digest' not in text or 'July 2024' not in text or 'July 2025' not in text:
        raise RuntimeError('Official notice no longer confirms both IBS digest editions')
    verified.append({'attempt': attempt, 'url': url, 'sha256': hashlib.sha256(r.content).hexdigest()})
for doc in m['pending'][:]:
    if doc['subjectId'] != 'final-6': continue
    doc = {**doc, 'approved': True, 'applicability': {'kind': 'exact', 'attempts': [s['attempt'] for s in verified], 'from': None, 'until': None}, 'applicabilitySource': verified[-1]['url'], 'applicabilitySources': verified, 'applicabilityNote': 'ICAI BoS attempt-specific material notices explicitly include the July 2024 and July 2025 IBS Case Study Digests.'}
    doc.pop('reason', None)
    m['documents'].append(doc)
m['pending'] = [d for d in m['pending'] if d['subjectId'] != 'final-6']
# The newer November notice resolves an old May-only heading for AFM's same edition.
for doc in m['documents'][:]:
    if doc['subjectId'] == 'final-2' and doc['editionURL'].endswith('sm-final-p2-may2026') and 'Nov 2026' not in doc['applicability'].get('attempts', []):
        updated = {**doc, 'id': hashlib.sha256((doc['id'] + '|Nov 2026').encode()).hexdigest()[:32], 'applicability': {'kind': 'exact', 'attempts': ['Nov 2026'], 'from': None, 'until': None}, 'applicabilitySource': verified[-1]['url'], 'applicabilityNote': 'ICAI November 2026 material notice: Advanced Financial Management, November 2024 edition (Reprint August 2025 and onwards).'}
        m['documents'].append(updated)
# IBS also draws on SPOM Sets A/B. Check edition pages against the official notices.
for set_name, page_url, initial_url, edition in [
    ('A', 'https://www.icai.org/post/self-paced-online-modules-set-a-nset-2026', 'https://resource.cdn.icai.org/88585bos-aps2625m1-ip.pdf', r'September\s*,?\s*2025'),
    ('B', 'https://www.icai.org/post/self-paced-online-modules-set-b-nset', 'https://resource.cdn.icai.org/75131bos60674-ip-m1.pdf', r'October\s*,?\s*2023')
]:
    initial = requests.get(initial_url, timeout=45)
    initial.raise_for_status()
    text = ' '.join(' '.join(p.get_text() for p in pymupdf.open(stream=initial.content, filetype='pdf')).split())
    if not re.search(edition, text, re.I): raise RuntimeError('SPOM edition no longer matches the official IBS applicability notice')
    response = requests.get(page_url, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    for a in soup.select('.table-responsive a[href]'):
        url = urllib.parse.urljoin(page_url, a['href'])
        host = urllib.parse.urlparse(url).hostname or ''
        if not url.startswith('https://') or not host.endswith('.icai.org') or not re.search(r'\.pdf(?:\?|$)', url, re.I): continue
        title = a.get_text(' ', strip=True)
        key = 'final-6|' + url + '|spom-ibs-2026'
        doc = {'id': hashlib.sha256(key.encode()).hexdigest()[:32], 'level': 'final', 'subjectId': 'final-6', 'subject': 'Integrated Business Solutions — SPOM Set ' + set_name, 'title': 'SPOM Set ' + set_name + ' — ' + title, 'url': url, 'type': 'study_material', 'applicability': {'kind': 'exact', 'attempts': [s['attempt'] for s in verified], 'from': None, 'until': None}, 'applicabilitySource': verified[-1]['url'], 'applicabilitySources': verified, 'editionURL': page_url, 'applicabilityNote': 'Edition checked against ICAI Final material notices for May and November 2026 IBS.', 'verification': 'official_source', 'approved': True}
        m['documents'].append(doc)
# November's Set A amendment is explicitly identified for IBS on ICAI's module hub.
hub_url = 'https://www.icai.org/post/self-paced-online-modules-nset'
response = requests.get(hub_url, timeout=45)
response.raise_for_status()
for a in BeautifulSoup(response.text, 'html.parser').select('a[href]'):
    title = a.get_text(' ', strip=True)
    if not ('Set A' in title and 'November 2026' in title and 'Amendment' in title): continue
    url = urllib.parse.urljoin(hub_url, a['href'])
    if not url.startswith('https://resource.cdn.icai.org/') or not url.lower().endswith('.pdf'): continue
    m['documents'].append({'id': hashlib.sha256(('final-6|' + url + '|Nov2026').encode()).hexdigest()[:32], 'level': 'final', 'subjectId': 'final-6', 'subject': 'Integrated Business Solutions', 'title': title, 'url': url, 'type': 'amendment', 'applicability': {'kind': 'exact', 'attempts': ['Nov 2026'], 'from': None, 'until': None}, 'applicabilitySource': hub_url, 'editionURL': hub_url, 'applicabilityNote': title, 'verification': 'official_source', 'approved': True})
m['documents'] = list({d['id']: d for d in m['documents']}.values())
file.write_text(json.dumps(m, indent=2), encoding='utf8')
print(json.dumps({'verifiedNotices': len(verified), 'documents': len(m['documents'])}))
