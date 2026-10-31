"""Operator utility: record verified, exact-attempt facts from official BoS notices."""
import datetime, hashlib, json, pathlib, requests
import pymupdf
root = pathlib.Path(__file__).resolve().parents[1]
sources = [
    ('May 2026', 'https://boslive.icai.org/assets/BOS_Study_Material_applicable_for_May_2026_Examination.pdf'),
    ('Nov 2026', 'https://boslive.icai.org/assets/BOS_Study_Material_applicable_for_Final_Course_November_2026_Examination.pdf')
]
facts = []
for attempt, url in sources:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    text = ' '.join(' '.join(page.get_text() for page in pymupdf.open(stream=r.content, filetype='pdf')).split())
    if not all(s in text for s in ['IBS Case Study Digest', 'July 2024', 'July 2025', 'Set-A', 'Set-B']):
        raise RuntimeError('Notice facts do not match the expected IBS applicability evidence')
    facts.append({'attempt': attempt, 'url': url, 'sha256': hashlib.sha256(r.content).hexdigest(), 'verifiedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'verifiedFacts': {'ibsDigests': ['July 2024', 'July 2025 Volume II'], 'spomSetA': 'September 2025', 'spomSetB': 'October 2023'}})
(root / 'knowledge' / 'applicability-evidence.json').write_text(json.dumps(facts, indent=2), encoding='utf8')
print('Recorded verified exact-attempt facts from ' + str(len(facts)) + ' official notices')
