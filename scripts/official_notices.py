"""Fetch and verify live ICAI notices, directly or through our authenticated cloud relay."""
import datetime
import hashlib
import os
import re
import requests
import pymupdf

ENDPOINT = 'https://ca-chatbot.arpitagarwala.workers.dev/api/internal/sync'
SOURCES = [
    ('May 2026', 'https://boslive.icai.org/assets/BOS_Study_Material_applicable_for_May_2026_Examination.pdf'),
    ('Nov 2026', 'https://boslive.icai.org/assets/BOS_Study_Material_applicable_for_Final_Course_November_2026_Examination.pdf'),
]

def fetch_notice(url):
    if url not in [s[1] for s in SOURCES]:
        raise ValueError('Notice URL not allowed')
    try:
        direct = requests.get(url, timeout=45, allow_redirects=False, headers={'User-Agent': 'CA-Bhaiya-Source-Checker/5.0 (+https://ca.arpitagarwala.com)'})
        direct.raise_for_status()
        if not direct.content.startswith(b'%PDF-'):
            raise ValueError('Official source did not return a PDF')
        return direct.content, 'direct'
    except (requests.RequestException, ValueError):
        if not os.environ.get('ACTIONS_ID_TOKEN_REQUEST_URL') or not os.environ.get('ACTIONS_ID_TOKEN_REQUEST_TOKEN'):
            raise
    # GitHub's short-lived job identity is the only credential used by the relay.
    identity = requests.get(os.environ['ACTIONS_ID_TOKEN_REQUEST_URL'], params={'audience': ENDPOINT}, headers={'Authorization': 'Bearer ' + os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']}, timeout=30)
    identity.raise_for_status()
    response = requests.post(ENDPOINT, json={'operation': 'fetch-notice', 'url': url}, headers={'Authorization': 'Bearer ' + identity.json()['value']}, timeout=60)
    if not response.ok:
        raise RuntimeError('Cloud notice fetch failed: HTTP ' + str(response.status_code) + ' ' + response.text[:200])
    if not response.content.startswith(b'%PDF-') or len(response.content) > 2000000:
        raise ValueError('Invalid notice PDF from cloud relay')
    return response.content, 'cloudflare-relay'

def verify_notice(attempt, url, data, transport):
    if len(data) > 2000000 or not data.startswith(b'%PDF-'):
        raise ValueError('Invalid notice PDF')
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        text = ' '.join(' '.join(p.get_text() for p in pdf).split())
    month = {'May 2026': 'May', 'Nov 2026': 'November'}[attempt]
    patterns = [
        r'BOS STUDY MATERIAL APPLICABLE FOR.{0,80}' + month + r'\s+2026\s+EXAMINATION',
        r'Advanced Financial Management\s+November,?\s*2024\s*\(Reprint August,?\s*2025 and onwards\)',
        r'SPOM Set-A\s*\(September,?\s*2025\)',
        r'SPOM Set-B\s*\(October,?\s*2023\)',
        r'IBS Case Study Digest\s*\(July\s+2024 edition\)',
        r'IBS Case Study Digest\s*\(July\s+2025 edition\)\s*VOL\s+II',
    ]
    if not all(re.search(p, text, re.I) for p in patterns):
        raise RuntimeError('Live official notice does not confirm the expected attempt and editions; publication refused')
    return {'attempt': attempt, 'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'verifiedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'transport': transport, 'verifiedFacts': {'ibsDigests': ['July 2024', 'July 2025 Volume II'], 'spomSetA': 'September 2025', 'spomSetB': 'October 2023', 'afm': 'November 2024 (Reprint August 2025 and onwards)'}}

def verify_notices():
    verified = []
    for attempt, url in SOURCES:
        data, transport = fetch_notice(url)
        verified.append(verify_notice(attempt, url, data, transport))
        print('Live notice verified: ' + attempt + ' via ' + transport, flush=True)
    return verified
