"""Download official PDFs, preserve page references, and build a local searchable corpus.
No embedding or LLM API is used. Unchanged PDFs and completed documents are cached.
"""
import concurrent.futures, hashlib, json, pathlib, re, sqlite3, time, urllib.parse, sys
import requests
import pymupdf as fitz

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / 'knowledge' / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT / 'knowledge' / 'manifest.json').read_text(encoding='utf-8-sig'))
db = sqlite3.connect(CACHE / 'corpus.sqlite')
db.executescript((ROOT / 'knowledge' / 'schema.sql').read_text(encoding='utf-8-sig'))

def official(url):
    u = urllib.parse.urlparse(url)
    return u.scheme == 'https' and (u.hostname == 'icai.org' or u.hostname.endswith('.icai.org'))

def download(doc):
    url = doc['url']
    if not official(url):
        raise ValueError('Unofficial source')
    file = CACHE / (hashlib.sha256(url.encode()).hexdigest() + '.pdf')
    # Every scheduled run revalidates the source with HTTP conditional requests.
    meta_file = file.with_suffix('.headers.json')
    headers = {'User-Agent': 'CA-Bhaiya-Source-Checker/5.0'}
    if file.exists() and meta_file.exists():
        prior = json.loads(meta_file.read_text())
        if prior.get('etag'): headers['If-None-Match'] = prior['etag']
        if prior.get('modified'): headers['If-Modified-Since'] = prior['modified']
    for trial in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=(15, 90), allow_redirects=True)
            if not official(r.url): raise ValueError('Redirect outside official ICAI hosts')
            if r.status_code == 304 and file.exists():
                data = file.read_bytes()
            else:
                r.raise_for_status()
                data = r.content
                if len(data) > 50_000_000 or not data.startswith(b'%PDF'):
                    raise ValueError('Invalid PDF or over 50MB')
                tmp = file.with_suffix('.part')
                tmp.write_bytes(data)
                tmp.replace(file)
                meta_file.write_text(json.dumps({'etag': r.headers.get('ETag'), 'modified': r.headers.get('Last-Modified')}))
            digest = hashlib.sha256(data).hexdigest()
            pdf = fitz.open(stream=data, filetype='pdf')
            chunks = []
            pending = ''
            start = 1
            end = 1
            for pno, page in enumerate(pdf, start=1):
                # Sorted blocks retain paragraph/newline structure and PDF page references.
                text = '\n'.join(b[4] for b in sorted(page.get_text('blocks'), key=lambda b: (round(b[1] / 8), b[0])) if len(b) > 4 and isinstance(b[4], str))
                text = re.sub(r'[ \t]+', ' ', text).strip()
                if len(text) < 50: continue
                for offset in range(0, len(text), 8500):
                    chunk = '[PDF page ' + str(pno) + ']\n' + text[offset:offset + 9000].strip()
                    if pending and len(pending) + len(chunk) > 12000:
                        chunks.append((start, end, pending))
                        pending = ''
                    if not pending: start = pno
                    pending += '\n' + chunk
                    end = pno
            if pending: chunks.append((start, end, pending))
            if not chunks: raise ValueError('Image-only PDF: OCR required')
            return doc, digest, chunks, len(pdf)
        except Exception:
            if trial == 2: raise
            time.sleep(1 + trial * 2)

documents = manifest['documents']
result = {'checkedAt': manifest['checkedAt'], 'documents': [], 'errors': [], 'pending': manifest.get('pending', []), 'warnings': manifest.get('warnings', [])}
# Download each URL only once. Different applicability mappings may share the same PDF.
groups = {}
for doc in documents:
    if not doc.get('approved') or not doc.get('applicability'): continue
    prior = db.execute('SELECT metadata FROM documents WHERE id=? AND active=1', (doc['id'],)).fetchone() if '--only-new' in sys.argv else None
    if prior:
        result['documents'].append(json.loads(prior[0]))
    else:
        groups.setdefault(doc['url'], []).append(doc)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(download, docs[0]): docs for docs in groups.values()}
    for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
        docs = futures[future]
        try:
            _, digest, chunks, pages = future.result()
            for doc in docs:
                metadata = {**doc, 'sha256': digest, 'pages': pages, 'chunks': len(chunks), 'checkedAt': manifest['checkedAt']}
                prior = db.execute('SELECT sha256,metadata FROM documents WHERE id=?', (doc['id'],)).fetchone()
                if not prior or prior[0] != digest:
                    db.execute('DELETE FROM documents WHERE id=?', (doc['id'],))
                    db.execute('INSERT INTO documents(id,level,subject_id,title,url,type,sha256,metadata,updated_at,active) VALUES(?,?,?,?,?,?,?,?,?,1)', (doc['id'], doc['level'], doc['subjectId'], doc['title'], doc['url'], doc['type'], digest, json.dumps(metadata), manifest['checkedAt']))
                    db.executemany('INSERT INTO chunks(document_id,page,page_end,title,text) VALUES(?,?,?,?,?)', [(doc['id'], p, end, doc['title'], text) for p, end, text in chunks])
                else:
                    db.execute('UPDATE documents SET metadata=?,updated_at=?,active=1 WHERE id=?', (json.dumps(metadata), manifest['checkedAt'], doc['id']))
                db.commit()
                result['documents'].append(metadata)
        except Exception as error:
            result['errors'].append({'url': docs[0]['url'], 'message': str(error)[:250], 'ids': [d['id'] for d in docs]})
            # Changed/unavailable sources are not eligible until a successful check.
            for doc in docs: db.execute('UPDATE documents SET active=0 WHERE id=?', (doc['id'],))
            db.commit()
        if number % 20 == 0 or number == len(futures):
            (ROOT / 'knowledge' / 'build-report.json').write_text(json.dumps(result, indent=2), encoding='utf8')
            print(json.dumps({'processedURLs': number, 'totalURLs': len(futures), 'documents': len(result['documents']), 'failed': len(result['errors'])}), flush=True)
# A document removed from ICAI's current directory must not remain active remotely.
active_ids = {d['id'] for d in result['documents']}
for (doc_id,) in db.execute('SELECT id FROM documents').fetchall():
    if doc_id not in active_ids: db.execute('UPDATE documents SET active=0 WHERE id=?', (doc_id,))
db.commit()
db.close()
(ROOT / 'knowledge' / 'build-report.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps({'complete': True, 'documents': len(result['documents']), 'failed': len(result['errors'])}), flush=True)
