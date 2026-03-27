/**
 * CA Bhaiya — ICAI Knowledge Base Scraper v2
 * ============================================
 * Dynamically crawls boslive.icai.org for Foundation, Intermediate, and Final
 * amendments, statutory updates, corrigendums, AND full study material PDFs.
 * Generates 384-dim embeddings via Cloudflare Workers AI REST API.
 * Uploads vectors to Cloudflare Vectorize.
 *
 * SETUP:
 *   1. npm install
 *   2. Create a .env file with CLOUDFLARE_API_TOKEN
 *      (Cloudflare Dashboard → My Profile → API Tokens → Create Token → Edit Workers template)
 *   3. npm run update
 *
 *   NOTE: If no API token is set, the scraper will save all chunks to chunks.json
 *   and vectors.ndjson without uploading. Then run:
 *     npx wrangler vectorize insert ca-knowledge-vectors --file=vectors.ndjson
 *   (wrangler uses your existing logged-in session automatically)
 */

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const https = require('https');
const cheerio = require('cheerio');
const pdfParse = require('pdf-parse');
const { v4: uuidv4 } = require('uuid');

// ─── Config ──────────────────────────────────────────────────────────────────
const CF_ACCOUNT_ID  = process.env.CLOUDFLARE_ACCOUNT_ID || 'bacdfb7c3105ba2451a6dfd467803073';
const CF_API_TOKEN   = process.env.CLOUDFLARE_API_TOKEN;  // Optional: if missing, save NDJSON for manual wrangler upload
const NVIDIA_API_KEY = process.env.NVIDIA_EMBED_API_KEY;
const VECTORIZE_INDEX = 'ca-knowledge-vectors';
const EMBED_MODEL    = 'baai/bge-m3'; // 1024-dim, available on NVIDIA NIM API
const NVIDIA_EMBED_URL = 'https://integrate.api.nvidia.com/v1/embeddings';

const OUTPUT_DIR  = path.join(__dirname, 'data');
const STATE_FILE  = path.join(__dirname, 'scrape-state.json');
const NDJSON_FILE = path.join(__dirname, 'vectors.ndjson');

const CHUNK_SIZE    = 380;  // words
const CHUNK_OVERLAP = 40;
const EMBED_BATCH   = 20;   // chunks per CF AI call
const REQUEST_DELAY = 1200; // ms between PDF downloads

/**
 * Attempt string mapping — maps ICAI "Applicable for..." text to short labels.
 * e.g. "Applicable for May 2025, September 2025, January 2026 Examination"
 *      → ["May 2025", "Sep 2025", "Jan 2026"]
 */
function parseAttempts(yearString) {
  const MAP = { january:'Jan', jan:'Jan', march:'Mar', mar:'Mar',
                may:'May', june:'Jun', jun:'Jun',
                september:'Sep', sep:'Sep', november:'Nov', nov:'Nov', december:'Dec', dec:'Dec' };
  const results = [];
  const rx = /(\b(?:january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\b)\s*(\d{4})/gi;
  let m;
  while ((m = rx.exec(yearString)) !== null) {
    const abbr = MAP[m[1].toLowerCase()];
    if (abbr) results.push(`${abbr} ${m[2]}`);
  }
  return [...new Set(results)];
}

/**
 * All three levels to crawl.
 */
const LEVELS = ['foundation', 'intermediate', 'final'];
const LEVEL_LABEL = { foundation: 'Foundation', intermediate: 'Inter', final: 'Final' };

// ─── HTTP helpers ────────────────────────────────────────────────────────────
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function fetchBuffer(url, redirects = 5) {
  return new Promise((resolve, reject) => {
    const mod = url.startsWith('https') ? https : require('http');
    const req = mod.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) CA-Scraper/2.0',
        'Accept': 'text/html,application/pdf,*/*',
      }
    }, res => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location && redirects > 0)
        return fetchBuffer(res.headers.location, redirects - 1).then(resolve).catch(reject);
      if (res.statusCode !== 200)
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve(Buffer.concat(chunks)));
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error(`Timeout: ${url}`)); });
  });
}

async function fetchHtml(url) {
  const buf = await fetchBuffer(url);
  return buf.toString('utf8');
}

// ─── Stage 1: Discover year strings per level ────────────────────────────────
async function discoverYears(level) {
  const baseUrl = `https://boslive.icai.org/study_material_new_paper_details.php?c=${level}&language=English`;
  console.log(`  🔍 Discovering years for ${LEVEL_LABEL[level]}...`);
  try {
    const html = await fetchHtml(baseUrl);
    const $ = cheerio.load(html);
    const years = new Set();

    // Find <option> values containing "Applicable"
    $('option').each((_, el) => {
      const val = $(el).attr('value') || '';
      const txt = $(el).text().trim();
      if (val.toLowerCase().includes('applicable') || val.toLowerCase().includes('may') || val.toLowerCase().includes('nov') || val.toLowerCase().includes('jan'))
        years.add(val.trim());
      if (txt.toLowerCase().includes('applicable'))
        years.add(txt);
    });

    // Find links with year= in href
    $('a[href*="year="]').each((_, el) => {
      const href = $(el).attr('href') || '';
      const match = href.match(/year=([^&"']+)/);
      if (match) years.add(decodeURIComponent(match[1]).trim());
    });

    // Find onclick attributes with year=
    $('[onclick*="year="]').each((_, el) => {
      const oc = $(el).attr('onclick') || '';
      const match = oc.match(/year=['"]([^'"]+)['"]/);
      if (match) years.add(match[1].trim());
    });

    const found = [...years].filter(y => y.length > 4);
    console.log(`    Found ${found.length} year string(s)`);
    return found;
  } catch (err) {
    console.warn(`  ⚠️  Year discovery failed for ${level}: ${err.message}`);
    return [];
  }
}

// ─── Stage 2: Scrape PDFs for a year page ────────────────────────────────────
async function scrapePDFLinks(level, yearString) {
  const url = `https://boslive.icai.org/study_material_new_paper_details.php?c=${level}&language=English&year=${encodeURIComponent(yearString)}`;
  let html;
  try {
    html = await fetchHtml(url);
  } catch (err) {
    console.warn(`  ⚠️  Failed to fetch page for ${level}/${yearString}: ${err.message}`);
    return [];
  }

  const $ = cheerio.load(html);
  const pdfs = [];

  // Walk all <a> tags pointing to a PDF on cdn.icai.org
  $('a[href$=".pdf"], a[href*=".pdf?"]').each((_, el) => {
    const href = $(el).attr('href') || '';
    if (!href.includes('icai.org') && !href.startsWith('/') && !href.startsWith('http')) return;
    const resolvedUrl = href.startsWith('http') ? href : `https://boslive.icai.org${href}`;

    const linkText  = $(el).text().trim().toLowerCase();
    const parentTxt = $(el).closest('tr, div, li, section').text().toLowerCase();

    // Classify document type by link text keywords
    let type = 'study_material';
    if (/amendment|amended/i.test(linkText + parentTxt))      type = 'amendment';
    else if (/statutory update|judicial update/i.test(linkText + parentTxt)) type = 'statutory_update';
    else if (/corrigendum|erratum/i.test(linkText + parentTxt)) type = 'corrigendum';
    else if (/mock test|mtp|rtp/i.test(linkText + parentTxt)) type = 'practice';

    // Try to extract subject from surrounding <h3>/<h4>/<strong> or the table row
    let subject = 'General';
    const heading = $(el).closest('table, div[class*="paper"], section')
                     .find('h3,h4,h5,strong,th').first().text().trim();
    if (heading) subject = heading.replace(/\s+/g, ' ').substring(0, 80);

    pdfs.push({ url: resolvedUrl, type, subject, linkText: $(el).text().trim() });
  });

  // Deduplicate by URL
  const seen = new Set();
  return pdfs.filter(p => {
    if (seen.has(p.url)) return false;
    seen.add(p.url);
    return true;
  });
}

// ─── Stage 3: Download & Chunk PDF text ──────────────────────────────────────
function chunkText(text) {
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const chunks = [];
  for (let i = 0; i < words.length; i += (CHUNK_SIZE - CHUNK_OVERLAP)) {
    const slice = words.slice(i, i + CHUNK_SIZE);
    if (slice.length < 30) break;
    chunks.push(slice.join(' '));
  }
  return chunks;
}

async function processPDF(pdf, level, attempts, state) {
  if (state.processed.has(pdf.url)) {
    console.log(`    ↩ Already processed: ${path.basename(pdf.url)}`);
    return [];
  }

  let buffer;
  try {
    buffer = await fetchBuffer(pdf.url);
  } catch (err) {
    console.warn(`    ❌ Download failed: ${err.message}`);
    return [];
  }

  let text = '';
  try {
    const parsed = await pdfParse(buffer);
    text = parsed.text || '';
  } catch (err) {
    console.warn(`    ❌ PDF parse failed: ${err.message}`);
    return [];
  }

  if (text.trim().length < 100) {
    console.warn(`    ⚠️  Near-empty PDF: ${pdf.url}`);
    return [];
  }

  const textChunks = chunkText(text);
  const results = textChunks.map((content, idx) => ({
    id: uuidv4(),
    content,
    metadata: {
      level,
      subject:  pdf.subject,
      type:     pdf.type,
      attempts: attempts.join('|'),
      latestAttempt: attempts[attempts.length - 1] || 'Unknown',
      source:   path.basename(pdf.url),
      chunkIdx: idx,
      text:     content.substring(0, 10000), // Ensure text fits within Vectorize limits
    }
  }));

  state.processed.add(pdf.url);
  return results;
}

// ─── Stage 4: Embed via NVIDIA NIM API (baai/bge-m3, 1024 dims) ───────────────
async function embedBatch(texts) {
  const body = JSON.stringify({
    model: EMBED_MODEL,
    input: texts,
    input_type: 'passage',
    encoding_format: 'float',
    truncate: 'END',
  });

  return new Promise((resolve, reject) => {
    const req = https.request(NVIDIA_EMBED_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${NVIDIA_API_KEY}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      }
    }, res => {
      const parts = [];
      res.on('data', d => parts.push(d));
      res.on('end', () => {
        try {
          const json = JSON.parse(Buffer.concat(parts).toString());
          if (json.error) return reject(new Error(json.error.message || JSON.stringify(json)));
          // NVIDIA returns { data: [ { embedding: [...] }, ... ] }
          const vectors = json.data.map(d => d.embedding);
          resolve(vectors);
        } catch(e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('Embed timeout')); });
    req.write(body);
    req.end();
  });
}

// ─── Stage 5: Upsert to Cloudflare Vectorize ─────────────────────────────────
async function upsertVectors(vectorRows) {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) {
    throw new Error('CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set in .env');
  }
  const ndjson = vectorRows.map(r => JSON.stringify(r)).join('\n');
  const url = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/vectorize/indexes/${VECTORIZE_INDEX}/upsert`;

  return new Promise((resolve, reject) => {
    const buf = Buffer.from(ndjson, 'utf8');
    const req = https.request(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CF_API_TOKEN}`,
        'Content-Type': 'application/x-ndjson',
        'Content-Length': buf.length,
      }
    }, res => {
      const parts = [];
      res.on('data', d => parts.push(d));
      res.on('end', () => {
        try {
          const json = JSON.parse(Buffer.concat(parts).toString());
          if (!json.success) return reject(new Error(JSON.stringify(json.errors)));
          resolve(json.result);
        } catch(e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(buf);
    req.end();
  });
}

// ─── Main ────────────────────────────────────────────────────────────────────
async function main() {
  console.log('\n🎓 CA Bhaiya — ICAI Knowledge Scraper v2\n');
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  // Load state so we can resume interrupted runs
  const state = { processed: new Set() };
  if (fs.existsSync(STATE_FILE)) {
    try {
      const saved = JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
      state.processed = new Set(saved.processed || []);
      console.log(`📂 Resuming — ${state.processed.size} PDFs already processed.\n`);
    } catch(_) {}
  }

  let totalChunks = 0;
  let totalVectors = 0;
  const ndjsonStream = fs.createWriteStream(NDJSON_FILE, { flags: 'a' });

  for (const level of LEVELS) {
    console.log(`\n━━━ ${LEVEL_LABEL[level].toUpperCase()} ━━━`);
    const yearStrings = await discoverYears(level);

    if (yearStrings.length === 0) {
      console.log(`  ⚠️  No years discovered. Skipping ${level}.`);
      continue;
    }

    for (const yearStr of yearStrings) {
      const attempts = parseAttempts(yearStr);
      const label = attempts.length > 0 ? attempts.join(', ') : yearStr.substring(0, 50);
      console.log(`\n  📅 ${label}`);

      const pdfLinks = await scrapePDFLinks(level, yearStr);
      console.log(`    📄 Found ${pdfLinks.length} PDFs`);
      await delay(800);

      for (const pdf of pdfLinks) {
        console.log(`    ⬇  [${pdf.type}] ${pdf.linkText || path.basename(pdf.url)}`);
        const chunks = await processPDF(pdf, level, attempts, state);

        if (chunks.length === 0) { await delay(REQUEST_DELAY); continue; }
        console.log(`       → ${chunks.length} chunks extracted`);
        totalChunks += chunks.length;

        // Embed in batches
        for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
          const batch = chunks.slice(i, i + EMBED_BATCH);
          let embeddings;
          try {
            embeddings = await embedBatch(batch.map(c => c.content));
          } catch (err) {
            console.warn(`       ⚠️  Embed failed: ${err.message}. Skipping batch.`);
            continue;
          }

          const vectorRows = batch.map((chunk, j) => ({
            id:       chunk.id,
            values:   embeddings[j],
            metadata: chunk.metadata,
          }));

          // Always write to local NDJSON (backup + wrangler CLI upload)
          vectorRows.forEach(r => ndjsonStream.write(JSON.stringify(r) + '\n'));

          // Upload to Vectorize via REST API (only if token is set)
          if (CF_API_TOKEN) {
            try {
              await upsertVectors(vectorRows);
              totalVectors += vectorRows.length;
            } catch (err) {
              console.warn(`       ⚠️  Vectorize upsert failed: ${err.message}`);
            }
          } else {
            totalVectors += vectorRows.length; // count for NDJSON
          }

          await delay(400);
        }

        // Save state after each PDF
        fs.writeFileSync(STATE_FILE, JSON.stringify({ processed: [...state.processed] }));
        await delay(REQUEST_DELAY);
      }
    }
  }

  ndjsonStream.end();

  console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  console.log(`✅ Done!  ${totalChunks} chunks extracted, ${totalVectors} vectors processed.`);
  console.log(`📄 Backup NDJSON saved to: vectors.ndjson`);
  if (!CF_API_TOKEN) {
    console.log(`\n⚠️  CLOUDFLARE_API_TOKEN not set — vectors were NOT uploaded automatically.`);
    console.log(`   Upload now by running:`);
    console.log(`   npx wrangler vectorize insert ca-knowledge-vectors --file=vectors.ndjson`);
  } else {
    console.log(`🚀 Vectors uploaded to Cloudflare Vectorize!`);
    console.log(`\n💡 To re-upload from backup manually:`);
    console.log(`   npx wrangler vectorize insert ca-knowledge-vectors --file=vectors.ndjson`);
  }
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
