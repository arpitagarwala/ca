/**
 * CA Bhaiya — Cloudflare Worker v2
 * =====================================
 * Hybrid RAG: Cloudflare Vectorize semantic search + cumulative amendment logic
 * LLM (text): Groq llama-3.3-70b-versatile
 * LLM (vision): NVIDIA google/gemma-3-27b-it
 * Embeddings: Cloudflare Workers AI bge-small-en-v1.5
 */

const GROQ_URL   = 'https://api.groq.com/openai/v1/chat/completions';
const NVIDIA_URL = 'https://integrate.api.nvidia.com/v1/chat/completions';
const NVIDIA_EMBED_URL = 'https://integrate.api.nvidia.com/v1/embeddings';
const EMBED_MODEL = 'baai/bge-m3';  // 1024-dim — must match scraper
const GROQ_MODEL = 'llama-3.3-70b-versatile';
const GROQ_FAST  = 'llama3-8b-8192';
const NVIDIA_VISION_MODEL = 'google/gemma-3-27b-it';
const TOP_K = 15;
const MAX_HISTORY = 8;

// Ordered by recency — used to sort context so newest amendment wins
const ATTEMPT_ORDER = [
  'Dec 2026','Sep 2026','Jun 2026','May 2026','Mar 2026','Jan 2026',
  'Nov 2025','Sep 2025','Jun 2025','May 2025','Mar 2025','Jan 2025',
  'Nov 2024','May 2024','Nov 2023','May 2023',
];

// ─── CORS ─────────────────────────────────────────────────────────────────────
function cors() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
  };
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...cors(), 'Content-Type': 'application/json' },
  });
}

// ─── Step 1: Embed query via NVIDIA NIM API (baai/bge-m3, 1024 dims) ────────────
async function embedQuery(text, env) {
  const res = await fetch(NVIDIA_EMBED_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.NVIDIA_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: EMBED_MODEL,
      input: [text],
      input_type: 'query',
      encoding_format: 'float',
      truncate: 'END',
    }),
  });
  if (!res.ok) throw new Error(`NVIDIA embed error: ${res.status}`);
  const data = await res.json();
  return data.data[0].embedding; // float32 array [1024]
}

// ─── Step 2: Semantic retrieval + cumulative logic ────────────────────────────
async function retrieveContext(query, level, attempt, env) {
  let queryVec;
  try {
    queryVec = await embedQuery(query, env);
  } catch (e) {
    console.error('Embed error:', e);
    return [];
  }

  // Build metadata filter — always filter by level if specified
  const filter = {};
  if (level && level !== 'all') filter.level = { $eq: level };

  let matches;
  try {
    const res = await env.VECTORIZE.query(queryVec, {
      topK: TOP_K,
      returnMetadata: 'all',
      filter: Object.keys(filter).length ? filter : undefined,
    });
    matches = res.matches;
  } catch (e) {
    return [{ text: 'DEBUG_DB_ERROR: ' + e.message, type: 'error', subject: 'Debug' }];
  }

  if (!matches || matches.length === 0) return [];

  // ── Cumulative Amendment Logic ──────────────────────────────────────────────
  // All versions of all laws are kept in context, but sorted so the LATEST
  // amendment appears first. The system prompt instructs the LLM to treat the
  // latest-dated source as currently in force, overriding earlier versions only
  // where explicitly contradicted. Previous-attempt amendments that are NOT
  // contradicted by a newer one remain fully valid.
  const ranked = matches
    .map(m => ({
      score:         m.score,
      text:          m.metadata.content || m.metadata.text || '',  // fallback
      source:        m.metadata.source   || 'Unknown',
      level:         m.metadata.level    || level,
      subject:       m.metadata.subject  || 'General',
      type:          m.metadata.type     || 'study_material',
      latestAttempt: m.metadata.latestAttempt || '',
      attempts:      m.metadata.attempts || '',
    }))
    .sort((a, b) => {
      // Amendments before study material
      const typeScore = (x) => x.type === 'amendment' ? 3 : x.type === 'statutory_update' ? 2 : x.type === 'corrigendum' ? 2 : 1;
      if (typeScore(a) !== typeScore(b)) return typeScore(b) - typeScore(a);

      // Newer attempt first
      const ai = ATTEMPT_ORDER.indexOf(a.latestAttempt);
      const bi = ATTEMPT_ORDER.indexOf(b.latestAttempt);
      if (ai === -1 && bi === -1) return b.score - a.score;
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi; // lower index = newer
    });

  return ranked;
}

// ─── Step 3: Build system prompt with context ─────────────────────────────────
function buildSystemPrompt(contextChunks, level, attempt) {
  const levelLabel  = { foundation: 'CA Foundation', intermediate: 'CA Intermediate', final: 'CA Final' };
  const levelStr    = levelLabel[level] || 'CA (Foundation/Inter/Final)';
  const attemptStr  = attempt || 'Latest Available';

  const contextBlock = contextChunks.length
    ? contextChunks.map((c, i) =>
        `[Source ${i+1} | ${c.type.toUpperCase()} | ${c.subject} | ${c.latestAttempt || 'General'}]\n${c.text}`
      ).join('\n\n---\n\n')
    : 'No specific RAG context found — rely on your comprehensive CA knowledge.';

  return `You are "CA Bhaiya" — a highly knowledgeable GenZ Chartered Accountant created by Arpit Agarwala. You mentor ${levelStr} students appearing for the ${attemptStr} attempt in India.

YOUR PERSONA & RESPONSE RULES:
1. GenZ, relatable, approachable — crisp explanations like that smart senior who's already cracked the exam.
2. No motivational fluff. Jump straight to the concept with bullet points, bold key terms, and tables for comparisons.
3. For legal provisions, ALWAYS cite the Section / Rule / Notification number.
4. If the user uploads an image (balance sheet, MCQ, journal entry), analyze it carefully and answer based on the visual.
5. If an extracted PDF document is provided, base your calculation on that document's data.

AMENDMENT PRECEDENCE RULE (CRITICAL):
The RAG context below may contain multiple entries covering the same legal provision at different exam attempt dates. Apply this rule:
- The entry with the MOST RECENT attempt date is the currently applicable version.
- If an older amendment is NOT contradicted by a newer one, it is STILL fully valid and applicable.
- Only override an old amendment with a newer one when they explicitly conflict.
- Always state the applicable attempt/version when citing an amendment.

SELECTED ATTEMPT CONTEXT: ${attemptStr} | Level: ${levelStr}

─── ICAI RAG KNOWLEDGE CONTEXT (sorted: newest amendment first) ───
${contextBlock}
──────────────────────────────────────────────────────────────────`;
}

// ─── Step 4a: Call Groq for text responses ────────────────────────────────────
async function callGroq(systemPrompt, history, query, pdfContext, env) {
  const apiKey = env.GROQ_API_KEY;
  if (!apiKey) throw new Error('GROQ_API_KEY secret is not set.');

  const messages = [{ role: 'system', content: systemPrompt }];

  // Add conversation history (alternate user/assistant)
  let lastRole = 'system';
  for (const msg of history.slice(-MAX_HISTORY)) {
    const role = msg.role === 'user' ? 'user' : 'assistant';
    if (role === lastRole) continue;
    messages.push({ role, content: msg.text || msg.content || '' });
    lastRole = role;
  }

  // Append current query (with optional PDF context)
  const userContent = pdfContext
    ? `[UPLOADED DOCUMENT]\n${pdfContext.substring(0, 8000)}\n\n[QUESTION]\n${query}`
    : query;

  if (lastRole !== 'user') {
    messages.push({ role: 'user', content: userContent });
  } else {
    messages[messages.length - 1].content += '\n\n' + userContent;
  }

  const res = await fetch(GROQ_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages,
      temperature: 0.65,
      max_tokens: 2048,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    // Fallback to smaller model on rate limit
    if (res.status === 429) return callGroqFallback(systemPrompt, history, query, pdfContext, env);
    throw new Error(`Groq error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content || 'I could not generate a response. Please try again.';
}

async function callGroqFallback(systemPrompt, history, query, pdfContext, env) {
  const apiKey = env.GROQ_API_KEY;
  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: pdfContext ? `[DOC]\n${pdfContext.substring(0, 4000)}\n\n${query}` : query }
  ];
  const res = await fetch(GROQ_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model: GROQ_FAST, messages, temperature: 0.65, max_tokens: 1500 }),
  });
  const data = await res.json();
  return data.choices?.[0]?.message?.content || 'Service is temporarily busy. Please try again in a moment.';
}

// ─── Step 4b: Call NVIDIA for vision (image) inputs ───────────────────────────
async function callNvidiaVision(systemPrompt, query, imageDataUrl, env) {
  const apiKey = env.NVIDIA_API_KEY;
  if (!apiKey) throw new Error('NVIDIA_API_KEY secret is not set.');

  // Convert data URL to base64 if needed, or pass URL directly
  let imageContent;
  if (imageDataUrl.startsWith('data:')) {
    imageContent = { type: 'image_url', image_url: { url: imageDataUrl } };
  } else {
    imageContent = { type: 'image_url', image_url: { url: imageDataUrl } };
  }

  const messages = [
    { role: 'system', content: systemPrompt },
    {
      role: 'user',
      content: [
        imageContent,
        { type: 'text', text: query },
      ],
    },
  ];

  const res = await fetch(NVIDIA_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      model: NVIDIA_VISION_MODEL,
      messages,
      max_tokens: 1024,
      temperature: 0.20,
      top_p: 0.70,
      stream: false,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`NVIDIA error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content || 'Could not analyze the image. Please describe it in text.';
}

// ─── Main Worker Handler ───────────────────────────────────────────────────────
export default {
  async fetch(request, env) {
    const headers = cors();

    if (request.method === 'OPTIONS')
      return new Response(null, { status: 204, headers });

    if (request.method !== 'POST')
      return json({ error: 'Method not allowed' }, 405);

    try {
      const { query, history = [], image_url = null, pdf_context = null, level = 'all', attempt = 'Latest' } = await request.json();

      if (!query || typeof query !== 'string' || !query.trim())
        return json({ error: 'Query is required' }, 400);

      const q = query.trim();

      // 1. Semantic retrieval with cumulative logic
      const contextChunks = await retrieveContext(q, level, attempt, env);

      // 2. Build system prompt
      const systemPrompt = buildSystemPrompt(contextChunks, level, attempt);

      // 3. Generate answer
      let answer;
      if (image_url) {
        // Vision model for images
        answer = await callNvidiaVision(systemPrompt, q, image_url, env);
      } else {
        // Text-only via Groq
        answer = await callGroq(systemPrompt, history, q, pdf_context, env);
      }

      return json({ answer, sources: contextChunks.slice(0, 3).map(c => ({
        subject: c.subject, type: c.type, attempt: c.latestAttempt
      })) });

    } catch (err) {
      console.error('Worker error:', err);
      return json({ error: 'Server Error', detail: err.message }, 500);
    }
  }
};
