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
const NV_EMBED_URL = 'https://integrate.api.nvidia.com/v1/embeddings';
const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
const EMBED_MODEL = 'baai/bge-m3'; 
const GROQ_MODEL = 'llama-3.3-70b-versatile';
const OPENROUTER_MODEL = 'meta-llama/llama-3.3-70b-instruct';
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

  // Fetch broadly from VectorDB, bypassing buggy internal unindexed filters
  let matches;
  try {
    const res = await env.VECTORIZE.query(queryVec, {
      topK: 50,
      returnMetadata: 'all'
    });
    matches = res.matches;
  } catch (e) {
    return [{ text: 'DEBUG_DB_ERROR: ' + e.message, type: 'error', subject: 'Debug' }];
  }

  if (!matches || matches.length === 0) return [];

  // ── Local Post-Filtering & Cumulative Amendment Logic ───────────────────
  // Filter manually to avoid Cloudflare Vectorize metadata indexing restrictions
  let filtered = matches;
  if (level && level !== 'all') {
    filtered = matches.filter(m => m.metadata && m.metadata.level === level);
  }

  // All versions of all laws are kept in context, but sorted so the LATEST
  // amendment appears first. The system prompt instructs the LLM to treat the
  // latest-dated source as currently in force.
  const ranked = filtered
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
    : 'No specific ICAI context found — rely on your comprehensive CA knowledge.';

    return `You are "CA Bhaiya" — a highly knowledgeable GenZ Chartered Accountant created by Arpit Agarwala. You mentor ${levelStr} students in India for the ${attemptStr} attempt.

PERSONALITY & TONE:
1. GenZ relatable, crisp, and smart. Like a senior who's cleared exams and is now your vibe check partner.
2. STICK TO ENGLISH/HINGLISH flow. NO "NAMASTE" or traditional greetings. Start with "Yo", "Wassup", or "Hey there".
3. NO CRINGE: Use Bollywood song hooks, jokes, or mnemonics ONLY when explaining text-heavy or boring concepts as "Memory Hacks". Do NOT use them for regular chatter.
4. If a question is simple, be direct and fast. If it's a dry logic (Tax, Audit, Law), drop a memory hook (e.g., using a trend or a song) to make it stick.

CRITICAL: If relevant data is missing for ${attemptStr}, admit it rather than hallucinating old 2020-2022 amendments.

AMENDMENT RULES:
- LATEST amendment in context wins.
- Cite Section/Rule numbers ALWAYS.

SELECTED ATTEMPT: ${attemptStr} | Level: ${levelStr}

─── ICAI KNOWLEDGE BASE CONTEXT ───
${contextBlock}
──────────────────────────────────────────────────────────────────`;
}

// ─── Step 4a: Call Groq for text responses ────────────────────────────────────
async function callLLM(provider, systemPrompt, history, query, env) {
  let url, key, model;
  if (provider === 'groq') {
    url = GROQ_URL; key = env.GROQ_API_KEY; model = GROQ_MODEL;
  } else if (provider === 'openrouter') {
    url = OPENROUTER_URL; key = env.OPENROUTER_API_KEY; model = OPENROUTER_MODEL;
  } else {
    throw new Error('Unknown provider');
  }

  const messages = [{ role: 'system', content: systemPrompt }];
  for (const msg of history.slice(-6)) {
    messages.push({ role: msg.role === 'user' ? 'user' : 'assistant', content: msg.text || msg.content || '' });
  }
  messages.push({ role: 'user', content: query });

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${key}` },
    body: JSON.stringify({ model, messages, temperature: 0.65, max_tokens: 1500 }),
  });

  if (!res.ok) return { error: res.status, detail: await res.text() };
  const data = await res.json();
  return { answer: data.choices?.[0]?.message?.content };
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

      // 3. Waterfall through providers
      let finalResult = await callLLM('groq', systemPrompt, history, q, env);
      
      if (finalResult.error === 429 && env.OPENROUTER_API_KEY) {
        finalResult = await callLLM('openrouter', systemPrompt, history, q, env);
      }

      const answer = finalResult.answer || "Yo, current traffic is high! Study some AS-2 while I reset. Catch you in a bit? (API Limit)";

      return json({ answer, sources: contextChunks.slice(0, 3).map(c => ({
        subject: c.subject, type: c.type, attempt: c.latestAttempt
      })) });

    } catch (err) {
      console.error('Worker error:', err);
      return json({ error: 'Server Error', detail: err.message }, 500);
    }
  }
};
