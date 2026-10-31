# CA Bhaiya v5

Live app: https://ca.arpitagarwala.com

Backend / alternate app: https://ca-chatbot.arpitagarwala.workers.dev

CA Bhaiya now retrieves page-referenced material from an indexed ICAI corpus before generating an answer. This is retrieval-augmented generation: the model reads applicable official passages for each question, so amendments can be incorporated without retraining model weights or buying embeddings.

## Current corpus

The completed initial ingestion on 31 October 2026 contains 1,150 unique official PDFs, represented by 1,167 applicability records and 10,233 searchable passages. Records span Foundation, Intermediate and Final, including all core papers, statutory/judicial updates and corrigenda linked from the course pages, edition/initial pages, IBS digests, and the SPOM Set A/B material identified by ICAI for IBS. It covers the editions and attempt mappings currently linked from those pages; it does not claim that every historical ICAI publication or optional SPOM elective has been indexed.

Verification records official HTTPS origin, the published applicability statement or attempt-specific BoS material notice, PDF content hash and PDF pages. It is automated source/provenance verification, not a human CA review of every explanation. Future/unpublished attempts and documents without unambiguous applicability are not silently assigned the newest law. A newer edition ends the preceding 'onwards' interval.

The runtime uses Cloudflare D1 with indexed FTS5/BM25 search, title/standard-number matching and exact/range attempt filters. It does not consume an embedding API or Vectorize storage. Sources are limited to the selected level/attempt; IBS may retrieve the applicable core Final papers as prerequisite material. The curriculum explorer uses the same attempt mappings to choose the actual PDF links.

## Free answer providers and billing controls

- Primary: Groq `openai/gpt-oss-120b` on the user's free Groq account; `qwen/qwen3.8-27b` is a second free Groq model when the first cannot provide a valid answer.
- Fallback: OpenRouter `openrouter/free`, with prompt/completion maximum prices set to zero. Its authenticated key endpoint reported 50 free requests/day during setup; the app conservatively allows at most 45/day.
- Cloudflare AI: **disabled, with no AI binding in the deployed Worker**. An app counter cannot guarantee zero overages on a paid account with shared AI usage. Cloudflare models passed basic connectivity tests but are not part of this deployment's answer path.
- The LlamaCloud key passed an authenticated organization check. The NVIDIA key passed token authentication but the tested Nemotron endpoint returned 404 (model unavailable for that account). They are not used by the free answer router because valid keys do not establish an ongoing free production allowance. LlamaCloud is a parsing service; local PyMuPDF handles ingestion without that API.

D1-backed atomic reservations apply request/token budgets before provider calls. Retry-After and provider cooldowns are respected. Keys are not rotated to evade limits. Paid custom endpoints require an explicit FREE_ONLY=false operator change. Cached answers include the corpus revision, context, language and history in their cache key. They cannot be reused across attempts. The app returns a capacity message when no free route is available.

Free APIs have hard limits and changing availability; this is not an unlimited service. Groq's exact free allowances are account/model dependent. Do not enable a paid plan or billing on a provider account if it must remain strictly free. The application does not change the user's existing Cloudflare subscription or account billing. D1/Workers hosting use included allowances; ingestion stops at a conservative daily write budget and must resume later if that budget is reached.

## Automatic amendment refresh

The public GitHub repository runs a standard Ubuntu GitHub-hosted runner daily at 00:30 UTC (06:00 India time), and supports manual workflow dispatch. GitHub schedules can be delayed; the exact minute is not guaranteed. No self-hosted runner or running PC is required. No caches or PDF artifacts are uploaded to GitHub storage. The workflow refuses to run if the repository becomes private, so it does not consume paid private-repository runner minutes.

GitHub issues a short-lived signed OpenID Connect identity for each job. The Worker verifies GitHub signature, issuer, audience, immutable repository/owner IDs, public visibility, main branch, exact workflow path, allowed event type and GitHub-hosted runner. The job receives only access to the corpus publication endpoint; it has no Cloudflare account API token, chat API keys or desktop OAuth token.

The cloud runner refreshes official ICAI pages, verifies applicability, parses/revalidates PDFs locally, then stages changed records through the authenticated Worker endpoint. Removed or changing records are withheld during publication. Final activation requires the full expected corpus and all three levels. The server enforces a persistent conservative 40,000 estimated ingestion writes per UTC day. An interrupted or quota-limited run can reuse staged records on its next attempt. Cloudflare AI inference remains disabled.

A small verified-check status file is committed after each successful run for audit and repository activity; raw textbook content stays out of the public repository. The live /api/status response records the cloud run URL and check time. The course directory is published into D1, so changing amendments does not require a new frontend deployment or a Cloudflare deployment credential.

Some historical BoS notice URLs return HTTP 403 to cloud runners. For those exact 2026 attempts only, the updater retains the previously verified facts in knowledge/applicability-evidence.json, preserves the original verification date, and reports a recheck warning. A missing, contradictory or newly unverified notice is not silently accepted. Current curriculum pages, textbooks and amendment documents are still checked normally.

The first complete cloud run succeeded on 31 October 2026: https://github.com/arpitagarwala/ca/actions/runs/41217426374. It published all 1,167 records with zero corpus errors and recorded two historical-notice recheck warnings. The previous Windows task is now disabled. Manual npm run sync remains available as an operator fallback, but is not the scheduled updater.

## Development and maintenance

Node 24+ and Python 3.12+ are used by the ingestion/evaluation tools.

```sh
npm ci
python -m pip install -r scripts/requirements-corpus.txt
npm run check
npm run dev
```

Local `wrangler dev` uses a local D1 database, which is empty unless explicitly populated; production uses the deployed corpus. Optional provider keys may use GROQ_API_KEY / OPENROUTER_API_KEY or the existing GROQ / OPENROUTER aliases in the ignored .env file. `node scripts/publish-secrets.mjs` sends only those two provider keys to Worker secrets, without printing their values. Never commit .env or cached corpus files.

Useful commands:

- `npm run discover`: refresh the official course/chapter directory.
- `npm run verify:sources`: derive applicability records from official pages.
- `venv/Scripts/python.exe scripts/verify_supplements.py`: verify IBS digests, SPOM prerequisites and the attempt-specific BoS notices.
- `venv/Scripts/python.exe scripts/build_corpus.py`: build/revalidate the local corpus.
- `npm run sync:remote`: incrementally bulk-upload approved corpus records to D1.
- `npm run test:live`: run a small three-level evaluation against actual providers; consumes free quota.
- `npm run prepare:pages`: prepare the root GitHub Pages entry point to use the deployed Worker API.
- `npm run deploy`: deploy the Worker and public assets.

The live version is the Worker URL above. The frontend is deployed to the existing ca-bhaiya Vercel project at ca.arpitagarwala.com; ca.arpitagarwala.online preserves its redirect there. Vercel deployment dpl_G3PZunaGASuxZt4rZ9Vhr9ZLMUZy is READY. Backend and database remain on Cloudflare. Source changes in this checkout are not a Git push.

## Validation and practical limits

Unit tests cover attempt boundaries, unofficial/legacy source rejection, provider budgets, cooldowns, paid-route exclusion, citation-number validation, error states and safe answer rendering. Three live tests (Foundation guarantee/surety, Intermediate AS 2 and Final Ind AS 12) returned official citations; the Final case exercised OpenRouter after Groq's quota controls. Production HTTP checks also confirmed the deployed chat works.

Citation-number checks establish that cited sources exist; they do not prove every generated statement follows from those passages. Numerical examples, exceptions, source extraction and amendment conflicts still need ongoing academic evaluation. The service must not promise perfect answers, predicted exam marks or exhaustive latest-law coverage. Image-only PDFs are reported as needing OCR rather than filled with guessed text. Student analytics measure actual browser activity and self-assessed completion, not mastery. Chats and saved answers are stored on the user's device, without account synchronization.

Reference sources: ICAI study material hub https://www.icai.org/post/study-material-nset; Groq limits https://console.groq.com/docs/rate-limits; OpenRouter limits https://openrouter.ai/docs/api/reference/limits; Cloudflare AI pricing https://developers.cloudflare.com/workers-ai/platform/pricing/; D1 pricing https://developers.cloudflare.com/d1/platform/pricing/.
