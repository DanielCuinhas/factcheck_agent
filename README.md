# Fact-check agent (LangGraph + FastAPI)

Takes a news URL or raw text and returns a structured dossier answering:
origin, source legitimacy, author verification, suspicion level, corroboration.

## Pipeline

```
START -> ingest -> extract_claims -> +-- origin ---------+
                                     +-- reputation -----+
                                     +-- authors --------+--> synthesize -> END
                                     +-- suspicion ------+
                                     +-- corroboration --+
```

| Question | Node | Evidence used |
|---|---|---|
| 1. Where did it first appear? | `origin` | web/news search, Wayback CDX, outbound links, cited-page peek |
| 2. Is the origin legit? | `reputation` | RDAP domain age, about/contact/corrections probe, search for flags |
| 3. Are the journalists real? | `authors` | byline search across the web |
| 4. Is it suspicious? | `suspicion` | content signals only (weak evidence by design) |
| 5. Is it well supported? | `corroboration` | per-claim news search + Google Fact Check API, independence check |

## Run

Choose the model backend with `MODEL_PROVIDER` (`anthropic`, `openai`, or `huggingface`). Anthropic is the
default and uses `ANTHROPIC_API_KEY` plus `MODEL_NAME`. OpenAI uses `OPENAI_API_KEY` and `OPENAI_MODEL_NAME`.
Hugging Face uses `HF_TOKEN` and `HUGGINGFACE_MODEL_NAME`, with `https://router.huggingface.co/v1` as its
default endpoint. `MODEL_BASE_URL` can override the endpoint for either OpenAI-compatible provider. Models
used with structured output must support tool calling.
The API caller can override the configured provider and model for one run with `model_provider` and `model`;
for example, `"model_provider": "huggingface"`. Omit either field to use its configured default. API keys
remain configured on the server.



```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add keys
uvicorn app.main:app --reload
```

```bash
curl -X POST localhost:8000/check -H 'content-type: application/json' \
  -d '{"url": "https://example.com/some-article", "model_provider": "openai", "model": "gpt-4o-mini"}'

# streaming progress (SSE)
curl -N -X POST localhost:8000/check/stream -H 'content-type: application/json' \
  -d '{"text": "Paste the news text here..."}'

# or CLI
python -m app.cli "https://example.com/some-article" --provider openai --model gpt-4o-mini
```

## Safeguards built in

- Fetched pages and search snippets are wrapped in `<untrusted>` tags and the system prompt forbids following instructions inside them.
- Any URL the model returns is dropped unless it was in the evidence we supplied.
- A "supported" or "contradicted" verdict with no matching evidence is downgraded to "unverified".
- The item's own publisher is excluded from corroboration.
- URL fetching resolves every hop (incl. redirects) and refuses non-public IPs (SSRF).
- `limitations` in the report lists everything that could not be checked.

## Known gaps / next steps

- No reverse image/video search (TinEye / Google Lens need paid or scraped access).
- Closed platforms (WhatsApp, Telegram groups, TikTok) can't be traced.
- Citation following is shallow (peeks 3 links). Replace with a bounded agent loop for multi-hop chains.
- No caching (domain/author lookups repeat across requests), no auth, no rate limiting.
- DNS-rebinding between the SSRF check and the request is not fully closed; pin the resolved IP for production.
- Add licensed reputation data (NewsGuard etc.) as another signal in `reputation`.
- Add tests with recorded fixtures plus a labelled eval set (known hoaxes vs. verified stories).
