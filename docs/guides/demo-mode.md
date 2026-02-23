# Demo Mode

Demo Mode allows running the project without real LLM provider keys.

## Behavior

- If no `OPENAI_API_KEY` / `GEMINI_API_KEY` is set, query/classification flows fall back to local logic.
- The app still supports ingestion, chunking, indexing, retrieval and evidence display.
- Optional provider keys can be added later via `.env`.

## Run

```bash
cp .env.example .env
docker compose up --build
```

## Optional: enable Docling conversion

Set:

```env
RAG_INSTALL_DOCLING=1
```

Then restart with rebuild.
