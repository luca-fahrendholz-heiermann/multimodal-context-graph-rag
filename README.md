# Multimodal Graph-RAG Architecture for Cross-Project Context Modeling

## Purpose

This repository is a **portfolio prototype**.  
It demonstrates how to build a robust RAG system with:
- Email ingestion
- Document ingestion
- Multimodal documents
- Transparent feedback and traceability

This is **not a production system**.

---

## Architecture Overview

- Ingestion Layer
- Document Conversion (Docling)
- Chunking and Embedding
- Vector Retrieval
- Chat UI with Evidence Viewer

---

## Features

- **Email + file ingestion**: Upload files or ingest emails from the demo inbox to feed the pipeline. 
- **Multimodal-ready conversion**: Docling-based conversion produces Markdown and metadata artifacts for downstream processing.
- **Chunking + embeddings**: Configurable chunk sizing/overlap with stored embedding artifacts for retrieval. 
- **Vector retrieval**: Hybrid im Backend – JSON-Store als Fallback + Chroma-Bridge für echte Vector-DB-Persistenz/Suche.
- **Vector DB status**: Chroma ist jetzt im Backend integriert (mit JSON-Fallback), Details in `VECTOR_DB_DECISION.md`.
- **RAG chat + evidence viewer**: Query the system and review retrieved chunks, confidence hints, and document previews.
- **Drag-and-drop auto ingestion**: Upload files in the UI and trigger upload → conversion → chunking → indexing in one step.
- **Virtual email composer ingestion**: Compose an email in the UI (to/from/subject/body) and ingest it like an email source without SMTP setup.
- **API key input in UI**: Provide a token for query/classification requests so a real LLM provider can be plugged in.
- **Classification with labels**: Optional LLM-style labeling with whitelist validation and confidence reporting.
- **Feedback and warnings**: Ingestion and classification warnings are surfaced to make the pipeline transparent.

---

## LLM Provider behavior (Gemini vs. ChatGPT)

Currently, this demo does **not** auto-detect or switch between Gemini and ChatGPT based on your key.

- The API key field is forwarded to backend endpoints.
- The backend currently runs a local/demo provider path and only adds informational warnings about the key.
- To use Gemini/OpenAI for real model calls, you would plug in a concrete provider implementation in the backend.

## Getting Started (Docker)

### Prerequisites

- Docker Desktop or Docker Engine with Compose support.

### Start the stack

From the repository root:

```bash
docker compose up --build
```

This will start the backend and frontend development containers using `docker-compose.yml`.

### Stop the stack

```bash
docker compose down
```

### Access services

- Frontend: http://localhost:3000  
- Backend API: http://localhost:8000  
- Mailpit (SMTP inbox UI): http://localhost:8025  

---


### Optional environment variables

- `RAG_DEFAULT_EMAIL_RECIPIENT`: pre-fills the recipient in the Email Composer UI.
- `RAG_INSTALL_DOCLING`: set to `1` to install Docling in the backend container. Default is `0`.

### Ist Docling im Docker-Environment schon installiert?

Kurz: **standardmäßig nein**.

- Im `docker-compose.yml` wird `docling` nur installiert, wenn `RAG_INSTALL_DOCLING=1` gesetzt ist.
- Ohne diese Variable startet das Backend absichtlich ohne `docling`.

Status prüfen:

```bash
curl http://localhost:8000/docling/status
```

### Docling korrekt in diesem Projekt aktivieren

**Empfohlen (plattformunabhängig):** `.env` im Repo-Root anlegen/ergänzen:

```env
RAG_INSTALL_DOCLING=1
```

Dann starten:

```bash
docker compose up --build
```

**Alternativ per Shell-Umgebungsvariable (einmalig):**

Linux/macOS (bash/zsh):

```bash
RAG_INSTALL_DOCLING=1 docker compose up --build
```

Windows PowerShell:

```powershell
$env:RAG_INSTALL_DOCLING = "1"
docker compose up --build
```

Windows CMD:

```cmd
set RAG_INSTALL_DOCLING=1 && docker compose up --build
```

Wenn der Stack bereits läuft, einmal neu bauen/starten:

```bash
docker compose down
docker compose up --build
```

Optional (nur ad hoc in laufendem Backend-Container, nicht reproduzierbar):

```bash
docker compose exec backend pip install --no-cache-dir -r backend/requirements-docling.txt
```

### Optional Docling installation (for PDF/DOCX/image conversion)

By default, the backend starts **without** installing `docling` to keep startup fast and avoid large ML/CUDA downloads.

If you need Docling conversion features, start Compose with:

```bash
RAG_INSTALL_DOCLING=1 docker compose up --build
```

Without Docling, `.txt` ingestion still works and the API returns a clear warning for conversion endpoints.

### Troubleshooting: Frontend zeigt nach Pull noch alten Stand

Wenn nach `git pull` + `docker compose up --build` alte UI-Elemente sichtbar sind, liegt es meist am Browser-Cache.

- Der Frontend-Container läuft jetzt mit `http-server -c-1` (Cache deaktiviert).
- Bitte einmal **Hard Reload** im Browser ausführen (`Ctrl+Shift+R` / `Cmd+Shift+R`).
- Verifizieren, was der Server wirklich ausliefert:

```bash
curl -s http://localhost:3000 | rg "RAG Datenbank ansehen"
```

Wenn die Zeile gefunden wird, wird die aktuelle HTML-Version ausgeliefert.

### Troubleshooting: "Backend nicht erreichbar"

If the frontend says `Backend nicht erreichbar`, check whether the backend is still installing dependencies.
A common cause is a very long `docling` installation (hundreds of MB/GB, especially `torch` wheels).

Quick checks:

```bash
docker compose ps
docker compose logs -f backend
```

If you see long-running downloads, restart without Docling (default), or only enable it when needed via `RAG_INSTALL_DOCLING=1`.

### Troubleshooting: Compose variable interpolation errors

If Compose reports an interpolation error like `${RAG_INSTALL_DOCLING:1}`, the expression is malformed.
Use the default-value form with `:-` (as in this repo): `${RAG_INSTALL_DOCLING:-0}`.


## References

- [Specification](SPEC.md)
- [Vector DB Decision](VECTOR_DB_DECISION.md)

---

## How to Use Codex on This Repo

Codex must:
- Follow this README and SPEC.md
- Implement tasks step by step
- Mark completed tasks below

---

## Task Checklist

### Phase 1: Project Setup
- [x] Initialize repository structure
- [x] Add Docker Compose for local development
- [x] Add basic README and SPEC references

### Phase 2: Ingestion
- [x] Implement file upload ingestion
- [x] Implement watch-folder ingestion
- [x] Implement SMTP demo inbox (Mailpit or MailHog)

### Phase 3: Document Conversion
- [x] Integrate Docling
- [x] Convert documents to Markdown
- [x] Store metadata and artifacts

### Phase 4: Chunking and Embeddings
- [x] Implement chunking pipeline
- [x] Store chunks and embeddings
- [x] Connect vector store

### Phase 5: Chat and Retrieval
- [x] Implement RAG query pipeline
- [x] Add model provider abstraction
- [x] Add token input in UI

### Phase 6: Evidence Viewer
- [x] Show referenced documents
- [x] Highlight used chunks
- [x] Enable document preview

### Phase 7: Classification (Optional Toggle)
- [x] Add label whitelist configuration
- [x] Implement LLM-based classification
- [x] Store classification metadata

### Phase 8: Feedback and Warnings
- [x] Add ingestion warnings
- [x] Add classification confidence display
- [x] Add retrieval confidence hints

---


### Follow-up TODOs (User Feedback 2026-02-12)
- [x] Ensure upload processing converts/chunks with fallback paths so retrieval still works when Docling conversion fails.
- [x] Fix dynamic search parsing so natural-language month queries (e.g. "alle dokumente von februrar") map to ISO date filters.
- [x] Move the "RAG Datenbank ansehen" button out of the chat card and place it above/in the database area.
- [x] Keep this README TODO list updated and mark completed follow-up tasks.

## Done Tasks Log

### SPEC Erweiterte Task-Liste (aktuell bearbeitet)
- Added explicit per-step performance budgets for ingestion observability (parse/normalize/embed/index/viewer_artifacts), including breach tracking in logs + aggregated metrics.
- 2026-02-14: End-to-End-Testmatrix je Formatfamilie ergänzt (Happy Path, Malformed, Large File, Encoding Edge Cases) über die Upload→Parse→Normalize→Index Pipeline mit Artefakt- und QA-Validierung.
- 2026-02-14: Zielschema als Pydantic-Modelle ergänzt und JSON-Schema-basierte Validierungstests für normalisierte Artefakte hinzugefügt.
- 2026-02-14: Dead-letter-Queue ergänzt: fehlgeschlagene Upload-Verarbeitung schreibt reproduzierbare Fehlerreports (JSON + Queue-Index) inkl. Fingerprint/Stacktrace.
- 2026-02-14: Idempotente Index-Upserts umgesetzt: RAG- und Vector-Store ersetzen nun ältere Einträge mit identischem SHA256, damit Re-Runs keine Dubletten erzeugen.

---

## DSGVO-orientiertes Graph-RAG Skeleton (neu)

Es wurde ein TypeScript-Skeleton für Multi-Tenant Graph-RAG mit Supabase ergänzt:

- **DB-Migrationen unter `/db`**: Schema, RLS-Policies, Indexe.
- **`/rag-core` Module**:
  - `pii_filter.ts` (PII-Reduktion vor Embedding)
  - `ingest.ts`, `query.ts`, `graph.ts`
  - `embedding_provider.ts` (Adapter + Dummy)
  - `supabase.ts` (Server-Client + Tenant-Guard)
  - `logging.ts` (PII-freies, gehashtes Logging)
- **`/api` Endpoints**:
  - `ingest.ts`
  - `query.ts`
  - `delete-account.ts` (`DELETE /api/account`)
  - `storage.ts` (private Bucket, presigned Upload/Download)
- **`/docs` Security-Dokus**:
  - `retention-policy.md`
  - `data-processing.md`
  - `security-notes.md`
- **Tests**: `tests/pii_filter.test.ts`, `tests/rls_guard.test.ts`

### Lokaler Start (bestehendes Backend + neue TS Tests)

1. **Backend starten (Python/FastAPI):**

```bash
docker compose up --build backend
```

2. **Backend Reachability prüfen:**

```bash
curl http://localhost:8000/health
```

Sollte `{"status":"ok"}` liefern.

3. **TypeScript Tests ausführen:**

```bash
npm install
npm test
```

### Neue DSGVO-Task-Checklist

- [x] DB Schema + RLS Policies unter `/db`
- [x] `/rag-core` Module für PII/ingest/query/graph/logging/supabase
- [x] `/api` Endpoints für ingest/query/delete-account
- [x] Private Storage Helpers mit presigned URLs
- [x] Logging Wrapper ohne PII/Prompt Logging
- [x] Minimale Tests für `stripPII` + Tenant Guard
- [x] Doku zu Retention, Processing, Security
