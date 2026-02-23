from __future__ import annotations

import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.chunking import chunk_stored_markdown, embed_text
from backend.classification import (
    classify_text,
    select_classification_labels,
    store_classification_metadata,
)
from backend.classification_config import load_label_whitelist
from backend.docling_integration import (
    convert_stored_upload_to_markdown,
    get_docling_status,
    get_document_preview,
    get_source_document_info,
    store_markdown_artifact,
)
from backend import ingestion
from backend.ingestion import ingest_smtp_inbox, ingest_watch_folder, store_upload, validate_upload
from backend.llm_provider import generate_text_with_gemini, generate_text_with_openai
from backend.rag import query_rag
from backend.vector_store import filter_overview_documents, get_store_overview, search_embeddings
from backend import graph_store, mcp_tools

app = FastAPI(title="RAG Ingestion API")


@app.middleware("http")
async def api_prefix_alias(request, call_next):
    """Accept both `/path` and `/api/path` for frontend compatibility."""
    if request.scope.get("path", "").startswith("/api/"):
        request.scope["path"] = request.scope["path"][4:]
    return await call_next(request)

DEFAULT_RECIPIENT = os.getenv("RAG_DEFAULT_EMAIL_RECIPIENT", "rag-inbox@example.local")
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("RAG_CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


class ConvertMarkdownRequest(BaseModel):
    stored_filename: str


class ChunkMarkdownRequest(BaseModel):
    stored_filename: str
    chunk_size: int = 1000
    overlap: int = 200


class RagQueryRequest(BaseModel):
    query_text: str
    top_k: int = 5
    stored_filename: str | None = None
    graph_id: str | None = None
    graph_version_id: str | None = None
    api_key: str | None = None
    provider: str | None = None


class DocumentPreviewRequest(BaseModel):
    stored_filename: str
    max_chars: int = 4000


class ClassificationRequest(BaseModel):
    text: str
    labels: list[str] | None = None
    stored_filename: str | None = None
    api_key: str | None = None
    provider: str | None = None


class ComposedEmailRequest(BaseModel):
    recipient: str = Field(default=DEFAULT_RECIPIENT)
    sender: str = Field(default="demo-sender@example.local")
    subject: str = Field(default="")
    body: str = Field(default="")


class LlmConnectionCheckRequest(BaseModel):
    api_key: str
    provider: str | None = None


class RagTableFilterRequest(BaseModel):
    query: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    classification: str | None = None
    api_key: str | None = None
    provider: str | None = None


class CreateGraphRequest(BaseModel):
    name: str | None = None


class CreateDraftRequest(BaseModel):
    from_version_id: str | None = None


class AddGraphEdgeRequest(BaseModel):
    from_doc_id: str
    to_doc_id: str
    type: str = "references"
    note: str | None = None


class McpToolCallRequest(BaseModel):
    name: str
    arguments: dict | None = None


def _parse_filter_query(query: str | None, provider: str | None = None, api_key: str | None = None) -> dict:
    raw = (query or "").strip()
    payload = {
        "text_query": raw or None,
        "date_from": None,
        "date_to": None,
        "classification": None,
        "file_extensions": [],
        "document_category": None,
        "month": None,
        "month_from": None,
        "month_to": None,
        "year": None,
        "warnings": [],
    }
    if not raw:
        return payload

    selected_provider = (provider or "chatgpt").strip().lower()
    if api_key and api_key.strip():
        prompt = (
            "Extrahiere Filter aus der Nutzeranfrage. "
            "Antworte nur als JSON mit Schlüsseln text_query, date_from, date_to, classification, file_extensions, document_category. "
            "date_from/date_to als ISO-8601 oder null. Anfrage: " + raw
        )
        if selected_provider in {"chatgpt", "openai"}:
            llm = generate_text_with_openai(
                api_key=api_key.strip(),
                model="gpt-4.1-mini",
                prompt=prompt,
                max_output_tokens=240,
            )
        elif selected_provider == "gemini":
            llm = generate_text_with_gemini(
                api_key=api_key.strip(),
                model="gemini-1.5-flash",
                prompt=prompt,
                max_output_tokens=240,
            )
        else:
            llm = None
            payload["warnings"].append(f"Unbekannter LLM Provider '{selected_provider}', nutze lokale Filterlogik.")

        if llm is not None:
            payload["warnings"].extend(llm.warnings)
            if llm.status == "success" and llm.raw_response:
                import json
                try:
                    parsed = json.loads(llm.raw_response)
                    if isinstance(parsed, dict):
                        payload["text_query"] = parsed.get("text_query") or payload["text_query"]
                        payload["date_from"] = parsed.get("date_from") or None
                        payload["date_to"] = parsed.get("date_to") or None
                        payload["classification"] = parsed.get("classification") or None
                        payload["file_extensions"] = parsed.get("file_extensions") or []
                        payload["document_category"] = parsed.get("document_category") or None
                        payload["month"] = parsed.get("month") or None
                        payload["year"] = parsed.get("year") or None
                        return payload
                except json.JSONDecodeError:
                    payload["warnings"].append("LLM Filter-Antwort war kein JSON, lokale Heuristik verwendet.")

    date_matches = re.findall(r"(\d{4}-\d{2}-\d{2})", raw)
    if len(date_matches) >= 2:
        payload["date_from"] = f"{date_matches[0]}T00:00:00+00:00"
        payload["date_to"] = f"{date_matches[1]}T23:59:59+00:00"
    elif len(date_matches) == 1:
        payload["date_from"] = f"{date_matches[0]}T00:00:00+00:00"

    lowered = raw.lower()
    month_aliases = {
        "januar": 1,
        "jänner": 1,
        "jan": 1,
        "februar": 2,
        "februrar": 2,
        "feb": 2,
        "märz": 3,
        "maerz": 3,
        "mrz": 3,
        "april": 4,
        "apr": 4,
        "mai": 5,
        "juni": 6,
        "jun": 6,
        "juli": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "oktober": 10,
        "okt": 10,
        "november": 11,
        "nov": 11,
        "dezember": 12,
        "dez": 12,
    }
    month_mentions: list[int] = []
    for alias, value in month_aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            month_mentions.append(value)

    month_number = month_mentions[0] if month_mentions else None
    range_months = month_mentions[:2] if len(month_mentions) >= 2 else []

    if month_number is not None and not payload["date_from"]:
        year_matches = re.findall(r"\b(20\d{2})\b", lowered)
        if year_matches:
            year = int(year_matches[0])
            payload["year"] = year
            start = datetime(year, month_number, 1, tzinfo=timezone.utc)
            if month_number == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, month_number + 1, 1, tzinfo=timezone.utc)
            payload["date_from"] = start.isoformat()
            payload["date_to"] = (end.replace(microsecond=0) - timedelta(seconds=1)).isoformat()
        else:
            payload["month"] = month_number

    if range_months and not payload["date_from"] and not payload["date_to"]:
        year_matches = re.findall(r"\b(20\d{2})\b", lowered)
        month_start = range_months[0]
        month_end = range_months[1]
        if year_matches:
            year = int(year_matches[0])
            payload["year"] = year
            start = datetime(year, month_start, 1, tzinfo=timezone.utc)
            if month_end == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, month_end + 1, 1, tzinfo=timezone.utc)
            payload["date_from"] = start.isoformat()
            payload["date_to"] = (end.replace(microsecond=0) - timedelta(seconds=1)).isoformat()
        else:
            payload["month_from"] = month_start
            payload["month_to"] = month_end
            payload["month"] = None

    for label in ["invoice", "purchase-order", "contract", "nda", "policy", "report", "cv-resume"]:
        if label in lowered:
            payload["classification"] = label
            break

    if "rechnung" in lowered and not payload["classification"]:
        payload["classification"] = "invoice"

    if any(token in lowered for token in ("präsentation", "praesentation", "folien", "slides", "powerpoint")):
        payload["document_category"] = "presentation"

    known_extensions = ["pdf", "doc", "docx", "txt", "md", "ppt", "pptx", "xls", "xlsx", "csv", "json", "xml"]
    payload["file_extensions"] = [
        extension
        for extension in known_extensions
        if re.search(rf"\b{re.escape(extension)}(?:-datei(?:en)?)?\b", lowered)
    ]

    has_temporal_filter = bool(
        payload["date_from"]
        or payload["date_to"]
        or payload["month"] is not None
        or payload["month_from"] is not None
        or payload["month_to"] is not None
        or payload["year"] is not None
    )
    if has_temporal_filter:
        generic_tokens = {
            "alle", "dokumente", "dokument", "dateien", "datei", "docs", "documents",
            "zeige", "zeig", "list", "liste", "gib", "mir", "bitte",
            "von", "im", "in", "aus", "zwischen", "bis", "und", "dem", "den", "der", "des",
        }
        temporal_tokens = set(month_aliases.keys()) | {"monat", "jahr"}
        tokens = re.findall(r"[a-zA-ZäöüÄÖÜß0-9-]+", lowered)
        meaningful_tokens = [
            token
            for token in tokens
            if token not in generic_tokens
            and token not in temporal_tokens
            and not re.fullmatch(r"20\d{2}", token)
            and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", token)
        ]
        if not meaningful_tokens:
            payload["text_query"] = None

    return payload



def _tokenize_filter_text(value: str | None) -> set[str]:
    generic_tokens = {
        "alle", "dokumente", "dokument", "dateien", "datei", "docs", "documents",
        "zeige", "zeig", "list", "liste", "gib", "mir", "bitte", "ein", "eine",
        "von", "im", "in", "aus", "zwischen", "bis", "und", "dem", "den", "der", "des",
    }
    return {
        token
        for token in re.findall(r"[a-zA-ZäöüÄÖÜß0-9-]+", (value or "").lower())
        if len(token) >= 3 and token not in generic_tokens
    }


def _metadata_text_match_document_ids(overview: dict, text_query: str | None) -> set[str]:
    query_tokens = _tokenize_filter_text(text_query)
    if not query_tokens:
        return set()

    matches: set[str] = set()
    for document in overview.get("documents") or []:
        classification_payload = document.get("classification") or {}
        haystack = " ".join(
            [
                str(document.get("stored_filename") or ""),
                str(document.get("source_filename") or ""),
                str(document.get("source_type") or ""),
                str(document.get("source_timestamp") or ""),
                str(document.get("indexed_at") or ""),
                str((classification_payload or {}).get("label") or ""),
            ]
        ).lower()
        if all(token in haystack for token in query_tokens):
            stored_filename = str(document.get("stored_filename") or "").strip()
            if stored_filename:
                matches.add(stored_filename)

    return matches


def _semantic_text_match_document_ids(
    text_query: str | None,
    *,
    top_k: int = 40,
) -> set[str]:
    if not text_query:
        return set()

    query_embedding = embed_text(text_query)
    semantic_matches = search_embeddings(
        query_embedding,
        top_k=top_k,
        query_text=text_query,
    )
    if not semantic_matches:
        return set()

    best_score_per_doc: dict[str, float] = {}
    for match in semantic_matches:
        stored_filename = str(match.get("stored_filename") or "").strip()
        if not stored_filename:
            continue
        raw_score = match.get("score")
        if raw_score is None:
            raw_score = match.get("vector_score")
        if raw_score is None:
            raw_score = 1.0
        score = float(raw_score)
        previous = best_score_per_doc.get(stored_filename)
        if previous is None or score > previous:
            best_score_per_doc[stored_filename] = score

    if not best_score_per_doc:
        return set()

    ranked = sorted(best_score_per_doc.items(), key=lambda item: item[1], reverse=True)
    best_score = ranked[0][1]
    # Keep high-confidence neighbors, but avoid empty result sets on low absolute score scales.
    min_score_threshold = max(best_score - 0.08, best_score * 0.85)

    selected = {
        doc_id
        for doc_id, score in best_score_per_doc.items()
        if score >= min_score_threshold
    }

    if selected:
        return selected

    # Final fallback: always keep the single strongest semantic hit.
    return {ranked[0][0]}

def _store_upload_fallback_markdown(
    stored_filename: str,
    source_extension: str | None = None,
) -> tuple[bool, list[str]]:
    fallback_path = ingestion.UPLOAD_DIR / stored_filename
    if not fallback_path.exists():
        return False, ["Stored upload missing, fallback conversion failed."]

    normalized_extension = (source_extension or fallback_path.suffix or "").lower()
    if normalized_extension != ".txt":
        normalized_path = ingestion.NORMALIZED_DIR / f"{stored_filename}.json"
        if not normalized_path.exists():
            return False, [
                "Emergency fallback requires normalized ingestion output, but none was found. "
                "Install Docling or optional extractors for PDFs/images/DOCX.",
            ]

        try:
            normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return False, [f"Emergency fallback could not read normalized artifact: {exc}"]

        canonical_text = str(normalized_payload.get("canonical_text") or "").strip()
        chunks = [
            str(chunk.get("text") or "").strip()
            for chunk in normalized_payload.get("chunks") or []
            if isinstance(chunk, dict)
        ]
        chunk_text = "\n\n".join(item for item in chunks if item)

        fallback_sections = [
            f"Source file reference: {stored_filename}",
            canonical_text,
            chunk_text,
        ]
        fallback_text = "\n\n".join(section for section in fallback_sections if section)
        if not fallback_text.strip():
            fallback_text = (
                "Source file reference: "
                f"{stored_filename}\n\n"
                "No extractable text was available; ingestion metadata was indexed as fallback context."
            )

        store_markdown_artifact(
            stored_filename,
            fallback_text,
            warnings=[
                "Primary conversion failed; fallback indexed normalized ingestion content.",
            ],
        )
        return True, []

    fallback_text = fallback_path.read_text(encoding="utf-8", errors="replace")
    if not fallback_text.strip():
        return False, ["Fallback conversion produced no readable text content."]

    store_markdown_artifact(
        stored_filename,
        fallback_text,
        warnings=[
            "Primary conversion failed; fallback decoded plain-text file as UTF-8.",
        ],
    )
    return True, []


def _auto_index_existing_uploads() -> dict:
    processed: list[str] = []
    warnings: list[str] = []

    if not ingestion.UPLOAD_DIR.exists():
        return {"processed": processed, "warnings": warnings}

    for file_path in sorted(path for path in ingestion.UPLOAD_DIR.iterdir() if path.is_file()):
        stored_filename = file_path.name
        artifact_path = ingestion.UPLOAD_DIR / "artifacts" / f"{stored_filename}.md"
        if artifact_path.exists():
            continue

        convert_result = convert_stored_upload_to_markdown(stored_filename)
        if convert_result.status != "success":
            warnings.append(f"{stored_filename}: {convert_result.message}")
            continue

        chunk_result = chunk_stored_markdown(stored_filename)
        if chunk_result.status != "success":
            warnings.append(f"{stored_filename}: {chunk_result.message}")
            continue

        processed.append(stored_filename)

    return {"processed": processed, "warnings": warnings}


def _append_key_warning(payload: dict, api_key: str | None, provider: str | None = None) -> dict:
    warnings = list(payload.get("warnings") or [])
    selected_provider = (provider or "chatgpt").strip() or "chatgpt"
    if api_key and api_key.strip():
        warnings.append(
            f"API key received for '{selected_provider}'. "
            "Backend will attempt provider routing and may fallback to local demo mode."
        )
    else:
        warnings.append(
            f"No API key provided for '{selected_provider}'. "
            "Running with the demo/local model provider."
        )
    payload["warnings"] = warnings
    return payload


@app.get("/config/frontend")
def frontend_config():
    return {
        "default_email_recipient": DEFAULT_RECIPIENT,
    }


@app.get("/classification/labels")
def classification_labels():
    return load_label_whitelist().to_dict()


@app.post("/classification/run")
def run_classification(request: ClassificationRequest):
    whitelist = load_label_whitelist()
    selection = select_classification_labels(request.labels, whitelist)

    if selection.status != "success":
        return JSONResponse(
            status_code=400,
            content=selection.to_dict(),
        )

    result = classify_text(
        request.text,
        selection.labels,
        provider=request.provider,
        api_key=request.api_key,
    )

    if result.status != "success":
        payload = _append_key_warning(result.to_dict(), request.api_key, request.provider)
        return JSONResponse(status_code=400, content=payload)

    payload = result.to_dict()

    if request.stored_filename:
        metadata_result = store_classification_metadata(request.stored_filename, result)
        if metadata_result["status"] == "success":
            payload["metadata_path"] = metadata_result["metadata_path"]
        else:
            payload["warnings"] = payload.get("warnings", []) + metadata_result.get(
                "warnings", []
            )

    return _append_key_warning(payload, request.api_key, request.provider)


@app.post("/llm/check-connection")
def check_llm_connection(request: LlmConnectionCheckRequest):
    api_key = request.api_key.strip()
    provider = (request.provider or "chatgpt").strip().lower()

    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "API key must not be empty.",
                "connected": False,
            },
        )

    prompt = "Reply with exactly one word: CONNECTED"
    if provider in {"chatgpt", "openai"}:
        result = generate_text_with_openai(
            api_key=api_key,
            model="gpt-4.1-mini",
            prompt=prompt,
            max_output_tokens=16,
        )
    elif provider == "gemini":
        result = generate_text_with_gemini(
            api_key=api_key,
            model="gemini-1.5-flash",
            prompt=prompt,
            max_output_tokens=16,
        )
    else:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Unknown provider '{provider}'.",
                "connected": False,
            },
        )

    if result.status != "success" or not (result.raw_response or "").strip():
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Connection test failed.",
                "connected": False,
                "warnings": result.warnings,
            },
        )

    return {
        "status": "success",
        "message": "Connection successful.",
        "connected": True,
        "response_preview": result.raw_response.strip(),
    }


@app.get("/docling/status")
def docling_status():
    return get_docling_status().to_dict()


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    provider: str | None = Form(None),
    api_key: str | None = Form(None),
):
    content = await file.read()
    result = validate_upload(file.filename or "", content)

    if result.status == "error":
        return JSONResponse(
            status_code=400,
            content={
                "status": result.status,
                "message": result.message,
                "warnings": result.warnings,
            },
        )

    if result.status == "warning":
        return JSONResponse(
            status_code=415,
            content={
                "status": result.status,
                "message": result.message,
                "warnings": result.warnings,
            },
        )

    try:
        metadata = store_upload(file.filename or "", content, file.content_type, provider=provider, api_key=api_key)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc), "warnings": []},
        )

    return {
        "status": result.status,
        "message": result.message,
        "warnings": result.warnings,
        "metadata": metadata,
    }


@app.post("/ingest/upload-and-process")
async def ingest_upload_and_process(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    overlap: int = Form(200),
    provider: str | None = Form(None),
    api_key: str | None = Form(None),
):
    content = await file.read()
    validation = validate_upload(file.filename or "", content)
    if validation.status == "error":
        return JSONResponse(
            status_code=400,
            content={
                "status": validation.status,
                "message": validation.message,
                "warnings": validation.warnings,
            },
        )
    if validation.status == "warning":
        return JSONResponse(
            status_code=415,
            content={
                "status": validation.status,
                "message": validation.message,
                "warnings": validation.warnings,
            },
        )

    try:
        metadata = store_upload(file.filename or "", content, file.content_type, provider=provider, api_key=api_key)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc), "warnings": []},
        )
    stored_filename = metadata["stored_filename"]

    extension = (file.filename or "").lower().rsplit(".", maxsplit=1)
    suffix = f".{extension[-1]}" if len(extension) > 1 else ""

    if suffix == ".txt":
        markdown = content.decode("utf-8", errors="replace")
        convert_result = store_markdown_artifact(stored_filename, markdown)
    else:
        convert_result = convert_stored_upload_to_markdown(stored_filename)

    payload: dict = {
        "status": "success",
        "message": "Upload stored.",
        "warnings": list(validation.warnings),
        "metadata": metadata,
        "conversion": convert_result.to_dict(),
    }

    if convert_result.status != "success":
        recovered, recovery_warnings = _store_upload_fallback_markdown(stored_filename, suffix)
        payload["warnings"] = payload["warnings"] + list(convert_result.warnings) + recovery_warnings
        if not recovered:
            payload["status"] = convert_result.status
            payload["message"] = (
                "Upload stored, but conversion failed. Convert/chunk manually for retrieval."
            )
            return JSONResponse(status_code=202, content=payload)
        payload["warnings"].append("Used emergency text fallback so the document can still be indexed.")

    chunk_result = chunk_stored_markdown(
        stored_filename,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    payload["chunking"] = chunk_result.to_dict()
    payload["message"] = "Upload ingested and indexed for RAG."

    if chunk_result.status != "success":
        payload["status"] = "warning"
        payload["warnings"] = payload["warnings"] + list(chunk_result.warnings)
        return JSONResponse(status_code=202, content=payload)

    payload["warnings"] = payload["warnings"] + list(convert_result.warnings)
    return payload


@app.post("/ingest/watch")
def ingest_watch():
    payload = ingest_watch_folder()
    auto = _auto_index_existing_uploads()
    payload["auto_indexed"] = auto["processed"]
    payload["warnings"] = list(payload.get("warnings") or []) + auto["warnings"]
    return payload


@app.post("/ingest/email")
def ingest_email():
    payload = ingest_smtp_inbox()
    auto = _auto_index_existing_uploads()
    payload["auto_indexed"] = auto["processed"]
    payload["warnings"] = list(payload.get("warnings") or []) + auto["warnings"]
    return payload


@app.post("/ingest/email/compose")
def ingest_composed_email(request: ComposedEmailRequest):
    if not request.body.strip():
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Email body must not be empty.",
                "warnings": [],
            },
        )

    safe_subject = request.subject.strip() or "untitled"
    filename = f"email_composed_{safe_subject[:32].replace(' ', '_')}.txt"
    try:
        metadata = store_upload(
            filename,
            request.body.encode("utf-8"),
            content_type="text/plain",
            source_type="email-compose",
            extra_metadata={
                "sender": request.sender,
                "recipient": request.recipient,
                "subject": request.subject,
            },
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc), "warnings": []},
        )

    convert_result = store_markdown_artifact(metadata["stored_filename"], request.body)
    chunk_result = chunk_stored_markdown(metadata["stored_filename"])

    return {
        "status": "success",
        "message": "Composed email ingested and indexed for retrieval.",
        "warnings": [],
        "metadata": metadata,
        "conversion": convert_result.to_dict(),
        "chunking": chunk_result.to_dict(),
    }


@app.post("/convert/markdown")
def convert_markdown(request: ConvertMarkdownRequest):
    result = convert_stored_upload_to_markdown(request.stored_filename)

    if result.status == "warning":
        return JSONResponse(
            status_code=415,
            content={
                "status": result.status,
                "message": result.message,
                "warnings": result.warnings,
            },
        )

    if result.status != "success":
        return JSONResponse(
            status_code=400,
            content={
                "status": result.status,
                "message": result.message,
                "warnings": result.warnings,
            },
        )

    return result.to_dict()


@app.post("/chunk/markdown")
def chunk_markdown(request: ChunkMarkdownRequest):
    result = chunk_stored_markdown(
        request.stored_filename,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )

    if result.status != "success":
        return JSONResponse(
            status_code=400,
            content={
                "status": result.status,
                "message": result.message,
                "warnings": result.warnings,
                "chunks": result.chunks,
            },
        )

    return result.to_dict()


@app.post("/rag/query")
def rag_query(request: RagQueryRequest):
    graph_doc_ids = None
    if request.graph_id and request.graph_version_id:
        try:
            graph_doc_ids = graph_store.resolve_graph_document_ids(request.graph_id, request.graph_version_id)
        except KeyError as exc:
            message = "Graph not found." if str(exc) == "'graph_not_found'" else "Version not found."
            return JSONResponse(status_code=404, content={"status": "warning", "message": message})

    result = query_rag(
        request.query_text,
        top_k=request.top_k,
        stored_filename=request.stored_filename,
        stored_filenames=graph_doc_ids,
        provider=request.provider,
        api_key=request.api_key,
    )

    payload = _append_key_warning(result.to_dict(), request.api_key, request.provider)

    if result.status == "warning":
        return JSONResponse(
            status_code=404,
            content=payload,
        )

    if result.status != "success":
        return JSONResponse(
            status_code=400,
            content=payload,
        )

    return payload




@app.get("/rag/store-overview")
def rag_store_overview(max_chunks_per_document: int = 3):
    safe_limit = max(0, min(max_chunks_per_document, 10))
    return {
        "status": "success",
        "message": "RAG index overview loaded.",
        "overview": get_store_overview(max_chunks_per_document=safe_limit),
    }


@app.post("/rag/store-overview/filter")
def rag_store_overview_filter(request: RagTableFilterRequest):
    overview = get_store_overview(max_chunks_per_document=4)
    parsed = _parse_filter_query(request.query, provider=request.provider, api_key=request.api_key)

    date_from = request.date_from or parsed.get("date_from")
    date_to = request.date_to or parsed.get("date_to")
    classification = request.classification or parsed.get("classification")
    text_query = parsed.get("text_query")
    month = parsed.get("month")
    month_from = parsed.get("month_from")
    month_to = parsed.get("month_to")
    year = parsed.get("year")
    file_extensions = parsed.get("file_extensions") or []
    document_category = parsed.get("document_category")

    semantic_doc_ids: set[str] | None = None
    if text_query:
        metadata_matches = _metadata_text_match_document_ids(overview, text_query)
        semantic_matches_set = _semantic_text_match_document_ids(text_query, top_k=40)

        combined_matches = metadata_matches | semantic_matches_set
        if combined_matches:
            semantic_doc_ids = combined_matches

    documents = filter_overview_documents(
        overview,
        text_query=text_query if semantic_doc_ids is None else None,
        class_label=classification,
        date_from=date_from,
        date_to=date_to,
        month=month,
        month_from=month_from,
        month_to=month_to,
        year=year,
        file_extensions=file_extensions,
        document_category=document_category,
        semantic_doc_ids=semantic_doc_ids,
    )

    return {
        "status": "success",
        "message": "Filtered RAG index overview loaded.",
        "filters": {
            "query": request.query,
            "text_query": text_query,
            "date_from": date_from,
            "date_to": date_to,
            "classification": classification,
            "month": month,
            "month_from": month_from,
            "month_to": month_to,
            "year": year,
            "file_extensions": file_extensions,
            "document_category": document_category,
        },
        "warnings": parsed.get("warnings") or [],
        "overview": {
            **overview,
            "document_count": len(documents),
            "total_chunks": sum(int(item.get("chunk_count") or 0) for item in documents),
            "documents": documents,
        },
    }


@app.get("/mcp/tools/list")
def mcp_tools_list():
    return {
        "status": "success",
        "message": "MCP tools loaded.",
        **mcp_tools.tools_list(),
    }


@app.post("/mcp/tools/call")
def mcp_tools_call(request: McpToolCallRequest):
    result = mcp_tools.tools_call(request.name, request.arguments or {})
    status_code = 400 if result.get("isError") else 200
    return JSONResponse(status_code=status_code, content={
        "status": "warning" if result.get("isError") else "success",
        "message": "MCP tool call failed." if result.get("isError") else "MCP tool call successful.",
        **result,
    })


@app.get("/graphs")
def list_graphs():
    payload = graph_store.list_graphs()
    return {
        "status": "success",
        "message": "Graphs loaded.",
        **payload,
    }


@app.post("/graphs")
def create_graph(request: CreateGraphRequest):
    graph = graph_store.create_graph(name=request.name)
    versions = graph_store.list_versions(graph["graph_id"])
    return {
        "status": "success",
        "message": "Graph created.",
        "graph": {
            "graph_id": graph["graph_id"],
            "name": graph.get("name"),
            "active_version_id": graph.get("active_version_id"),
        },
        "versions": versions,
    }


@app.get("/graphs/{graph_id}/versions")
def list_graph_versions(graph_id: str):
    try:
        payload = graph_store.list_versions(graph_id)
    except KeyError:
        return JSONResponse(status_code=404, content={"status": "warning", "message": "Graph not found."})
    return {
        "status": "success",
        "message": "Graph versions loaded.",
        **payload,
    }


@app.post("/graphs/{graph_id}/versions/draft")
def create_graph_draft(graph_id: str, request: CreateDraftRequest):
    try:
        draft = graph_store.create_draft(graph_id, from_version_id=request.from_version_id)
    except KeyError as exc:
        message = "Graph not found." if str(exc) == "'graph_not_found'" else "Version not found."
        return JSONResponse(status_code=404, content={"status": "warning", "message": message})
    return {
        "status": "success",
        "message": "Draft created and activated.",
        "draft": draft,
    }


@app.post("/graphs/{graph_id}/versions/{version_id}/commit")
def commit_graph_version(graph_id: str, version_id: str):
    try:
        version = graph_store.commit_version(graph_id, version_id)
    except KeyError as exc:
        message = "Graph not found." if str(exc) == "'graph_not_found'" else "Version not found."
        return JSONResponse(status_code=404, content={"status": "warning", "message": message})
    return {
        "status": "success",
        "message": "Version committed and activated.",
        "version": version,
    }


@app.post("/graphs/{graph_id}/versions/{version_id}/rollback")
def rollback_graph_version(graph_id: str, version_id: str):
    try:
        version = graph_store.rollback_active_version(graph_id, version_id)
    except KeyError as exc:
        message = "Graph not found." if str(exc) == "'graph_not_found'" else "Version not found."
        return JSONResponse(status_code=404, content={"status": "warning", "message": message})
    return {
        "status": "success",
        "message": "Active version rolled back.",
        "version": version,
    }


@app.get("/graphs/{graph_id}/versions/{version_id}/view")
def get_graph_view(graph_id: str, version_id: str):
    try:
        view = graph_store.get_graph_view(graph_id, version_id)
    except KeyError as exc:
        message = "Graph not found." if str(exc) == "'graph_not_found'" else "Version not found."
        return JSONResponse(status_code=404, content={"status": "warning", "message": message})
    return {
        "status": "success",
        "message": "Graph view loaded.",
        **view,
    }


@app.post("/graphs/{graph_id}/versions/{version_id}/edges")
def add_graph_edge(graph_id: str, version_id: str, request: AddGraphEdgeRequest):
    try:
        view = graph_store.add_edge(
            graph_id,
            version_id,
            from_doc_id=request.from_doc_id,
            to_doc_id=request.to_doc_id,
            relation_type=request.type,
            note=request.note,
        )
    except ValueError:
        return JSONResponse(status_code=400, content={"status": "error", "message": "from_doc_id and to_doc_id are required."})
    except KeyError as exc:
        message = "Graph not found." if str(exc) == "'graph_not_found'" else "Version not found."
        return JSONResponse(status_code=404, content={"status": "warning", "message": message})

    return {
        "status": "success",
        "message": "Relation added and graph re-laid out.",
        **view,
    }


@app.get("/documents/source-info/{stored_filename}")
def document_source_info(stored_filename: str):
    result = get_source_document_info(stored_filename)

    if result.status == "warning":
        return JSONResponse(status_code=404, content=result.to_dict())

    if result.status != "success":
        return JSONResponse(status_code=400, content=result.to_dict())

    return result.to_dict()


@app.get("/documents/source-file/{stored_filename}")
def document_source_file(stored_filename: str, download: bool = False, viewer: bool = False):
    result = get_source_document_info(stored_filename)

    if result.status == "warning":
        return JSONResponse(status_code=404, content=result.to_dict())

    selected_path = result.source_path
    selected_media_type = result.source_mime_type
    selected_filename = result.stored_filename or stored_filename

    if viewer and not download and result.viewer_source_path:
        selected_path = result.viewer_source_path
        selected_media_type = result.viewer_source_mime_type or selected_media_type
        selected_filename = Path(result.viewer_source_path).name or selected_filename

    if result.status != "success" or not selected_path:
        return JSONResponse(status_code=400, content=result.to_dict())

    media_type = selected_media_type or "application/octet-stream"
    content_disposition_type = "attachment" if download else "inline"
    return FileResponse(
        path=selected_path,
        media_type=media_type,
        filename=selected_filename,
        content_disposition_type=content_disposition_type,
    )


@app.get("/documents/chunks/{stored_filename}")
def document_chunks(stored_filename: str):
    metadata_path = ingestion.METADATA_DIR / f"{stored_filename}.json"
    if not metadata_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "status": "warning",
                "message": "No chunk metadata found for stored filename.",
                "stored_filename": stored_filename,
                "results": [],
            },
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    chunks: list[dict] = []
    if isinstance(payload, dict):
        legacy_chunks = payload.get("chunks")
        if isinstance(legacy_chunks, list):
            chunks = [item for item in legacy_chunks if isinstance(item, dict)]

        if not chunks:
            normalized_path_raw = str(payload.get("normalized_artifact_path") or "").strip()
            if normalized_path_raw:
                normalized_path = Path(normalized_path_raw)
                if normalized_path.exists():
                    try:
                        normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        normalized_payload = {}
                    normalized_chunks = normalized_payload.get("chunks") if isinstance(normalized_payload, dict) else []
                    if isinstance(normalized_chunks, list):
                        chunks = [item for item in normalized_chunks if isinstance(item, dict)]

    results = [
        {
            "stored_filename": stored_filename,
            "chunk_id": chunk.get("chunk_id"),
            "chunk_index": chunk.get("index") if isinstance(chunk.get("index"), int) else index,
            "text": chunk.get("text"),
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "confidence_hint": "manual",
            "score": None,
        }
        for index, chunk in enumerate(chunks)
    ]

    return {
        "status": "success",
        "message": "Loaded chunk list for document.",
        "stored_filename": stored_filename,
        "results": results,
    }


@app.post("/documents/preview")
def document_preview(request: DocumentPreviewRequest):
    result = get_document_preview(
        request.stored_filename,
        max_chars=request.max_chars,
    )

    if result.status == "warning":
        return JSONResponse(status_code=404, content=result.to_dict())

    if result.status != "success":
        return JSONResponse(status_code=400, content=result.to_dict())

    return result.to_dict()
