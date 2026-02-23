from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import tempfile

from backend import ingestion
from backend import chroma_store


@dataclass(frozen=True)
class VectorStoreEntry:
    stored_filename: str
    embedding_dimensions: int
    chunk_count: int
    embeddings: list[dict]
    chunk_metadata_path: str
    embeddings_path: str
    indexed_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _vector_store_path() -> Path:
    return ingestion.UPLOAD_DIR / "vector_store.json"


def _rag_index_path() -> Path:
    return ingestion.UPLOAD_DIR / "rag_index.json"


def _chroma_store_path() -> Path:
    return ingestion.UPLOAD_DIR / "chroma_store"


def _load_vector_store() -> dict:
    store_path = _vector_store_path()
    if not store_path.exists():
        return {"updated_at": None, "documents": {}}
    return json.loads(store_path.read_text(encoding="utf-8"))


def _load_rag_index() -> dict:
    index_path = _rag_index_path()
    if not index_path.exists():
        return {
            "updated_at": None,
            "documents": {},
            "provenance_graph": {
                "node_count": 0,
                "edge_count": 0,
                "nodes": [],
                "edges": [],
            },
        }
    return json.loads(index_path.read_text(encoding="utf-8"))


def _build_provenance_graph(documents: dict) -> dict:
    nodes: dict[str, dict] = {}
    edge_map: dict[tuple[str, str, str], dict] = {}

    for stored_filename, document in documents.items():
        relations = document.get("relations") or []
        for relation in relations:
            if not isinstance(relation, dict):
                continue

            relation_type = str(relation.get("type") or "").strip().lower()
            if relation_type not in {"contains", "derived_from", "derived-from", "abgeleitet-von"}:
                continue

            source = str(relation.get("source") or stored_filename).strip()
            target = str(relation.get("target") or "").strip()
            if not source or not target:
                continue

            nodes.setdefault(
                source,
                {
                    "id": source,
                    "origin_stored_filename": stored_filename,
                },
            )
            target_node = {
                "id": target,
                "origin_stored_filename": stored_filename,
            }
            target_sha = relation.get("target_sha256")
            if target_sha:
                target_node["sha256"] = target_sha
            nodes.setdefault(target, target_node)

            edge_key = (relation_type, source, target)
            if edge_key in edge_map:
                continue

            edge_payload = {
                "type": relation_type,
                "source": source,
                "target": target,
                "document": stored_filename,
            }
            if target_sha:
                edge_payload["target_sha256"] = target_sha
            edge_map[edge_key] = edge_payload

    return {
        "node_count": len(nodes),
        "edge_count": len(edge_map),
        "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
        "edges": sorted(edge_map.values(), key=lambda item: (item["type"], item["source"], item["target"])),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _metadata_sha_for_stored_filename(stored_filename: str) -> str | None:
    metadata_path = ingestion.METADATA_DIR / f"{stored_filename}.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    sha256 = str(payload.get("sha256") or "").strip()
    return sha256 or None


def _drop_duplicate_sha_entries(
    documents: dict,
    *,
    target_sha256: str | None,
    keep_stored_filename: str,
    metadata_lookup: callable,
) -> dict:
    if not target_sha256:
        return documents

    deduped: dict = {}
    for stored_filename, entry in documents.items():
        if stored_filename == keep_stored_filename:
            deduped[stored_filename] = entry
            continue
        current_sha = metadata_lookup(stored_filename, entry)
        if current_sha and current_sha == target_sha256:
            continue
        deduped[stored_filename] = entry
    return deduped


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def upsert_embeddings(embedding_payload: dict) -> dict:
    ingestion.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    store = _load_vector_store()
    documents = store.get("documents", {})

    stored_filename = embedding_payload["stored_filename"]
    target_sha256 = _metadata_sha_for_stored_filename(stored_filename)
    documents = _drop_duplicate_sha_entries(
        documents,
        target_sha256=target_sha256,
        keep_stored_filename=stored_filename,
        metadata_lookup=lambda filename, _entry: _metadata_sha_for_stored_filename(filename),
    )
    removed_documents = [name for name in store.get("documents", {}).keys() if name not in documents]

    entry = VectorStoreEntry(
        stored_filename=stored_filename,
        embedding_dimensions=embedding_payload["embedding_dimensions"],
        chunk_count=embedding_payload["chunk_count"],
        embeddings=embedding_payload["embeddings"],
        chunk_metadata_path=embedding_payload["chunk_metadata_path"],
        embeddings_path=embedding_payload["embeddings_path"],
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )

    documents[entry.stored_filename] = entry.to_dict()
    store["documents"] = documents
    store["updated_at"] = datetime.now(timezone.utc).isoformat()

    store_path = _vector_store_path()
    _atomic_write_json(store_path, store)

    chroma_result = chroma_store.chroma_upsert_embeddings(
        persist_dir=_chroma_store_path(),
        embedding_payload=embedding_payload,
        remove_stored_filenames=removed_documents,
    )

    return {"path": str(store_path), "entry": entry.to_dict(), "chroma": chroma_result}


def upsert_rag_document(
    *,
    stored_filename: str,
    metadata: dict,
    chunks: list[dict],
    relations: list[dict],
) -> dict:
    ingestion.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    index = _load_rag_index()
    documents = index.get("documents", {})
    target_sha256 = str(metadata.get("sha256") or "").strip() or None
    documents = _drop_duplicate_sha_entries(
        documents,
        target_sha256=target_sha256,
        keep_stored_filename=stored_filename,
        metadata_lookup=lambda _filename, entry: str(((entry.get("metadata") or {}).get("sha256")) or "").strip() or None,
    )
    documents[stored_filename] = {
        "stored_filename": stored_filename,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "chunks": chunks,
        "relations": relations,
        "chunk_count": len(chunks),
        "relation_count": len(relations),
    }
    index["documents"] = documents
    index["provenance_graph"] = _build_provenance_graph(documents)
    index["updated_at"] = datetime.now(timezone.utc).isoformat()

    index_path = _rag_index_path()
    _atomic_write_json(index_path, index)
    return {"path": str(index_path), "entry": documents[stored_filename]}


def _tokenize_for_lexical_match(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-ZäöüÄÖÜß0-9-]+", (text or "").lower())
        if len(token) >= 3
    }


def _lexical_overlap_score(query_text: str | None, chunk_text: str | None) -> float:
    query_tokens = _tokenize_for_lexical_match(query_text or "")
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenize_for_lexical_match(chunk_text or "")
    if not chunk_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    return overlap / max(1, len(query_tokens))


def search_embeddings(
    query_embedding: list[float],
    top_k: int = 5,
    stored_filename: str | None = None,
    stored_filenames: list[str] | None = None,
    query_text: str | None = None,
) -> list[dict]:
    if top_k <= 0:
        return []

    chroma_results = chroma_store.chroma_search_embeddings(
        persist_dir=_chroma_store_path(),
        query_embedding=query_embedding,
        top_k=top_k,
        stored_filename=stored_filename,
        stored_filenames=stored_filenames,
    )
    if chroma_results is not None:
        return chroma_results

    store = _load_vector_store()
    documents = store.get("documents", {})
    candidates: list[dict] = []

    for entry in documents.values():
        if stored_filename and entry.get("stored_filename") != stored_filename:
            continue
        if stored_filenames and entry.get("stored_filename") not in stored_filenames:
            continue
        for item in entry.get("embeddings", []):
            base_score = _dot_product(query_embedding, item.get("embedding", []))
            lexical_score = _lexical_overlap_score(query_text, item.get("text"))
            combined_score = base_score + (0.2 * lexical_score)
            candidates.append(
                {
                    "stored_filename": entry.get("stored_filename"),
                    "chunk_index": item.get("index"),
                    "score": combined_score,
                    "vector_score": base_score,
                    "lexical_score": lexical_score,
                    "chunk_metadata_path": entry.get("chunk_metadata_path"),
                    "embedding_dimensions": entry.get("embedding_dimensions"),
                    "_embedding": item.get("embedding", []),
                }
            )

    candidates.sort(key=lambda result: result["score"], reverse=True)

    selected: list[dict] = []
    lambda_mult = 0.8

    while candidates and len(selected) < top_k:
        best_index = 0
        best_mmr = None
        for index, candidate in enumerate(candidates):
            relevance = candidate["score"]
            if not selected:
                mmr = relevance
            else:
                max_similarity = max(
                    _dot_product(candidate.get("_embedding", []), chosen.get("_embedding", []))
                    for chosen in selected
                )
                mmr = (lambda_mult * relevance) - ((1 - lambda_mult) * max_similarity)

            if best_mmr is None or mmr > best_mmr:
                best_mmr = mmr
                best_index = index

        selected.append(candidates.pop(best_index))

    for item in selected:
        item.pop("_embedding", None)

    return selected


def get_store_overview(max_chunks_per_document: int = 3) -> dict:
    store = _load_vector_store()
    documents = store.get("documents", {})

    overview_documents: list[dict] = []
    for entry in documents.values():
        stored_filename = entry.get("stored_filename")
        source_metadata_path = ingestion.METADATA_DIR / f"{stored_filename}.json"
        source_metadata: dict = {}
        if source_metadata_path.exists():
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))

        classification = source_metadata.get("classification")
        embeddings = entry.get("embeddings", [])
        chunk_preview = [
            {
                "index": item.get("index"),
                "vector_dimensions": len(item.get("embedding", [])),
            }
            for item in embeddings[:max(0, max_chunks_per_document)]
        ]
        overview_documents.append(
            {
                "stored_filename": stored_filename,
                "source_filename": source_metadata.get("filename"),
                "source_type": source_metadata.get("source_type"),
                "source_timestamp": source_metadata.get("timestamp"),
                "size_bytes": source_metadata.get("size_bytes"),
                "chunk_count": entry.get("chunk_count", 0),
                "embedding_dimensions": entry.get("embedding_dimensions", 0),
                "indexed_at": entry.get("indexed_at"),
                "index_status": "indexed",
                "classification": {
                    "label": (classification or {}).get("label") if isinstance(classification, dict) else None,
                    "confidence": (classification or {}).get("confidence") if isinstance(classification, dict) else None,
                },
                "chunk_metadata_path": entry.get("chunk_metadata_path"),
                "embeddings_path": entry.get("embeddings_path"),
                "chunk_preview": chunk_preview,
            }
        )

    overview_documents.sort(
        key=lambda item: item.get("indexed_at") or "",
        reverse=True,
    )

    total_chunks = sum(int(item.get("chunk_count") or 0) for item in overview_documents)
    return {
        "updated_at": store.get("updated_at"),
        "document_count": len(overview_documents),
        "total_chunks": total_chunks,
        "documents": overview_documents,
    }


def _parse_iso_date(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def filter_overview_documents(
    overview: dict,
    *,
    text_query: str | None = None,
    class_label: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    month: int | None = None,
    month_from: int | None = None,
    month_to: int | None = None,
    year: int | None = None,
    file_extensions: list[str] | None = None,
    document_category: str | None = None,
    semantic_doc_ids: set[str] | None = None,
) -> list[dict]:
    documents = list(overview.get("documents") or [])
    filtered: list[dict] = []

    query_tokens = [token for token in re.split(r"\s+", (text_query or "").lower()) if token]
    from_dt = _parse_iso_date(date_from)
    to_dt = _parse_iso_date(date_to)
    label = (class_label or "").strip().lower()
    normalized_extensions = {
        ((value or "").strip().lower().lstrip("."))
        for value in (file_extensions or [])
        if (value or "").strip()
    }
    normalized_category = (document_category or "").strip().lower()
    normalized_semantic_doc_ids = {doc_id for doc_id in (semantic_doc_ids or set()) if doc_id}

    for document in documents:
        classification_payload = document.get("classification") or {}
        source_filename = str(document.get("source_filename") or "")
        stored_filename = str(document.get("stored_filename") or "")
        extension_source = Path(source_filename or stored_filename).suffix.lower().lstrip(".")

        if normalized_extensions and extension_source not in normalized_extensions:
            continue

        if normalized_semantic_doc_ids and stored_filename not in normalized_semantic_doc_ids:
            continue

        if normalized_category == "presentation":
            combined_name = f"{source_filename} {stored_filename}".lower()
            looks_like_presentation = extension_source in {"ppt", "pptx"} or (
                extension_source == "pdf"
                and any(token in combined_name for token in ("presentation", "präsentation", "praesentation", "slides", "folie", "folien", "deck"))
            )
            if not looks_like_presentation:
                continue

        haystack = " ".join(
            [
                str(document.get("stored_filename") or ""),
                str(document.get("source_filename") or ""),
                str(document.get("source_type") or ""),
                str(document.get("source_timestamp") or ""),
                str(document.get("indexed_at") or ""),
                str(document.get("index_status") or ""),
                str(document.get("chunk_count") or ""),
                str(document.get("embedding_dimensions") or ""),
                str(classification_payload.get("label") or ""),
                str(classification_payload.get("confidence") or ""),
            ]
        ).lower()

        if label:
            current_label = (
                (((document.get("classification") or {}).get("label")) or "").strip().lower()
            )
            if current_label != label:
                continue

        candidate = _parse_iso_date(document.get("source_timestamp") or document.get("indexed_at"))

        if from_dt or to_dt:
            if candidate is None:
                continue
            if from_dt and candidate < from_dt:
                continue
            if to_dt and candidate > to_dt:
                continue

        if month is not None:
            if candidate is not None:
                if candidate.month != month:
                    continue
            else:
                month_token_match = bool(re.search(rf"(?:^|[^\d]){month:02d}(?:[^\d]|$)", haystack))
                if not month_token_match:
                    continue

        if year is not None:
            if candidate is not None:
                if candidate.year != year:
                    continue
            elif str(year) not in haystack:
                continue

        if month_from is not None and month_to is not None:
            if candidate is not None:
                if not (month_from <= candidate.month <= month_to):
                    continue
            else:
                month_range_tokens = [f"{value:02d}" for value in range(month_from, month_to + 1)]
                if not any(token in haystack for token in month_range_tokens):
                    continue

        if query_tokens:
            if not all(token in haystack for token in query_tokens):
                continue

        filtered.append(document)

    return filtered
