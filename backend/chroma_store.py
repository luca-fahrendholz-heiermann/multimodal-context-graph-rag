from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


CHROMA_COLLECTION = "rag_chunks"
CHROMA_SPACE = "cosine"


def chroma_available() -> bool:
    return importlib.util.find_spec("chromadb") is not None


def _load_chromadb_module():
    if not chroma_available():
        return None
    try:
        return importlib.import_module("chromadb")
    except Exception:
        return None


def _safe_text(value: object) -> str:
    return str(value or "")


def _normalize_embedding(embedding: object) -> list[float]:
    if not isinstance(embedding, list):
        return []
    normalized: list[float] = []
    for value in embedding:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            return []
    return normalized


def chroma_upsert_embeddings(
    *,
    persist_dir: Path,
    embedding_payload: dict,
    remove_stored_filenames: list[str],
) -> dict | None:
    chromadb = _load_chromadb_module()
    if chromadb is None:
        return None

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": CHROMA_SPACE})

    for stale_filename in remove_stored_filenames:
        collection.delete(where={"stored_filename": stale_filename})

    stored_filename = embedding_payload["stored_filename"]
    expected_dimensions = int(embedding_payload.get("embedding_dimensions") or 0)
    collection.delete(where={"stored_filename": stored_filename})

    embeddings = embedding_payload.get("embeddings") or []
    if not embeddings:
        return {"backend": "chroma", "written": 0}

    ids: list[str] = []
    vectors: list[list[float]] = []
    metadatas: list[dict] = []
    documents: list[str] = []

    for item in embeddings:
        vector = _normalize_embedding(item.get("embedding"))
        if expected_dimensions > 0 and len(vector) != expected_dimensions:
            continue

        chunk_index = int(item.get("index", 0))
        ids.append(f"{stored_filename}:{chunk_index}")
        vectors.append(vector)
        metadatas.append(
            {
                "stored_filename": stored_filename,
                "chunk_index": chunk_index,
                "embedding_dimensions": len(vector),
                "chunk_metadata_path": _safe_text(embedding_payload.get("chunk_metadata_path")),
                "embeddings_path": _safe_text(embedding_payload.get("embeddings_path")),
            }
        )
        documents.append(_safe_text(item.get("text")))

    if not ids:
        return {"backend": "chroma", "written": 0}

    collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents)
    return {"backend": "chroma", "written": len(ids)}


def chroma_search_embeddings(
    *,
    persist_dir: Path,
    query_embedding: list[float],
    top_k: int,
    stored_filename: str | None,
    stored_filenames: list[str] | None = None,
) -> list[dict] | None:
    chromadb = _load_chromadb_module()
    if chromadb is None:
        return None

    normalized_query = _normalize_embedding(query_embedding)
    if not normalized_query:
        return []

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": CHROMA_SPACE})

    where = None
    if stored_filename:
        where = {"stored_filename": stored_filename}
    elif stored_filenames:
        where = {"stored_filename": {"$in": [str(item) for item in stored_filenames if str(item).strip()]}}
    query_result = collection.query(
        query_embeddings=[normalized_query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = (query_result.get("documents") or [[]])[0]
    metadatas = (query_result.get("metadatas") or [[]])[0]
    distances = (query_result.get("distances") or [[]])[0]

    normalized: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        distance_value = float(distance)
        vector_score = 1.0 - distance_value
        normalized.append(
            {
                "stored_filename": metadata.get("stored_filename"),
                "chunk_index": metadata.get("chunk_index"),
                "document": _safe_text(document),
                "score": vector_score,
                "vector_score": vector_score,
                "distance": distance_value,
                "lexical_score": 0.0,
                "chunk_metadata_path": metadata.get("chunk_metadata_path"),
                "embedding_dimensions": int(metadata.get("embedding_dimensions") or len(normalized_query)),
                "metric": CHROMA_SPACE,
            }
        )

    return normalized
