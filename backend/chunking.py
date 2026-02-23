from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend import ingestion, model_provider, vector_store


@dataclass(frozen=True)
class ChunkingResult:
    status: str
    message: str
    chunks: list[dict]
    warnings: list[str]
    metadata_path: str | None = None
    embeddings_path: str | None = None
    vector_store_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_dirs() -> None:
    ingestion.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ingestion.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    _embeddings_dir().mkdir(parents=True, exist_ok=True)


def _chunks_dir() -> Path:
    return ingestion.UPLOAD_DIR / "chunks"


def _embeddings_dir() -> Path:
    return ingestion.UPLOAD_DIR / "embeddings"


def _validate_chunk_params(chunk_size: int, overlap: int) -> str | None:
    if chunk_size <= 0:
        return "Chunk size must be greater than zero."
    if overlap < 0:
        return "Chunk overlap must be zero or greater."
    if overlap >= chunk_size:
        return "Chunk overlap must be smaller than the chunk size."
    return None


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    segments = _segment_text_for_chunking(text)
    chunks: list[dict] = []
    index = 0

    for segment in segments:
        segment_chunks = _chunk_segment(
            segment_text=segment["text"],
            segment_start=segment["start"],
            chunk_size=chunk_size,
            overlap=overlap,
            start_index=index,
        )
        chunks.extend(segment_chunks)
        index += len(segment_chunks)

    return chunks


def _chunk_segment(
    *,
    segment_text: str,
    segment_start: int,
    chunk_size: int,
    overlap: int,
    start_index: int,
) -> list[dict]:
    chunks: list[dict] = []
    start = 0
    index = start_index
    length = len(segment_text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk_text = segment_text[start:end]
        chunks.append(
            {
                "index": index,
                "start": segment_start + start,
                "end": segment_start + end,
                "text": chunk_text,
            }
        )
        if end >= length:
            break
        start = end - overlap
        index += 1

    return chunks


def _segment_text_for_chunking(text: str) -> list[dict]:
    units = _paragraph_units(text)
    if not units:
        return []

    segments: list[dict] = []
    current: dict | None = None

    for unit in units:
        noisy = _is_likely_ocr_noise(unit["text"])
        if current is None:
            current = {"start": unit["start"], "end": unit["end"], "noisy": noisy}
            continue

        if noisy != current["noisy"]:
            segments.append(
                {
                    "start": current["start"],
                    "end": current["end"],
                    "text": text[current["start"] : current["end"]],
                }
            )
            current = {"start": unit["start"], "end": unit["end"], "noisy": noisy}
            continue

        current["end"] = unit["end"]

    if current is not None:
        segments.append(
            {
                "start": current["start"],
                "end": current["end"],
                "text": text[current["start"] : current["end"]],
            }
        )

    return [segment for segment in segments if segment["text"].strip()]


def _paragraph_units(text: str) -> list[dict]:
    units: list[dict] = []

    start = 0
    for match in re.finditer(r"\n\s*\n+", text):
        end = match.start()
        chunk_text = text[start:end]
        if chunk_text.strip():
            units.append({"start": start, "end": end, "text": chunk_text})
        start = match.end()

    tail = text[start:]
    if tail.strip():
        units.append({"start": start, "end": len(text), "text": tail})

    return units


def _is_likely_ocr_noise(text: str) -> bool:
    sample = text.strip()
    if len(sample) < 32:
        return False

    total = len(sample)
    alpha_count = sum(char.isalpha() for char in sample)
    digit_count = sum(char.isdigit() for char in sample)
    symbol_count = sum(not (char.isalnum() or char.isspace()) for char in sample)
    mojibake_count = sum(sample.count(marker) for marker in ("Ã", "Ð", "Ê", "Ï", "Ý", "À", "ë", "õ", "Ì", "ï"))
    delimiter_count = sample.count("|") + sample.count("=")

    alpha_ratio = alpha_count / total
    symbol_ratio = symbol_count / total
    numeric_ratio = digit_count / total

    if mojibake_count >= 2 and (symbol_ratio > 0.12 or delimiter_count >= 3):
        return True
    if alpha_ratio < 0.45 and (symbol_ratio > 0.16 or numeric_ratio > 0.2):
        return True
    return False


def embed_text(text: str, dimensions: int = 8) -> list[float]:
    provider = model_provider.get_embedding_provider(dimensions=dimensions)
    return provider.embed(text)


def chunk_stored_markdown(
    stored_filename: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> ChunkingResult:
    warnings: list[str] = []
    error = _validate_chunk_params(chunk_size, overlap)
    if error:
        return ChunkingResult(
            status="error",
            message=error,
            chunks=[],
            warnings=warnings,
        )

    artifact_path = ingestion.ARTIFACTS_DIR / f"{stored_filename}.md"
    if not artifact_path.exists():
        return ChunkingResult(
            status="error",
            message="Markdown artifact not found for chunking.",
            chunks=[],
            warnings=warnings,
        )

    markdown = artifact_path.read_text(encoding="utf-8")
    if not markdown.strip():
        return ChunkingResult(
            status="error",
            message="Markdown artifact was empty.",
            chunks=[],
            warnings=warnings,
        )

    chunks = _chunk_text(markdown, chunk_size, overlap)
    _ensure_dirs()
    chunks_dir = _chunks_dir()
    chunks_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = _embeddings_dir()
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "stored_filename": stored_filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(artifact_path),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "warnings": warnings,
    }

    source_metadata_path = ingestion.METADATA_DIR / f"{stored_filename}.json"
    if source_metadata_path.exists():
        payload["source_metadata"] = json.loads(
            source_metadata_path.read_text(encoding="utf-8")
        )

    metadata_path = chunks_dir / f"{stored_filename}.chunks.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    embedding_payload = {
        "stored_filename": stored_filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(artifact_path),
        "chunk_metadata_path": str(metadata_path),
        "embedding_dimensions": 8,
        "chunk_count": len(chunks),
        "embeddings": [
            {"index": chunk["index"], "text": chunk["text"], "embedding": embed_text(chunk["text"])}
            for chunk in chunks
        ],
    }
    embeddings_path = embeddings_dir / f"{stored_filename}.embeddings.json"
    embeddings_path.write_text(
        json.dumps(embedding_payload, indent=2), encoding="utf-8"
    )

    embedding_payload["embeddings_path"] = str(embeddings_path)
    vector_result = vector_store.upsert_embeddings(embedding_payload)

    return ChunkingResult(
        status="success",
        message="Markdown chunked successfully.",
        chunks=chunks,
        warnings=warnings,
        metadata_path=str(metadata_path),
        embeddings_path=str(embeddings_path),
        vector_store_path=vector_result["path"],
    )
