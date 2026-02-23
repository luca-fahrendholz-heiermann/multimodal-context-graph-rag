from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChunkModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk_id: str
    text: str = ""
    modality: str = "text"


class EmbeddingInputModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk_id: str
    text: str


class NormalizedArtifactModel(BaseModel):
    """Canonical normalized document schema persisted for RAG indexing."""

    model_config = ConfigDict(extra="allow")

    canonical_text: str
    chunks: list[ChunkModel]
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    embeddings_inputs: list[EmbeddingInputModel]
    render_hints: dict[str, Any]
    provenance: dict[str, Any]
    warnings: list[str]


def validate_normalized_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a normalized-artifact payload before persisting it."""

    return NormalizedArtifactModel.model_validate(payload).model_dump()


def normalized_artifact_json_schema() -> dict[str, Any]:
    """Expose JSON schema for tests and tooling."""

    return NormalizedArtifactModel.model_json_schema()
