from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.schema_models import (
    normalized_artifact_json_schema,
    validate_normalized_artifact_payload,
)


def test_normalized_artifact_schema_exposes_required_top_level_fields():
    schema = normalized_artifact_json_schema()

    required = set(schema.get("required", []))
    assert {"canonical_text", "chunks", "embeddings_inputs"}.issubset(required)


def test_validate_normalized_artifact_payload_accepts_valid_payload():
    payload = {
        "canonical_text": "Invoice 2026-01",
        "chunks": [{"chunk_id": "c-1", "text": "Invoice", "modality": "text"}],
        "entities": [],
        "relations": [],
        "embeddings_inputs": [{"chunk_id": "c-1", "text": "Invoice"}],
        "render_hints": {},
        "provenance": {"parser": "test"},
        "warnings": [],
    }

    validated = validate_normalized_artifact_payload(payload)

    assert validated["chunks"][0]["chunk_id"] == "c-1"


def test_validate_normalized_artifact_payload_rejects_missing_required_fields():
    payload = {
        "chunks": [{"chunk_id": "c-1", "text": "Invoice", "modality": "text"}],
        "embeddings_inputs": [{"chunk_id": "c-1", "text": "Invoice"}],
    }

    with pytest.raises(ValidationError):
        validate_normalized_artifact_payload(payload)
