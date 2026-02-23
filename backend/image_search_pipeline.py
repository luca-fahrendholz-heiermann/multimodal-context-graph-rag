from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import os

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.llm_provider import describe_image_with_gemini, describe_image_with_openai

logger = logging.getLogger(__name__)

IMAGE_MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),
]


class VisionDescriptionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    ocr_text: str = ""
    metadata: dict = Field(default_factory=dict)


class VisionDocumentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    ocr_text: str = ""
    metadata: dict = Field(default_factory=dict)
    source_uri: str
    sha256: str


def detect_image_mime_type(*, filename: str, content_type: str | None, content_bytes: bytes) -> tuple[str, str]:
    for signature, mime in IMAGE_MAGIC_SIGNATURES:
        if content_bytes.startswith(signature):
            return mime, "magic_bytes"

    guessed_mime, _ = mimetypes.guess_type(filename)
    normalized_content_type = (content_type or "").strip().lower()

    if normalized_content_type.startswith("image/"):
        return normalized_content_type, "content_type"
    if guessed_mime and guessed_mime.startswith("image/"):
        return guessed_mime, "filename"
    return "application/octet-stream", "fallback"


def resize_image_if_needed(content_bytes: bytes, *, max_dimension: int = 2048) -> tuple[bytes, dict]:
    try:
        from PIL import Image
        import io
    except Exception:
        return content_bytes, {"resized": False, "reason": "pillow_unavailable"}

    try:
        image = Image.open(io.BytesIO(content_bytes))
        width, height = image.size
        if max(width, height) <= max_dimension:
            return content_bytes, {"resized": False, "width": width, "height": height}

        scale = max_dimension / float(max(width, height))
        new_size = (int(width * scale), int(height * scale))
        resized = image.resize(new_size)

        output = io.BytesIO()
        fmt = image.format or "PNG"
        resized.save(output, format=fmt)
        return output.getvalue(), {
            "resized": True,
            "original_width": width,
            "original_height": height,
            "new_width": new_size[0],
            "new_height": new_size[1],
        }
    except Exception as exc:
        logger.warning("Image resize failed: %s", exc)
        return content_bytes, {"resized": False, "reason": "resize_failed"}


def _parse_json(raw_response: str | None) -> dict | None:
    if not raw_response:
        return None
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return None


def _get_vision_callable() -> Callable[[bytes, str], tuple[str | None, list[str]]]:
    provider = (os.getenv("RAG_IMAGE_DESCRIPTION_PROVIDER") or "").strip().lower()

    if provider in {"openai", "chatgpt"}:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        model = (os.getenv("RAG_IMAGE_DESCRIPTION_OPENAI_MODEL") or "gpt-4.1-mini").strip()

        def _call(image_bytes: bytes, prompt: str) -> tuple[str | None, list[str]]:
            if not api_key:
                return None, ["OPENAI_API_KEY not configured."]
            result = describe_image_with_openai(api_key=api_key, model=model, image_bytes=image_bytes, prompt=prompt)
            return result.raw_response, list(result.warnings)

        return _call

    if provider == "gemini":
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        model = (os.getenv("RAG_IMAGE_DESCRIPTION_GEMINI_MODEL") or "gemini-1.5-flash").strip()

        def _call(image_bytes: bytes, prompt: str) -> tuple[str | None, list[str]]:
            if not api_key:
                return None, ["GEMINI_API_KEY not configured."]
            result = describe_image_with_gemini(api_key=api_key, model=model, image_bytes=image_bytes, prompt=prompt)
            return result.raw_response, list(result.warnings)

        return _call

    return lambda _image_bytes, _prompt: (None, ["Vision provider disabled."])


def _default_fallback(*, source_uri: str, sha256: str, reason: str) -> dict:
    payload = VisionDocumentModel(
        caption="Image upload with unavailable structured description.",
        tags=["image", "fallback"],
        objects=[],
        ocr_text="",
        metadata={"fallback_reason": reason},
        source_uri=source_uri,
        sha256=sha256,
    )
    return payload.model_dump()


def generate_structured_description(
    *,
    image_bytes: bytes,
    source_uri: str,
    sha256: str,
    vision_callable: Callable[[bytes, str], tuple[str | None, list[str]]] | None = None,
) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    prompt = (
        "Return STRICT JSON only with keys: caption (string), tags (string[]), objects (string[]), "
        "ocr_text (string), metadata (object). Do not wrap with markdown."
    )
    repair_prompt = (
        "Your previous response was invalid. Return ONLY valid JSON with exactly keys: caption, tags, objects, ocr_text, metadata."
    )

    caller = vision_callable or _get_vision_callable()
    raw, call_warnings = caller(image_bytes, prompt)
    warnings.extend(call_warnings)

    parsed = _parse_json(raw)
    if parsed is None:
        raw, retry_warnings = caller(image_bytes, repair_prompt)
        warnings.extend(retry_warnings)
        parsed = _parse_json(raw)

    if parsed is None:
        return _default_fallback(source_uri=source_uri, sha256=sha256, reason="invalid_json"), warnings

    try:
        description = VisionDescriptionModel.model_validate(parsed)
    except ValidationError:
        raw, retry_warnings = caller(image_bytes, repair_prompt)
        warnings.extend(retry_warnings)
        parsed = _parse_json(raw)
        if parsed is None:
            return _default_fallback(source_uri=source_uri, sha256=sha256, reason="schema_validation_failed"), warnings
        try:
            description = VisionDescriptionModel.model_validate(parsed)
        except ValidationError:
            return _default_fallback(source_uri=source_uri, sha256=sha256, reason="schema_validation_failed"), warnings

    if not description.caption.strip():
        warnings.append("Low-confidence caption generated; using minimal caption fallback.")
        description.caption = "Image content detected."

    payload = VisionDocumentModel(
        caption=description.caption,
        tags=description.tags,
        objects=description.objects,
        ocr_text=description.ocr_text,
        metadata=description.metadata,
        source_uri=source_uri,
        sha256=sha256,
    )
    return payload.model_dump(), warnings


def derive_text_for_embedding(vector_document: dict) -> str:
    parts = [
        vector_document.get("caption") or "",
        " ".join(vector_document.get("tags") or []),
        " ".join(vector_document.get("objects") or []),
        vector_document.get("ocr_text") or "",
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def store_blob(content_bytes: bytes, *, sha256: str, blob_dir: Path) -> str:
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"{sha256}.bin"
    if not blob_path.exists():
        blob_path.write_bytes(content_bytes)
    return f"blob://{blob_path}"


def process_image_for_search(
    *,
    filename: str,
    content_type: str | None,
    upload_bytes: bytes,
    blob_dir: Path,
    idempotency_index_path: Path,
    vision_callable: Callable[[bytes, str], tuple[str | None, list[str]]] | None = None,
) -> dict:
    sha256 = hashlib.sha256(upload_bytes).hexdigest()
    detected_mime_type, detection_source = detect_image_mime_type(
        filename=filename,
        content_type=content_type,
        content_bytes=upload_bytes,
    )
    if not detected_mime_type.startswith("image/"):
        raise ValueError("Uploaded file is not a supported image type.")

    resized_bytes, resize_meta = resize_image_if_needed(upload_bytes)

    idempotency_index_path.parent.mkdir(parents=True, exist_ok=True)
    if idempotency_index_path.exists():
        idempotency_index = json.loads(idempotency_index_path.read_text(encoding="utf-8"))
    else:
        idempotency_index = {"records": {}}

    should_refresh_from_vision = False
    existing = (idempotency_index.get("records") or {}).get(sha256)
    if existing:
        existing_vector_document = existing.get("vector_document") or {}
        existing_metadata = existing_vector_document.get("metadata") or {}
        fallback_reason = str(existing_metadata.get("fallback_reason") or "").strip()

        should_refresh_from_vision = bool(vision_callable) and bool(fallback_reason)
        if not should_refresh_from_vision:
            existing["idempotent_hit"] = True
            return existing

    blob_uri = store_blob(upload_bytes, sha256=sha256, blob_dir=blob_dir)

    vector_document, warnings = generate_structured_description(
        image_bytes=resized_bytes,
        source_uri=blob_uri,
        sha256=sha256,
        vision_callable=vision_callable,
    )
    vector_document["metadata"] = {
        **(vector_document.get("metadata") or {}),
        "detected_mime_type": detected_mime_type,
        "mime_detection_source": detection_source,
        "resize": resize_meta,
        "filename": filename,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    derived_text = derive_text_for_embedding(vector_document)
    if not derived_text.strip():
        derived_text = "Image content detected."
        warnings.append("Derived text for embedding was empty; used fallback text.")

    if existing and should_refresh_from_vision:
        warnings.append("Idempotency cache refreshed with vision description output.")

    result = {
        "blob_uri": blob_uri,
        "vector_document": vector_document,
        "derived_text": derived_text,
        "embeddings_text": derived_text,
        "warnings": warnings,
        "sha256": sha256,
        "detected_mime_type": detected_mime_type,
    }

    idempotency_index.setdefault("records", {})[sha256] = result
    idempotency_index_path.write_text(json.dumps(idempotency_index, indent=2), encoding="utf-8")
    return result
