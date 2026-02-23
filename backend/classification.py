from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable

from backend import ingestion
from backend.classification_config import LabelWhitelist
from backend.llm_provider import classify_with_gemini, classify_with_openai


@dataclass(frozen=True)
class LabelSelectionResult:
    status: str
    message: str
    warnings: list[str]
    labels: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationResult:
    status: str
    message: str
    warnings: list[str]
    label: str | None
    confidence: float | None
    labels_used: list[str]
    raw_response: str | None
    metadata_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_labels(values: Iterable[object]) -> list[str]:
    labels: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned and cleaned not in labels:
            labels.append(cleaned)
    return labels


def select_classification_labels(
    requested: list[str] | None,
    whitelist: LabelWhitelist,
) -> LabelSelectionResult:
    warnings: list[str] = []

    if requested is None:
        return LabelSelectionResult(
            status="success",
            message="Using default label whitelist.",
            warnings=warnings,
            labels=list(whitelist.labels),
        )

    normalized = _normalize_labels(requested)
    if not normalized:
        return LabelSelectionResult(
            status="error",
            message="Select at least one classification label.",
            warnings=warnings,
            labels=[],
        )

    allowed = set(whitelist.labels)
    invalid = [label for label in normalized if label not in allowed]
    if invalid:
        return LabelSelectionResult(
            status="error",
            message="Selected labels must come from the configured whitelist.",
            warnings=[f"Unsupported labels: {', '.join(invalid)}."],
            labels=[],
        )

    return LabelSelectionResult(
        status="success",
        message="Using requested labels.",
        warnings=warnings,
        labels=normalized,
    )


def _keyword_scores(text: str, labels: list[str]) -> dict[str, int]:
    keywords = {
        "finance": ["invoice", "payment", "budget", "revenue", "expense"],
        "legal": ["contract", "agreement", "policy", "compliance", "law"],
        "hr": ["employee", "benefits", "hiring", "onboarding", "payroll"],
    }
    scores: dict[str, int] = {label: 0 for label in labels}
    lowered = text.lower()
    for label in labels:
        for keyword in keywords.get(label, []):
            if keyword in lowered:
                scores[label] += 1
    return scores


def _simulate_llm_response(text: str, labels: list[str]) -> str:
    scores = _keyword_scores(text, labels)
    best_label = max(labels, key=lambda label: scores.get(label, 0))
    best_score = scores.get(best_label, 0)
    confidence = min(0.95, 0.45 + (0.1 * best_score))
    if best_score == 0:
        confidence = 0.4

    payload = {
        "label": best_label,
        "confidence": round(confidence, 3),
    }
    return json.dumps(payload, separators=(",", ":"))


def classify_text(
    text: str,
    labels: list[str],
    provider: str | None = None,
    api_key: str | None = None,
) -> ClassificationResult:
    warnings: list[str] = []

    if not text.strip():
        return ClassificationResult(
            status="error",
            message="Classification text must not be empty.",
            warnings=warnings,
            label=None,
            confidence=None,
            labels_used=labels,
            raw_response=None,
        )

    if not labels:
        return ClassificationResult(
            status="error",
            message="No classification labels provided.",
            warnings=warnings,
            label=None,
            confidence=None,
            labels_used=labels,
            raw_response=None,
        )

    selected_provider = (provider or "local").lower().strip()
    raw_response: str | None = None

    if selected_provider in {"openai", "chatgpt"}:
        if api_key and api_key.strip():
            remote = classify_with_openai(
                api_key=api_key.strip(),
                model="gpt-4.1-nano",
                text=text,
                labels=labels,
            )
            warnings.extend(remote.warnings)
            if remote.status == "success":
                raw_response = remote.raw_response
            else:
                warnings.append("Falling back to local demo classifier.")
        else:
            warnings.append("No OpenAI key supplied; using local demo classifier.")

    elif selected_provider == "gemini":
        if api_key and api_key.strip():
            remote = classify_with_gemini(
                api_key=api_key.strip(),
                model="gemini-1.5-flash",
                text=text,
                labels=labels,
            )
            warnings.extend(remote.warnings)
            if remote.status == "success":
                raw_response = remote.raw_response
            else:
                warnings.append("Falling back to local demo classifier.")
        else:
            warnings.append("No Gemini key supplied; using local demo classifier.")

    if raw_response is None:
        raw_response = _simulate_llm_response(text, labels)
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return ClassificationResult(
            status="error",
            message=f"LLM response was not valid JSON: {exc}",
            warnings=warnings,
            label=None,
            confidence=None,
            labels_used=labels,
            raw_response=raw_response,
        )

    label = parsed.get("label") if isinstance(parsed, dict) else None
    confidence = parsed.get("confidence") if isinstance(parsed, dict) else None

    if label not in labels:
        return ClassificationResult(
            status="error",
            message="LLM returned a label outside the requested set.",
            warnings=warnings,
            label=None,
            confidence=None,
            labels_used=labels,
            raw_response=raw_response,
        )

    if not isinstance(confidence, (int, float)):
        return ClassificationResult(
            status="error",
            message="LLM response missing confidence score.",
            warnings=warnings,
            label=label,
            confidence=None,
            labels_used=labels,
            raw_response=raw_response,
        )

    return ClassificationResult(
        status="success",
        message="Classification completed via LLM.",
        warnings=warnings,
        label=label,
        confidence=float(confidence),
        labels_used=labels,
        raw_response=raw_response,
    )


def store_classification_metadata(
    stored_filename: str,
    result: ClassificationResult,
) -> dict:
    warnings: list[str] = []

    if not stored_filename.strip():
        return {
            "status": "error",
            "message": "Stored filename is required to save classification metadata.",
            "warnings": warnings,
            "metadata_path": None,
        }

    if result.status != "success":
        return {
            "status": "error",
            "message": "Classification must succeed before storing metadata.",
            "warnings": warnings,
            "metadata_path": None,
        }

    ingestion.METADATA_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "stored_filename": stored_filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": result.status,
        "message": result.message,
        "warnings": result.warnings,
        "label": result.label,
        "confidence": result.confidence,
        "labels_used": result.labels_used,
        "raw_response": result.raw_response,
    }

    source_metadata_path = ingestion.METADATA_DIR / f"{stored_filename}.json"
    if source_metadata_path.exists():
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        source_metadata["classification"] = {
            "label": result.label,
            "confidence": result.confidence,
            "labels_used": result.labels_used,
            "timestamp": payload["timestamp"],
        }
        source_metadata_path.write_text(
            json.dumps(source_metadata, indent=2), encoding="utf-8"
        )
        payload["source_metadata"] = source_metadata

    metadata_path = ingestion.METADATA_DIR / f"{stored_filename}.classification.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "message": "Classification metadata stored.",
        "warnings": warnings,
        "metadata_path": str(metadata_path),
    }
