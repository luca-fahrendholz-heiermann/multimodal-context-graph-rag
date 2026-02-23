from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_LABELS = ["finance", "legal", "hr"]


@dataclass
class LabelWhitelist:
    labels: list[str]
    source: str

    def to_dict(self) -> dict:
        return {"labels": self.labels, "source": self.source}


def _normalize_labels(values: Iterable[object]) -> list[str]:
    labels: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned and cleaned not in labels:
            labels.append(cleaned)
    return labels


def load_label_whitelist(path: str | None = None) -> LabelWhitelist:
    configured_path = path or os.getenv("CLASSIFICATION_LABELS_PATH", "data/classification_labels.json")
    file_path = Path(configured_path)

    if file_path.exists():
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return LabelWhitelist(labels=DEFAULT_LABELS, source="default")

        if isinstance(raw, dict):
            raw = raw.get("labels", [])

        if isinstance(raw, list):
            labels = _normalize_labels(raw)
            if labels:
                return LabelWhitelist(labels=labels, source=str(file_path))

    return LabelWhitelist(labels=DEFAULT_LABELS, source="default")
