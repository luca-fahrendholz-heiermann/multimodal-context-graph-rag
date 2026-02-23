from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, text: str) -> list[float]:
        ...


@dataclass
class SimpleEmbeddingProvider:
    dimensions: int = 8
    name: str = "simple"

    def embed(self, text: str) -> list[float]:
        if self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")

        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\wäöüÄÖÜß]+", text.lower())

        if not tokens:
            return vector

        for token in tokens:
            bucket = hash(token) % self.dimensions
            vector[bucket] += 1.0

        total = sum(vector)
        if total:
            vector = [value / total for value in vector]
        return vector


def list_embedding_providers() -> list[str]:
    return ["simple"]


def get_embedding_provider(
    provider_name: str | None = None,
    *,
    dimensions: int = 8,
) -> EmbeddingProvider:
    selected = provider_name or os.getenv("RAG_EMBEDDING_PROVIDER", "simple")
    if selected == "simple":
        return SimpleEmbeddingProvider(dimensions=dimensions)
    raise ValueError(
        f"Unknown embedding provider '{selected}'. "
        f"Available providers: {', '.join(list_embedding_providers())}."
    )
