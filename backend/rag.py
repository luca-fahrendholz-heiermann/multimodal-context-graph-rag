from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from backend import chunking, mcp_tools, vector_store
from backend.llm_provider import generate_text_with_gemini, generate_text_with_openai


@dataclass(frozen=True)
class RagQueryResult:
    status: str
    message: str
    warnings: list[str]
    query_embedding: list[float]
    results: list[dict]
    answer: str | None = None
    tool_payload: dict | None = None
    used_tool: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def _load_chunk_metadata(path: str) -> dict | None:
    if not path:
        return None
    metadata_path = Path(path)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _confidence_hint(score: float, max_score: float) -> str:
    if max_score <= 0:
        return "low"
    ratio = score / max_score
    if ratio >= 0.85:
        return "high"
    if ratio >= 0.6:
        return "medium"
    return "low"


def _build_context(results: list[dict]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(
            f"[{index}] Datei={result.get('stored_filename')} Abschnitt={result.get('chunk_index')} "
            f"Score={result.get('score')}:\n{result.get('text') or ''}"
        )
    return "\n\n".join(lines)


def _build_rag_prompt(query_text: str, results: list[dict]) -> str:
    context = _build_context(results)
    return (
        "Du bist ein Assistent für Dokumentfragen. "
        "Nutze ausschließlich die bereitgestellten Belegstellen. "
        "Wenn etwas nicht in den Belegen steht, sage das klar und direkt. "
        "Antworte auf Deutsch in maximal 4 Sätzen. "
        "Jede inhaltliche Aussage muss mindestens eine Beleg-ID im Format [1], [2], ... enthalten. "
        "Wenn keine Aussage möglich ist, antworte: 'Nicht in den Belegstellen enthalten.'\n\n"
        f"Frage: {query_text}\n\n"
        f"Belegstellen:\n{context}\n\n"
        "Antwort mit Beleg-IDs:"
    )


def _local_answer(query_text: str, results: list[dict]) -> str:
    if not results:
        return "Keine Evidenz gefunden."

    synonym_groups = [
        {"technical", "technisch", "technik", "skills", "fähigkeiten", "faehigkeiten", "kompetenzen"},
        {"experience", "erfahrung", "work", "arbeit"},
        {"python", "pytorch", "tensorflow"},
    ]

    def tokenize(text: str) -> set[str]:
        tokens = {token for token in re.findall(r"[a-zA-ZäöüÄÖÜß0-9-]+", text.lower()) if token}
        expanded = set(tokens)
        for group in synonym_groups:
            if tokens & group:
                expanded |= group
        return expanded

    query_tokens = tokenize(query_text)

    def lexical_score(entry: dict) -> int:
        text_tokens = tokenize(str(entry.get("text") or ""))
        return len(query_tokens & text_tokens)

    best_index, best = max(
        enumerate(results, start=1),
        key=lambda pair: lexical_score(pair[1]),
    )
    return (
        f"Ich habe {len(results)} Evidenz-Stellen gefunden. "
        f"Relevanteste Referenz: [{best_index}]."
    )


def _response_claims_missing_information(answer: str) -> bool:
    lowered = (answer or "").lower()
    missing_patterns = (
        "nicht enthalten",
        "kein kurzprofil",
        "keine information",
        "nicht in den beleg",
        "nicht ersichtlich",
    )
    return any(pattern in lowered for pattern in missing_patterns)


def _query_terms_present_in_results(query_text: str, results: list[dict]) -> bool:
    query_tokens = {
        token
        for token in re.findall(r"[a-zA-ZäöüÄÖÜß0-9-]+", (query_text or "").lower())
        if len(token) >= 4
    }
    if not query_tokens:
        return False

    for result in results:
        chunk_text = str(result.get("text") or "").lower()
        if any(token in chunk_text for token in query_tokens):
            return True
    return False


def _response_contains_valid_citation(answer: str, max_citation: int) -> bool:
    if max_citation <= 0:
        return False
    citation_ids = {int(value) for value in re.findall(r"\[(\d+)\]", answer or "")}
    return any(1 <= value <= max_citation for value in citation_ids)


def _extract_format_filters(query_text: str) -> list[str]:
    normalized = (query_text or "").lower()
    known_formats = [
        "pdf", "doc", "docx", "txt", "md", "ppt", "pptx", "xls", "xlsx",
        "csv", "json", "xml", "png", "jpg", "jpeg", "gif", "svg", "html",
    ]
    selected: list[str] = []
    for extension in known_formats:
        if re.search(rf"\b{re.escape(extension)}(?:-datei(?:en)?)?\b", normalized):
            selected.append(extension)
    return selected


def _extract_document_category(query_text: str) -> str | None:
    normalized = (query_text or "").lower()
    presentation_terms = (
        "powerpoint",
        "präsentation",
        "praesentation",
        "präsentationen",
        "praesentationen",
        "folie",
        "folien",
        "slides",
        "slide deck",
        "slide-deck",
        "pitch deck",
        "pitch-deck",
    )
    if any(term in normalized for term in presentation_terms):
        return "presentation"
    return None


def _is_document_count_query(query_text: str) -> bool:
    normalized = (query_text or "").strip().lower()
    if not normalized:
        return False

    document_terms = (
        "dokument", "dokumente", "datei", "dateien", "documents", "docs",
        "präsentation", "präsentationen", "praesentation", "praesentationen",
        "powerpoint", "slides", "folien",
    )
    count_terms = ("wie viele", "anzahl", "gesamtzahl", "count", "how many")

    return any(term in normalized for term in document_terms) and any(
        term in normalized for term in count_terms
    )


def _is_format_histogram_query(query_text: str) -> bool:
    normalized = (query_text or "").lower()
    chart_terms = ("diagramm", "chart", "histogramm", "häufigkeit", "haeufigkeit")
    format_terms = ("dateiformat", "dateiformate", "formate", "file format")
    return any(term in normalized for term in chart_terms) and any(term in normalized for term in format_terms)


def _generate_answer(
    query_text: str,
    results: list[dict],
    provider: str | None,
    api_key: str | None,
) -> tuple[str, list[str]]:
    warnings: list[str] = []

    if not results:
        return "Keine Evidenz gefunden.", warnings

    selected_provider = (provider or "chatgpt").strip().lower()
    prompt = _build_rag_prompt(query_text, results)

    if api_key and api_key.strip():
        if selected_provider in {"chatgpt", "openai"}:
            remote = generate_text_with_openai(
                api_key=api_key.strip(),
                model="gpt-4.1-mini",
                prompt=prompt,
                max_output_tokens=400,
            )
            warnings.extend(remote.warnings)
            if remote.status == "success" and remote.raw_response:
                answer_text = remote.raw_response.strip()
                if _response_claims_missing_information(answer_text) and _query_terms_present_in_results(query_text, results):
                    warnings.append(
                        "Model answer contradicted retrieved evidence. Using local extractive answer."
                    )
                    return _local_answer(query_text, results), warnings
                if not _response_contains_valid_citation(answer_text, len(results)):
                    warnings.append(
                        "Model answer missed evidence citations. Using local extractive answer."
                    )
                    return _local_answer(query_text, results), warnings
                return answer_text, warnings
            warnings.append("Falling back to local extractive answer.")
        elif selected_provider == "gemini":
            remote = generate_text_with_gemini(
                api_key=api_key.strip(),
                model="gemini-1.5-flash",
                prompt=prompt,
                max_output_tokens=400,
            )
            warnings.extend(remote.warnings)
            if remote.status == "success" and remote.raw_response:
                answer_text = remote.raw_response.strip()
                if _response_claims_missing_information(answer_text) and _query_terms_present_in_results(query_text, results):
                    warnings.append(
                        "Model answer contradicted retrieved evidence. Using local extractive answer."
                    )
                    return _local_answer(query_text, results), warnings
                if not _response_contains_valid_citation(answer_text, len(results)):
                    warnings.append(
                        "Model answer missed evidence citations. Using local extractive answer."
                    )
                    return _local_answer(query_text, results), warnings
                return answer_text, warnings
            warnings.append("Falling back to local extractive answer.")
        else:
            warnings.append(f"Unknown provider '{selected_provider}'. Using local extractive answer.")
    else:
        warnings.append("No API key provided. Using local extractive answer.")

    return _local_answer(query_text, results), warnings


def query_rag(
    query_text: str,
    top_k: int = 5,
    stored_filename: str | None = None,
    stored_filenames: list[str] | None = None,
    provider: str | None = None,
    api_key: str | None = None,
) -> RagQueryResult:
    warnings: list[str] = []

    if not query_text.strip():
        return RagQueryResult(
            status="error",
            message="Query text must not be empty.",
            warnings=warnings,
            query_embedding=[],
            results=[],
        )

    tool_catalog = mcp_tools.tools_list()
    available_tools = {tool.get("name") for tool in tool_catalog.get("tools", [])}

    if _is_format_histogram_query(query_text):
        if "rag_format_histogram" not in available_tools:
            warnings.append("MCP tool rag_format_histogram is not available.")
        else:
            tool_result = mcp_tools.tools_call(
                "rag_format_histogram",
                {"stored_filename": stored_filename, "stored_filenames": stored_filenames},
            )
            if not tool_result.get("isError"):
                contents = tool_result.get("content") or []
                chart_payload = {}
                if contents and isinstance(contents[0], dict):
                    first = contents[0]
                    if first.get("type") == "json" and isinstance(first.get("json"), dict):
                        chart_payload = first["json"]
                return RagQueryResult(
                    status="success",
                    message="Returned file-format histogram via MCP tool call.",
                    warnings=warnings,
                    query_embedding=[],
                    results=[],
                    answer="Hier ist das Häufigkeitsdiagramm über Dateiformate im aktuellen RAG-Index.",
                    tool_payload=chart_payload,
                    used_tool="rag_format_histogram",
                )
            warnings.append("MCP format histogram tool call failed. Falling back to retrieval.")

    if _is_document_count_query(query_text):
        if "rag_document_stats" not in available_tools:
            warnings.append("MCP tool rag_document_stats is not available.")
        else:
            format_filters = _extract_format_filters(query_text)
            document_category = _extract_document_category(query_text)
            tool_result = mcp_tools.tools_call(
                "rag_document_stats",
                {
                    "stored_filename": stored_filename,
                    "stored_filenames": stored_filenames,
                    "file_extensions": format_filters,
                    "document_category": document_category,
                },
            )
            if not tool_result.get("isError"):
                contents = tool_result.get("content") or []
                stats_payload = {}
                if contents and isinstance(contents[0], dict):
                    first = contents[0]
                    if first.get("type") == "json" and isinstance(first.get("json"), dict):
                        stats_payload = first["json"]

                document_count = int(stats_payload.get("document_count") or 0)
                if document_category == "presentation":
                    answer_text = f"Im aktuellen Index sind {document_count} Präsentationsdokumente vorhanden (PowerPoint-Formate und erkannte Präsentations-PDFs)."
                elif format_filters:
                    ext_text = ", ".join(format_filters)
                    answer_text = f"Im aktuellen Index sind {document_count} Dokumente mit Format {ext_text} vorhanden."
                else:
                    answer_text = f"Im aktuellen Index sind {document_count} Dokumente vorhanden."

                return RagQueryResult(
                    status="success",
                    message="Returned document count via MCP tool call.",
                    warnings=warnings,
                    query_embedding=[],
                    results=[],
                    answer=answer_text,
                    tool_payload=stats_payload,
                    used_tool="rag_document_stats",
                )

            warnings.append("MCP stats tool call failed. Falling back to retrieval.")

    query_embedding = chunking.embed_text(query_text)
    normalized_embedding = _normalize_vector(query_embedding)
    matches = vector_store.search_embeddings(
        normalized_embedding,
        top_k=top_k,
        stored_filename=stored_filename,
        stored_filenames=stored_filenames,
        query_text=query_text,
    )

    if not matches:
        return RagQueryResult(
            status="warning",
            message="No indexed chunks found for retrieval.",
            warnings=warnings,
            query_embedding=normalized_embedding,
            results=[],
            answer="Keine Evidenz gefunden.",
            used_tool="rag_similarity_search",
        )

    max_score = max(match.get("score", 0) for match in matches)
    results: list[dict] = []
    metadata_cache: dict[str, dict] = {}
    chunk_cache: dict[str, dict[int, dict]] = {}

    for match in matches:
        metadata_path = match.get("chunk_metadata_path") or ""
        if metadata_path not in metadata_cache:
            metadata_cache[metadata_path] = _load_chunk_metadata(metadata_path) or {}
        metadata = metadata_cache.get(metadata_path, {})

        if metadata_path not in chunk_cache:
            chunks = metadata.get("chunks", [])
            chunk_cache[metadata_path] = {
                chunk.get("index"): chunk for chunk in chunks
            }

        chunk_map = chunk_cache.get(metadata_path, {})
        chunk = chunk_map.get(match.get("chunk_index"))
        if not chunk:
            warnings.append(
                f"Chunk metadata missing for index {match.get('chunk_index')}."
            )
            continue

        results.append(
            {
                "stored_filename": match.get("stored_filename"),
                "chunk_index": chunk.get("index"),
                "score": match.get("score"),
                "vector_score": match.get("vector_score"),
                "lexical_score": match.get("lexical_score"),
                "confidence_hint": _confidence_hint(match.get("score", 0), max_score),
                "text": chunk.get("text"),
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "chunk_metadata_path": metadata_path,
            }
        )

    answer, answer_warnings = _generate_answer(
        query_text=query_text,
        results=results,
        provider=provider,
        api_key=api_key,
    )
    warnings.extend(answer_warnings)

    return RagQueryResult(
        status="success",
        message="Retrieved top chunks for query.",
        warnings=warnings,
        query_embedding=normalized_embedding,
        results=results,
        answer=answer,
        used_tool="rag_similarity_search",
    )
