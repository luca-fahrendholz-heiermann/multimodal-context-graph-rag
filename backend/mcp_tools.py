from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json

from backend import ingestion, vector_store


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: dict


def _resolve_extension(document: dict) -> str:
    source_filename = str(document.get("source_filename") or "").strip()
    fallback_filename = str(document.get("stored_filename") or "").strip()
    candidate = source_filename or fallback_filename
    suffix = Path(candidate).suffix.lower().strip()
    return suffix or "unknown"


def _normalize_extensions(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    normalized: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        if not token:
            continue
        normalized.add(token if token.startswith(".") else f".{token}")
    return normalized


def _looks_like_presentation_document(document: dict) -> bool:
    extension = _resolve_extension(document)
    if extension in {".ppt", ".pptx", ".odp", ".key"}:
        return True

    source_filename = str(document.get("source_filename") or "").lower()
    stored_filename = str(document.get("stored_filename") or "").lower()
    combined = f"{source_filename} {stored_filename}"
    presentation_tokens = (
        "presentation",
        "praesentation",
        "präsentation",
        "powerpoint",
        "slides",
        "slide-deck",
        "folien",
        "pitchdeck",
        "pitch-deck",
    )

    return extension == ".pdf" and any(token in combined for token in presentation_tokens)


def _matches_document_category(document: dict, category: str) -> bool:
    normalized = category.strip().lower()
    if not normalized:
        return True
    if normalized == "presentation":
        return _looks_like_presentation_document(document)
    return False


def _scoped_documents(arguments: dict) -> list[dict]:
    overview = vector_store.get_store_overview(max_chunks_per_document=1)
    documents = list(overview.get("documents") or [])

    stored_filename = arguments.get("stored_filename")
    stored_filenames = arguments.get("stored_filenames")
    extensions = _normalize_extensions(arguments.get("file_extensions"))
    category = str(arguments.get("document_category") or "").strip().lower()

    if stored_filename:
        documents = [doc for doc in documents if doc.get("stored_filename") == stored_filename]
    elif isinstance(stored_filenames, list) and stored_filenames:
        allowed = {str(item) for item in stored_filenames}
        documents = [doc for doc in documents if doc.get("stored_filename") in allowed]

    if extensions:
        documents = [doc for doc in documents if _resolve_extension(doc) in extensions]

    if category:
        documents = [doc for doc in documents if _matches_document_category(doc, category)]

    return documents


def _generate_svg_bar_chart(title: str, labels: list[str], values: list[int]) -> str:
    width = 760
    height = 360
    margin_left = 80
    margin_bottom = 90
    margin_top = 50
    margin_right = 30
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1
    bar_count = max(1, len(labels))
    slot = plot_width / bar_count
    bar_width = max(22, slot * 0.58)

    bars: list[str] = []
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = (value / max_value) * plot_height if max_value else 0
        x = margin_left + (index * slot) + ((slot - bar_width) / 2)
        y = margin_top + (plot_height - bar_height)
        label_x = margin_left + (index * slot) + (slot / 2)
        escaped_label = label.replace("&", "&amp;").replace("<", "&lt;")
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="6" fill="#2563eb" />'
        )
        bars.append(
            f'<text x="{label_x:.2f}" y="{margin_top + plot_height + 22}" text-anchor="middle" font-size="12" fill="#cbd5e1">{escaped_label}</text>'
        )
        bars.append(
            f'<text x="{label_x:.2f}" y="{max(18, y - 6):.2f}" text-anchor="middle" font-size="12" fill="#e2e8f0">{value}</text>'
        )

    axis = (
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#64748b" stroke-width="1" />'
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#64748b" stroke-width="1" />'
    )

    grid = []
    for tick in range(0, 5):
        value = round((max_value / 4) * tick) if max_value else 0
        y = margin_top + plot_height - ((tick / 4) * plot_height)
        grid.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#1e293b" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#94a3b8">{value}</text>'
        )

    escaped_title = title.replace("&", "&amp;").replace("<", "&lt;")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#0f172a" rx="12" />'
        f'<text x="{margin_left}" y="30" font-size="18" fill="#f8fafc" font-weight="700">{escaped_title}</text>'
        + "".join(grid)
        + axis
        + "".join(bars)
        + "</svg>"
    )


def _document_stats(arguments: dict) -> dict:
    documents = _scoped_documents(arguments)
    counts = Counter(_resolve_extension(doc) for doc in documents)

    by_extension = {ext: int(counts.get(ext, 0)) for ext in sorted(ingestion.ALLOWED_EXTENSIONS)}
    if counts.get("unknown"):
        by_extension["unknown"] = int(counts["unknown"])

    return {
        "document_count": len(documents),
        "total_chunks": sum(int(doc.get("chunk_count") or 0) for doc in documents),
        "by_extension": by_extension,
    }


def _format_histogram(arguments: dict) -> dict:
    documents = _scoped_documents(arguments)
    counts = Counter(_resolve_extension(doc) for doc in documents)
    if not counts:
        return {
            "document_count": 0,
            "chart": {
                "type": "bar",
                "title": "Dateiformat-Häufigkeit",
                "labels": [],
                "values": [],
                "svg": _generate_svg_bar_chart("Dateiformat-Häufigkeit", [], []),
            },
        }

    items = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    labels = [ext for ext, _value in items]
    values = [value for _ext, value in items]

    return {
        "document_count": len(documents),
        "chart": {
            "type": "bar",
            "title": "Dateiformat-Häufigkeit",
            "labels": labels,
            "values": values,
            "svg": _generate_svg_bar_chart("Dateiformat-Häufigkeit", labels, values),
        },
    }


def _tool_manifest() -> dict[str, tuple[McpTool, callable]]:
    return {
        "rag_document_stats": (
            McpTool(
                name="rag_document_stats",
                description="Returns deterministic index statistics with optional file-format and scope filtering.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "stored_filename": {"type": "string", "description": "Optional single-document scope."},
                        "stored_filenames": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional graph/document subset scope.",
                        },
                        "file_extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional format filter, e.g. ['.pdf','.docx'] (without dot also accepted).",
                        },
                        "document_category": {
                            "type": "string",
                            "description": "Optional semantic category filter, currently supports 'presentation'.",
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            _document_stats,
        ),
        "rag_format_histogram": (
            McpTool(
                name="rag_format_histogram",
                description="Returns a frequency histogram for file formats (bar chart as SVG).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "stored_filename": {"type": "string"},
                        "stored_filenames": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": False,
                },
            ),
            _format_histogram,
        ),
    }


def tools_list() -> dict:
    manifests = _tool_manifest()
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool, _handler in manifests.values()
        ]
    }


def _validate_arguments(tool: McpTool, arguments: dict) -> tuple[bool, str | None]:
    if not isinstance(arguments, dict):
        return False, "Arguments must be an object."

    allowed = set(((tool.input_schema.get("properties") or {}).keys()))
    for key in arguments.keys():
        if key not in allowed:
            return False, f"Unexpected argument '{key}'."
    return True, None


def tools_call(name: str, arguments: dict | None = None) -> dict:
    manifests = _tool_manifest()
    tool_entry = manifests.get(name)
    if tool_entry is None:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Unknown tool '{name}'."}],
        }

    tool, handler = tool_entry
    payload_args = arguments or {}
    valid, error = _validate_arguments(tool, payload_args)
    if not valid:
        return {
            "isError": True,
            "content": [{"type": "text", "text": error or "Invalid arguments."}],
        }

    payload = handler(payload_args)
    return {
        "isError": False,
        "content": [{"type": "json", "json": payload}],
    }


def _mcp_tools_store_path() -> Path:
    return ingestion.UPLOAD_DIR / "mcp_tools.json"


def persist_tool_call(name: str, arguments: dict, result: dict) -> str:
    path = _mcp_tools_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "arguments": arguments,
        "result": result,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
