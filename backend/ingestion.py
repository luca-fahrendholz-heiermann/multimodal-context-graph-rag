from __future__ import annotations

import io
import importlib
import json
import os
import re
import csv
import string
import base64
import hashlib
import mimetypes
import shutil
import time
import zipfile
import zlib
import urllib.error
import urllib.request
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Callable, Iterable
from xml.etree import ElementTree as ET

from backend import vector_store
from backend.image_search_pipeline import process_image_for_search
from backend.vision_analyze import analyze_image_bytes_with_provider
from backend.schema_models import validate_normalized_artifact_payload

ALLOWED_EXTENSIONS = {
    ".zip",
    ".pdf",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".gcode",
    ".nc",
    ".cnc",
    ".tap",
    ".src",
    ".dat",
    ".sub",
    ".mod",
    ".sys",
    ".prg",
    ".tp",
    ".ls",
    ".script",
    ".urscript",
    ".st",
    ".il",
    ".ld",
    ".scl",
    ".awl",
    ".json",
    ".jsonl",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".stl",
    ".obj",
    ".ply",
    ".3mf",
    ".glb",
    ".gltf",
    ".step",
    ".iges",
    ".ifc",
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".srt",
    ".vtt",
    ".pptx",
    ".ppt",
}

UPLOAD_DIR = Path("data/uploads")
METADATA_DIR = UPLOAD_DIR / "metadata"
ARTIFACTS_DIR = UPLOAD_DIR / "artifacts"
PARSED_DIR = ARTIFACTS_DIR / "parsed"
NORMALIZED_DIR = ARTIFACTS_DIR / "normalized"
VIEWER_ARTIFACTS_DIR = ARTIFACTS_DIR / "viewer"
DEAD_LETTER_DIR = ARTIFACTS_DIR / "dead_letter"
IMAGE_BLOB_DIR = UPLOAD_DIR / "blobs"
IMAGE_DESCRIPTIONS_DIR = UPLOAD_DIR / "img_descriptions"
IMAGE_IDEMPOTENCY_INDEX_PATH = ARTIFACTS_DIR / "image_idempotency_index.json"
DEAD_LETTER_QUEUE_PATH = DEAD_LETTER_DIR / "queue.jsonl"
OBSERVABILITY_LOG_PATH = ARTIFACTS_DIR / "processing_logs.jsonl"
OBSERVABILITY_METRICS_PATH = ARTIFACTS_DIR / "processing_metrics.json"
PERFORMANCE_BUDGETS_MS = {
    "parse": 1200.0,
    "normalize": 600.0,
    "embed": 300.0,
    "index": 300.0,
    "viewer_artifacts": 400.0,
}
WATCH_DIR = Path("data/watch")
WATCH_PROCESSED_DIR = WATCH_DIR / "processed"
WATCH_REJECTED_DIR = WATCH_DIR / "rejected"
MAILPIT_API_URL = os.getenv("MAILPIT_API_URL", "http://localhost:8025")
SMTP_PROVIDER = os.getenv("SMTP_PROVIDER", "mailpit")
SMTP_MAX_MESSAGES = int(os.getenv("SMTP_MAX_MESSAGES", "50"))
CONTAINER_DEPTH_LIMIT = 3
CONTAINER_MAX_ENTRY_SIZE_BYTES = 50 * 1024 * 1024
CONTAINER_MAX_TOTAL_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
CONTAINER_MAX_COMPRESSION_RATIO = 200


@dataclass
class ValidationResult:
    status: str
    message: str
    warnings: list[str]


@dataclass(frozen=True)
class ParsedDoc:
    parser: str
    text: str
    layout: list[dict]
    tables: list[dict]
    media: list[dict]
    object_structure: dict
    warnings: list[str]


@dataclass(frozen=True)
class NormalizedDoc:
    canonical_text: str
    chunks: list[dict]
    entities: list[dict]
    relations: list[dict]
    embeddings_inputs: list[dict]
    render_hints: dict
    provenance: dict
    warnings: list[str]


def run_pipeline_qa(*, metadata: dict, normalized_doc: NormalizedDoc) -> dict:
    """Run format-aware QA checks and classify findings as recoverable/fatal."""

    checks: list[dict] = []
    errors: list[dict] = []

    def _record_check(name: str, passed: bool, details: str) -> None:
        checks.append({"name": name, "passed": passed, "details": details})

    def _record_error(error_class: str, reason: str) -> None:
        errors.append({"class": error_class, "reason": reason})

    stored_filename = metadata.get("stored_filename") or "unknown"
    extension = _normalize_extension(metadata.get("filename") or stored_filename)
    modality = _detect_modality(
        metadata.get("detected_mime_type"),
        extension,
    )

    has_chunks = bool(normalized_doc.chunks)
    _record_check("chunks_present", has_chunks, f"chunk_count={len(normalized_doc.chunks)}")
    if not has_chunks:
        _record_error("fatal", "Normalization produced no chunks.")

    chunk_ids = [chunk.get("chunk_id") for chunk in normalized_doc.chunks]
    unique_chunk_ids = len([value for value in chunk_ids if value]) == len(set(value for value in chunk_ids if value))
    _record_check("chunk_ids_unique", unique_chunk_ids, f"chunk_ids={len(chunk_ids)}")
    if not unique_chunk_ids:
        _record_error("fatal", "Duplicate chunk_id values detected.")

    non_empty_embeddings = [item for item in normalized_doc.embeddings_inputs if (item.get("text") or "").strip()]
    embeddings_match = len(non_empty_embeddings) == len(normalized_doc.embeddings_inputs)
    _record_check(
        "embedding_inputs_non_empty",
        embeddings_match,
        f"non_empty={len(non_empty_embeddings)}/{len(normalized_doc.embeddings_inputs)}",
    )
    if not embeddings_match:
        _record_error("fatal", "Embedding queue contains empty text payloads.")

    if modality == "text":
        has_text = bool(normalized_doc.canonical_text.strip())
        _record_check("text_canonical_content", has_text, f"canonical_chars={len(normalized_doc.canonical_text)}")
        if not has_text:
            _record_error("fatal", "Text format requires canonical_text but it was empty.")
    else:
        has_modality_chunk = any((chunk.get("modality") or "").lower() == modality for chunk in normalized_doc.chunks)
        _record_check("modality_chunk_present", has_modality_chunk, f"expected_modality={modality}")
        if not has_modality_chunk:
            _record_error("recoverable", f"No {modality} chunk found; text fallback was used.")

    status = "passed"
    if any(error["class"] == "fatal" for error in errors):
        status = "failed"
    elif errors:
        status = "warning"

    return {
        "status": status,
        "modality": modality,
        "checks": checks,
        "errors": errors,
    }


def build_viewer_artifacts(*, normalized_doc: NormalizedDoc, metadata: dict) -> dict:
    """Create viewer-ready artifacts per detected chunk modality."""
    _ensure_dirs([VIEWER_ARTIFACTS_DIR])

    stored_filename = metadata["stored_filename"]
    by_modality: dict[str, list[dict]] = {}
    for chunk in normalized_doc.chunks:
        modality = (chunk.get("modality") or "text").lower()
        by_modality.setdefault(modality, []).append(chunk)

    artifact_manifest = {
        "stored_filename": stored_filename,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [],
    }

    for modality, chunks in sorted(by_modality.items()):
        artifact_payload: dict
        if modality == "text":
            artifact_payload = {
                "type": "text_highlight_map",
                "entries": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "text": chunk.get("text") or "",
                        "highlight_spans": [
                            {"start": 0, "end": len((chunk.get("text") or ""))}
                        ],
                    }
                    for chunk in chunks
                ],
            }
        elif modality == "table":
            parsed_tables: list[dict] = []
            parsed_artifact_path = metadata.get("parsed_artifact_path")
            if parsed_artifact_path:
                parsed_path = Path(parsed_artifact_path)
                if parsed_path.exists():
                    try:
                        parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
                        parsed_tables = list(parsed_payload.get("tables") or [])
                    except (json.JSONDecodeError, OSError):
                        parsed_tables = []

            sheet_navigation: list[dict] = []
            row_deep_links: list[dict] = []
            filter_columns: list[str] = []
            for chunk in chunks:
                source = chunk.get("source") or {}
                table_index = source.get("table_index", 0)
                table = parsed_tables[table_index] if isinstance(table_index, int) and 0 <= table_index < len(parsed_tables) else {}
                sheet_name = table.get("sheet_name") or f"Sheet{table_index + 1}"
                nav_entry = next((entry for entry in sheet_navigation if entry["table_index"] == table_index), None)
                if nav_entry is None:
                    nav_entry = {
                        "table_index": table_index,
                        "sheet_name": sheet_name,
                        "row_count": table.get("row_count") if isinstance(table, dict) else None,
                        "column_count": len(table.get("header") or []),
                    }
                    sheet_navigation.append(nav_entry)

                column_name = source.get("column_name")
                if column_name and column_name not in filter_columns:
                    filter_columns.append(column_name)

                row_index = source.get("row_index")
                if isinstance(row_index, int):
                    row_deep_links.append(
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "table_index": table_index,
                            "sheet_name": sheet_name,
                            "row_index": row_index,
                            "viewer_link": f"table://{stored_filename}/{sheet_name}#row={row_index + 1}",
                        }
                    )

            artifact_payload = {
                "type": "table_view",
                "rows": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "preview": (chunk.get("text") or "")[:280],
                    }
                    for chunk in chunks
                ],
                "filter_metadata": {
                    "available_columns": sorted(filter_columns),
                    "chunk_strategies": sorted(
                        {
                            (chunk.get("chunk_strategy") or "unknown")
                            for chunk in chunks
                        }
                    ),
                },
                "sheet_navigation": sorted(sheet_navigation, key=lambda item: item["table_index"]),
                "row_deep_links": row_deep_links,
            }
        elif modality == "image":
            artifact_payload = {
                "type": "ocr_overlay",
                "boxes": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "text": chunk.get("text") or "",
                        "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
                    }
                    for chunk in chunks
                ],
            }
        elif modality == "json_xml":
            parsed_payload: dict = {}
            parsed_artifact_path = metadata.get("parsed_artifact_path")
            if parsed_artifact_path:
                parsed_path = Path(parsed_artifact_path)
                if parsed_path.exists():
                    try:
                        parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        parsed_payload = {}

            object_structure = parsed_payload.get("object_structure") or {}
            path_extraction = object_structure.get("path_extraction") or {}
            extracted_nodes = list(path_extraction.get("nodes") or [])
            extracted_relations = list(object_structure.get("relations") or [])

            node_lookup = {
                str(node.get("node_id") or ""): node
                for node in extracted_nodes
                if str(node.get("node_id") or "")
            }
            value_to_node_id: dict[str, str] = {}
            for node in extracted_nodes:
                value = node.get("value")
                node_id = str(node.get("node_id") or "")
                if isinstance(value, str) and value.strip() and node_id:
                    value_to_node_id.setdefault(value.strip(), node_id)

            resolved_relations: list[dict] = []
            for relation in extracted_relations:
                target_node_id = str(relation.get("target_node_id") or "")
                target_value = relation.get("target_value")
                fallback_node_id = (
                    value_to_node_id.get(target_value.strip())
                    if isinstance(target_value, str) and target_value.strip()
                    else None
                )
                resolved_target_node_id = target_node_id or fallback_node_id or None

                resolved_relations.append(
                    {
                        **relation,
                        "resolved_target_node_id": resolved_target_node_id,
                        "resolution_status": (
                            "resolved"
                            if resolved_target_node_id and resolved_target_node_id in node_lookup
                            else "unresolved"
                        ),
                    }
                )

            tree_nodes = []
            for node in extracted_nodes:
                path = str(node.get("path") or "")
                if path.startswith("$"):
                    parent_path = None
                    if "." in path:
                        parent_path = path.rsplit(".", 1)[0]
                    if "[" in path and path.endswith("]"):
                        parent_path = path.rsplit("[", 1)[0]
                    if path == "$":
                        parent_path = None
                else:
                    parent_path = path.rsplit("/", 1)[0] if "/" in path else None
                    if parent_path == "":
                        parent_path = None

                tree_nodes.append(
                    {
                        "node_id": node.get("node_id"),
                        "path": path,
                        "parent_path": parent_path,
                        "value_type": node.get("value_type"),
                        "depth": path.count(".") + path.count("/") + path.count("["),
                        "text": node.get("text") or "",
                    }
                )

            artifact_payload = {
                "type": "structured_tree_graph_view",
                "engine": path_extraction.get("engine"),
                "tree_view": {
                    "nodes": tree_nodes,
                    "path_search_index": [
                        {
                            "node_id": node.get("node_id"),
                            "path": node.get("path"),
                            "search_text": f"{node.get('path') or ''} {node.get('text') or ''}".strip(),
                            "search_terms": [
                                token
                                for token in re.split(r"[^a-zA-Z0-9_]+", f"{node.get('path') or ''} {node.get('text') or ''}".lower())
                                if token
                            ],
                        }
                        for node in extracted_nodes
                    ],
                },
                "graph_view": {
                    "nodes": [
                        {
                            "node_id": node.get("node_id"),
                            "path": node.get("path"),
                            "value_type": node.get("value_type"),
                        }
                        for node in extracted_nodes
                    ],
                    "edges": resolved_relations,
                    "reference_resolution": {
                        "resolved": sum(1 for edge in resolved_relations if edge.get("resolution_status") == "resolved"),
                        "unresolved": sum(1 for edge in resolved_relations if edge.get("resolution_status") == "unresolved"),
                        "unresolved_references": [
                            {
                                "type": edge.get("type"),
                                "source": edge.get("source"),
                                "target": edge.get("target"),
                            }
                            for edge in resolved_relations
                            if edge.get("resolution_status") == "unresolved"
                        ],
                    },
                },
            }
        elif modality in {"audio", "video"}:
            timeline_entries = []
            search_hit_jump_targets = []
            for chunk in chunks:
                source = chunk.get("source") or {}
                start_sec = source.get("start_sec")
                end_sec = source.get("end_sec")
                if start_sec is None and end_sec is None:
                    continue
                viewer_link = f"media://{stored_filename}#t={start_sec}"
                timeline_entries.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "text": chunk.get("text") or "",
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "viewer_link": viewer_link,
                    }
                )
                search_hit_jump_targets.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "viewer_link": viewer_link,
                        "seek_to_sec": start_sec,
                        "end_sec": end_sec,
                    }
                )

            alignment_hints = [
                hint
                for hint in (normalized_doc.render_hints.get("transcript_timeline_alignment") or [])
                if hint.get("chunk_id") in {chunk.get("chunk_id") for chunk in chunks}
            ]

            artifact_payload = {
                "type": "transcript_timeline_view",
                "timeline_entries": timeline_entries,
                "deep_link_support": {
                    "click_to_seek": bool(timeline_entries),
                    "scheme": f"media://{stored_filename}#t=<seconds>",
                },
                "transcript_alignment": alignment_hints,
                "search_hit_jump_targets": search_hit_jump_targets,
            }
        elif modality == "3d":
            source_path = UPLOAD_DIR / stored_filename
            canonical_path = VIEWER_ARTIFACTS_DIR / f"{stored_filename}.canonical.glb"
            canonical_meta_path = VIEWER_ARTIFACTS_DIR / f"{stored_filename}.canonical.meta.json"
            preview_path = VIEWER_ARTIFACTS_DIR / f"{stored_filename}.preview.png"
            features_path = VIEWER_ARTIFACTS_DIR / f"{stored_filename}.features.json"
            conversion_status = str(metadata.get("model_3d_conversion_status") or "missing_source")
            intermediate_artifact_path = metadata.get("model_3d_intermediate_artifact_path")
            conversion_warnings = list(metadata.get("model_3d_conversion_warnings") or [])
            preview_warnings = list(metadata.get("model_3d_preview_warnings") or [])

            if not canonical_path.exists() and source_path.exists():
                extension = _normalize_extension(stored_filename)
                conversion_status, conversion_warnings, intermediate_artifact_path = _convert_3d_to_canonical_glb(
                    source_path=source_path,
                    canonical_path=canonical_path,
                    extension=extension,
                )
                metadata["model_3d_conversion_status"] = conversion_status
                metadata["model_3d_conversion_warnings"] = conversion_warnings
                metadata["model_3d_intermediate_artifact_path"] = intermediate_artifact_path

            if not preview_path.exists():
                if canonical_path.exists() and conversion_status in {"passthrough_glb", "converted_to_glb"}:
                    preview_bytes, preview_warnings = _render_3d_preview_from_glb(canonical_glb_path=canonical_path)
                    preview_path.write_bytes(preview_bytes)
                else:
                    preview_path.write_bytes(_build_3d_preview_png())
                metadata["model_3d_preview_warnings"] = preview_warnings

            artifact_payload = {
                "type": "model_3d_view",
                "canonical_viewer_format": "glb",
                "canonical_glb_path": str(canonical_path),
                "canonical_meta_path": str(canonical_meta_path),
                "preview_path": str(preview_path),
                "features_path": str(features_path),
                "conversion_status": conversion_status,
                "conversion_warnings": conversion_warnings,
                "preview_warnings": preview_warnings,
                "intermediate_artifact_path": intermediate_artifact_path,
                "items": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "text": chunk.get("text") or "",
                    }
                    for chunk in chunks
                ],
            }
            canonical_meta_payload = _build_3d_canonical_meta(metadata=metadata, chunks=chunks)
            artifact_payload["meta_mapping"] = _build_3d_viewer_meta_mapping(canonical_meta_payload)
            canonical_meta_path.write_text(json.dumps(canonical_meta_payload, indent=2), encoding="utf-8")
            features_payload = _build_3d_features(metadata=metadata, chunks=chunks, canonical_meta=canonical_meta_payload)
            features_path.write_text(json.dumps(features_payload, indent=2), encoding="utf-8")
        elif modality == "container":
            parsed_payload: dict = {}
            parsed_artifact_path = metadata.get("parsed_artifact_path")
            if parsed_artifact_path:
                parsed_path = Path(parsed_artifact_path)
                if parsed_path.exists():
                    try:
                        parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        parsed_payload = {}

            container = (parsed_payload.get("object_structure") or {}).get("container") or {}
            entries = list(container.get("entries") or [])

            inherited_source_metadata = {
                "source_type": metadata.get("source_type"),
                "source_version": metadata.get("source_version"),
                "filename": metadata.get("filename"),
                "stored_filename": stored_filename,
                "sha256": metadata.get("sha256"),
            }

            tree_nodes = [
                {
                    "node_id": "root",
                    "parent_id": None,
                    "name": str(metadata.get("filename") or stored_filename),
                    "kind": "container",
                    "entry_path": "",
                    "entry_status": "root",
                    "inherited_source_metadata": inherited_source_metadata,
                }
            ]
            node_index = {"": "root"}

            for entry in entries:
                entry_name = str(entry.get("entry_name") or "")
                if not entry_name:
                    continue
                parts = [segment for segment in entry_name.split("/") if segment]
                parent_path = ""
                for part_index, part_name in enumerate(parts):
                    path = "/".join(parts[: part_index + 1])
                    if path in node_index:
                        parent_path = path
                        continue

                    is_leaf = part_index == len(parts) - 1
                    node_id = f"node:{path}"
                    node_index[path] = node_id
                    tree_nodes.append(
                        {
                            "node_id": node_id,
                            "parent_id": node_index[parent_path],
                            "name": part_name,
                            "kind": "entry" if is_leaf else "directory",
                            "entry_path": path,
                            "entry_status": entry.get("status") if is_leaf else "virtual",
                            "depth": path.count("/") + 1,
                            "inherited_source_metadata": inherited_source_metadata,
                            "entry_metadata": entry if is_leaf else None,
                        }
                    )
                    parent_path = path

            artifact_payload = {
                "type": "package_tree_view",
                "tree": {
                    "root_id": "root",
                    "nodes": tree_nodes,
                },
                "entries": entries,
                "source_metadata_inheritance": {
                    "inherited_fields": sorted(inherited_source_metadata.keys()),
                    "description": "Each package tree node includes inherited source metadata from the original container upload.",
                },
            }
        else:
            artifact_payload = {
                "type": "generic_view",
                "items": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "text": chunk.get("text") or "",
                    }
                    for chunk in chunks
                ],
            }

        artifact_path = VIEWER_ARTIFACTS_DIR / f"{stored_filename}.{modality}.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
        artifact_manifest["artifacts"].append(
            {
                "modality": modality,
                "type": artifact_payload["type"],
                "path": str(artifact_path),
            }
        )

    manifest_path = VIEWER_ARTIFACTS_DIR / f"{stored_filename}.viewer.json"
    manifest_path.write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
    return {
        "path": str(manifest_path),
        "count": len(artifact_manifest["artifacts"]),
        "modalities": [item["modality"] for item in artifact_manifest["artifacts"]],
    }


def _detect_modality(detected_mime_type: str | None, extension: str | None) -> str:
    mime = (detected_mime_type or "").lower()
    ext = (extension or "").lower()

    if mime.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".svg"}:
        return "image"
    if mime.startswith("audio/") or ext in {".wav", ".mp3", ".m4a", ".flac"}:
        return "audio"
    if mime.startswith("video/") or ext in {".mp4", ".mov", ".mkv", ".webm"}:
        return "video"
    if mime in {"application/zip", "application/x-tar", "application/gzip", "application/x-7z-compressed"} or ext in {
        ".zip",
        ".tar",
        ".gz",
        ".7z",
    }:
        return "container"
    if ext in {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".sql", ".gcode"}:
        return "code"
    if ext in {".stl", ".obj", ".ply", ".3mf", ".glb", ".gltf", ".step", ".iges", ".ifc"}:
        return "3d"
    if mime in {"application/json", "application/xml", "text/xml"} or ext in {
        ".json",
        ".jsonl",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
    }:
        return "json_xml"
    if mime in {"text/csv", "application/vnd.ms-excel"} or ext in {
        ".csv",
        ".tsv",
        ".xlsx",
        ".xls",
        ".ods",
    }:
        return "table"
    return "text"


def _is_cad_bim_format(extension: str | None) -> bool:
    return (extension or "").lower() in {".step", ".iges", ".ifc"}


def _build_3d_canonical_meta(*, metadata: dict, chunks: list[dict]) -> dict:
    source_id = str(metadata.get("sha256") or metadata.get("stored_filename") or "unknown-source")
    source_label = str(metadata.get("filename") or metadata.get("stored_filename") or "3d-object")

    nodes: list[dict] = []
    for index, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("chunk_id") or f"chunk:{index}")
        object_hash = hashlib.sha256(f"{source_id}:{chunk_id}".encode("utf-8")).hexdigest()[:16]
        nodes.append(
            {
                "object_id": f"obj_{object_hash}",
                "source_id": source_id,
                "labels": [
                    source_label,
                    str(chunk.get("modality") or "3d"),
                    str(chunk.get("chunk_strategy") or "3d_fallback"),
                ],
                "bbox": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "width": 1.0,
                    "height": 1.0,
                    "depth": 1.0,
                },
            }
        )

    return {
        "type": "canonical_3d_meta",
        "source_id": source_id,
        "node_count": len(nodes),
        "nodes": nodes,
    }


def _build_3d_preview_png() -> bytes:
    # 1x1 transparent PNG placeholder preview artifact.
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+lmFoAAAAASUVORK5CYII="
    )


def _convert_3d_to_canonical_glb(*, source_path: Path, canonical_path: Path, extension: str) -> tuple[str, list[str], str | None]:
    warnings: list[str] = []
    intermediate_artifact_path: str | None = None

    if extension == ".glb":
        shutil.copyfile(source_path, canonical_path)
        return "passthrough_glb", warnings, intermediate_artifact_path

    if _is_cad_bim_format(extension):
        intermediate_path = VIEWER_ARTIFACTS_DIR / f"{source_path.name}.tessellation.json"
        intermediate_payload = {
            "type": "cad_bim_tessellation_intermediate",
            "source_extension": extension,
            "source_path": str(source_path),
            "target_format": "glb",
            "status": "pending_glb_export",
        }
        intermediate_path.write_text(json.dumps(intermediate_payload, indent=2), encoding="utf-8")
        intermediate_artifact_path = str(intermediate_path)
        canonical_path.write_bytes(
            b"GLB export pending: CAD/BIM tessellation intermediate generated for this source format."
        )
        return "pending_tessellation_and_conversion", warnings, intermediate_artifact_path

    try:
        glb_payload = _convert_mesh_like_file_to_glb(source_path=source_path, extension=extension)
        canonical_path.write_bytes(glb_payload)
        return "converted_to_glb", warnings, intermediate_artifact_path
    except Exception as exc:
        warnings.append(f"3D to GLB conversion failed for {source_path.name}: {exc}")
        canonical_path.write_bytes(
            b"GLB conversion failed for this source format; see warnings in metadata."
        )
        return "conversion_failed", warnings, intermediate_artifact_path


def _render_3d_preview_from_glb(*, canonical_glb_path: Path) -> tuple[bytes, list[str]]:
    warnings: list[str] = []
    warnings.append("3D preview rendering backend not configured; using placeholder preview.")
    return _build_3d_preview_png(), warnings


def _convert_mesh_like_file_to_glb(*, source_path: Path, extension: str) -> bytes:
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    if extension == ".obj":
        vertices, triangles = _parse_obj_mesh(text)
    elif extension == ".stl":
        vertices, triangles = _parse_ascii_stl_mesh(text)
    elif extension == ".ply":
        vertices, triangles = _parse_ascii_ply_mesh(text)
    else:
        raise ValueError(f"Unsupported conversion source extension: {extension}")

    if not vertices or not triangles:
        raise ValueError("No mesh vertices/faces detected for GLB conversion")

    return _build_minimal_glb(vertices=vertices, triangles=triangles)


def _parse_obj_mesh(text: str) -> tuple[list[tuple[float, float, float]], list[int]]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[int] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            refs = [segment for segment in line.split()[1:] if segment]
            face_indices: list[int] = []
            for ref in refs:
                idx_token = ref.split("/")[0].strip()
                if not idx_token:
                    continue
                raw_idx = int(idx_token)
                idx = raw_idx - 1 if raw_idx > 0 else len(vertices) + raw_idx
                face_indices.append(idx)
            for i in range(1, len(face_indices) - 1):
                triangles.extend([face_indices[0], face_indices[i], face_indices[i + 1]])
    return vertices, triangles


def _parse_ascii_stl_mesh(text: str) -> tuple[list[tuple[float, float, float]], list[int]]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[int] = []
    vertex_map: dict[tuple[float, float, float], int] = {}
    current_face: list[int] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lower()
        if not line.startswith("vertex "):
            continue
        parts = raw_line.strip().split()
        if len(parts) < 4:
            continue
        vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
        idx = vertex_map.get(vertex)
        if idx is None:
            idx = len(vertices)
            vertex_map[vertex] = idx
            vertices.append(vertex)
        current_face.append(idx)
        if len(current_face) == 3:
            triangles.extend(current_face)
            current_face = []
    return vertices, triangles


def _parse_ascii_ply_mesh(text: str) -> tuple[list[tuple[float, float, float]], list[int]]:
    lines = text.splitlines()
    vertex_count = 0
    face_count = 0
    header_end = 0
    for idx, line in enumerate(lines):
        token = line.strip().lower()
        if token.startswith("element vertex"):
            vertex_count = int(token.split()[-1])
        elif token.startswith("element face"):
            face_count = int(token.split()[-1])
        elif token == "end_header":
            header_end = idx + 1
            break

    vertices: list[tuple[float, float, float]] = []
    for line in lines[header_end : header_end + vertex_count]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))

    triangles: list[int] = []
    face_start = header_end + vertex_count
    for line in lines[face_start : face_start + face_count]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        count = int(parts[0])
        indices = [int(item) for item in parts[1 : 1 + count]]
        for i in range(1, len(indices) - 1):
            triangles.extend([indices[0], indices[i], indices[i + 1]])
    return vertices, triangles


def _build_minimal_glb(*, vertices: list[tuple[float, float, float]], triangles: list[int]) -> bytes:
    import struct

    vertex_bytes = b"".join(struct.pack("<fff", *vertex) for vertex in vertices)
    index_bytes = b"".join(struct.pack("<I", index) for index in triangles)
    vertex_offset = 0
    index_offset = len(vertex_bytes)
    bin_chunk = vertex_bytes + index_bytes
    while len(bin_chunk) % 4 != 0:
        bin_chunk += b"\x00"

    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    zs = [vertex[2] for vertex in vertices]
    json_payload = {
        "asset": {"version": "2.0", "generator": "rag-ingestion-lab"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": vertex_offset, "byteLength": len(vertex_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5125,
                "count": len(triangles),
                "type": "SCALAR",
            },
        ],
    }

    json_bytes = json.dumps(json_payload, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    bin_chunk_wrapped = struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
    total_len = 12 + len(json_chunk) + len(bin_chunk_wrapped)
    header = struct.pack("<4sII", b"glTF", 2, total_len)
    return header + json_chunk + bin_chunk_wrapped


def _build_3d_features(*, metadata: dict, chunks: list[dict], canonical_meta: dict) -> dict:
    source_id = str(metadata.get("sha256") or metadata.get("stored_filename") or "unknown-source")
    return {
        "type": "model_3d_features",
        "source_id": source_id,
        "node_count": canonical_meta.get("node_count", 0),
        "vertex_count": None,
        "surface_area": None,
        "volume": None,
        "notes": [
            "Geometry-derived metrics are optional and remain unavailable until full mesh analysis is integrated.",
            f"Generated from {len(chunks)} normalized 3d chunk(s).",
        ],
    }


def _build_3d_analysis_text(metadata: dict) -> str:
    vector_document = metadata.get("model_3d_vector_document") or {}
    if not isinstance(vector_document, dict):
        return ""

    caption = str(vector_document.get("caption") or "").strip()
    tags = [str(item).strip() for item in (vector_document.get("tags") or []) if str(item).strip()]
    analysis = str((vector_document.get("metadata") or {}).get("analysis") or "").strip()
    open_questions = [
        str(item).strip()
        for item in ((vector_document.get("metadata") or {}).get("open_questions") or [])
        if str(item).strip()
    ]

    sections: list[str] = []
    if caption:
        sections.append(f"3D Vorschau-Beschreibung:\n{caption}")
    if analysis:
        sections.append(f"3D Vorschau-Analyse:\n{analysis}")
    if tags:
        sections.append("Tags: " + ", ".join(tags))
    if open_questions:
        sections.append("Offene Fragen:\n" + "\n".join(f"- {item}" for item in open_questions))
    return "\n\n".join(sections)


def _prepare_3d_pipeline_artifacts(*, metadata: dict, provider: str, api_key: str | None) -> None:
    stored_filename = str(metadata.get("stored_filename") or "")
    if not stored_filename:
        return

    source_path = UPLOAD_DIR / stored_filename
    if not source_path.exists():
        return

    extension = _normalize_extension(stored_filename)
    modality = _detect_modality(str(metadata.get("detected_mime_type") or ""), extension)
    if modality != "3d":
        return

    canonical_path = VIEWER_ARTIFACTS_DIR / f"{stored_filename}.canonical.glb"
    preview_path = VIEWER_ARTIFACTS_DIR / f"{stored_filename}.preview.png"

    conversion_status, conversion_warnings, intermediate_artifact_path = _convert_3d_to_canonical_glb(
        source_path=source_path,
        canonical_path=canonical_path,
        extension=extension,
    )
    metadata["model_3d_conversion_status"] = conversion_status
    metadata["model_3d_conversion_warnings"] = conversion_warnings
    metadata["model_3d_intermediate_artifact_path"] = intermediate_artifact_path

    preview_warnings: list[str] = []
    if canonical_path.exists() and conversion_status in {"passthrough_glb", "converted_to_glb"}:
        preview_bytes, preview_warnings = _render_3d_preview_from_glb(canonical_glb_path=canonical_path)
        preview_path.write_bytes(preview_bytes)
    else:
        preview_path.write_bytes(_build_3d_preview_png())
    metadata["model_3d_preview_warnings"] = preview_warnings
    metadata["model_3d_preview_path"] = str(preview_path)
    metadata["model_3d_canonical_glb_path"] = str(canonical_path)

    preview_bytes = preview_path.read_bytes()
    try:
        image_analysis_result = analyze_image_bytes_with_provider(
            filename=f"{stored_filename}.preview.png",
            content_bytes=preview_bytes,
            provider=provider,
            api_key=api_key,
        )
    except Exception as exc:
        metadata["model_3d_analysis_warnings"] = [f"3D preview analysis failed: {exc}"]
        return

    description_text = str(image_analysis_result.description_text or "").strip()
    analysis_text = str(image_analysis_result.analysis_text or "").strip()
    open_questions = [
        str(item).strip()
        for item in (image_analysis_result.open_questions or [])
        if str(item).strip()
    ]
    provider_name = str((image_analysis_result.meta or {}).get("provider") or "").strip()
    metadata["model_3d_vector_document"] = {
        "caption": description_text,
        "tags": ["3d", "preview", "llm-analysis", *([provider_name] if provider_name else [])],
        "metadata": {
            "analysis": analysis_text,
            "open_questions": open_questions,
            "provider": provider_name or None,
            "source": "3d_preview_pipeline",
        },
    }


def _build_3d_viewer_meta_mapping(canonical_meta: dict) -> dict:
    nodes = list(canonical_meta.get("nodes") or [])
    highlight_targets = []
    isolate_targets = []
    fit_targets = []

    for node in nodes:
        object_id = node.get("object_id")
        if not object_id:
            continue

        mapping_target = {
            "object_id": object_id,
            "bbox": node.get("bbox") or {},
            "labels": list(node.get("labels") or []),
            "source_id": node.get("source_id"),
        }
        highlight_targets.append(mapping_target)
        isolate_targets.append(mapping_target)
        fit_targets.append(mapping_target)

    return {
        "type": "object_meta_mapping",
        "source_id": canonical_meta.get("source_id"),
        "interaction_support": {
            "highlight": bool(highlight_targets),
            "isolate": bool(isolate_targets),
            "fit_to_object": bool(fit_targets),
        },
        "highlight_targets": highlight_targets,
        "isolate_targets": isolate_targets,
        "fit_to_object_targets": fit_targets,
    }



def _build_image_analysis_text(metadata: dict) -> str:
    image_vector_document = metadata.get("image_vector_document") or {}
    if not isinstance(image_vector_document, dict):
        image_vector_document = {}

    caption = str(image_vector_document.get("caption") or "").strip()
    objects = [str(item).strip() for item in (image_vector_document.get("objects") or []) if str(item).strip()]
    tags = [str(item).strip() for item in (image_vector_document.get("tags") or []) if str(item).strip()]
    ocr_text = str(image_vector_document.get("ocr_text") or "").strip()
    if ocr_text and _looks_like_binary_image_metadata_text(ocr_text):
        ocr_text = ""
    analysis = str((image_vector_document.get("metadata") or {}).get("analysis") or "").strip()

    sections: list[str] = []
    if caption:
        sections.append(f"Bildbeschreibung:\n{caption}")
    if analysis:
        sections.append(f"Bildanalyse:\n{analysis}")
    if objects:
        sections.append("Erkannte Objekte: " + ", ".join(objects))
    if tags:
        sections.append("Tags: " + ", ".join(tags))
    if ocr_text:
        sections.append(f"Extrahierter Text (OCR):\n{ocr_text}")

    assembled = "\n\n".join(section for section in sections if section.strip())
    if assembled:
        fallback_reason = str((image_vector_document.get("metadata") or {}).get("fallback_reason") or "").strip()
        if fallback_reason:
            return ""
        return assembled

    return ""

def _build_image_description_from_parsed_doc(parsed_doc: ParsedDoc) -> str:
    image_ocr = (parsed_doc.object_structure or {}).get("ocr") or {}
    description_meta = image_ocr.get("description") or {}
    if not isinstance(description_meta, dict):
        return ""

    description_text = str(description_meta.get("description_text") or "").strip()
    analysis_text = str(description_meta.get("analysis_text") or "").strip()
    open_questions = [
        str(item).strip()
        for item in (description_meta.get("open_questions") or [])
        if str(item).strip()
    ]

    sections: list[str] = []
    if description_text:
        sections.append(f"Bildbeschreibung:\n{description_text}")
    if analysis_text:
        sections.append(f"Bildanalyse:\n{analysis_text}")
    if open_questions:
        sections.append("Offene Fragen:\n" + "\n".join(f"- {item}" for item in open_questions))

    return "\n\n".join(sections)


def _build_modality_chunks(parsed_doc: ParsedDoc, metadata: dict) -> tuple[list[dict], list[str]]:
    chunks: list[dict] = []
    warnings: list[str] = []
    stored_filename = metadata["stored_filename"]
    extension = _normalize_extension(metadata.get("filename") or stored_filename)
    detected_mime_type = metadata.get("detected_mime_type")
    modality = _detect_modality(detected_mime_type, extension)

    canonical_text = parsed_doc.text.strip()
    if canonical_text and modality == "text":
        paragraphs = [segment.strip() for segment in canonical_text.split("\n\n") if segment.strip()]
        for index, paragraph in enumerate(paragraphs or [canonical_text]):
            chunks.append(
                {
                    "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                    "text": paragraph,
                    "modality": "text",
                    "chunk_strategy": "text_paragraph",
                    "source": {
                        "parser": parsed_doc.parser,
                        "layout_index": index,
                    },
                }
            )

    for index, table in enumerate(parsed_doc.tables):
        chunks.append(
            {
                "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                "text": table.get("text") or json.dumps(table, ensure_ascii=False),
                "modality": "table",
                "chunk_strategy": "table_object",
                "source": {"parser": parsed_doc.parser, "table_index": index},
            }
        )

        header = list(table.get("header") or table.get("headers") or [])
        rows = list(table.get("rows") or [])

        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            row_pairs = []
            for col_index, value in enumerate(row):
                column_name = header[col_index] if col_index < len(header) else f"column_{col_index + 1}"
                row_pairs.append(f"{column_name}={value}")

            row_text = " | ".join(item for item in row_pairs if item)
            if not row_text:
                continue

            chunks.append(
                {
                    "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                    "text": row_text,
                    "modality": "table",
                    "chunk_strategy": "table_row_as_doc",
                    "source": {
                        "parser": parsed_doc.parser,
                        "table_index": index,
                        "row_index": row_index,
                    },
                }
            )

        for col_index, column_name in enumerate(header):
            column_values = [str(row[col_index]).strip() for row in rows if isinstance(row, list) and col_index < len(row) and str(row[col_index]).strip()]
            if not column_values:
                continue

            chunks.append(
                {
                    "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                    "text": f"{column_name}: " + " | ".join(column_values),
                    "modality": "table",
                    "chunk_strategy": "table_column_block",
                    "source": {
                        "parser": parsed_doc.parser,
                        "table_index": index,
                        "column_index": col_index,
                        "column_name": column_name,
                    },
                }
            )

    if modality == "image":
        image_analysis_text = _build_image_analysis_text(metadata)
        if not image_analysis_text:
            image_analysis_text = _build_image_description_from_parsed_doc(parsed_doc)
        image_ref = metadata.get("image_blob_uri") or stored_filename
        image_source = {
            "image_ref": image_ref,
            "stored_filename": stored_filename,
            "sha256": metadata.get("sha256"),
        }
        if image_analysis_text:
            chunks.append(
                {
                    "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                    "text": image_analysis_text,
                    "modality": "image",
                    "chunk_strategy": "image_llm_analysis",
                    "source": {
                        "parser": parsed_doc.parser,
                        "provider": ((metadata.get("image_vector_document") or {}).get("metadata") or {}).get("provider"),
                        **image_source,
                    },
                }
            )

        ocr_text = str((((parsed_doc.object_structure or {}).get("ocr") or {}).get("text") or canonical_text)).strip()
        if ocr_text and _looks_like_binary_image_metadata_text(ocr_text):
            ocr_text = ""
            warnings.append("Skipped image OCR chunk because text resembled binary metadata.")
        if ocr_text:
            chunks.append(
                {
                    "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                    "text": ocr_text,
                    "modality": "image",
                    "chunk_strategy": "image_ocr_layout",
                    "source": {
                        "parser": parsed_doc.parser,
                        "ocr_blocks": len(parsed_doc.layout),
                        **image_source,
                    },
                }
            )

    if modality in {"audio", "video"} and canonical_text:
        asr_payload = parsed_doc.object_structure.get("asr", {})
        chapter_chunks_added = False
        for chapter in asr_payload.get("chapters") or []:
            chapter_text = str(chapter.get("text") or "").strip()
            if not chapter_text:
                continue
            chunks.append(
                {
                    "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                    "text": chapter_text,
                    "modality": modality,
                    "chunk_strategy": "asr_chapter_semantic_timeline",
                    "source": {
                        "parser": parsed_doc.parser,
                        "chapter_id": chapter.get("chapter_id"),
                        "start_sec": chapter.get("start_sec"),
                        "end_sec": chapter.get("end_sec"),
                        "segment_ids": chapter.get("segment_ids") or [],
                        "boundary": chapter.get("boundary"),
                    },
                }
            )
            chapter_chunks_added = True

        if not chapter_chunks_added:
            chunks.append(
                {
                    "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                    "text": canonical_text,
                    "modality": modality,
                    "chunk_strategy": "asr_transcript_segment",
                    "source": {
                        "parser": parsed_doc.parser,
                        "asr_segments": len(parsed_doc.object_structure.get("asr", {}).get("segments") or []),
                    },
                }
            )

    if modality == "container":
        container = parsed_doc.object_structure.get("container") or {}
        for entry in container.get("entries") or []:
            entry_name = str(entry.get("entry_name") or "")
            if not entry_name:
                continue
            status = str(entry.get("status") or "unknown")
            chunks.append(
                {
                    "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                    "text": f"{entry_name} ({status})",
                    "modality": "container",
                    "chunk_strategy": "container_entry",
                    "source": {
                        "parser": parsed_doc.parser,
                        "entry_name": entry_name,
                        "depth": entry.get("depth"),
                        "status": status,
                        "sha256": entry.get("sha256"),
                    },
                }
            )

    if modality == "3d":
        object_structure = parsed_doc.object_structure if isinstance(parsed_doc.object_structure, dict) else {}
        format_hint = str(object_structure.get("format") or extension.lstrip("."))
        source_filename = metadata.get("filename") or stored_filename
        analysis_text = _build_3d_analysis_text(metadata)

        if analysis_text:
            chunks.append(
                {
                    "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                    "text": analysis_text,
                    "modality": "3d",
                    "chunk_strategy": "model_3d_preview_analysis",
                    "source": {
                        "parser": parsed_doc.parser,
                        "mime_type": detected_mime_type,
                        "source_format": format_hint,
                    },
                }
            )

        chunks.append(
            {
                "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                "text": (
                    f"3D Modellanalyse für {source_filename}. "
                    f"Format: {format_hint}. "
                    "Viewer nutzt kanonisches GLB; Vorschaubild wird für Bildanalyse bereitgestellt."
                ),
                "modality": "3d",
                "chunk_strategy": "model_3d_summary",
                "source": {
                    "parser": parsed_doc.parser,
                    "mime_type": detected_mime_type,
                    "source_format": format_hint,
                },
            }
        )

    if not chunks and modality != "text":
        chunks.append(
            {
                "chunk_id": f"{stored_filename}:chunk:{len(chunks)}",
                "text": (
                    f"{modality.upper()} placeholder chunk for {metadata.get('filename') or stored_filename}. "
                    "No parser-specific extraction was available."
                ),
                "modality": modality,
                "chunk_strategy": f"{modality}_fallback",
                "source": {
                    "parser": parsed_doc.parser,
                    "mime_type": detected_mime_type,
                },
            }
        )
        warnings.append(
            f"Applied {modality} fallback chunk strategy because parsed text/tables were empty."
        )

    return chunks, warnings


@dataclass(frozen=True)
class ProcessorBase:
    name: str
    priority: int
    mime_types: set[str]
    extensions: set[str]
    magic_types: set[str]

    def capability_score(self, mime: str, ext: str, magic_bytes: str | None) -> int:
        normalized_mime = (mime or "").lower()
        normalized_ext = (ext or "").lower()
        normalized_magic = (magic_bytes or "").lower()

        score = 0
        if normalized_magic and normalized_magic in self.magic_types:
            score += 8
        if normalized_mime in self.mime_types:
            score += 4
        if normalized_ext in self.extensions:
            score += 2
        return score

    def can_handle(self, mime: str, ext: str, magic_bytes: str | None) -> bool:
        return self.capability_score(mime, ext, magic_bytes) > 0


@dataclass(frozen=True)
class ProcessorRoute(ProcessorBase):
    pass


@dataclass(frozen=True)
class ProcessorRegistry:
    routes: tuple[ProcessorRoute, ...]

    def select_route(self, mime: str, ext: str, magic_bytes: str | None) -> str:
        scored: list[tuple[int, int, str]] = []
        for route in self.routes:
            capability = route.capability_score(mime, ext, magic_bytes)
            if capability <= 0:
                continue
            scored.append((capability, route.priority, route.name))

        if not scored:
            return "generic_binary_processor"

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]


@dataclass(frozen=True)
class OptionalConverterPlugin(ProcessorBase):
    """Optional converter definition with isolated tooling and license metadata."""

    plugin_id: str
    route_name: str
    license_name: str
    tooling: str
    converter: Callable[..., ParsedDoc]


def _load_optional_converter_plugins() -> tuple[OptionalConverterPlugin, ...]:
    configured_plugins = [
        plugin_name.strip()
        for plugin_name in os.getenv("RAG_OPTIONAL_CONVERTERS", "").split(",")
        if plugin_name.strip()
    ]
    discovered_plugins: list[OptionalConverterPlugin] = []

    for plugin_name in configured_plugins:
        try:
            module = importlib.import_module(f"backend.optional_converters.{plugin_name}")
        except ModuleNotFoundError:
            continue

        build_plugin = getattr(module, "build_plugin", None)
        if not callable(build_plugin):
            continue

        plugin = build_plugin()
        if isinstance(plugin, OptionalConverterPlugin):
            discovered_plugins.append(plugin)

    return tuple(sorted(discovered_plugins, key=lambda plugin: plugin.priority, reverse=True))


OPTIONAL_CONVERTER_PLUGINS: tuple[OptionalConverterPlugin, ...] = _load_optional_converter_plugins()


PROCESSOR_REGISTRY = ProcessorRegistry(
    routes=(
    ProcessorRoute(
        name="container_processor",
        priority=95,
        mime_types={"application/zip"},
        extensions={".zip"},
        magic_types={"application/zip"},
    ),
    ProcessorRoute(
        name="pdf_processor",
        priority=100,
        mime_types={"application/pdf"},
        extensions={".pdf"},
        magic_types={"application/pdf"},
    ),
    ProcessorRoute(
        name="docx_processor",
        priority=90,
        mime_types={
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        },
        extensions={".docx"},
        magic_types=set(),
    ),
    ProcessorRoute(
        name="text_processor",
        priority=80,
        mime_types={"text/plain"},
        extensions={".txt"},
        magic_types=set(),
    ),
    ProcessorRoute(
        name="image_processor",
        priority=70,
        mime_types={"image/png", "image/jpeg", "image/svg+xml"},
        extensions={".png", ".jpg", ".jpeg", ".svg"},
        magic_types={"image/png", "image/jpeg"},
    ),
    )
)


def select_processor_route(mime: str, ext: str, magic_bytes: str | None) -> str:
    for plugin in sorted(OPTIONAL_CONVERTER_PLUGINS, key=lambda candidate: candidate.priority, reverse=True):
        if plugin.can_handle(mime, ext, magic_bytes):
            return plugin.route_name
    return PROCESSOR_REGISTRY.select_route(mime, ext, magic_bytes)


def _parse_with_optional_converter(
    *,
    filename: str,
    content_bytes: bytes,
    detected_mime_type: str,
    magic_bytes_type: str | None,
) -> ParsedDoc | None:
    extension = _normalize_extension(filename)
    for plugin in OPTIONAL_CONVERTER_PLUGINS:
        if not plugin.can_handle(detected_mime_type, extension, magic_bytes_type):
            continue

        return plugin.converter(
            filename=filename,
            content_bytes=content_bytes,
            detected_mime_type=detected_mime_type,
            magic_bytes_type=magic_bytes_type,
        )
    return None


def _detect_magic_type(content_bytes: bytes) -> str | None:
    signatures: list[tuple[bytes, str]] = [
        (b"%PDF", "application/pdf"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"PK\x03\x04", "application/zip"),
    ]
    for signature, mime in signatures:
        if content_bytes.startswith(signature):
            return mime
    return None


def _best_effort_text_extraction(content_bytes: bytes) -> tuple[str, int]:
    decoded = content_bytes.decode("utf-8", errors="replace")
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", decoded)
    compact = re.sub(r"\s+", " ", normalized).strip()
    printable_ratio = 0
    if content_bytes:
        printable_bytes = sum(1 for byte in content_bytes if 32 <= byte <= 126 or byte in {9, 10, 13})
        printable_ratio = int((printable_bytes / len(content_bytes)) * 100)
    return compact, printable_ratio


def _looks_like_unreadable_pdf_text(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return True

    lowered = normalized.lower()
    if "%pdf-" in lowered and "xref" in lowered and "/type /catalog" in lowered:
        return True

    printable = sum(1 for char in normalized if char in string.printable or char in "äöüÄÖÜß€")
    printable_ratio = printable / max(1, len(normalized))
    replacement_char_ratio = normalized.count("�") / max(1, len(normalized))

    return printable_ratio < 0.75 or replacement_char_ratio > 0.05


def _extract_pdf_text_from_bytes(content_bytes: bytes) -> tuple[str, list[str], dict]:
    warnings: list[str] = []
    details: dict[str, int | bool] = {
        "has_pypdf": False,
        "total_pages": 0,
        "pages_with_text": 0,
    }

    if importlib.util.find_spec("pypdf") is None:
        warnings.append("PDF parsing dependency 'pypdf' is not installed; extracted text is unavailable.")
        return "", warnings, details

    try:
        from pypdf import PdfReader
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to import pypdf reader: {exc}")
        return "", warnings, details

    details["has_pypdf"] = True

    try:
        reader = PdfReader(io.BytesIO(content_bytes))
        pages: list[str] = []
        details["total_pages"] = len(reader.pages)
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(page_text)

        details["pages_with_text"] = len(pages)
        extracted = "\n\n".join(pages).strip()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"PDF text extraction failed: {exc}")
        return "", warnings, details

    if _looks_like_unreadable_pdf_text(extracted):
        warnings.append(
            "PDF text extraction returned unreadable/empty content; the file may be scanned, encrypted, or malformed."
        )
        return "", warnings, details

    return extracted, warnings, details


def _detect_mime_type(filename: str, content_type: str | None, content_bytes: bytes) -> str:
    magic_mime = _detect_magic_type(content_bytes)
    guessed_mime, _ = mimetypes.guess_type(filename)
    return magic_mime or (content_type or "").strip() or guessed_mime or "application/octet-stream"


def _extract_text_structure_sections(text: str) -> list[dict]:
    sections: list[dict] = []
    lines = text.splitlines()
    pending_table: list[str] = []

    def _flush_table() -> None:
        nonlocal pending_table
        if pending_table:
            sections.append(
                {
                    "type": "table",
                    "rows": pending_table,
                }
            )
            pending_table = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            _flush_table()
            continue

        if re.match(r"^#{1,6}\s+", line):
            _flush_table()
            level = len(line.split(" ", 1)[0])
            sections.append(
                {
                    "type": "heading",
                    "level": level,
                    "text": line[level:].strip(),
                    "line": line_number,
                }
            )
            continue

        if re.match(r"^(?:-|\*|\+|\d+\.)\s+", line):
            _flush_table()
            sections.append(
                {
                    "type": "list_item",
                    "text": re.sub(r"^(?:-|\*|\+|\d+\.)\s+", "", line),
                    "line": line_number,
                }
            )
            continue

        if "|" in line and line.count("|") >= 2:
            pending_table.append(line)
            continue

        if re.match(r"^(\[\^?\d+\]|\[\^[^\]]+\]:)", line):
            _flush_table()
            sections.append(
                {
                    "type": "footnote",
                    "text": line,
                    "line": line_number,
                }
            )

    _flush_table()
    return sections


def _extract_html_structure_sections(html_text: str) -> list[dict]:
    sections: list[dict] = []
    heading_matches = re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>", html_text, flags=re.IGNORECASE | re.DOTALL)
    for match in heading_matches:
        sections.append(
            {
                "type": "heading",
                "level": int(match.group(1)),
                "text": re.sub(r"<[^>]+>", "", match.group(2)).strip(),
            }
        )

    list_item_matches = re.finditer(r"<li[^>]*>(.*?)</li>", html_text, flags=re.IGNORECASE | re.DOTALL)
    for match in list_item_matches:
        sections.append(
            {
                "type": "list_item",
                "text": re.sub(r"<[^>]+>", "", match.group(1)).strip(),
            }
        )

    table_matches = re.finditer(r"<table[^>]*>(.*?)</table>", html_text, flags=re.IGNORECASE | re.DOTALL)
    for match in table_matches:
        table_text = re.sub(r"<[^>]+>", " ", match.group(1))
        sections.append(
            {
                "type": "table",
                "text": re.sub(r"\s+", " ", table_text).strip(),
            }
        )

    footnote_matches = re.finditer(r"<footnote[^>]*>(.*?)</footnote>", html_text, flags=re.IGNORECASE | re.DOTALL)
    for match in footnote_matches:
        sections.append(
            {
                "type": "footnote",
                "text": re.sub(r"<[^>]+>", "", match.group(1)).strip(),
            }
        )

    return sections


def _extract_svg_semantic_regions(content_bytes: bytes) -> tuple[str, list[dict], list[str]]:
    warnings: list[str] = []
    try:
        root = ET.fromstring(content_bytes.decode("utf-8", errors="replace"))
    except ET.ParseError:
        return "", [], ["SVG parser failed to parse XML payload; semantic regions were skipped."]

    regions: list[dict] = []
    text_fragments: list[str] = []

    for node in root.iter():
        tag_name = node.tag.split("}")[-1].lower()
        attributes = dict(node.attrib)

        if tag_name == "text":
            text_value = " ".join(fragment.strip() for fragment in node.itertext() if fragment.strip())
            if not text_value:
                continue
            text_fragments.append(text_value)
            regions.append(
                {
                    "type": "text",
                    "text": text_value,
                    "attributes": attributes,
                }
            )
            continue

        if tag_name in {"path", "line", "polyline", "polygon", "rect", "circle", "ellipse"}:
            label = attributes.get("aria-label") or attributes.get("id") or attributes.get("class")
            regions.append(
                {
                    "type": "path",
                    "shape": tag_name,
                    "label": label,
                    "attributes": attributes,
                }
            )
            continue

        if tag_name in {"title", "desc"}:
            label_text = " ".join(fragment.strip() for fragment in node.itertext() if fragment.strip())
            if label_text:
                regions.append(
                    {
                        "type": "label",
                        "label": label_text,
                        "attributes": attributes,
                    }
                )

    if not regions:
        warnings.append("SVG parser found no semantic regions (text/paths/labels).")

    return "\n".join(text_fragments).strip(), regions, warnings


def _extract_email_quoted_blocks(body_text: str) -> list[dict]:
    blocks: list[dict] = []
    current: list[str] = []
    start_line = 0

    for line_index, raw_line in enumerate(body_text.splitlines()):
        stripped = raw_line.lstrip()
        if stripped.startswith(">"):
            if not current:
                start_line = line_index
            current.append(raw_line)
            continue

        if current:
            blocks.append(
                {
                    "line_start": start_line,
                    "line_end": line_index - 1,
                    "text": "\n".join(current),
                }
            )
            current = []

    if current:
        blocks.append(
            {
                "line_start": start_line,
                "line_end": len(body_text.splitlines()) - 1,
                "text": "\n".join(current),
            }
        )

    return blocks


def _infer_table_column_type(values: list[str]) -> str:
    non_null_values = [value for value in values if value.strip()]
    if not non_null_values:
        return "null"

    def _is_int(value: str) -> bool:
        return re.fullmatch(r"[+-]?\d+", value.strip()) is not None

    def _is_float(value: str) -> bool:
        return re.fullmatch(r"[+-]?(?:\d+\.\d+|\d+)", value.strip()) is not None

    if all(_is_int(value) for value in non_null_values):
        return "integer"
    if all(_is_float(value) for value in non_null_values):
        return "float"

    normalized_values = [value.strip().lower() for value in non_null_values]
    if all(value in {"true", "false", "yes", "no", "0", "1"} for value in normalized_values):
        return "boolean"
    return "string"


def _infer_table_schema(*, header: list[str], rows: list[list[str]], sheet_name: str) -> dict:
    row_count = len(rows)
    columns: list[dict] = []
    key_columns: list[str] = []

    for index, column_name in enumerate(header):
        column_values = [row[index] if index < len(row) else "" for row in rows]
        null_count = sum(1 for value in column_values if not value.strip())
        null_ratio = round((null_count / row_count), 4) if row_count else 0.0
        inferred_type = _infer_table_column_type(column_values)
        non_null_values = [value.strip() for value in column_values if value.strip()]
        is_unique_key = bool(non_null_values) and len(non_null_values) == row_count and len(set(non_null_values)) == row_count

        if is_unique_key:
            key_columns.append(column_name)

        columns.append(
            {
                "name": column_name,
                "inferred_type": inferred_type,
                "null_ratio": null_ratio,
                "sample_values": non_null_values[:3],
            }
        )

    return {
        "sheet_name": sheet_name,
        "row_count": row_count,
        "column_count": len(header),
        "columns": columns,
        "key_columns": key_columns,
    }


def _normalize_header_and_units(header: list[str]) -> tuple[list[str], list[dict]]:
    normalized_header: list[str] = []
    units: list[dict] = []

    for index, original_name in enumerate(header):
        column_name = original_name.strip()
        normalized_name = column_name
        detected_unit: str | None = None

        bracket_match = re.match(r"^(.*?)\s*[\(\[]\s*([A-Za-z%°/]+)\s*[\)\]]\s*$", column_name)
        if bracket_match:
            normalized_name = bracket_match.group(1).strip() or f"column_{index + 1}"
            detected_unit = bracket_match.group(2).strip().lower()
        else:
            suffix_match = re.match(r"^(.*?)_([a-z]{1,6})$", column_name)
            known_units = {"c", "f", "kg", "g", "mg", "lb", "m", "cm", "mm", "km", "s", "ms", "usd", "eur", "gb", "mb", "kb"}
            if suffix_match and len(suffix_match.group(1)) > 1:
                suffix = suffix_match.group(2).strip().lower()
                if suffix in known_units:
                    normalized_name = suffix_match.group(1).strip() or f"column_{index + 1}"
                    detected_unit = suffix

        normalized_name = normalized_name or f"column_{index + 1}"
        normalized_header.append(normalized_name)

        if detected_unit:
            units.append(
                {
                    "column": normalized_name,
                    "unit": detected_unit,
                    "source": "header",
                }
            )

    return normalized_header, units


def _collect_formula_metadata(*, rows: list[list[str]], header: list[str]) -> list[dict]:
    formulas: list[dict] = []
    for row_index, row in enumerate(rows, start=2):
        for col_index, value in enumerate(row):
            stripped = value.strip()
            if stripped.startswith("="):
                column_name = header[col_index] if col_index < len(header) else f"column_{col_index + 1}"
                formulas.append(
                    {
                        "row": row_index,
                        "column": column_name,
                        "expression": stripped,
                    }
                )
    return formulas


def _collect_pivot_ranges(*, rows: list[list[str]]) -> list[str]:
    ranges: set[str] = set()
    cell_range_pattern = re.compile(r"\b[A-Z]{1,3}\d+:[A-Z]{1,3}\d+\b")
    for row in rows:
        for value in row:
            stripped = value.strip()
            upper = stripped.upper()
            if "PIVOT" not in upper and "GETPIVOTDATA" not in upper:
                continue
            for match in cell_range_pattern.findall(upper):
                ranges.add(match)
    return sorted(ranges)


def _parse_delimited_table(*, filename: str, content_bytes: bytes, extension: str) -> tuple[str, list[dict], list[str], dict]:
    warnings: list[str] = []
    text = content_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    delimiter = "\t" if extension == ".tsv" else ","
    if extension == ".csv":
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            warnings.append("CSV delimiter detection fallback used ',' due to ambiguous source data.")

    reader = csv.reader(lines, delimiter=delimiter)
    records = list(reader)
    if not records:
        return text, [], warnings, {"format": "table", "sheets": []}

    raw_header = [column.strip() or f"column_{index + 1}" for index, column in enumerate(records[0])]
    header, units = _normalize_header_and_units(raw_header)
    rows = [[cell.strip() for cell in row] for row in records[1:]]
    schema = _infer_table_schema(header=header, rows=rows, sheet_name="Sheet1")
    formulas = _collect_formula_metadata(rows=rows, header=header)
    pivot_ranges = _collect_pivot_ranges(rows=rows)
    header_rows = [
        {
            "row": 1,
            "raw": raw_header,
            "normalized": header,
        }
    ]

    table_text_rows = [" | ".join(header)]
    table_text_rows.extend(" | ".join(row + [""] * (len(header) - len(row))) for row in rows)
    table_text = "\n".join(table_text_rows).strip()

    tables = [
        {
            "sheet_name": "Sheet1",
            "delimiter": delimiter,
            "header": header,
            "raw_header": raw_header,
            "rows": rows,
            "row_count": len(rows),
            "text": table_text,
            "schema_inference": schema,
            "formulas": formulas,
            "pivot_ranges": pivot_ranges,
            "header_rows": header_rows,
            "units": units,
        }
    ]
    structure = {
        "format": "table",
        "sheets": [schema],
        "header_rows": header_rows,
        "units": units,
        "formulas": formulas,
        "pivot_ranges": pivot_ranges,
    }
    return table_text, tables, warnings, structure


def _run_ocr_layout_analysis(content_bytes: bytes) -> tuple[str, list[dict], list[str]]:
    """Best-effort OCR-like extraction with block/line/word confidence output."""
    warnings: list[str] = []
    raw_text = content_bytes.decode("latin-1", errors="ignore")
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9_\-/:.,]{1,}", raw_text)
    filtered_tokens = [token for token in tokens if not token.startswith("\\x")][:128]

    if not filtered_tokens:
        warnings.append("Image OCR found no readable text tokens; layout/confidence output is empty.")
        return "", [], warnings

    max_tokens_per_line = 8
    lines = [filtered_tokens[index:index + max_tokens_per_line] for index in range(0, len(filtered_tokens), max_tokens_per_line)]

    layout_blocks: list[dict] = []
    line_entries: list[dict] = []
    global_word_index = 0
    for line_index, line_tokens in enumerate(lines):
        words: list[dict] = []
        for token in line_tokens:
            alpha_ratio = sum(char.isalpha() for char in token) / max(len(token), 1)
            confidence = round(min(0.98, 0.55 + alpha_ratio * 0.4), 3)
            words.append(
                {
                    "text": token,
                    "confidence": confidence,
                    "bbox": {
                        "x": round((global_word_index % max_tokens_per_line) * 0.11, 3),
                        "y": round(line_index * 0.08, 3),
                        "width": 0.1,
                        "height": 0.06,
                    },
                }
            )
            global_word_index += 1

        line_text = " ".join(item["text"] for item in words)
        line_entries.append(
            {
                "type": "line",
                "line_index": line_index,
                "text": line_text,
                "confidence": round(sum(word["confidence"] for word in words) / max(len(words), 1), 3),
                "words": words,
            }
        )

    layout_blocks.append(
        {
            "type": "block",
            "block_index": 0,
            "text": "\n".join(item["text"] for item in line_entries),
            "confidence": round(sum(item["confidence"] for item in line_entries) / max(len(line_entries), 1), 3),
            "lines": line_entries,
        }
    )

    canonical_text = "\n".join(item["text"] for item in line_entries)
    return canonical_text, layout_blocks, warnings


def _looks_like_binary_image_metadata_text(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True

    base64_like_segments = re.findall(r"[a-z0-9+/]{24,}={0,2}", normalized)
    if len(base64_like_segments) >= 1:
        return True

    tokens = re.findall(r"[a-z]{2,}", normalized)
    if not tokens:
        return True

    metadata_markers = {
        "png",
        "ihdr",
        "idat",
        "iend",
        "srgb",
        "gama",
        "phys",
        "jfif",
        "exif",
        "riff",
        "webp",
    }
    marker_hits = sum(1 for token in tokens if token in metadata_markers)
    if marker_hits >= 2:
        non_marker_tokens = [token for token in tokens if token not in metadata_markers]
        if len(non_marker_tokens) < 6:
            return True

    avg_token_length = sum(len(token) for token in tokens) / max(len(tokens), 1)
    if avg_token_length > 14:
        return True

    return False


def _has_meaningful_embedded_image_text(raw_text: str) -> bool:
    """Heuristic guard against treating binary image payload bytes as extracted text."""
    normalized = (raw_text or "").strip()
    if not normalized:
        return False

    non_whitespace_chars = [char for char in normalized if not char.isspace()]
    if non_whitespace_chars:
        control_chars = [char for char in non_whitespace_chars if ord(char) < 32]
        if len(control_chars) / len(non_whitespace_chars) > 0.01:
            return False

        punctuation_chars = [char for char in non_whitespace_chars if not char.isalnum()]
        if len(punctuation_chars) / len(non_whitespace_chars) > 0.45:
            return False

    if _looks_like_binary_image_metadata_text(normalized):
        return False

    lexical_tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]{2,}", normalized)
    filtered_tokens = [
        token
        for token in lexical_tokens
        if token.lower() not in {"png", "jfif", "exif", "webp", "riff", "ihdr", "idat", "iend"}
    ]
    if len(filtered_tokens) < 3:
        return False

    long_alpha_tokens = [token for token in filtered_tokens if len(token) >= 4 and any(ch.isalpha() for ch in token)]
    return bool(long_alpha_tokens)


def _extract_tables_from_image_text(raw_text: str) -> list[dict]:
    """Detect simple table-like line groups from OCR source text."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    candidate_rows: list[list[str]] = []
    for line in lines:
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        elif "\t" in line:
            cells = [cell.strip() for cell in line.split("\t") if cell.strip()]
        elif "," in line:
            cells = [cell.strip() for cell in line.split(",") if cell.strip()]
        else:
            cells = re.split(r"\s{2,}", line)
            cells = [cell.strip() for cell in cells if cell.strip()]

        if len(cells) >= 2:
            candidate_rows.append(cells)

    if len(candidate_rows) < 2:
        return []

    width_counts: dict[int, int] = {}
    for row in candidate_rows:
        width_counts[len(row)] = width_counts.get(len(row), 0) + 1

    column_count, row_count = max(width_counts.items(), key=lambda item: item[1])
    if column_count < 2 or row_count < 2:
        return []

    consistent_rows = [row for row in candidate_rows if len(row) == column_count]
    if len(consistent_rows) < 2:
        return []

    header = consistent_rows[0]
    body = consistent_rows[1:]
    table_text = "\n".join([" | ".join(header)] + [" | ".join(row) for row in body])

    return [
        {
            "table_index": 0,
            "headers": header,
            "rows": body,
            "text": table_text,
            "source": "image_ocr_table_detection",
        }
    ]


def _build_email_relations(*, headers: dict, attachments: list[dict], quoted_blocks: list[dict]) -> list[dict]:
    message_id = headers.get("message_id") or "email:message:unknown"
    relations: list[dict] = []

    def _parse_recipients(value: str | None) -> list[str]:
        if not value:
            return []
        return [entry.strip() for entry in value.split(",") if entry.strip()]

    sender = (headers.get("from") or "").strip()
    if sender:
        relations.append(
            {
                "type": "sent_by",
                "from": message_id,
                "to": f"email:address:{sender.lower()}",
                "metadata": {"header": "from", "value": sender},
            }
        )

    for header_name in ("to", "cc", "bcc"):
        for recipient in _parse_recipients(headers.get(header_name)):
            relations.append(
                {
                    "type": "sent_to",
                    "from": message_id,
                    "to": f"email:address:{recipient.lower()}",
                    "metadata": {"header": header_name, "value": recipient},
                }
            )

    subject = (headers.get("subject") or "").strip()
    if subject:
        relations.append(
            {
                "type": "has_subject",
                "from": message_id,
                "to": f"{message_id}:subject",
                "metadata": {"value": subject},
            }
        )

    in_reply_to = headers.get("in_reply_to")
    if in_reply_to:
        relations.append(
            {
                "type": "reply_to",
                "from": message_id,
                "to": in_reply_to,
            }
        )

    for reference in headers.get("references") or []:
        relations.append(
            {
                "type": "references",
                "from": message_id,
                "to": reference,
            }
        )

    for index, attachment in enumerate(attachments):
        attachment_id = f"{message_id}:attachment:{index}"
        relations.append(
            {
                "type": "has_attachment",
                "from": message_id,
                "to": attachment_id,
                "metadata": {
                    "filename": attachment.get("filename"),
                    "content_type": attachment.get("content_type"),
                },
            }
        )

    for index, block in enumerate(quoted_blocks):
        quote_node_id = f"{message_id}:quote:{index}"
        relations.append(
            {
                "type": "quoted_block",
                "from": message_id,
                "to": quote_node_id,
                "metadata": {
                    "line_start": block.get("line_start"),
                    "line_end": block.get("line_end"),
                },
            }
        )
        if in_reply_to:
            relations.append(
                {
                    "type": "quotes_message",
                    "from": quote_node_id,
                    "to": in_reply_to,
                }
            )

    return relations


def _detect_signature_compliance_markers(
    *,
    filename: str,
    detected_mime_type: str,
    content_bytes: bytes,
) -> dict:
    """Best-effort detection for signature/compliance markers (e.g. PAdES/XAdES)."""
    extension = _normalize_extension(filename)
    mime = (detected_mime_type or "").lower()
    if mime in {"image/svg+xml", "text/html", "application/xhtml+xml", "application/xml", "text/xml", "application/json"}:
        return None
    payload_lower = content_bytes.lower()

    markers: list[str] = []
    schemes: list[str] = []


    if detected_mime_type == "application/pdf" or extension == ".pdf":
        pdf_checks = {
            "pdf_sig_dictionary": b"/sig" in payload_lower,
            "pdf_byte_range": b"/byterange" in payload_lower,
            "pdf_subfilter": b"/subfilter" in payload_lower,
            "pdf_pades_token": b"etsi.cades.detached" in payload_lower,
        }
        markers.extend([name for name, matched in pdf_checks.items() if matched])
        if any(pdf_checks.values()):
            schemes.append("PAdES")

    if detected_mime_type in {"application/xml", "text/xml"} or extension == ".xml":
        xades_checks = {
            "xades_qualifying_properties": b"qualifyingproperties" in payload_lower,
            "xades_signed_properties": b"signedproperties" in payload_lower,
            "xades_signature_policy": b"signaturepolicyidentifier" in payload_lower,
        }
        markers.extend([name for name, matched in xades_checks.items() if matched])
        if any(xades_checks.values()):
            schemes.append("XAdES")

    return {
        "signature_present": bool(schemes),
        "signature_schemes": schemes,
        "compliance_markers": markers,
    }





def _extract_pptx_text(content_bytes: bytes) -> tuple[str, list[str], dict]:
    warnings: list[str] = []
    slides: list[dict] = []
    try:
        with zipfile.ZipFile(io.BytesIO(content_bytes)) as archive:
            slide_paths = sorted(
                [
                    name
                    for name in archive.namelist()
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml")
                ],
                key=lambda value: int(re.search(r"slide(\d+)\.xml$", value).group(1)) if re.search(r"slide(\d+)\.xml$", value) else 0,
            )
            for index, slide_path in enumerate(slide_paths, start=1):
                try:
                    payload = archive.read(slide_path)
                    root = ET.fromstring(payload)
                    texts = [node.text.strip() for node in root.iter() if node.tag.endswith("}t") and isinstance(node.text, str) and node.text.strip()]
                    notes = [node.text.strip() for node in root.iter() if node.tag.endswith("}txBody") and isinstance(node.text, str) and node.text.strip()]
                except (KeyError, ET.ParseError):
                    warnings.append(f"Failed to parse slide XML '{slide_path}'.")
                    continue

                slide_text = "\n".join(texts).strip()
                slides.append({
                    "slide_index": index,
                    "slide_path": slide_path,
                    "text": slide_text,
                    "notes": notes,
                    "text_count": len(texts),
                })
    except zipfile.BadZipFile:
        warnings.append("PPTX parser failed: invalid ZIP container.")
        return "", warnings, {"format": "pptx", "slides": []}

    joined = "\n\n".join(
        f"Slide {slide['slide_index']}:\n{slide['text']}" for slide in slides if slide.get("text")
    ).strip()
    if not joined:
        warnings.append("PPTX parser found no readable slide text.")

    return joined, warnings, {"format": "pptx", "slides": slides}


def _format_image_description_output(*, filename: str, analysis_result) -> str:
    sections = [f"Datei: {filename}"]

    if analysis_result.description_text:
        sections.append(f"Beschreibung:\n{analysis_result.description_text}")
    if analysis_result.analysis_text:
        sections.append(f"Analyse:\n{analysis_result.analysis_text}")
    if analysis_result.open_questions:
        questions = "\n".join(f"- {item}" for item in analysis_result.open_questions)
        sections.append(f"Offene Fragen:\n{questions}")

    if not any([analysis_result.description_text, analysis_result.analysis_text, analysis_result.open_questions]):
        sections.append("Keine LLM-Bildbeschreibung erzeugt.")

    warnings = [str(item).strip() for item in (analysis_result.warnings or []) if str(item).strip()]
    if warnings:
        sections.append("Warnungen:\n" + "\n".join(f"- {warning}" for warning in warnings))

    return "\n\n".join(section for section in sections if section.strip()).strip() + "\n"


def _store_image_description_text(*, stored_filename: str, text_payload: str) -> Path:
    IMAGE_DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = IMAGE_DESCRIPTIONS_DIR / f"{stored_filename}.txt"
    output_path.write_text(text_payload, encoding="utf-8")
    return output_path


def _generate_image_description_with_llm(*, filename: str, content_bytes: bytes, provider: str | None = None, api_key: str | None = None) -> tuple[str | None, list[str], dict]:
    result = analyze_image_bytes_with_provider(filename=filename, content_bytes=content_bytes, provider=provider, api_key=api_key)
    print(f"[image-vision] Datei: {filename}")
    if result.combined_text:
        if result.description_text:
            print("[image-vision] Beschreibung:")
            print(result.description_text)
        if result.analysis_text:
            print("[image-vision] Analyse:")
            print(result.analysis_text)
        if result.open_questions:
            print("[image-vision] Offene Fragen:")
            for question in result.open_questions:
                print(f"- {question}")
    else:
        print("[image-vision] Keine LLM-Bildbeschreibung erzeugt.")

    for warning in result.warnings:
        print(f"[image-vision][warning] {warning}")

    return result.combined_text, result.warnings, result.meta




def _run_image_ocr_pipeline(*, filename: str, content_bytes: bytes) -> dict:
    warnings: list[str] = []
    blocks: list[dict] = []
    lines: list[dict] = []
    words: list[dict] = []

    has_pil = importlib.util.find_spec("PIL") is not None
    has_tesseract = importlib.util.find_spec("pytesseract") is not None
    if not has_pil or not has_tesseract:
        warnings.append("OCR dependencies (Pillow + pytesseract) are not installed; image OCR returned empty output.")
        return {"text": "", "layout": [], "blocks": blocks, "lines": lines, "words": words, "warnings": warnings}

    PIL_Image = importlib.import_module("PIL.Image")
    pytesseract = importlib.import_module("pytesseract")

    image = PIL_Image.open(io.BytesIO(content_bytes))
    ocr_text = (pytesseract.image_to_string(image) or "").strip()
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    seen_blocks: set[tuple[int, int]] = set()
    seen_lines: set[tuple[int, int, int]] = set()

    for idx, token in enumerate(data.get("text", [])):
        text_value = (token or "").strip()
        if not text_value:
            continue

        raw_conf = data.get("conf", ["-1"])[idx]
        try:
            confidence = max(0.0, min(1.0, float(raw_conf) / 100.0))
        except (TypeError, ValueError):
            confidence = 0.0

        left = int(data.get("left", [0])[idx])
        top = int(data.get("top", [0])[idx])
        width = int(data.get("width", [0])[idx])
        height = int(data.get("height", [0])[idx])

        block_num = int(data.get("block_num", [0])[idx])
        line_num = int(data.get("line_num", [0])[idx])
        par_num = int(data.get("par_num", [0])[idx])

        block_key = (block_num, par_num)
        if block_key not in seen_blocks:
            seen_blocks.add(block_key)
            blocks.append({
                "block_id": f"block-{block_num}-{par_num}",
                "bbox": {"x": left, "y": top, "width": width, "height": height},
                "confidence": confidence,
            })

        line_key = (block_num, par_num, line_num)
        if line_key not in seen_lines:
            seen_lines.add(line_key)
            lines.append({
                "line_id": f"line-{block_num}-{par_num}-{line_num}",
                "block_id": f"block-{block_num}-{par_num}",
                "bbox": {"x": left, "y": top, "width": width, "height": height},
                "confidence": confidence,
            })

        words.append({
            "text": text_value,
            "line_id": f"line-{block_num}-{par_num}-{line_num}",
            "bbox": {"x": left, "y": top, "width": width, "height": height},
            "confidence": confidence,
        })

    layout = [
        {"type": "ocr_block", **block} for block in blocks
    ]

    return {
        "text": ocr_text,
        "layout": layout,
        "blocks": blocks,
        "lines": lines,
        "words": words,
        "warnings": warnings,
    }


def _normalize_dicom_metadata(*, content_bytes: bytes) -> tuple[dict, list[str]]:
    """Extract and privacy-normalize a minimal set of DICOM metadata."""

    warnings: list[str] = []
    has_pydicom = importlib.util.find_spec("pydicom") is not None
    if not has_pydicom:
        warnings.append("DICOM metadata normalization skipped: optional dependency 'pydicom' is not installed.")
        return {
            "format": "dicom",
            "schema": "dicom_minimal_v1",
            "deidentified": False,
            "available": False,
            "tags": {},
            "pseudonymized": {},
            "redacted_fields": [],
        }, warnings

    pydicom = importlib.import_module("pydicom")
    try:
        dataset = pydicom.dcmread(io.BytesIO(content_bytes), stop_before_pixels=True, force=True)
    except Exception:
        warnings.append("DICOM metadata normalization failed: file could not be parsed safely.")
        return {
            "format": "dicom",
            "schema": "dicom_minimal_v1",
            "deidentified": False,
            "available": False,
            "tags": {},
            "pseudonymized": {},
            "redacted_fields": [],
        }, warnings

    tag_map = {
        "modality": "Modality",
        "study_date": "StudyDate",
        "study_time": "StudyTime",
        "series_date": "SeriesDate",
        "series_time": "SeriesTime",
        "manufacturer": "Manufacturer",
        "body_part_examined": "BodyPartExamined",
    }
    tags: dict[str, str] = {}
    for output_key, attr_name in tag_map.items():
        value = getattr(dataset, attr_name, None)
        if value is not None and str(value).strip():
            tags[output_key] = str(value).strip()

    pseudonymized: dict[str, str] = {}
    for output_key, attr_name in {
        "study_instance_uid_sha256": "StudyInstanceUID",
        "series_instance_uid_sha256": "SeriesInstanceUID",
        "sop_instance_uid_sha256": "SOPInstanceUID",
        "patient_id_sha256": "PatientID",
    }.items():
        value = getattr(dataset, attr_name, None)
        if value is not None and str(value).strip():
            pseudonymized[output_key] = hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()

    redacted_fields = [
        field
        for field in ["PatientName", "PatientBirthDate", "PatientSex", "AccessionNumber"]
        if getattr(dataset, field, None)
    ]

    normalized = {
        "format": "dicom",
        "schema": "dicom_minimal_v1",
        "deidentified": True,
        "available": True,
        "tags": tags,
        "pseudonymized": pseudonymized,
        "redacted_fields": redacted_fields,
    }
    return normalized, warnings


_AUTOMATION_DIALECT_EXTENSIONS: dict[str, set[str]] = {
    "gcode": {".gcode", ".nc", ".cnc", ".tap"},
    "kuka_krl": {".src", ".dat", ".sub"},
    "abb_rapid": {".mod", ".sys", ".prg"},
    "fanuc_tp": {".tp", ".ls"},
    "urscript": {".script", ".urscript"},
    "iec_61131": {".st", ".il", ".ld", ".scl", ".awl"},
    "plcopen_xml": {".xml"},
}


def _detect_automation_dialect(*, filename: str, text: str, detected_mime_type: str | None = None) -> str | None:
    extension = _normalize_extension(filename)
    mime = (detected_mime_type or "").lower()
    if mime in {"image/svg+xml", "text/html", "application/xhtml+xml", "application/xml", "text/xml", "application/json"}:
        return None
    for dialect, extensions in _AUTOMATION_DIALECT_EXTENSIONS.items():
        if extension in extensions:
            return dialect

    sample = text[:6000].lower()
    if re.search(r"\b[gmt]\d+(?:\.\d+)?\b", sample):
        return "gcode"
    if "&access" in sample or re.search(r"\bdef\s+[a-z_][a-z0-9_]*\s*\(", sample):
        return "kuka_krl"
    if "endmodule" in sample or "endproc" in sample:
        return "abb_rapid"
    if re.search(r"\bmove[ljp]\b", sample) and "def " in sample:
        return "urscript"
    if "plcopen.org/xml/tc6" in sample or "<pou" in sample:
        return "plcopen_xml"
    return None


def _parse_automation_document(*, filename: str, content_bytes: bytes, warnings: list[str]) -> ParsedDoc:
    text = content_bytes.decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in text.splitlines()]
    dialect = _detect_automation_dialect(filename=filename, text=text, detected_mime_type=None)
    parser_mode = "dialect"
    if not dialect:
        dialect = "fallback"
        parser_mode = "fallback"
        warnings.append(
            "Automation parser fallback was used because no known dialect signature was detected."
        )

    parser_name = f"automation_{dialect}_parser"
    command_regex = {
        "gcode": re.compile(r"\b(?:[GMTFSXYZIJKRAN]|N\d+)\s*[-+]?\d*\.?\d+\b", re.IGNORECASE),
        "kuka_krl": re.compile(r"\b(?:DEF|END|PTP|LIN|CIRC|BAS)\b", re.IGNORECASE),
        "abb_rapid": re.compile(r"\b(?:MODULE|PROC|ENDPROC|ENDMODULE|MoveJ|MoveL)\b", re.IGNORECASE),
        "fanuc_tp": re.compile(r"\b(?:J\s*P\[\d+\]|L\s*P\[\d+\]|UFRAME_NUM|UTOOL_NUM)\b", re.IGNORECASE),
        "urscript": re.compile(r"\b(?:def|end|movej|movel|movep|set_digital_out)\b", re.IGNORECASE),
        "iec_61131": re.compile(r"\b(?:PROGRAM|FUNCTION_BLOCK|END_PROGRAM|VAR|END_VAR)\b", re.IGNORECASE),
        "plcopen_xml": re.compile(r"\b(?:pou|transition|action|step)\b", re.IGNORECASE),
        "fallback": re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b"),
    }.get(dialect, re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b"))

    program_blocks: list[dict] = []
    structural_nodes: list[dict] = []

    comment_prefixes = [";", "//", "#", "!"]
    subroutine_start_regex = re.compile(r"\b(?:DEF|PROC|PROGRAM|FUNCTION_BLOCK)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
    parameter_regex = re.compile(r"\b([A-Z_][A-Z0-9_]*)\s*=\s*([-+]?\d*\.?\d+|\"[^\"]*\"|'[^']*'|[A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

    def _extract_comment(raw_line: str) -> str | None:
        for marker in comment_prefixes:
            index = raw_line.find(marker)
            if index >= 0:
                comment = raw_line[index + len(marker):].strip()
                if comment:
                    return comment
        return None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        comment = _extract_comment(stripped)
        if comment:
            structural_nodes.append(
                {
                    "node_type": "comment",
                    "line": index + 1,
                    "text": comment,
                }
            )

        tokens = command_regex.findall(stripped)

        subroutine_match = subroutine_start_regex.search(stripped)
        if subroutine_match:
            structural_nodes.append(
                {
                    "node_type": "subroutine",
                    "line": index + 1,
                    "name": subroutine_match.group(1),
                    "raw": stripped,
                }
            )

        parameters = [
            {"name": match.group(1), "value": match.group(2)}
            for match in parameter_regex.finditer(stripped)
        ]
        for parameter in parameters:
            structural_nodes.append(
                {
                    "node_type": "parameter",
                    "line": index + 1,
                    **parameter,
                }
            )

        has_structural_signal = bool(tokens or parameters or subroutine_match or comment)
        if not has_structural_signal:
            continue

        program_blocks.append(
            {
                "line": index + 1,
                "raw": stripped,
                "tokens": tokens[:12],
                "parameters": parameters,
                "comment": comment,
            }
        )

        structural_nodes.append(
            {
                "node_type": "program_block",
                "line": index + 1,
                "raw": stripped,
                "token_count": len(tokens[:12]),
            }
        )

    if not program_blocks and dialect != "fallback":
        parser_mode = "fallback"
        dialect = "fallback"
        parser_name = "automation_fallback_parser"
        warnings.append(
            "Automation dialect parser yielded no blocks; fallback parser emitted warning-level baseline output."
        )
        fallback_regex = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            tokens = fallback_regex.findall(stripped)
            if not tokens:
                continue
            program_blocks.append(
                {
                    "line": index + 1,
                    "raw": stripped,
                    "tokens": tokens[:12],
                }
            )

    return ParsedDoc(
        parser=parser_name,
        text=text,
        layout=[{"type": "line", "index": item["line"] - 1, "text": item["raw"]} for item in program_blocks],
        tables=[],
        media=[],
        object_structure={
            "filename": filename,
            "format": "automation_code",
            "automation": {
                "dialect": dialect,
                "parser_mode": parser_mode,
                "program_blocks": program_blocks,
                "structural_nodes": structural_nodes,
            },
        },
        warnings=warnings,
    )


def _stable_path_node_id(*, engine: str, path: str) -> str:
    digest = hashlib.sha256(f"{engine}:{path}".encode("utf-8")).hexdigest()[:16]
    return f"{engine}_node_{digest}"


def _extract_jsonpath_nodes(payload: object) -> list[dict]:
    nodes: list[dict] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            nodes.append(
                {
                    "node_id": _stable_path_node_id(engine="jsonpath", path=path),
                    "path": path,
                    "value": None,
                    "value_type": "object",
                    "text": f"{path} = <object>",
                }
            )
            for key, child in value.items():
                walk(child, f"{path}.{key}")
            return
        if isinstance(value, list):
            nodes.append(
                {
                    "node_id": _stable_path_node_id(engine="jsonpath", path=path),
                    "path": path,
                    "value": None,
                    "value_type": "array",
                    "text": f"{path} = <array[{len(value)}]>",
                }
            )
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
            return

        value_repr = "" if value is None else str(value)
        nodes.append(
            {
                "node_id": _stable_path_node_id(engine="jsonpath", path=path),
                "path": path,
                "value": value,
                "value_type": type(value).__name__,
                "text": f"{path} = {value_repr}".strip(),
            }
        )

    walk(payload, "$")
    return nodes


def _is_unsafe_archive_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    parts = [segment for segment in normalized.split("/") if segment]
    return any(segment == ".." for segment in parts)


def _is_dangerous_archive_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    parts = [segment for segment in normalized.split("/") if segment]
    reserved_windows_names = {
        "con",
        "prn",
        "aux",
        "nul",
        "com1",
        "com2",
        "com3",
        "com4",
        "com5",
        "com6",
        "com7",
        "com8",
        "com9",
        "lpt1",
        "lpt2",
        "lpt3",
        "lpt4",
        "lpt5",
        "lpt6",
        "lpt7",
        "lpt8",
        "lpt9",
    }
    for segment in parts:
        if any(ord(char) < 32 for char in segment):
            return True
        if segment.strip() != segment or segment.endswith("."):
            return True
        if ":" in segment:
            return True
        base = segment.split(".", 1)[0].lower()
        if base in reserved_windows_names:
            return True
    return False


def _archive_entry_depth(name: str) -> int:
    normalized = name.replace("\\", "/")
    return len([segment for segment in normalized.split("/") if segment])


def _parse_container_document(
    *,
    filename: str,
    content_bytes: bytes,
    detected_mime_type: str,
    magic_bytes_type: str | None,
    signature_compliance: dict,
) -> ParsedDoc:
    warnings: list[str] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(content_bytes))
    except zipfile.BadZipFile:
        warnings.append("Container could not be read (corrupt ZIP archive).")
        return ParsedDoc(
            parser="container_parser",
            text="",
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "container",
                "container": {
                    "entries": [],
                    "depth_limit": CONTAINER_DEPTH_LIMIT,
                    "deduplicated_entries": 0,
                },
                "relations": [],
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    entries: list[dict] = []
    relations: list[dict] = []
    extracted_text_fragments: list[str] = []
    seen_hashes: set[str] = set()
    deduplicated_entries = 0
    total_uncompressed_bytes = 0

    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            safe_name = info.filename.replace("\\", "/")
            depth = _archive_entry_depth(info.filename)
            entry_record = {
                "entry_name": safe_name,
                "depth": depth,
                "size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size,
            }

            if _is_unsafe_archive_name(info.filename):
                entry_record["status"] = "blocked"
                entry_record["reason"] = "unsafe_path"
                warnings.append(f"Blocked unsafe container entry path: {safe_name}")
                entries.append(entry_record)
                continue

            if _is_dangerous_archive_name(info.filename):
                entry_record["status"] = "blocked"
                entry_record["reason"] = "dangerous_filename"
                warnings.append(f"Blocked dangerous container entry filename: {safe_name}")
                entries.append(entry_record)
                continue

            if depth > CONTAINER_DEPTH_LIMIT:
                entry_record["status"] = "skipped"
                entry_record["reason"] = "depth_limit"
                warnings.append(f"Skipped container entry '{safe_name}' due to depth limit ({CONTAINER_DEPTH_LIMIT}).")
                entries.append(entry_record)
                continue

            if info.flag_bits & 0x1:
                entry_record["status"] = "password_protected"
                warnings.append(f"Container entry '{safe_name}' is password protected and was skipped.")
                entries.append(entry_record)
                continue

            if info.file_size > CONTAINER_MAX_ENTRY_SIZE_BYTES:
                entry_record["status"] = "blocked"
                entry_record["reason"] = "zip_bomb_guard"
                warnings.append(f"Blocked oversized container entry '{safe_name}'.")
                entries.append(entry_record)
                continue

            if info.file_size > 0 and info.compress_size > 0 and (info.file_size / info.compress_size) > CONTAINER_MAX_COMPRESSION_RATIO:
                entry_record["status"] = "blocked"
                entry_record["reason"] = "zip_bomb_guard"
                warnings.append(f"Blocked suspicious compression ratio for '{safe_name}'.")
                entries.append(entry_record)
                continue

            if (total_uncompressed_bytes + info.file_size) > CONTAINER_MAX_TOTAL_UNCOMPRESSED_BYTES:
                entry_record["status"] = "blocked"
                entry_record["reason"] = "zip_bomb_guard"
                warnings.append(f"Blocked container entry '{safe_name}' due to total uncompressed-size guard.")
                entries.append(entry_record)
                continue

            try:
                member_bytes = archive.read(info)
            except RuntimeError:
                entry_record["status"] = "password_protected"
                warnings.append(f"Container entry '{safe_name}' requires a password and was skipped.")
                entries.append(entry_record)
                continue
            except (zipfile.BadZipFile, zlib.error, OSError):
                entry_record["status"] = "corrupt"
                warnings.append(f"Container entry '{safe_name}' appears corrupt and was skipped.")
                entries.append(entry_record)
                continue

            member_hash = hashlib.sha256(member_bytes).hexdigest()
            if member_hash in seen_hashes:
                deduplicated_entries += 1
                entry_record["status"] = "deduplicated"
                entry_record["sha256"] = member_hash
                entries.append(entry_record)
                continue

            seen_hashes.add(member_hash)
            entry_record["status"] = "parsed"
            entry_record["sha256"] = member_hash
            entries.append(entry_record)
            total_uncompressed_bytes += info.file_size

            ext = _normalize_extension(safe_name)
            if ext in {".txt", ".md", ".csv", ".json", ".jsonl", ".xml"}:
                extracted_text_fragments.append(f"{safe_name}\n{member_bytes.decode('utf-8', errors='replace').strip()}")

            relations.append(
                {
                    "type": "contains",
                    "source": filename,
                    "target": safe_name,
                    "target_sha256": member_hash,
                }
            )

    return ParsedDoc(
        parser="container_parser",
        text="\n\n".join(fragment for fragment in extracted_text_fragments if fragment),
        layout=[],
        tables=[],
        media=[],
        object_structure={
            "filename": filename,
            "mime_type": detected_mime_type,
            "magic_bytes_type": magic_bytes_type,
            "format": "container",
            "container": {
                "entries": entries,
                "depth_limit": CONTAINER_DEPTH_LIMIT,
                "deduplicated_entries": deduplicated_entries,
            },
            "relations": relations,
            "signature_compliance": signature_compliance,
        },
        warnings=warnings,
    )


def _extract_xpath_nodes(root: ET.Element) -> list[dict]:
    nodes: list[dict] = []

    def local_name(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def walk(element: ET.Element, path: str) -> None:
        nodes.append(
            {
                "node_id": _stable_path_node_id(engine="xpath", path=path),
                "path": path,
                "value": None,
                "value_type": "element",
                "text": f"{path} = <{local_name(element.tag)}>",
            }
        )

        for attr_key, attr_value in sorted(element.attrib.items()):
            attr_path = f"{path}/@{local_name(attr_key)}"
            nodes.append(
                {
                    "node_id": _stable_path_node_id(engine="xpath", path=attr_path),
                    "path": attr_path,
                    "value": attr_value,
                    "value_type": "attribute",
                    "text": f"{attr_path} = {attr_value}",
                }
            )

        text = (element.text or "").strip()
        if text:
            nodes.append(
                {
                    "node_id": _stable_path_node_id(engine="xpath", path=path),
                    "path": path,
                    "value": text,
                    "value_type": "str",
                    "text": f"{path} = {text}",
                }
            )

        children = [child for child in list(element) if isinstance(child.tag, str)]
        sibling_counts: dict[str, int] = {}
        for child in children:
            name = local_name(child.tag)
            sibling_counts[name] = sibling_counts.get(name, 0) + 1
            walk(child, f"{path}/{name}[{sibling_counts[name]}]")

    walk(root, f"/{local_name(root.tag)}[1]")
    return nodes


def _extract_structured_relations(*, engine: str, nodes: list[dict]) -> list[dict]:
    """Derive graph edges (`references`, `part_of`, `same_as`) from structured path nodes."""

    def _path_tail(path: str) -> str:
        if engine == "jsonpath":
            normalized = re.sub(r"\[\d+\]", "", path)
            return (normalized.rsplit(".", 1)[-1] if "." in normalized else normalized).lower()
        normalized = re.sub(r"\[\d+\]", "", path)
        normalized = normalized.split("/@")[-1] if "/@" in normalized else normalized.rsplit("/", 1)[-1]
        return normalized.lower()

    identifier_aliases = {"id", "identifier", "uuid", "uid"}
    relation_labels = {
        "references": {"reference", "references", "ref"},
        "part_of": {"partof", "part_of", "memberof", "member_of"},
        "same_as": {"sameas", "same_as", "equivalent", "equivalentto", "equivalent_to"},
    }

    id_value_to_node: dict[str, str] = {}
    for node in nodes:
        path = str(node.get("path") or "")
        value = node.get("value")
        if not isinstance(value, str):
            continue
        if _path_tail(path) in identifier_aliases and value.strip():
            id_value_to_node[value.strip()] = str(node.get("node_id") or "")

    relations: list[dict] = []
    for node in nodes:
        path = str(node.get("path") or "")
        source_node_id = str(node.get("node_id") or "")
        relation_value = node.get("value")
        if not source_node_id or not isinstance(relation_value, str) or not relation_value.strip():
            continue

        tail = _path_tail(path)
        relation_type = next(
            (
                label
                for label, aliases in relation_labels.items()
                if tail in aliases
            ),
            None,
        )
        if relation_type is None:
            continue

        target_value = relation_value.strip()
        target_node_id = id_value_to_node.get(target_value)
        relation = {
            "type": relation_type,
            "source": source_node_id,
            "target": target_node_id or target_value,
            "engine": engine,
            "path": path,
        }
        if target_node_id:
            relation["target_node_id"] = target_node_id
        else:
            relation["target_external"] = target_value
        relations.append(relation)

    return relations


def _detect_structured_schema(
    *,
    extension: str,
    detected_mime_type: str,
    json_payload: object | None = None,
    xml_root: ET.Element | None = None,
) -> dict:
    schema_name: str | None = None
    schema_version: str | None = None
    validation_errors: list[str] = []

    if isinstance(json_payload, dict) and isinstance(json_payload.get("resourceType"), str):
        schema_name = "fhir"
        candidate_version = json_payload.get("fhirVersion")
        if isinstance(candidate_version, str) and candidate_version.strip():
            schema_version = candidate_version.strip()

        if schema_version is None:
            meta = json_payload.get("meta")
            profiles = meta.get("profile") if isinstance(meta, dict) else None
            if isinstance(profiles, list):
                for profile in profiles:
                    if not isinstance(profile, str):
                        continue
                    if "/4.0/" in profile or profile.rstrip("/").endswith("4.0.1"):
                        schema_version = "R4"
                        break
                    if "/5.0/" in profile or profile.rstrip("/").endswith("5.0.0"):
                        schema_version = "R5"
                        break


    if xml_root is not None:
        namespace_uri = ""
        if isinstance(xml_root.tag, str) and xml_root.tag.startswith("{"):
            namespace_uri = xml_root.tag[1:].split("}", 1)[0]

        iso_match = re.search(
            r"urn:iso:std:iso:20022:tech:xsd:([A-Za-z0-9.]+)$",
            namespace_uri,
        )
        if iso_match:
            schema_name = "iso20022"
            schema_version = iso_match.group(1)
            if len(list(xml_root)) == 0:
                validation_errors.append("ISO20022 validation failed: message body is empty.")

    validation_status = "valid" if not validation_errors else "invalid"
    return {
        "schema": schema_name,
        "version": schema_version,
        "validation_status": validation_status,
        "errors": validation_errors,
        "detected_from": {
            "extension": extension,
            "mime_type": detected_mime_type,
        },
    }


def parse_structured_document(
    *,
    filename: str,
    content_bytes: bytes,
    detected_mime_type: str,
    magic_bytes_type: str | None,
    provider: str | None = None,
    api_key: str | None = None,
) -> ParsedDoc:
    extension = _normalize_extension(filename)
    warnings: list[str] = []
    signature_compliance = _detect_signature_compliance_markers(
        filename=filename,
        detected_mime_type=detected_mime_type,
        content_bytes=content_bytes,
    )

    if detected_mime_type == "application/zip" or extension == ".zip":
        return _parse_container_document(
            filename=filename,
            content_bytes=content_bytes,
            detected_mime_type=detected_mime_type,
            magic_bytes_type=magic_bytes_type,
            signature_compliance=signature_compliance,
        )

    plugin_parsed_doc = _parse_with_optional_converter(
        filename=filename,
        content_bytes=content_bytes,
        detected_mime_type=detected_mime_type,
        magic_bytes_type=magic_bytes_type,
    )
    if plugin_parsed_doc is not None:
        plugin_structure = dict(plugin_parsed_doc.object_structure)
        plugin_structure.setdefault("mime_type", detected_mime_type)
        plugin_structure.setdefault("magic_bytes_type", magic_bytes_type)
        plugin_structure.setdefault("signature_compliance", signature_compliance)

        return ParsedDoc(
            parser=plugin_parsed_doc.parser,
            text=plugin_parsed_doc.text,
            layout=plugin_parsed_doc.layout,
            tables=plugin_parsed_doc.tables,
            media=plugin_parsed_doc.media,
            object_structure=plugin_structure,
            warnings=plugin_parsed_doc.warnings,
        )

    detected_dialect = _detect_automation_dialect(
        filename=filename,
        text=content_bytes.decode("utf-8", errors="replace"),
        detected_mime_type=detected_mime_type,
    )
    if detected_dialect or extension in {
        ".gcode", ".nc", ".cnc", ".tap", ".src", ".dat", ".sub", ".mod", ".sys", ".prg",
        ".tp", ".ls", ".script", ".urscript", ".st", ".il", ".ld", ".scl", ".awl",
    }:
        parsed = _parse_automation_document(
            filename=filename,
            content_bytes=content_bytes,
            warnings=warnings,
        )
        parsed.object_structure.update(
            {
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "signature_compliance": signature_compliance,
            }
        )
        return parsed

    if detected_mime_type in {"application/dicom", "application/dicom+json"} or extension in {".dcm", ".dicom"}:
        medical_metadata, medical_warnings = _normalize_dicom_metadata(content_bytes=content_bytes)
        warnings.extend(medical_warnings)
        return ParsedDoc(
            parser="dicom_metadata_parser",
            text="",
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "dicom",
                "medical_metadata": medical_metadata,
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type == "application/json" or extension in {".json", ".jsonl"}:
        try:
            if extension == ".jsonl":
                rows = [json.loads(line) for line in content_bytes.decode("utf-8", errors="replace").splitlines() if line.strip()]
                json_payload: object = rows
            else:
                json_payload = json.loads(content_bytes.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            warnings.append("JSON parser failed to decode payload; returning empty structured fallback.")
            json_payload = {}

        path_nodes = _extract_jsonpath_nodes(json_payload)
        relations = _extract_structured_relations(engine="jsonpath", nodes=path_nodes)
        canonical_text = "\n".join(node["text"] for node in path_nodes)
        return ParsedDoc(
            parser="jsonpath_parser",
            text=canonical_text,
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "json",
                "path_extraction": {
                    "engine": "jsonpath",
                    "nodes": path_nodes,
                },
                "relations": relations,
                "schema_validation": _detect_structured_schema(
                    extension=extension,
                    detected_mime_type=detected_mime_type,
                    json_payload=json_payload,
                ),
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type in {"application/xml", "text/xml"} or extension == ".xml":
        root: ET.Element | None = None
        try:
            root = ET.fromstring(content_bytes)
            path_nodes = _extract_xpath_nodes(root)
            relations = _extract_structured_relations(engine="xpath", nodes=path_nodes)
            canonical_text = "\n".join(node["text"] for node in path_nodes)
        except ET.ParseError:
            warnings.append("XML parser failed to decode payload; returning empty structured fallback.")
            path_nodes = []
            relations = []
            canonical_text = ""

        return ParsedDoc(
            parser="xpath_parser",
            text=canonical_text,
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "xml",
                "path_extraction": {
                    "engine": "xpath",
                    "nodes": path_nodes,
                },
                "relations": relations,
                "schema_validation": _detect_structured_schema(
                    extension=extension,
                    detected_mime_type=detected_mime_type,
                    xml_root=root,
                ),
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type == "text/plain" or extension == ".txt":
        text = content_bytes.decode("utf-8", errors="replace")
        sections = _extract_text_structure_sections(text)
        paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
        layout = [
            {
                "type": "paragraph",
                "index": index,
                "text": paragraph,
            }
            for index, paragraph in enumerate(paragraphs)
        ]
        return ParsedDoc(
            parser="plain_text_parser",
            text=text,
            layout=layout,
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "paragraph_count": len(layout),
                "structure": {
                    "sections": sections,
                },
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type in {"text/html", "application/xhtml+xml"} or extension in {".html", ".htm", ".xhtml"}:
        html_text = content_bytes.decode("utf-8", errors="replace")
        extracted_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_text)).strip()
        sections = _extract_html_structure_sections(html_text)
        return ParsedDoc(
            parser="html_parser",
            text=extracted_text,
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "html",
                "structure": {
                    "sections": sections,
                },
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type in {"text/csv", "application/vnd.ms-excel"} or extension in {".csv", ".tsv"}:
        table_text, tables, table_warnings, structure = _parse_delimited_table(
            filename=filename,
            content_bytes=content_bytes,
            extension=extension,
        )
        warnings.extend(table_warnings)
        return ParsedDoc(
            parser="delimited_table_parser",
            text=table_text,
            layout=[],
            tables=tables,
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "table",
                "structure": structure,
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if (
        detected_mime_type.startswith("audio/")
        or detected_mime_type.startswith("video/")
        or extension in {".wav", ".mp3", ".m4a", ".flac", ".mp4", ".mov", ".mkv", ".webm", ".srt", ".vtt"}
    ):
        raw_text = content_bytes.decode("utf-8", errors="ignore").replace("\x00", " ")
        words = [token for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", raw_text) if token.strip()]
        has_caption_timestamps = bool(re.search(r"\d{2}:\d{2}:\d{2}[\.,]\d{3}", raw_text))

        if not words:
            words = ["Audio", "transcript", "unavailable"]
            warnings.append(
                "ASR baseline used fallback transcript because no readable tokens were found in media bytes."
            )

        token_duration = 0.45
        segment_size = 8
        segments: list[dict] = []
        word_timestamps: list[dict] = []

        for index, word in enumerate(words):
            start_sec = round(index * token_duration, 3)
            end_sec = round(start_sec + token_duration, 3)
            word_entry = {
                "word": word,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
            if index % 2 == 0:
                word_entry["speaker"] = "speaker_1"
            else:
                word_entry["speaker"] = "speaker_2"
            word_timestamps.append(word_entry)

        for segment_index in range(0, len(word_timestamps), segment_size):
            segment_words = word_timestamps[segment_index: segment_index + segment_size]
            if not segment_words:
                continue
            speakers = sorted({entry.get("speaker") for entry in segment_words if entry.get("speaker")})
            segments.append(
                {
                    "segment_id": f"seg_{segment_index // segment_size}",
                    "start_sec": segment_words[0]["start_sec"],
                    "end_sec": segment_words[-1]["end_sec"],
                    "text": " ".join(entry["word"] for entry in segment_words),
                    "speaker": speakers[0] if len(speakers) == 1 else "multi_speaker",
                }
            )

        transcript_text = "\n".join(
            f"[{segment['start_sec']:.2f}-{segment['end_sec']:.2f}] {segment['text']}"
            for segment in segments
        )
        chapters = _detect_asr_chapters(segments)

        if not has_caption_timestamps:
            warnings.append(
                "ASR timestamps are synthetic baseline timings; integrate a speech model for production-grade alignment."
            )

        return ParsedDoc(
            parser="asr_transcript_parser",
            text=transcript_text,
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "audio_video",
                "asr": {
                    "segments": segments,
                    "chapters": chapters,
                    "words": word_timestamps,
                    "diarization": {
                        "enabled": True,
                        "speaker_count": len({entry.get('speaker') for entry in word_timestamps if entry.get('speaker')}),
                        "method": "baseline_token_speaker_split",
                    },
                },
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type in {"message/rfc822", "application/eml"} or extension in {".eml", ".msg"}:
        message = BytesParser(policy=policy.default).parsebytes(content_bytes)
        body_parts: list[str] = []
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type().startswith("text/") and not part.get_filename():
                    body_parts.append(part.get_content())
        else:
            body_parts.append(message.get_content())
        body_text = "\n".join(str(part) for part in body_parts if str(part).strip()).strip()
        sections = _extract_text_structure_sections(body_text)
        references = [value.strip() for value in (message.get("References") or "").split() if value.strip()]
        headers = {
            "from": message.get("From"),
            "to": message.get("To"),
            "cc": message.get("Cc"),
            "bcc": message.get("Bcc"),
            "subject": message.get("Subject"),
            "date": message.get("Date"),
            "message_id": message.get("Message-ID"),
            "in_reply_to": message.get("In-Reply-To"),
            "references": references,
        }
        attachments = [
            {
                "filename": attachment.get_filename(),
                "content_type": attachment.get_content_type(),
            }
            for attachment in message.iter_attachments()
        ]
        quoted_blocks = _extract_email_quoted_blocks(body_text)
        relations = _build_email_relations(headers=headers, attachments=attachments, quoted_blocks=quoted_blocks)
        return ParsedDoc(
            parser="email_parser",
            text=body_text,
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "email",
                "headers": headers,
                "attachments": attachments,
                "threading": {
                    "message_id": headers.get("message_id"),
                    "in_reply_to": headers.get("in_reply_to"),
                    "references": references,
                },
                "quoted_blocks": quoted_blocks,
                "relations": relations,
                "structure": {
                    "sections": sections,
                },
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or extension == ".docx":
        document_text = ""
        sections: list[dict] = []
        try:
            with zipfile.ZipFile(io.BytesIO(content_bytes)) as archive:
                xml_payload = archive.read("word/document.xml")
            root = ET.fromstring(xml_payload)
            text_nodes = [node.text for node in root.iter() if node.tag.endswith("}t") and node.text]
            document_text = "\n".join(text_nodes)
            sections = _extract_text_structure_sections(document_text)
        except (zipfile.BadZipFile, KeyError, ET.ParseError):
            warnings.append("DOCX parser failed to read word/document.xml; returning empty text fallback.")

        return ParsedDoc(
            parser="docx_parser",
            text=document_text,
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "docx",
                "structure": {
                    "sections": sections,
                },
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation" or extension == ".pptx":
        pptx_text, pptx_warnings, pptx_structure = _extract_pptx_text(content_bytes)
        warnings.extend(pptx_warnings)
        sections = _extract_text_structure_sections(pptx_text)
        return ParsedDoc(
            parser="pptx_parser",
            text=pptx_text,
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "pptx",
                "presentation": {
                    "slide_count": len(pptx_structure.get("slides") or []),
                    "slides": pptx_structure.get("slides") or [],
                },
                "structure": {"sections": sections},
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if extension == ".ppt":
        warnings.append("Legacy .ppt files are not parsed natively. Please upload .pptx or enable Docling conversion.")
        return ParsedDoc(
            parser="ppt_legacy_parser",
            text="",
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "ppt",
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type.startswith("image/") or extension in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".tiff", ".tif", ".svg"}:
        if detected_mime_type == "image/svg+xml" or extension == ".svg":
            svg_text, svg_regions, svg_warnings = _extract_svg_semantic_regions(content_bytes)
            warnings.extend(svg_warnings)
            return ParsedDoc(
                parser="svg_vector_parser",
                text=svg_text,
                layout=svg_regions,
                tables=[],
                media=[],
                object_structure={
                    "filename": filename,
                    "mime_type": detected_mime_type,
                    "magic_bytes_type": magic_bytes_type,
                    "format": "svg",
                    "vector_regions": svg_regions,
                    "signature_compliance": signature_compliance,
                },
                warnings=warnings,
            )

        raw_text = content_bytes.decode("latin-1", errors="ignore")
        contains_embedded_text = _has_meaningful_embedded_image_text(raw_text)
        image_description, description_warnings, description_meta = _generate_image_description_with_llm(
            filename=filename,
            content_bytes=content_bytes,
            provider=provider,
            api_key=api_key,
        )
        warnings.extend(description_warnings)

        description_layout_entry = None
        if image_description:
            description_layout_entry = {
                "type": "image_description",
                "line_id": "llm-description",
                "text": image_description,
                "confidence": 0.5,
            }

        if not contains_embedded_text:
            ocr_result = _run_image_ocr_pipeline(filename=filename, content_bytes=content_bytes)
            warnings.extend(ocr_result["warnings"])

            if not (ocr_result.get("text") or "").strip():
                heuristic_text, heuristic_layout, heuristic_warnings = _run_ocr_layout_analysis(content_bytes)
                warnings.extend(heuristic_warnings)
                if heuristic_text:
                    if _looks_like_binary_image_metadata_text(heuristic_text):
                        warnings.append(
                            "Heuristic OCR output resembled binary image metadata; ignoring fallback text."
                        )
                    else:
                        ocr_result["text"] = heuristic_text
                        ocr_result["layout"] = heuristic_layout
                        heuristic_tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]{2,}", heuristic_text)
                        meaningful_tokens = [
                            token
                            for token in heuristic_tokens
                            if token.lower() not in {"png", "jfif", "exif", "webp", "riff"}
                        ]
                        if meaningful_tokens:
                            warnings = [
                                warning
                                for warning in warnings
                                if "OCR dependencies" not in warning
                            ]

            extracted_tables = _extract_tables_from_image_text(ocr_result.get("text", ""))
            if not extracted_tables:
                extracted_tables = _extract_tables_from_image_text(raw_text)

            ocr_text = (ocr_result.get("text") or "").strip()
            ocr_layout = list(ocr_result.get("layout") or [])
            should_use_llm_description = not ocr_text or _looks_like_binary_image_metadata_text(ocr_text)
            if should_use_llm_description and ocr_text:
                warnings.append(
                    "OCR output resembled binary image metadata; trying vision-language description fallback."
                )

            if should_use_llm_description:
                parsed_text = image_description or ""
                parsed_layout = [description_layout_entry] if description_layout_entry else []
            else:
                parsed_text_parts = [part for part in [ocr_text, image_description] if part]
                parsed_text = "\n\n".join(parsed_text_parts)
                parsed_layout = list(ocr_layout)
                if description_layout_entry:
                    parsed_layout.append(description_layout_entry)

            if should_use_llm_description and not image_description and ocr_text:
                ocr_text = ""
                ocr_layout = []

            return ParsedDoc(
                parser="image_ocr_parser",
                text=parsed_text,
                layout=parsed_layout,
                tables=extracted_tables,
                media=[],
                object_structure={
                    "filename": filename,
                    "mime_type": detected_mime_type,
                    "magic_bytes_type": magic_bytes_type,
                    "format": "image",
                    "ocr": {
                        "text": ocr_text,
                        "layout": ocr_layout,
                        "blocks": ocr_result["blocks"],
                        "lines": ocr_result["lines"],
                        "words": ocr_result["words"],
                        "table_count": len(extracted_tables),
                        "description": description_meta,
                    },
                    "signature_compliance": signature_compliance,
                },
                warnings=warnings,
            )

        ocr_text, ocr_layout_blocks, ocr_warnings = _run_ocr_layout_analysis(content_bytes)
        warnings.extend(ocr_warnings)
        extracted_tables = _extract_tables_from_image_text(ocr_text)
        if not extracted_tables and contains_embedded_text:
            extracted_tables = _extract_tables_from_image_text(raw_text)

        line_count = sum(len(block.get("lines") or []) for block in ocr_layout_blocks)
        word_count = sum(
            len(line.get("words") or [])
            for block in ocr_layout_blocks
            for line in block.get("lines") or []
        )
        return ParsedDoc(
            parser="ocr_image_parser",
            text=ocr_text,
            layout=ocr_layout_blocks,
            tables=extracted_tables,
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "image",
                "ocr": {
                    "text": ocr_text,
                    "layout": ocr_layout_blocks,
                    "blocks": ocr_layout_blocks,
                    "line_count": line_count,
                    "word_count": word_count,
                    "table_count": len(extracted_tables),
                    "description": description_meta,
                },
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    if detected_mime_type == "application/pdf" or extension == ".pdf":
        extracted_text, pdf_warnings, pdf_details = _extract_pdf_text_from_bytes(content_bytes)
        warnings.extend(pdf_warnings)
        return ParsedDoc(
            parser="pdf_parser",
            text=extracted_text,
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "filename": filename,
                "mime_type": detected_mime_type,
                "magic_bytes_type": magic_bytes_type,
                "format": "pdf",
                "size_bytes": len(content_bytes),
                "pdf": pdf_details,
                "signature_compliance": signature_compliance,
            },
            warnings=warnings,
        )

    extracted_text, printable_ratio = _best_effort_text_extraction(content_bytes)
    warnings.append(
        "LEGACY_FALLBACK_BEST_EFFORT_TEXT: No dedicated parser/converter route matched; "
        "using best-effort text extraction with baseline metadata."
    )
    return ParsedDoc(
        parser="legacy_best_effort_parser",
        text=extracted_text,
        layout=[],
        tables=[],
        media=[],
        object_structure={
            "filename": filename,
            "mime_type": detected_mime_type,
            "magic_bytes_type": magic_bytes_type,
            "size_bytes": len(content_bytes),
            "fallback_strategy": {
                "id": "best_effort_text_extraction",
                "legacy_format": extension or "unknown",
                "printable_ratio_pct": printable_ratio,
                "structured_warnings": [
                    {
                        "code": "LEGACY_FALLBACK_BEST_EFFORT_TEXT",
                        "severity": "warning",
                        "message": "Dedicated parser/converter unavailable; best-effort text extraction applied.",
                    }
                ],
            },
            "signature_compliance": signature_compliance,
        },
        warnings=warnings,
    )


def normalize_parsed_document(
    *,
    parsed_doc: ParsedDoc,
    metadata: dict,
) -> NormalizedDoc:
    canonical_text = parsed_doc.text.strip()
    chunks, chunk_warnings = _build_modality_chunks(parsed_doc=parsed_doc, metadata=metadata)

    provenance = {
        "source_type": metadata.get("source_type"),
        "filename": metadata.get("filename"),
        "stored_filename": metadata.get("stored_filename"),
        "source_version": metadata.get("source_version"),
        "sha256": metadata.get("sha256"),
        "detected_mime_type": metadata.get("detected_mime_type"),
        "magic_bytes_type": metadata.get("magic_bytes_type"),
        "parser": parsed_doc.parser,
        "signature_compliance": parsed_doc.object_structure.get("signature_compliance")
        or {
            "signature_present": False,
            "signature_schemes": [],
            "compliance_markers": [],
        },
    }

    text_viewer_hints = _build_text_viewer_hints(canonical_text=canonical_text, chunks=chunks)
    render_hints = {
        "layout": parsed_doc.layout,
        "tables": parsed_doc.tables,
        "media": parsed_doc.media,
        "overlay": _build_image_overlay_hints(parsed_doc=parsed_doc),
        "page_map": text_viewer_hints["page_map"],
        "text_highlight_spans": text_viewer_hints["text_highlight_spans"],
        "automation": _build_automation_viewer_hints(parsed_doc=parsed_doc),
        "transcript_timeline_alignment": _build_transcript_timeline_alignment(
            parsed_doc=parsed_doc,
            canonical_text=canonical_text,
            chunks=chunks,
        ),
    }

    embeddings_inputs = [
        {
            "chunk_id": chunk.get("chunk_id") or f"{metadata.get('stored_filename')}:chunk:{index}",
            "modality": chunk.get("modality") or "text",
            "text": chunk.get("text") or "",
            "source": {
                "stored_filename": metadata.get("stored_filename"),
                "source_version": metadata.get("source_version"),
                "sha256": metadata.get("sha256"),
            },
        }
        for index, chunk in enumerate(chunks)
        if (chunk.get("text") or "").strip()
    ]

    warnings = list(parsed_doc.warnings)
    warnings.extend(chunk_warnings)
    if not canonical_text:
        warnings.append(
            "Normalization produced no canonical_text; downstream chunk/embed steps should use format-specific fallbacks."
        )

    relations = list(parsed_doc.object_structure.get("relations") or [])
    entities = _extract_domain_entities(parsed_doc=parsed_doc)

    return NormalizedDoc(
        canonical_text=canonical_text,
        chunks=chunks,
        entities=entities,
        relations=relations,
        embeddings_inputs=embeddings_inputs,
        render_hints=render_hints,
        provenance=provenance,
        warnings=warnings,
    )


def _extract_domain_entities(*, parsed_doc: ParsedDoc) -> list[dict]:
    automation = parsed_doc.object_structure.get("automation")
    if not isinstance(automation, dict):
        return []

    dialect = automation.get("dialect") or "unknown"
    entities: list[dict] = []
    for block in automation.get("program_blocks") or []:
        if not isinstance(block, dict):
            continue

        line = block.get("line")
        raw = str(block.get("raw") or "")
        for token in block.get("tokens") or []:
            normalized_token = str(token).strip()
            if not normalized_token:
                continue
            prefix = normalized_token[0].upper()
            if prefix in {"X", "Y", "Z", "A", "B", "C"}:
                entity_type = "axis"
            elif prefix == "F":
                entity_type = "feed"
            elif prefix in {"T", "H", "D"}:
                entity_type = "tool"
            elif re.match(r"^(?:ALARM\d+|ERR\d+|E\d+)$", normalized_token, re.IGNORECASE):
                entity_type = "alarmcode"
            else:
                continue

            entities.append(
                {
                    "entity_type": entity_type,
                    "value": normalized_token,
                    "line": line,
                    "dialect": dialect,
                    "source": "automation_token",
                }
            )

        for parameter in block.get("parameters") or []:
            if not isinstance(parameter, dict):
                continue
            name = str(parameter.get("name") or "").strip()
            value = str(parameter.get("value") or "").strip()
            if not name:
                continue

            upper_name = name.upper()
            if "TOOL" in upper_name:
                entity_type = "tool"
            elif upper_name in {"F", "FEED", "SPEED", "VEL", "VELOCITY"}:
                entity_type = "feed"
            elif "ALARM" in upper_name or upper_name.startswith("ERR"):
                entity_type = "alarmcode"
            elif upper_name.startswith(("X", "Y", "Z", "A", "B", "C")):
                entity_type = "axis"
            else:
                continue

            entities.append(
                {
                    "entity_type": entity_type,
                    "value": value or name,
                    "name": name,
                    "line": line,
                    "dialect": dialect,
                    "source": "automation_parameter",
                }
            )

        if re.search(r"\b(?:ALARM\d+|ERROR|FAULT|ERR\d+)\b", raw, re.IGNORECASE):
            entities.append(
                {
                    "entity_type": "alarmcode",
                    "value": raw,
                    "line": line,
                    "dialect": dialect,
                    "source": "automation_line",
                }
            )

    deduped: list[dict] = []
    seen: set[tuple] = set()
    for entity in entities:
        key = (
            entity.get("entity_type"),
            entity.get("name"),
            entity.get("value"),
            entity.get("line"),
            entity.get("dialect"),
            entity.get("source"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def enqueue_embedding_inputs(stored_filename: str, embeddings_inputs: list[dict]) -> dict:
    _ensure_dirs([ARTIFACTS_DIR])
    queue_record = {
        "stored_filename": stored_filename,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "count": len(embeddings_inputs),
        "items": embeddings_inputs,
    }
    queue_path = ARTIFACTS_DIR / "embedding_queue.jsonl"
    with queue_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(queue_record, ensure_ascii=False) + "\n")
    return {
        "path": str(queue_path),
        "count": len(embeddings_inputs),
    }


def _build_text_viewer_hints(*, canonical_text: str, chunks: list[dict]) -> dict:
    page_map: list[dict] = []
    text_highlight_spans: list[dict] = []

    if not canonical_text:
        return {
            "page_map": page_map,
            "text_highlight_spans": text_highlight_spans,
        }

    cursor = 0
    for chunk in chunks:
        if (chunk.get("modality") or "").lower() != "text":
            continue

        chunk_text = (chunk.get("text") or "").strip()
        if not chunk_text:
            continue

        start = canonical_text.find(chunk_text, cursor)
        if start < 0:
            start = canonical_text.find(chunk_text)
        if start < 0:
            continue

        end = start + len(chunk_text)
        cursor = end
        page_hint = {
            "page": 1,
            "start": start,
            "end": end,
            "chunk_id": chunk.get("chunk_id"),
        }
        page_map.append(page_hint)
        text_highlight_spans.append(page_hint.copy())

    return {
        "page_map": page_map,
        "text_highlight_spans": text_highlight_spans,
    }


def _build_image_overlay_hints(*, parsed_doc: ParsedDoc) -> list[dict]:
    if not parsed_doc.parser.startswith("image_") and parsed_doc.object_structure.get("format") != "image":
        return []

    overlay_entries: list[dict] = []
    seen_entries: set[tuple[str, str, int, int, int, int]] = set()

    for item in parsed_doc.layout or []:
        bbox = item.get("bbox") or {}
        normalized_bbox = {
            "x": int(bbox.get("x", 0)),
            "y": int(bbox.get("y", 0)),
            "width": int(bbox.get("width", 0)),
            "height": int(bbox.get("height", 0)),
        }
        entry = {
            "type": item.get("type") or "ocr_block",
            "id": item.get("block_id") or item.get("line_id") or item.get("word_id"),
            "text": item.get("text") or "",
            "bbox": normalized_bbox,
            "confidence": item.get("confidence"),
        }
        dedupe_key = (
            entry["type"],
            str(entry["id"]),
            normalized_bbox["x"],
            normalized_bbox["y"],
            normalized_bbox["width"],
            normalized_bbox["height"],
        )
        if dedupe_key in seen_entries:
            continue
        seen_entries.add(dedupe_key)
        overlay_entries.append(entry)

    return overlay_entries


def _build_automation_viewer_hints(*, parsed_doc: ParsedDoc) -> dict:
    automation = parsed_doc.object_structure.get("automation")
    if not isinstance(automation, dict):
        return {
            "jump_markers": [],
            "block_folding": [],
            "parameter_panel": [],
        }

    jump_markers: list[dict] = []
    block_folding: list[dict] = []
    parameter_panel: list[dict] = []

    for node in automation.get("structural_nodes") or []:
        if not isinstance(node, dict):
            continue

        node_type = str(node.get("node_type") or "")
        line = node.get("line")

        if node_type == "subroutine":
            jump_markers.append(
                {
                    "line": line,
                    "label": node.get("name") or "subroutine",
                    "node_type": "subroutine",
                }
            )

        if node_type in {"program_block", "subroutine"}:
            block_folding.append(
                {
                    "line": line,
                    "label": node.get("raw") or node.get("name") or "block",
                    "foldable": True,
                }
            )

        if node_type == "parameter":
            parameter_panel.append(
                {
                    "line": line,
                    "name": node.get("name"),
                    "value": node.get("value"),
                }
            )

    return {
        "jump_markers": jump_markers,
        "block_folding": block_folding,
        "parameter_panel": parameter_panel,
    }


def _build_transcript_timeline_alignment(*, parsed_doc: ParsedDoc, canonical_text: str, chunks: list[dict]) -> list[dict]:
    asr_payload = parsed_doc.object_structure.get("asr")
    if not isinstance(asr_payload, dict):
        return []

    normalized_text = canonical_text or ""
    if not normalized_text.strip():
        return []

    alignments: list[dict] = []
    cursor = 0
    for chunk in chunks:
        modality = (chunk.get("modality") or "").lower()
        if modality not in {"audio", "video"}:
            continue

        chunk_text = str(chunk.get("text") or "").strip()
        if not chunk_text:
            continue

        start_index = normalized_text.find(chunk_text, cursor)
        if start_index == -1:
            start_index = normalized_text.find(chunk_text)
        if start_index == -1:
            continue

        end_index = start_index + len(chunk_text)
        cursor = end_index
        source = chunk.get("source") or {}
        alignments.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "span": {"start": start_index, "end": end_index},
                "timeline": {
                    "start_sec": source.get("start_sec"),
                    "end_sec": source.get("end_sec"),
                    "chapter_id": source.get("chapter_id"),
                    "segment_ids": source.get("segment_ids") or [],
                },
            }
        )

    return alignments


def _detect_asr_chapters(segments: list[dict]) -> list[dict]:
    if not segments:
        return []

    topic_boundary_markers = {
        "agenda",
        "topic",
        "kapitel",
        "abschnitt",
        "nächste",
        "next",
        "summary",
        "fazit",
    }

    chapters: list[dict] = []
    current_segment_ids: list[str] = []
    current_text_parts: list[str] = []
    current_start = float(segments[0].get("start_sec") or 0.0)
    previous_end = current_start

    for index, segment in enumerate(segments):
        segment_id = str(segment.get("segment_id") or f"seg_{index}")
        segment_text = str(segment.get("text") or "").strip()
        segment_start = float(segment.get("start_sec") or previous_end)
        segment_end = float(segment.get("end_sec") or segment_start)

        normalized_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-zÀ-ÿ]+", segment_text)
        }
        is_topic_boundary = bool(topic_boundary_markers.intersection(normalized_tokens))
        long_time_gap = (segment_start - previous_end) >= 12
        has_payload = bool(current_segment_ids)

        if has_payload and (is_topic_boundary or long_time_gap):
            chapters.append(
                {
                    "chapter_id": f"chapter_{len(chapters)}",
                    "start_sec": round(current_start, 3),
                    "end_sec": round(previous_end, 3),
                    "segment_ids": list(current_segment_ids),
                    "text": " ".join(current_text_parts).strip(),
                    "boundary": "semantic" if is_topic_boundary else "time_gap",
                }
            )
            current_segment_ids = []
            current_text_parts = []
            current_start = segment_start

        current_segment_ids.append(segment_id)
        if segment_text:
            current_text_parts.append(segment_text)
        previous_end = max(previous_end, segment_end)

    if current_segment_ids:
        chapters.append(
            {
                "chapter_id": f"chapter_{len(chapters)}",
                "start_sec": round(current_start, 3),
                "end_sec": round(previous_end, 3),
                "segment_ids": list(current_segment_ids),
                "text": " ".join(current_text_parts).strip(),
                "boundary": "end_of_transcript",
            }
        )

    return chapters

def _compute_source_version(filename: str, source_type: str) -> int:
    if not METADATA_DIR.exists():
        return 1

    version = 0
    for metadata_file in METADATA_DIR.glob("*.json"):
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if payload.get("filename") == filename and payload.get("source_type") == source_type:
            version = max(version, int(payload.get("source_version") or 1))
    return version + 1


def _normalize_extension(filename: str) -> str:
    return Path(filename).suffix.lower().strip()


def validate_upload(filename: str, content_bytes: bytes) -> ValidationResult:
    warnings: list[str] = []
    extension = _normalize_extension(filename)

    if not content_bytes:
        return ValidationResult(
            status="error",
            message="Empty uploads are not allowed.",
            warnings=warnings,
        )

    if extension not in ALLOWED_EXTENSIONS:
        warnings.append(
            f"Unsupported file type '{extension or 'unknown'}'. "
            "Supported types: PDF, DOCX, TXT, PNG, JPG, JPEG, SVG, JSON/XML/YAML/TOML, plus machine-code formats (e.g. GCODE/NC/KRL/RAPID/URScript/IEC files)."
        )
        return ValidationResult(
            status="warning",
            message="Unsupported file type.",
            warnings=warnings,
        )

    return ValidationResult(
        status="success",
        message="File accepted for ingestion.",
        warnings=warnings,
    )


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _record_observability(*, metadata: dict, step_durations_ms: dict[str, float], status: str) -> dict:
    _ensure_dirs([ARTIFACTS_DIR])

    extension = _normalize_extension(metadata.get("filename") or metadata.get("stored_filename") or "")
    detected_mime = metadata.get("detected_mime_type")
    format_key = extension or (detected_mime or "unknown")
    qa_errors = metadata.get("qa_errors") or []

    budget_status = {
        "budgets_ms": PERFORMANCE_BUDGETS_MS,
        "breaches": [
            {
                "step": step_name,
                "duration_ms": float(duration),
                "budget_ms": float(PERFORMANCE_BUDGETS_MS[step_name]),
            }
            for step_name, duration in step_durations_ms.items()
            if step_name in PERFORMANCE_BUDGETS_MS and float(duration) > float(PERFORMANCE_BUDGETS_MS[step_name])
        ],
    }

    log_event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stored_filename": metadata.get("stored_filename"),
        "filename": metadata.get("filename"),
        "source_type": metadata.get("source_type"),
        "detected_mime_type": detected_mime,
        "extension": extension,
        "format_key": format_key,
        "status": status,
        "qa_status": metadata.get("qa_status"),
        "qa_error_classes": [error.get("class", "unknown") for error in qa_errors],
        "coverage": {
            "chunks": len(metadata.get("chunk_ids") or []),
            "embeddings_inputs": int(metadata.get("embedding_inputs_count") or 0),
            "viewer_modalities": len(metadata.get("viewer_artifact_modalities") or []),
        },
        "durations_ms": step_durations_ms,
        "performance_budget": budget_status,
    }
    with OBSERVABILITY_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_event, ensure_ascii=False) + "\n")

    if OBSERVABILITY_METRICS_PATH.exists():
        try:
            metrics_payload = json.loads(OBSERVABILITY_METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics_payload = {}
    else:
        metrics_payload = {}

    by_format = metrics_payload.setdefault("by_format", {})
    format_metrics = by_format.setdefault(
        format_key,
        {
            "count": 0,
            "errors": 0,
            "fatal_errors": 0,
            "warning_runs": 0,
            "coverage": {"chunks": 0, "embeddings_inputs": 0, "viewer_modalities": 0},
            "durations_ms": {},
            "performance_budget": {
                "budgets_ms": PERFORMANCE_BUDGETS_MS,
                "breach_counts": {step_name: 0 for step_name in PERFORMANCE_BUDGETS_MS},
            },
        },
    )
    format_metrics["count"] += 1
    format_metrics["errors"] += int(status != "success")
    format_metrics["fatal_errors"] += int(any(error.get("class") == "fatal" for error in qa_errors))
    format_metrics["warning_runs"] += int(metadata.get("qa_status") == "warning")

    for coverage_key, value in log_event["coverage"].items():
        format_metrics["coverage"][coverage_key] += int(value)

    for step_name, duration in step_durations_ms.items():
        step_entry = format_metrics["durations_ms"].setdefault(step_name, {"total": 0.0, "max": 0.0, "avg": 0.0})
        step_entry["total"] += float(duration)
        step_entry["max"] = max(float(step_entry["max"]), float(duration))
        step_entry["avg"] = step_entry["total"] / format_metrics["count"]

    breach_counts = format_metrics.setdefault("performance_budget", {}).setdefault(
        "breach_counts",
        {step_name: 0 for step_name in PERFORMANCE_BUDGETS_MS},
    )
    for breach in budget_status["breaches"]:
        breach_counts[breach["step"]] = int(breach_counts.get(breach["step"], 0)) + 1

    metrics_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    OBSERVABILITY_METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return {
        "log_path": str(OBSERVABILITY_LOG_PATH),
        "metrics_path": str(OBSERVABILITY_METRICS_PATH),
        "format_key": format_key,
        "performance_budget": budget_status,
    }


def _record_dead_letter(
    *,
    filename: str,
    source_type: str,
    content_bytes: bytes,
    content_type: str | None,
    detected_mime_type: str,
    magic_type: str | None,
    source_version: int,
    stored_filename: str,
    extra_metadata: dict | None,
    exception: Exception,
    stage: str,
) -> dict:
    _ensure_dirs([DEAD_LETTER_DIR])

    content_hash = hashlib.sha256(content_bytes).hexdigest()
    stack_trace = traceback.format_exc()
    error_fingerprint = hashlib.sha256(
        f"{type(exception).__name__}:{str(exception)}:{stage}".encode("utf-8")
    ).hexdigest()
    occurred_at = datetime.now(timezone.utc).isoformat()

    report = {
        "occurred_at": occurred_at,
        "filename": filename,
        "stored_filename": stored_filename,
        "source_type": source_type,
        "source_version": source_version,
        "content_type": content_type,
        "detected_mime_type": detected_mime_type,
        "magic_bytes_type": magic_type,
        "size_bytes": len(content_bytes),
        "sha256": content_hash,
        "error": {
            "stage": stage,
            "type": type(exception).__name__,
            "message": str(exception),
            "fingerprint": error_fingerprint,
            "stack_trace": stack_trace,
        },
        "processor_route": select_processor_route(
            mime=detected_mime_type,
            ext=_normalize_extension(filename),
            magic_bytes=magic_type,
        ),
        "extra_metadata": extra_metadata or {},
    }

    report_path = DEAD_LETTER_DIR / f"{content_hash}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    queue_entry = {
        "occurred_at": occurred_at,
        "sha256": content_hash,
        "report_path": str(report_path),
        "error_fingerprint": error_fingerprint,
        "stored_filename": stored_filename,
        "filename": filename,
        "source_type": source_type,
    }
    with DEAD_LETTER_QUEUE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(queue_entry, ensure_ascii=False) + "\n")

    return {
        "report_path": str(report_path),
        "queue_path": str(DEAD_LETTER_QUEUE_PATH),
        "sha256": content_hash,
        "error_fingerprint": error_fingerprint,
    }


def store_upload(
    filename: str,
    content_bytes: bytes,
    content_type: str | None,
    source_type: str = "upload",
    extra_metadata: dict | None = None,
    provider: str | None = None,
    api_key: str | None = None,
) -> dict:
    _ensure_dirs([UPLOAD_DIR, METADATA_DIR, ARTIFACTS_DIR, PARSED_DIR, NORMALIZED_DIR, VIEWER_ARTIFACTS_DIR, DEAD_LETTER_DIR, IMAGE_BLOB_DIR, IMAGE_DESCRIPTIONS_DIR])
    step_durations_ms: dict[str, float] = {}

    timestamp = datetime.now(timezone.utc).isoformat()
    extension = _normalize_extension(filename)
    safe_name = Path(filename).stem.replace(" ", "_")
    stored_name = f"{safe_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{extension}"
    file_path = UPLOAD_DIR / stored_name
    source_version = _compute_source_version(filename=filename, source_type=source_type)

    detected_mime_type = _detect_mime_type(
        filename=filename,
        content_type=content_type,
        content_bytes=content_bytes,
    )
    magic_type = _detect_magic_type(content_bytes)
    sha256 = hashlib.sha256(content_bytes).hexdigest()

    file_path.write_bytes(content_bytes)

    metadata = {
        "source_type": source_type,
        "filename": filename,
        "stored_filename": stored_name,
        "source_version": source_version,
        "timestamp": timestamp,
        "content_type": content_type,
        "detected_mime_type": detected_mime_type,
        "magic_bytes_type": magic_type,
        "sha256": sha256,
        "size_bytes": len(content_bytes),
        "processor_route": select_processor_route(
            mime=detected_mime_type,
            ext=extension,
            magic_bytes=magic_type,
        ),
    }
    try:
        started = time.perf_counter()
        parsed_doc = parse_structured_document(
            filename=filename,
            content_bytes=content_bytes,
            detected_mime_type=detected_mime_type,
            magic_bytes_type=magic_type,
            provider=provider,
            api_key=api_key,
        )
        step_durations_ms["parse"] = round((time.perf_counter() - started) * 1000, 3)
        parsed_path = PARSED_DIR / f"{stored_name}.json"
        parsed_path.write_text(
            json.dumps(
                {
                    "parser": parsed_doc.parser,
                    "text": parsed_doc.text,
                    "layout": parsed_doc.layout,
                    "tables": parsed_doc.tables,
                    "media": parsed_doc.media,
                    "object_structure": parsed_doc.object_structure,
                    "warnings": parsed_doc.warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        metadata["parsed_artifact_path"] = str(parsed_path)
        metadata["parse_warnings"] = parsed_doc.warnings

        if detected_mime_type.startswith("image/"):
            image_analysis_result = analyze_image_bytes_with_provider(
                filename=filename,
                content_bytes=content_bytes,
                provider=provider,
                api_key=api_key,
            )
            image_description_text = _format_image_description_output(
                filename=filename,
                analysis_result=image_analysis_result,
            )
            image_description_path = _store_image_description_text(
                stored_filename=stored_name,
                text_payload=image_description_text,
            )
            metadata["image_description_path"] = str(image_description_path)

            description_text = str(image_analysis_result.description_text or "").strip()
            analysis_text = str(image_analysis_result.analysis_text or "").strip()
            open_questions = [
                str(item).strip()
                for item in (image_analysis_result.open_questions or [])
                if str(item).strip()
            ]
            provider_name = str((image_analysis_result.meta or {}).get("provider") or "").strip()

            vision_callable = None
            if description_text or analysis_text:
                structured_payload = {
                    "caption": description_text or analysis_text,
                    "tags": ["image", "llm-analysis", *([provider_name] if provider_name else [])],
                    "objects": [],
                    "ocr_text": "",
                    "metadata": {
                        "analysis": analysis_text,
                        "open_questions": open_questions,
                        "provider": provider_name or None,
                        "source": "image_description_pipeline",
                    },
                }
                structured_json = json.dumps(structured_payload, ensure_ascii=False)

                def _description_vision_callable(_image: bytes, _prompt: str) -> tuple[str | None, list[str]]:
                    return structured_json, []

                vision_callable = _description_vision_callable

            try:
                image_pipeline = process_image_for_search(
                    filename=filename,
                    content_type=detected_mime_type,
                    upload_bytes=content_bytes,
                    blob_dir=IMAGE_BLOB_DIR,
                    idempotency_index_path=IMAGE_IDEMPOTENCY_INDEX_PATH,
                    vision_callable=vision_callable,
                )
            except Exception as image_exc:
                metadata["image_pipeline_warnings"] = [
                    "Image search pipeline failed and was skipped.",
                    f"image_pipeline_error={image_exc}",
                    *(image_analysis_result.warnings or []),
                ]
            else:
                metadata["image_blob_uri"] = image_pipeline.get("blob_uri")
                metadata["image_vector_document"] = image_pipeline.get("vector_document")
                metadata["image_embeddings_text"] = image_pipeline.get("embeddings_text")
                metadata["image_pipeline_warnings"] = [
                    *(image_pipeline.get("warnings") or []),
                    *(image_analysis_result.warnings or []),
                ]

        _prepare_3d_pipeline_artifacts(
            metadata=metadata,
            provider=provider,
            api_key=api_key,
        )

        started = time.perf_counter()
        normalized_doc = normalize_parsed_document(parsed_doc=parsed_doc, metadata=metadata)
        step_durations_ms["normalize"] = round((time.perf_counter() - started) * 1000, 3)
        normalized_path = NORMALIZED_DIR / f"{stored_name}.json"
        normalized_payload = validate_normalized_artifact_payload(
            {
                "canonical_text": normalized_doc.canonical_text,
                "chunks": normalized_doc.chunks,
                "entities": normalized_doc.entities,
                "relations": normalized_doc.relations,
                "embeddings_inputs": normalized_doc.embeddings_inputs,
                "render_hints": normalized_doc.render_hints,
                "provenance": normalized_doc.provenance,
                "warnings": normalized_doc.warnings,
            }
        )
        normalized_path.write_text(json.dumps(normalized_payload, indent=2), encoding="utf-8")
        metadata["normalized_artifact_path"] = str(normalized_path)
        metadata["normalize_warnings"] = normalized_doc.warnings

        started = time.perf_counter()
        embedding_queue_result = enqueue_embedding_inputs(
            stored_filename=stored_name,
            embeddings_inputs=normalized_doc.embeddings_inputs,
        )
        step_durations_ms["embed"] = round((time.perf_counter() - started) * 1000, 3)
        metadata["embedding_queue_path"] = embedding_queue_result["path"]
        metadata["embedding_inputs_count"] = embedding_queue_result["count"]
        metadata["chunk_ids"] = [chunk.get("chunk_id") for chunk in normalized_doc.chunks if chunk.get("chunk_id")]

        qa_result = run_pipeline_qa(metadata=metadata, normalized_doc=normalized_doc)
        qa_path = ARTIFACTS_DIR / f"{stored_name}.qa.json"
        qa_path.write_text(json.dumps(qa_result, indent=2), encoding="utf-8")
        metadata["qa_artifact_path"] = str(qa_path)
        metadata["qa_status"] = qa_result["status"]
        metadata["qa_errors"] = qa_result["errors"]

        started = time.perf_counter()
        rag_index_result = vector_store.upsert_rag_document(
            stored_filename=stored_name,
            metadata=metadata,
            chunks=normalized_doc.chunks,
            relations=normalized_doc.relations,
        )
        step_durations_ms["index"] = round((time.perf_counter() - started) * 1000, 3)
        metadata["rag_index_path"] = rag_index_result["path"]

        started = time.perf_counter()
        viewer_artifacts_result = build_viewer_artifacts(
            normalized_doc=normalized_doc,
            metadata=metadata,
        )
        step_durations_ms["viewer_artifacts"] = round((time.perf_counter() - started) * 1000, 3)
        metadata["viewer_artifacts_path"] = viewer_artifacts_result["path"]
        metadata["viewer_artifacts_count"] = viewer_artifacts_result["count"]
        metadata["viewer_artifact_modalities"] = viewer_artifacts_result["modalities"]

        if extra_metadata:
            metadata.update(extra_metadata)

        observability_result = _record_observability(
            metadata=metadata,
            step_durations_ms=step_durations_ms,
            status="error" if metadata.get("qa_status") == "failed" else "success",
        )
        metadata["observability_log_path"] = observability_result["log_path"]
        metadata["observability_metrics_path"] = observability_result["metrics_path"]
        metadata["observability_format_key"] = observability_result["format_key"]
        metadata["performance_budget"] = observability_result["performance_budget"]

        metadata_path = METADATA_DIR / f"{stored_name}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - exercised via targeted tests
        dead_letter = _record_dead_letter(
            filename=filename,
            source_type=source_type,
            content_bytes=content_bytes,
            content_type=content_type,
            detected_mime_type=detected_mime_type,
            magic_type=magic_type,
            source_version=source_version,
            stored_filename=stored_name,
            extra_metadata=extra_metadata,
            exception=exc,
            stage="store_upload",
        )
        raise RuntimeError(
            "Upload processing failed and was moved to dead-letter queue. "
            f"report={dead_letter['report_path']}"
        ) from exc

    return metadata


def _unique_destination(target_dir: Path, filename: str) -> Path:
    target = target_dir / filename
    if not target.exists():
        return target

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def ingest_watch_folder() -> dict:
    _ensure_dirs(
        [
            WATCH_DIR,
            WATCH_PROCESSED_DIR,
            WATCH_REJECTED_DIR,
            UPLOAD_DIR,
            METADATA_DIR,
        ]
    )

    files = [path for path in WATCH_DIR.iterdir() if path.is_file()]
    results: list[dict] = []

    if not files:
        return {
            "status": "success",
            "message": "No files found in watch folder.",
            "files": results,
        }

    for path in sorted(files):
        content = path.read_bytes()
        validation = validate_upload(path.name, content)
        entry = {
            "filename": path.name,
            "status": validation.status,
            "message": validation.message,
            "warnings": validation.warnings,
        }

        if validation.status == "success":
            try:
                metadata = store_upload(
                    path.name,
                    content,
                    content_type=None,
                    source_type="watch-folder",
                )
                entry["metadata"] = metadata
                destination_dir = WATCH_PROCESSED_DIR
            except RuntimeError as exc:
                entry["status"] = "error"
                entry["message"] = str(exc)
                destination_dir = WATCH_REJECTED_DIR
        else:
            destination_dir = WATCH_REJECTED_DIR

        destination = _unique_destination(destination_dir, path.name)
        path.rename(destination)
        entry["moved_to"] = str(destination)
        results.append(entry)

    has_warnings = any(entry.get("warnings") for entry in results)
    status = "warning" if has_warnings else "success"
    message = (
        "Processed watch folder files with warnings."
        if has_warnings
        else "Processed watch folder files."
    )

    return {
        "status": status,
        "message": message,
        "files": results,
    }


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach SMTP inbox API at {url}.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON returned from SMTP inbox API at {url}.") from exc


def _format_sender(sender: object) -> str | None:
    if isinstance(sender, dict):
        address = sender.get("Address") or sender.get("address")
        name = sender.get("Name") or sender.get("name")
        if name and address:
            return f"{name} <{address}>"
        return address or name
    if isinstance(sender, list):
        formatted = [_format_sender(item) for item in sender]
        formatted = [item for item in formatted if item]
        return ", ".join(formatted) if formatted else None
    if isinstance(sender, str):
        return sender
    return None


def _sanitize_filename(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    if not cleaned:
        return fallback
    return cleaned[:50]


def _fetch_mailpit_messages(api_url: str, max_messages: int) -> list[dict]:
    data = _fetch_json(f"{api_url}/api/v1/messages")
    messages = data.get("messages", [])
    return messages[:max_messages]


def _fetch_mailpit_message(api_url: str, message_id: str) -> dict:
    return _fetch_json(f"{api_url}/api/v1/message/{message_id}")


def _parse_mailpit_message(message: dict) -> tuple[str, str | None, str, list[str]]:
    subject = message.get("Subject") or "untitled"
    sender = _format_sender(message.get("From"))
    text = message.get("Text") or ""
    html = message.get("HTML") or ""
    warnings: list[str] = []

    body = text.strip()
    if not body and html.strip():
        body = html.strip()
        warnings.append("Email contained only HTML; stored raw HTML in .txt.")

    return subject, sender, body, warnings


def _fetch_mailhog_messages(api_url: str, max_messages: int) -> list[dict]:
    data = _fetch_json(f"{api_url}/api/v2/messages")
    messages = data.get("items", [])
    return messages[:max_messages]


def _first_header(headers: dict, key: str) -> str:
    values = headers.get(key) or headers.get(key.lower()) or []
    if isinstance(values, list):
        return values[0] if values else ""
    return values or ""


def _parse_mailhog_message(message: dict) -> tuple[str, str | None, str, list[str]]:
    content = message.get("Content", {})
    headers = content.get("Headers", {})
    subject = _first_header(headers, "Subject") or "untitled"
    sender = _first_header(headers, "From") or None
    body = content.get("Body") or ""
    warnings: list[str] = []
    if not body.strip():
        body = content.get("Raw", "") or ""
        if body:
            warnings.append("Email body was empty; stored raw content.")
    return subject, sender, body.strip(), warnings


def ingest_smtp_inbox(
    api_url: str | None = None,
    provider: str | None = None,
    max_messages: int | None = None,
) -> dict:
    _ensure_dirs([UPLOAD_DIR, METADATA_DIR, ARTIFACTS_DIR, PARSED_DIR, NORMALIZED_DIR])

    api_url = (api_url or MAILPIT_API_URL).rstrip("/")
    provider = (provider or SMTP_PROVIDER).lower().strip()
    max_messages = max_messages or SMTP_MAX_MESSAGES

    try:
        if provider == "mailhog":
            messages = _fetch_mailhog_messages(api_url, max_messages)
        else:
            messages = _fetch_mailpit_messages(api_url, max_messages)
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc), "files": []}

    if not messages:
        return {
            "status": "success",
            "message": "No emails found in SMTP inbox.",
            "files": [],
        }

    results: list[dict] = []
    for message in messages:
        message_id = message.get("ID") or message.get("Id") or ""
        entry = {"message_id": message_id}

        try:
            if provider == "mailhog":
                subject, sender, body, warnings = _parse_mailhog_message(message)
            else:
                detail = _fetch_mailpit_message(api_url, message_id)
                subject, sender, body, warnings = _parse_mailpit_message(detail)
        except RuntimeError as exc:
            entry.update({"status": "error", "message": str(exc), "warnings": []})
            results.append(entry)
            continue

        if not body:
            entry.update(
                {
                    "status": "error",
                    "message": "Email body was empty.",
                    "warnings": warnings,
                }
            )
            results.append(entry)
            continue

        safe_subject = _sanitize_filename(subject, "email")
        filename = f"email_{safe_subject}_{message_id or 'unknown'}.txt"
        try:
            metadata = store_upload(
                filename,
                body.encode("utf-8"),
                content_type="text/plain",
                source_type="smtp",
                extra_metadata={"sender": sender, "subject": subject, "message_id": message_id},
            )
            entry.update(
                {
                    "status": "success",
                    "message": "Email stored for ingestion.",
                    "warnings": warnings,
                    "metadata": metadata,
                }
            )
        except RuntimeError as exc:
            entry.update(
                {
                    "status": "error",
                    "message": str(exc),
                    "warnings": warnings,
                }
            )
        results.append(entry)

    has_warnings = any(entry.get("warnings") for entry in results)
    status = "warning" if has_warnings else "success"
    message = (
        "Processed SMTP inbox messages with warnings."
        if has_warnings
        else "Processed SMTP inbox messages."
    )

    return {
        "status": status,
        "message": message,
        "files": results,
    }
