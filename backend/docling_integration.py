from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import mimetypes
import json
from pathlib import Path
import re
import string
import zipfile
import xml.etree.ElementTree as ET

from backend.ingestion import (
    ALLOWED_EXTENSIONS,
    ARTIFACTS_DIR,
    METADATA_DIR,
    UPLOAD_DIR,
    VIEWER_ARTIFACTS_DIR,
    _convert_3d_to_canonical_glb,
)


@dataclass(frozen=True)
class DoclingStatus:
    available: bool
    message: str
    version: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConversionResult:
    status: str
    message: str
    markdown: str | None
    warnings: list[str]
    artifact_path: str | None = None
    metadata_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DocumentPreviewResult:
    status: str
    message: str
    warnings: list[str]
    stored_filename: str | None
    preview: str | None
    artifact_path: str | None = None
    truncated: bool = False
    total_chars: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SourceDocumentInfoResult:
    status: str
    message: str
    warnings: list[str]
    stored_filename: str | None
    source_path: str | None
    source_extension: str | None
    source_mime_type: str | None
    source_kind: str | None
    viewer_source_path: str | None = None
    viewer_source_extension: str | None = None
    viewer_source_mime_type: str | None = None
    viewer_source_kind: str | None = None
    viewer_source_ready: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def get_docling_status(module_name: str = "docling") -> DoclingStatus:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return DoclingStatus(
            available=False,
            message=(
                "Docling is not installed. Install the 'docling' package to enable "
                "document conversion."
            ),
        )

    try:
        version = importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        version = None

    return DoclingStatus(
        available=True,
        message="Docling is available for document conversion.",
        version=version,
    )


def _docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


def _extract_docx_text(file_path: Path) -> str:
    with zipfile.ZipFile(file_path) as archive:
        with archive.open("word/document.xml") as document_xml:
            xml_content = document_xml.read()

    root = ET.fromstring(xml_content)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []

    for paragraph in root.findall('.//w:p', namespace):
        runs = [node.text or "" for node in paragraph.findall('.//w:t', namespace)]
        line = "".join(runs).strip()
        if line:
            paragraphs.append(line)

    return "\n\n".join(paragraphs)


def _extract_pptx_text(file_path: Path) -> str:
    with zipfile.ZipFile(file_path) as archive:
        slide_paths = sorted(
            [
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ],
            key=lambda value: int(value.split("slide")[-1].split(".xml")[0]),
        )

        slide_texts: list[str] = []
        for slide_path in slide_paths:
            with archive.open(slide_path) as slide_xml:
                root = ET.fromstring(slide_xml.read())
            text_nodes = [node.text or "" for node in root.iter() if node.tag.endswith("}t") and node.text]
            text = "\n".join(value.strip() for value in text_nodes if value.strip()).strip()
            if text:
                slide_texts.append(text)

    return "\n\n".join(slide_texts)


def _extract_pdf_text(file_path: Path) -> str | None:
    if importlib.util.find_spec("pypdf") is None:
        return None

    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())

    return "\n\n".join(pages)


def _looks_like_unreadable_text(text: str) -> bool:
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


def _looks_like_binary_image_metadata_text(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return True

    lowered = normalized.lower()
    if re.search(r"(?:[a-z0-9+/]{24,}={0,2}){2,}", lowered):
        return True

    marker_hits = sum(
        marker in lowered
        for marker in ["jfif", "exif", "adobe", "icc_profile", "photoshop", "dqt", "dht", "soi", "eoi"]
    )
    if marker_hits >= 3:
        return True

    control_chars = sum(
        1
        for char in normalized
        if ord(char) < 32 and char not in {"\n", "\r", "\t"}
    )
    control_ratio = control_chars / max(1, len(normalized))
    if control_ratio > 0.01:
        return True

    tokens = re.findall(r"[A-Za-z0-9_]{2,}", lowered)
    if not tokens:
        return True

    metadata_tokens = {"jfif", "exif", "adobe", "rgb", "rgb8", "icc", "profile", "dqt", "dht", "sof", "sos"}
    metadata_token_hits = sum(token in metadata_tokens for token in tokens)
    if metadata_token_hits >= 4 and metadata_token_hits >= max(2, len(tokens) // 10):
        return True

    return False


def _mime_type_for_extension(extension: str) -> str | None:
    if extension == ".pdf":
        return "application/pdf"
    if extension == ".txt":
        return "text/plain; charset=utf-8"
    if extension == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if extension == ".pptx":
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if extension == ".png":
        return "image/png"
    if extension == ".jpg":
        return "image/jpeg"
    if extension == ".jpeg":
        return "image/jpeg"
    if extension == ".obj":
        return "model/obj"
    if extension == ".stl":
        return "model/stl"
    if extension == ".ply":
        return "application/ply"
    if extension == ".glb":
        return "model/gltf-binary"
    return None




def _is_valid_glb_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"glTF"
    except OSError:
        return False


def _source_kind_for_extension(extension: str) -> str:
    if extension == ".pdf":
        return "pdf"
    if extension in {".png", ".jpg", ".jpeg"}:
        return "image"
    if extension == ".docx":
        return "docx"
    if extension == ".pptx":
        return "pptx"
    if extension == ".txt":
        return "text"
    if extension in {".obj", ".stl", ".ply", ".glb", ".gltf", ".off"}:
        return "3d"
    return "binary"


def _extract_image_text(file_path: Path) -> str | None:
    if importlib.util.find_spec("PIL") is None or importlib.util.find_spec("pytesseract") is None:
        return None

    from PIL import Image
    import pytesseract

    with Image.open(file_path) as image:
        text = pytesseract.image_to_string(image)

    return text.strip()


def _convert_without_docling(file_path: Path, extension: str) -> ConversionResult:
    warnings: list[str] = []

    if extension == ".txt":
        markdown = file_path.read_text(encoding="utf-8", errors="replace")
        return ConversionResult(
            status="success",
            message="Document converted to Markdown via plain-text fallback.",
            markdown=markdown,
            warnings=warnings,
        )

    if extension == ".docx":
        try:
            markdown = _extract_docx_text(file_path)
        except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
            return ConversionResult(
                status="error",
                message=f"DOCX fallback conversion failed: {exc}",
                markdown=None,
                warnings=warnings,
            )

        if not markdown.strip():
            warnings.append("DOCX fallback conversion returned empty text.")

        return ConversionResult(
            status="success",
            message="Document converted to Markdown via DOCX fallback.",
            markdown=markdown,
            warnings=warnings,
        )

    if extension == ".pptx":
        try:
            markdown = _extract_pptx_text(file_path)
        except (zipfile.BadZipFile, KeyError, ET.ParseError, ValueError) as exc:
            return ConversionResult(
                status="error",
                message=f"PPTX fallback conversion failed: {exc}",
                markdown=None,
                warnings=warnings,
            )

        if not markdown.strip():
            warnings.append("PPTX fallback conversion returned empty text.")

        return ConversionResult(
            status="success",
            message="Document converted to Markdown via PPTX XML fallback.",
            markdown=markdown,
            warnings=warnings,
        )

    if extension == ".pdf":
        try:
            markdown = _extract_pdf_text(file_path)
        except Exception as exc:  # noqa: BLE001
            return ConversionResult(
                status="error",
                message=f"PDF fallback conversion failed: {exc}",
                markdown=None,
                warnings=warnings,
            )
        if markdown is not None and _looks_like_unreadable_text(markdown):
            return ConversionResult(
                status="error",
                message=(
                    "PDF fallback produced unreadable text. "
                    "The file is likely scanned, encrypted, or malformed."
                ),
                markdown=None,
                warnings=warnings,
            )
        if markdown is not None:
            return ConversionResult(
                status="success",
                message="Document converted to Markdown via PDF fallback.",
                markdown=markdown,
                warnings=warnings,
            )

    if extension in {".png", ".jpg", ".jpeg"}:
        try:
            markdown = _extract_image_text(file_path)
        except Exception as exc:  # noqa: BLE001
            return ConversionResult(
                status="error",
                message=f"Image OCR fallback failed: {exc}",
                markdown=None,
                warnings=warnings,
            )
        if markdown is not None:
            return ConversionResult(
                status="success",
                message="Document converted to Markdown via OCR image fallback.",
                markdown=markdown,
                warnings=warnings,
            )

    warnings.append(
        "Docling is not installed. Install the 'docling' package to convert this file type."
    )
    return ConversionResult(
        status="error",
        message="Docling is not installed and no fallback converter is available for this file type.",
        markdown=None,
        warnings=warnings,
    )


def convert_document_to_markdown(file_path: Path) -> ConversionResult:
    warnings: list[str] = []

    if not file_path.exists():
        return ConversionResult(
            status="error",
            message="Document not found for conversion.",
            markdown=None,
            warnings=warnings,
        )

    extension = file_path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        warnings.append(
            f"Unsupported file type '{extension or 'unknown'}'. "
            "Supported types: PDF, DOCX, TXT, PNG, JPG, JPEG."
        )
        return ConversionResult(
            status="warning",
            message="Unsupported file type.",
            markdown=None,
            warnings=warnings,
        )

    if extension in {".obj", ".stl", ".ply", ".glb", ".gltf", ".off"}:
        warnings.append("Skipped Docling conversion for 3D model; generated viewer-oriented Markdown summary.")
        return ConversionResult(
            status="success",
            message="3D model is prepared for viewer rendering; text extraction was skipped.",
            markdown=(
                "# 3D model upload\n\n"
                f"Source file: `{file_path.name}`\n\n"
                "This asset is intended for the 3D viewer in Evidence View."
            ),
            warnings=warnings,
        )

    if extension in {".png", ".jpg", ".jpeg"}:
        image_fallback = _convert_without_docling(file_path, extension)
        if image_fallback.status == "success":
            return image_fallback
        warnings.extend(image_fallback.warnings)
        return ConversionResult(
            status="error",
            message=(
                "Image conversion requires OCR fallback output, but no usable OCR extractor was available."
            ),
            markdown=None,
            warnings=warnings,
        )

    if not _docling_available():
        return _convert_without_docling(file_path, extension)

    from docling.document_converter import DocumentConverter

    try:
        converter = DocumentConverter()
        conversion = converter.convert(str(file_path))
        document = conversion.document if hasattr(conversion, "document") else conversion
        if not hasattr(document, "export_to_markdown"):
            raise ValueError("Docling conversion did not return a Markdown-capable document.")
        markdown = document.export_to_markdown()

        if _looks_like_unreadable_text(markdown) or (
            extension in {".png", ".jpg", ".jpeg"}
            and _looks_like_binary_image_metadata_text(markdown)
        ):
            fallback_result = _convert_without_docling(file_path, extension)
            if fallback_result.status == "success":
                combined_warnings = list(fallback_result.warnings)
                combined_warnings.append(
                    "Docling output appeared unreadable; fallback extractor output was used instead."
                )
                return ConversionResult(
                    status="success",
                    message=f"{fallback_result.message} (Quality fallback).",
                    markdown=fallback_result.markdown,
                    warnings=combined_warnings,
                )
            return ConversionResult(
                status="error",
                message=(
                    "Docling conversion produced unreadable text and no usable fallback was available."
                ),
                markdown=None,
                warnings=list(fallback_result.warnings),
            )
    except Exception as exc:  # noqa: BLE001 - surface conversion failures explicitly
        fallback_result = _convert_without_docling(file_path, extension)
        if fallback_result.status == "success":
            combined_warnings = list(fallback_result.warnings)
            combined_warnings.append(f"Docling conversion failed, fallback used: {exc}")
            return ConversionResult(
                status="success",
                message=f"{fallback_result.message} (Docling fallback).",
                markdown=fallback_result.markdown,
                warnings=combined_warnings,
            )
        return ConversionResult(
            status="error",
            message=f"Docling conversion failed: {exc}",
            markdown=None,
            warnings=warnings + list(fallback_result.warnings),
        )

    return ConversionResult(
        status="success",
        message="Document converted to Markdown.",
        markdown=markdown,
        warnings=warnings,
    )


def get_document_preview(
    stored_filename: str,
    max_chars: int = 4000,
) -> DocumentPreviewResult:
    warnings: list[str] = []

    if not stored_filename.strip():
        return DocumentPreviewResult(
            status="error",
            message="Stored filename is required for preview.",
            warnings=warnings,
            stored_filename=None,
            preview=None,
        )

    artifact_path = ARTIFACTS_DIR / f"{stored_filename}.md"
    if not artifact_path.exists():
        return DocumentPreviewResult(
            status="warning",
            message="No Markdown preview available. Convert the document first.",
            warnings=warnings,
            stored_filename=stored_filename,
            preview=None,
            artifact_path=str(artifact_path),
        )

    markdown = artifact_path.read_text(encoding="utf-8")
    if not markdown.strip():
        return DocumentPreviewResult(
            status="warning",
            message="Markdown preview is empty.",
            warnings=warnings,
            stored_filename=stored_filename,
            preview="",
            artifact_path=str(artifact_path),
        )

    safe_max = max(1, max_chars)
    preview = markdown[:safe_max]
    truncated = len(markdown) > safe_max

    return DocumentPreviewResult(
        status="success",
        message="Loaded Markdown preview.",
        warnings=warnings,
        stored_filename=stored_filename,
        preview=preview,
        artifact_path=str(artifact_path),
        truncated=truncated,
        total_chars=len(markdown),
    )


def get_source_document_info(stored_filename: str) -> SourceDocumentInfoResult:
    warnings: list[str] = []

    normalized = (stored_filename or "").strip()
    if not normalized:
        return SourceDocumentInfoResult(
            status="error",
            message="Stored filename is required for source document info.",
            warnings=warnings,
            stored_filename=None,
            source_path=None,
            source_extension=None,
            source_mime_type=None,
            source_kind=None,
        )

    source_path = UPLOAD_DIR / normalized
    if not source_path.exists():
        return SourceDocumentInfoResult(
            status="warning",
            message="Stored source document was not found.",
            warnings=warnings,
            stored_filename=normalized,
            source_path=str(source_path),
            source_extension=None,
            source_mime_type=None,
            source_kind=None,
        )

    extension = source_path.suffix.lower()
    resolved_mime = _mime_type_for_extension(extension)
    if resolved_mime is None:
        guessed_mime, _ = mimetypes.guess_type(source_path.name)
        resolved_mime = guessed_mime

    if resolved_mime in {"text/plain", "text/markdown", "application/json", "application/xml", "text/xml"}:
        resolved_mime = f"{resolved_mime}; charset=utf-8"

    source_kind = _source_kind_for_extension(extension)
    viewer_source_path = source_path
    viewer_source_extension = extension
    viewer_source_mime_type = resolved_mime
    viewer_source_kind = source_kind

    if source_kind == "3d":
        metadata_path = METADATA_DIR / f"{normalized}.json"
        metadata_payload: dict = {}
        if metadata_path.exists():
            try:
                metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                metadata_payload = {}

            canonical_glb_raw = str((metadata_payload or {}).get("model_3d_canonical_glb_path") or "").strip()
            conversion_status = str((metadata_payload or {}).get("model_3d_conversion_status") or "")
            if canonical_glb_raw:
                canonical_glb_path = Path(canonical_glb_raw)
                if canonical_glb_path.exists() and conversion_status in {"converted_to_glb", "passthrough_glb"} and _is_valid_glb_file(canonical_glb_path):
                    viewer_source_path = canonical_glb_path
                    viewer_source_extension = canonical_glb_path.suffix.lower() or ".glb"
                    viewer_source_mime_type = "model/gltf-binary"
                    viewer_source_kind = "3d"
                elif canonical_glb_path.exists() and conversion_status:
                    warnings.append(
                        "Stored canonical GLB is unavailable for interactive viewing; falling back to original 3D source."
                    )

        if viewer_source_path == source_path:
            canonical_glb_path = VIEWER_ARTIFACTS_DIR / f"{normalized}.canonical.glb"
            if not canonical_glb_path.exists():
                try:
                    _convert_3d_to_canonical_glb(
                        source_path=source_path,
                        canonical_path=canonical_glb_path,
                        extension=extension,
                    )
                except Exception as exc:
                    warnings.append(f"Could not prepare canonical GLB for viewer: {exc}")

            if canonical_glb_path.exists() and _is_valid_glb_file(canonical_glb_path):
                viewer_source_path = canonical_glb_path
                viewer_source_extension = ".glb"
                viewer_source_mime_type = "model/gltf-binary"
                viewer_source_kind = "3d"
                if metadata_path.exists() and isinstance(metadata_payload, dict):
                    metadata_payload["model_3d_canonical_glb_path"] = str(canonical_glb_path)
                    try:
                        metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    except OSError:
                        warnings.append("Could not persist canonical GLB path to metadata.")
            elif canonical_glb_path.exists():
                warnings.append(
                    "Generated canonical GLB was invalid; using original 3D source for viewer loading."
                )

    return SourceDocumentInfoResult(
        status="success",
        message="Loaded source document info.",
        warnings=warnings,
        stored_filename=normalized,
        source_path=str(source_path),
        source_extension=extension,
        source_mime_type=resolved_mime,
        source_kind=source_kind,
        viewer_source_path=str(viewer_source_path),
        viewer_source_extension=viewer_source_extension,
        viewer_source_mime_type=viewer_source_mime_type,
        viewer_source_kind=viewer_source_kind,
        viewer_source_ready=viewer_source_path != source_path,
    )


def convert_stored_upload_to_markdown(stored_filename: str) -> ConversionResult:
    file_path = UPLOAD_DIR / stored_filename
    result = convert_document_to_markdown(file_path)
    if result.status != "success" or result.markdown is None:
        return result

    artifact_path, metadata_path = _store_conversion_artifacts(
        stored_filename,
        result.markdown,
        result.warnings,
    )
    return ConversionResult(
        status=result.status,
        message=result.message,
        markdown=result.markdown,
        warnings=result.warnings,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )


def store_markdown_artifact(
    stored_filename: str,
    markdown: str,
    warnings: list[str] | None = None,
) -> ConversionResult:
    normalized_warnings = list(warnings or [])
    artifact_path, metadata_path = _store_conversion_artifacts(
        stored_filename,
        markdown,
        normalized_warnings,
    )
    return ConversionResult(
        status="success",
        message="Markdown artifact stored.",
        markdown=markdown,
        warnings=normalized_warnings,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )


def _store_conversion_artifacts(
    stored_filename: str,
    markdown: str,
    warnings: list[str],
) -> tuple[str, str]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    artifact_path = ARTIFACTS_DIR / f"{stored_filename}.md"
    artifact_path.write_text(markdown, encoding="utf-8")

    conversion_metadata = {
        "stored_filename": stored_filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact": {"type": "markdown", "path": str(artifact_path)},
        "warnings": warnings,
    }

    source_metadata_path = METADATA_DIR / f"{stored_filename}.json"
    if source_metadata_path.exists():
        conversion_metadata["source_metadata"] = json.loads(
            source_metadata_path.read_text(encoding="utf-8")
        )

    metadata_path = METADATA_DIR / f"{stored_filename}.conversion.json"
    metadata_path.write_text(json.dumps(conversion_metadata, indent=2), encoding="utf-8")

    return str(artifact_path), str(metadata_path)
