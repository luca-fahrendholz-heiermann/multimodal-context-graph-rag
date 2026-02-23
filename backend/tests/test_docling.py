import importlib.util
import json
import sys
import zipfile
from types import ModuleType, SimpleNamespace

from backend.docling_integration import (
    _looks_like_binary_image_metadata_text,
    _looks_like_unreadable_text,
    convert_document_to_markdown,
    convert_stored_upload_to_markdown,
    get_docling_status,
    get_source_document_info,
    store_markdown_artifact,
)


def test_get_docling_status_missing_module():
    status = get_docling_status(module_name="docling_missing_for_test")

    assert status.available is False
    assert "not installed" in status.message


def test_convert_document_to_markdown_with_fake_docling(tmp_path, monkeypatch):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello", encoding="utf-8")

    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "# Hello"

    class FakeConverter:
        def convert(self, path: str) -> SimpleNamespace:
            return SimpleNamespace(document=FakeDocument())

    fake_docling = ModuleType("docling")
    fake_converter_module = ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FakeConverter

    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)

    def fake_find_spec(name: str):
        return object() if name == "docling" else None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    result = convert_document_to_markdown(sample_file)

    assert result.status == "success"
    assert result.markdown == "# Hello"


def test_convert_stored_upload_to_markdown_stores_artifacts(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    metadata_dir = upload_dir / "metadata"
    artifacts_dir = upload_dir / "artifacts"
    upload_dir.mkdir()
    metadata_dir.mkdir()

    stored_filename = "sample.txt"
    sample_file = upload_dir / stored_filename
    sample_file.write_text("hello", encoding="utf-8")

    source_metadata = {"stored_filename": stored_filename, "source_type": "upload"}
    (metadata_dir / f"{stored_filename}.json").write_text(
        json.dumps(source_metadata),
        encoding="utf-8",
    )

    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "# Stored Hello"

    class FakeConverter:
        def convert(self, path: str) -> SimpleNamespace:
            return SimpleNamespace(document=FakeDocument())

    fake_docling = ModuleType("docling")
    fake_converter_module = ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FakeConverter

    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)

    def fake_find_spec(name: str):
        return object() if name == "docling" else None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        "backend.docling_integration.UPLOAD_DIR",
        upload_dir,
    )
    monkeypatch.setattr(
        "backend.docling_integration.METADATA_DIR",
        metadata_dir,
    )
    monkeypatch.setattr(
        "backend.docling_integration.ARTIFACTS_DIR",
        artifacts_dir,
    )

    result = convert_stored_upload_to_markdown(stored_filename)

    assert result.status == "success"
    assert result.artifact_path is not None
    assert result.metadata_path is not None
    artifact_path = artifacts_dir / f"{stored_filename}.md"
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == "# Stored Hello"
    metadata_path = metadata_dir / f"{stored_filename}.conversion.json"
    assert metadata_path.exists()
    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert stored_metadata["stored_filename"] == stored_filename
    assert stored_metadata["source_metadata"] == source_metadata


def test_store_markdown_artifact_writes_files(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    metadata_dir = upload_dir / "metadata"
    artifacts_dir = upload_dir / "artifacts"
    upload_dir.mkdir()
    metadata_dir.mkdir()

    stored_filename = "email_sample.txt"
    source_metadata = {"stored_filename": stored_filename, "source_type": "email-compose"}
    (metadata_dir / f"{stored_filename}.json").write_text(json.dumps(source_metadata), encoding="utf-8")

    monkeypatch.setattr("backend.docling_integration.UPLOAD_DIR", upload_dir)
    monkeypatch.setattr("backend.docling_integration.METADATA_DIR", metadata_dir)
    monkeypatch.setattr("backend.docling_integration.ARTIFACTS_DIR", artifacts_dir)

    result = store_markdown_artifact(stored_filename, "# Email content")

    assert result.status == "success"
    assert (artifacts_dir / f"{stored_filename}.md").exists()
    metadata_path = metadata_dir / f"{stored_filename}.conversion.json"
    assert metadata_path.exists()
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved["source_metadata"] == source_metadata


def test_convert_document_to_markdown_docx_fallback_without_docling(tmp_path, monkeypatch):
    docx_path = tmp_path / "fallback.docx"
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>Erfahrung in Python und FastAPI</w:t></w:r></w:p>
                <w:p><w:r><w:t>Kenntnisse in RAG und Embeddings</w:t></w:r></w:p>
              </w:body>
            </w:document>
            """,
        )

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    result = convert_document_to_markdown(docx_path)

    assert result.status == "success"
    assert "fallback" in result.message.lower()
    assert "Erfahrung in Python" in (result.markdown or "")


def test_convert_document_to_markdown_pdf_without_docling_returns_error(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    result = convert_document_to_markdown(pdf_path)

    assert result.status == "error"
    assert "no fallback converter" in result.message.lower()


def test_convert_document_to_markdown_docling_failure_uses_txt_fallback(tmp_path, monkeypatch):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("fallback text", encoding="utf-8")

    class FailingConverter:
        def convert(self, path: str):
            raise RuntimeError("docling exploded")

    fake_docling = ModuleType("docling")
    fake_converter_module = ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FailingConverter

    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "docling" else None)

    result = convert_document_to_markdown(sample_file)

    assert result.status == "success"
    assert "fallback" in result.message.lower()
    assert result.markdown == "fallback text"
    assert any("docling conversion failed" in warning.lower() for warning in result.warnings)


def test_looks_like_unreadable_text_detects_pdf_binary_garbage():
    unreadable = "%PDF-1.6 xref /Type /Catalog \x00\x01���"
    assert _looks_like_unreadable_text(unreadable) is True


def test_convert_document_to_markdown_uses_pdf_fallback_when_docling_output_is_unreadable(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "%PDF-1.6 xref /Type /Catalog \x00\x01���"

    class FakeConverter:
        def convert(self, path: str) -> SimpleNamespace:
            return SimpleNamespace(document=FakeDocument())

    fake_docling = ModuleType("docling")
    fake_converter_module = ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FakeConverter

    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)

    def fake_find_spec(name: str):
        if name in {"docling", "pypdf"}:
            return object()
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    class FakePage:
        def extract_text(self):
            return "Senior Applied ML Engineer"

    class FakePdfReader:
        def __init__(self, _: str):
            self.pages = [FakePage()]

    fake_pypdf = ModuleType("pypdf")
    fake_pypdf.PdfReader = FakePdfReader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    result = convert_document_to_markdown(pdf_path)

    assert result.status == "success"
    assert "quality fallback" in result.message.lower()
    assert result.markdown == "Senior Applied ML Engineer"
    assert any("unreadable" in warning.lower() for warning in result.warnings)


def test_convert_document_to_markdown_pptx_fallback_without_docling(tmp_path, monkeypatch):
    pptx_path = tmp_path / "slides.pptx"
    with zipfile.ZipFile(pptx_path, "w") as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            """<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Roadmap Q4</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>""",
        )

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    result = convert_document_to_markdown(pptx_path)

    assert result.status == "success"
    assert "pptx" in result.message.lower()
    assert "Roadmap Q4" in (result.markdown or "")


def test_looks_like_binary_image_metadata_text_detects_jpeg_header_noise():
    noisy = "\xff\xd8\xff\xe0 JFIF EXIF Adobe ICC_PROFILE DQT DHT SOI EOI"
    assert _looks_like_binary_image_metadata_text(noisy) is True




def test_looks_like_binary_image_metadata_text_detects_base64_payload_noise():
    noisy = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 3
    assert _looks_like_binary_image_metadata_text(noisy) is True

def test_convert_document_to_markdown_prefers_image_ocr_fallback_for_images(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "drawing.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0fake")

    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "\x00\x10JFIF\x00\x01\x01 EXIF ICC_PROFILE DQT DHT SOI EOI"

    class FakeConverter:
        def convert(self, path: str) -> SimpleNamespace:
            return SimpleNamespace(document=FakeDocument())

    fake_docling = ModuleType("docling")
    fake_converter_module = ModuleType("docling.document_converter")
    fake_converter_module.DocumentConverter = FakeConverter

    monkeypatch.setitem(sys.modules, "docling", fake_docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_converter_module)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name in {"docling", "PIL", "pytesseract"} else None)

    fake_pil = ModuleType("PIL")

    class _FakeImageContext:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeImageModule:
        @staticmethod
        def open(path):
            return _FakeImageContext()

    fake_pil.Image = FakeImageModule
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    fake_tesseract = ModuleType("pytesseract")
    fake_tesseract.image_to_string = lambda _image: "Maschinenbauzeichnung mit Stückliste"
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract)

    result = convert_document_to_markdown(image_path)

    assert result.status == "success"
    assert "ocr image fallback" in result.message.lower()
    assert result.markdown == "Maschinenbauzeichnung mit Stückliste"


def test_convert_document_to_markdown_skips_docling_for_3d_models(tmp_path, monkeypatch):
    model_path = tmp_path / "part.obj"
    model_path.write_text("o part\nv 0 0 0\n", encoding="utf-8")

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "docling" else None)

    result = convert_document_to_markdown(model_path)

    assert result.status == "success"
    assert "3d model" in result.message.lower()
    assert "3d viewer" in (result.markdown or "").lower()
    assert any("skipped docling conversion" in warning.lower() for warning in result.warnings)


def test_get_source_document_info_falls_back_to_obj_when_canonical_glb_is_invalid(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    metadata_dir = upload_dir / "metadata"
    viewer_artifacts_dir = upload_dir / "viewer-artifacts"
    upload_dir.mkdir()
    metadata_dir.mkdir()
    viewer_artifacts_dir.mkdir()

    stored_filename = "beam.obj"
    (upload_dir / stored_filename).write_text("o beam\nv 0 0 0\n", encoding="utf-8")

    canonical_path = viewer_artifacts_dir / f"{stored_filename}.canonical.glb"
    canonical_path.write_bytes(b"GLB conversion failed")

    (metadata_dir / f"{stored_filename}.json").write_text(
        json.dumps(
            {
                "stored_filename": stored_filename,
                "model_3d_canonical_glb_path": str(canonical_path),
                "model_3d_conversion_status": "conversion_failed",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("backend.docling_integration.UPLOAD_DIR", upload_dir)
    monkeypatch.setattr("backend.docling_integration.METADATA_DIR", metadata_dir)
    monkeypatch.setattr("backend.docling_integration.VIEWER_ARTIFACTS_DIR", viewer_artifacts_dir)

    result = get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_extension == ".obj"
    assert result.viewer_source_extension == ".obj"
    assert result.viewer_source_path == str(upload_dir / stored_filename)
    assert any("falling back to original 3d source" in warning.lower() for warning in result.warnings)


def test_get_source_document_info_uses_valid_converted_glb(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    metadata_dir = upload_dir / "metadata"
    viewer_artifacts_dir = upload_dir / "viewer-artifacts"
    upload_dir.mkdir()
    metadata_dir.mkdir()
    viewer_artifacts_dir.mkdir()

    stored_filename = "beam.obj"
    (upload_dir / stored_filename).write_text("o beam\nv 0 0 0\n", encoding="utf-8")

    canonical_path = viewer_artifacts_dir / f"{stored_filename}.canonical.glb"
    canonical_path.write_bytes(b"glTF" + b"\x02\x00\x00\x00" + b"\x00" * 12)

    (metadata_dir / f"{stored_filename}.json").write_text(
        json.dumps(
            {
                "stored_filename": stored_filename,
                "model_3d_canonical_glb_path": str(canonical_path),
                "model_3d_conversion_status": "converted_to_glb",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("backend.docling_integration.UPLOAD_DIR", upload_dir)
    monkeypatch.setattr("backend.docling_integration.METADATA_DIR", metadata_dir)
    monkeypatch.setattr("backend.docling_integration.VIEWER_ARTIFACTS_DIR", viewer_artifacts_dir)

    result = get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.viewer_source_extension == ".glb"
    assert result.viewer_source_path == str(canonical_path)

