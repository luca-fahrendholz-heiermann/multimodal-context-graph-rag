from pathlib import Path
import io
import zipfile
from backend import docling_integration
import json


def _build_sample_xlsx_bytes() -> bytes:
    workbook_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
              xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
    </workbook>
    """
    rels_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
    </Relationships>
    """
    shared_strings_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="4" uniqueCount="4">
      <si><t>id</t></si><si><t>name</t></si><si><t>Alice</t></si><si><t>Bob</t></si>
    </sst>
    """
    sheet_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <sheetData>
        <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
        <row r="2"><c r="A2"><v>1</v></c><c r="B2" t="s"><v>2</v></c></row>
        <row r="3"><c r="A3"><v>2</v></c><c r="B3" t="s"><v>3</v></c></row>
      </sheetData>
    </worksheet>
    """

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        archive.writestr("xl/sharedStrings.xml", shared_strings_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return payload.getvalue()


def test_get_document_preview_success_and_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "ARTIFACTS_DIR", tmp_path)
    stored_filename = "sample.txt"
    artifact_path = tmp_path / f"{stored_filename}.md"
    artifact_path.write_text("Hello world", encoding="utf-8")

    result = docling_integration.get_document_preview(stored_filename)

    assert result.status == "success"
    assert result.preview == "Hello world"
    assert result.truncated is False
    assert result.total_chars == 11

    truncated = docling_integration.get_document_preview(stored_filename, max_chars=5)

    assert truncated.preview == "Hello"
    assert truncated.truncated is True
    assert truncated.total_chars == 11


def test_get_document_preview_missing_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "ARTIFACTS_DIR", tmp_path)

    result = docling_integration.get_document_preview("missing.txt")

    assert result.status == "warning"


def test_get_source_document_info_for_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    stored_filename = "resume.pdf"
    (tmp_path / stored_filename).write_bytes(b"%PDF-1.4")

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_extension == ".pdf"
    assert result.source_kind == "pdf"
    assert result.source_mime_type == "application/pdf"


def test_get_source_document_info_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)

    result = docling_integration.get_source_document_info("missing.pdf")

    assert result.status == "warning"


def test_get_source_document_info_uses_mimetypes_guess_for_json(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    stored_filename = "payload.json"
    (tmp_path / stored_filename).write_text('{"ok": true}', encoding="utf-8")

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_mime_type == "application/json; charset=utf-8"


def test_get_source_document_info_sets_utf8_charset_for_text_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    stored_filename = "note.md"
    (tmp_path / stored_filename).write_text("# Hallo", encoding="utf-8")

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_mime_type == "text/markdown; charset=utf-8"


def test_get_source_document_info_for_obj_is_3d(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    stored_filename = "sample.obj"
    (tmp_path / stored_filename).write_text("o cube\nv 0 0 0\n", encoding="utf-8")

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "3d"
    assert result.source_mime_type == "model/obj"


def test_get_source_document_info_for_ply_is_3d(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    stored_filename = "sample.ply"
    (tmp_path / stored_filename).write_text("ply\nformat ascii 1.0\n", encoding="utf-8")

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "3d"
    assert result.source_mime_type == "application/ply"


def test_get_source_document_info_prefers_canonical_glb_for_3d_viewer(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "METADATA_DIR", tmp_path)

    stored_filename = "sample.obj"
    source_path = tmp_path / stored_filename
    source_path.write_text("o cube\nv 0 0 0\n", encoding="utf-8")

    canonical_path = tmp_path / f"{stored_filename}.canonical.glb"
    canonical_path.write_bytes(b"glTF")
    (tmp_path / f"{stored_filename}.json").write_text(
        f'{{"model_3d_canonical_glb_path": "{canonical_path}"}}',
        encoding="utf-8",
    )

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.viewer_source_path == str(canonical_path)
    assert result.viewer_source_extension == ".glb"
    assert result.viewer_source_mime_type == "model/gltf-binary"
    assert result.viewer_source_ready is True


def test_get_source_document_info_keeps_original_when_canonical_glb_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "METADATA_DIR", tmp_path)

    stored_filename = "sample.obj"
    source_path = tmp_path / stored_filename
    source_path.write_text("o cube\nv 0 0 0\n", encoding="utf-8")

    (tmp_path / f"{stored_filename}.json").write_text(
        '{"model_3d_canonical_glb_path": "/tmp/does-not-exist.glb"}',
        encoding="utf-8",
    )

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.viewer_source_path == str(source_path)
    assert result.viewer_source_extension == ".obj"
    assert result.viewer_source_ready is False


def test_get_source_document_info_generates_canonical_glb_on_demand(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "METADATA_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "VIEWER_ARTIFACTS_DIR", tmp_path)

    stored_filename = "sample.obj"
    source_path = tmp_path / stored_filename
    source_path.write_text("o cube\nv 0 0 0\n", encoding="utf-8")

    canonical_path = tmp_path / f"{stored_filename}.canonical.glb"

    def _fake_convert(*, source_path, canonical_path, extension):
        canonical_path.write_bytes(b"glTF")
        return "converted_to_glb", [], None

    monkeypatch.setattr(docling_integration, "_convert_3d_to_canonical_glb", _fake_convert)

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.viewer_source_path == str(canonical_path)
    assert result.viewer_source_extension == ".glb"
    assert result.viewer_source_mime_type == "model/gltf-binary"
    assert result.viewer_source_ready is True


def test_get_source_document_info_for_xlsx_uses_table_viewer_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "METADATA_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "VIEWER_ARTIFACTS_DIR", tmp_path / "viewer")

    stored_filename = "orders.xlsx"
    (tmp_path / stored_filename).write_bytes(b"PK\x03\x04")
    parsed_path = tmp_path / f"{stored_filename}.parsed.json"
    parsed_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "sheet_name": "Sheet1",
                        "header": ["id", "name"],
                        "rows": [["1", "Alice"], ["2", "Bob"]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{stored_filename}.json").write_text(
        json.dumps({"parsed_artifact_path": str(parsed_path)}),
        encoding="utf-8",
    )

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "table"
    assert result.viewer_source_extension == ".html"
    assert result.viewer_source_kind == "table"
    assert result.viewer_source_path is not None
    assert "table-viewer.html" in result.viewer_source_path


def test_get_source_document_info_for_txt_is_text_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    stored_filename = "notes.txt"
    (tmp_path / stored_filename).write_text("hello", encoding="utf-8")

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "text"


def test_get_source_document_info_for_eml_creates_email_text_viewer(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "METADATA_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "VIEWER_ARTIFACTS_DIR", tmp_path / "viewer")

    stored_filename = "mail.eml"
    (tmp_path / stored_filename).write_text(
        "From: sender@example.com\nTo: inbox@example.com\nSubject: Hallo\nDate: Tue, 01 Jan 2026 10:00:00 +0000\n\nBody text",
        encoding="utf-8",
    )

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "email"
    assert result.viewer_source_extension == ".txt"
    assert result.viewer_source_mime_type == "text/plain; charset=utf-8"
    assert result.viewer_source_path is not None
    rendered = Path(result.viewer_source_path).read_text(encoding="utf-8")
    assert "From: sender@example.com" in rendered
    assert "Subject: Hallo" in rendered


def test_get_source_document_info_for_xml_creates_collapsible_html_viewer(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "VIEWER_ARTIFACTS_DIR", tmp_path / "viewer")

    stored_filename = "payload.xml"
    (tmp_path / stored_filename).write_text(
        "<root><invoice id=\"42\"><amount currency=\"EUR\">100</amount></invoice></root>",
        encoding="utf-8",
    )

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "xml"
    assert result.viewer_source_extension == ".html"
    assert result.viewer_source_kind == "xml"
    assert result.viewer_source_path is not None
    rendered = Path(result.viewer_source_path).read_text(encoding="utf-8")
    assert "<details class='xml-node'>" in rendered
    assert "&lt;invoice id=\"42\"&gt;" in rendered


def test_get_source_document_info_for_xlsx_builds_viewer_without_metadata_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "METADATA_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "VIEWER_ARTIFACTS_DIR", tmp_path / "viewer")

    stored_filename = "fallback.xlsx"
    (tmp_path / stored_filename).write_bytes(_build_sample_xlsx_bytes())

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "table"
    assert result.viewer_source_extension == ".html"
    assert result.viewer_source_kind == "table"
    assert result.viewer_source_path is not None
    rendered = Path(result.viewer_source_path).read_text(encoding="utf-8")
    assert "<table class='rag-db-table'>" in rendered
    assert "Sheet1" in rendered



def test_get_source_document_info_for_sample_xlsx_renders_viewer_from_raw_file(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "METADATA_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "VIEWER_ARTIFACTS_DIR", tmp_path / "viewer")

    sample_path = Path(__file__).resolve().parents[2] / "data" / "sample_documents" / "commercial" / "02_kostenkalkulation.xlsx"
    stored_filename = sample_path.name
    (tmp_path / stored_filename).write_bytes(sample_path.read_bytes())

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "table"
    assert result.viewer_source_extension == ".html"
    assert result.viewer_source_kind == "table"
    assert result.viewer_source_path is not None
    rendered = Path(result.viewer_source_path).read_text(encoding="utf-8")
    assert "rag-db-table" in rendered

def test_get_source_document_info_for_dxf_creates_html_viewer(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "VIEWER_ARTIFACTS_DIR", tmp_path / "viewer")

    stored_filename = "plan.dxf"
    (tmp_path / stored_filename).write_text(
        "\n".join(
            [
                "0", "SECTION", "2", "ENTITIES",
                "0", "LINE", "8", "WALL", "10", "0", "20", "0", "11", "100", "21", "0",
                "0", "CIRCLE", "8", "WALL", "10", "50", "20", "50", "40", "10",
                "0", "LWPOLYLINE", "8", "WALL", "70", "1", "10", "10", "20", "10", "10", "20", "20", "20",
                "0", "ENDSEC", "0", "EOF",
            ]
        ),
        encoding="utf-8",
    )

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "dxf"
    assert result.viewer_source_kind == "dxf"
    assert result.viewer_source_extension == ".html"
    assert result.viewer_source_path is not None
    rendered = Path(result.viewer_source_path).read_text(encoding="utf-8")
    assert "<svg" in rendered
    assert "LINE 1" in rendered


def test_get_source_document_info_for_xlsx_falls_back_when_parsed_artifact_has_no_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_integration, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "METADATA_DIR", tmp_path)
    monkeypatch.setattr(docling_integration, "VIEWER_ARTIFACTS_DIR", tmp_path / "viewer")

    stored_filename = "fallback_empty_tables.xlsx"
    (tmp_path / stored_filename).write_bytes(_build_sample_xlsx_bytes())

    parsed_path = tmp_path / f"{stored_filename}.parsed.json"
    parsed_path.write_text(json.dumps({"tables": []}), encoding="utf-8")
    (tmp_path / f"{stored_filename}.json").write_text(
        json.dumps({"parsed_artifact_path": str(parsed_path)}),
        encoding="utf-8",
    )

    result = docling_integration.get_source_document_info(stored_filename)

    assert result.status == "success"
    assert result.source_kind == "table"
    assert result.viewer_source_extension == ".html"
    assert result.viewer_source_kind == "table"
    assert result.viewer_source_path is not None
    rendered = Path(result.viewer_source_path).read_text(encoding="utf-8")
    assert "<table class='rag-db-table'>" in rendered
    assert "Sheet1" in rendered
