import io
import json
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from backend import ingestion
from backend.ingestion import ingest_smtp_inbox, ingest_watch_folder, validate_upload


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


class TestIngestionValidation(unittest.TestCase):
    def test_empty_upload_rejected(self):
        result = validate_upload("sample.pdf", b"")
        self.assertEqual(result.status, "error")
        self.assertIn("Empty uploads", result.message)

    def test_unsupported_extension_warned(self):
        result = validate_upload("sample.exe", b"data")
        self.assertEqual(result.status, "warning")
        self.assertTrue(result.warnings)

    def test_supported_extension_success(self):
        result = validate_upload("sample.pdf", b"data")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.warnings, [])

    def test_svg_extension_success(self):
        result = validate_upload("diagram.svg", b"<svg></svg>")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.warnings, [])

    def test_xlsx_extension_success(self):
        result = validate_upload("report.xlsx", b"xlsx-bytes")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.warnings, [])

    def test_eml_extension_success(self):
        result = validate_upload("mail.eml", b"From: a@example.com\nTo: b@example.com\n\nHello")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.warnings, [])

    def test_dxf_extension_success(self):
        result = validate_upload("drawing.dxf", b"0\nSECTION\n2\nENTITIES\n0\nEOF")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.warnings, [])


class TestWatchFolderIngestion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.watch_dir = base / "watch"
        self.upload_dir = base / "uploads"
        self.metadata_dir = self.upload_dir / "metadata"
        self.processed_dir = self.watch_dir / "processed"
        self.rejected_dir = self.watch_dir / "rejected"
        self.artifacts_dir = self.upload_dir / "artifacts"
        self.parsed_dir = self.artifacts_dir / "parsed"
        self.normalized_dir = self.artifacts_dir / "normalized"
        self.viewer_dir = self.artifacts_dir / "viewer"

        self.original_dirs = (
            ingestion.WATCH_DIR,
            ingestion.WATCH_PROCESSED_DIR,
            ingestion.WATCH_REJECTED_DIR,
            ingestion.UPLOAD_DIR,
            ingestion.METADATA_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.PARSED_DIR,
            ingestion.NORMALIZED_DIR,
            ingestion.VIEWER_ARTIFACTS_DIR,
            ingestion.DEAD_LETTER_DIR,
            ingestion.DEAD_LETTER_QUEUE_PATH,
            ingestion.IMAGE_DESCRIPTIONS_DIR,
            ingestion.OBSERVABILITY_LOG_PATH,
            ingestion.OBSERVABILITY_METRICS_PATH,
        )

        ingestion.WATCH_DIR = self.watch_dir
        ingestion.WATCH_PROCESSED_DIR = self.processed_dir
        ingestion.WATCH_REJECTED_DIR = self.rejected_dir
        ingestion.UPLOAD_DIR = self.upload_dir
        ingestion.METADATA_DIR = self.metadata_dir
        ingestion.ARTIFACTS_DIR = self.artifacts_dir
        ingestion.PARSED_DIR = self.parsed_dir
        ingestion.NORMALIZED_DIR = self.normalized_dir
        ingestion.VIEWER_ARTIFACTS_DIR = self.viewer_dir
        ingestion.DEAD_LETTER_DIR = self.artifacts_dir / "dead_letter"
        ingestion.DEAD_LETTER_QUEUE_PATH = ingestion.DEAD_LETTER_DIR / "queue.jsonl"
        ingestion.OBSERVABILITY_LOG_PATH = self.artifacts_dir / 'processing_logs.jsonl'
        ingestion.OBSERVABILITY_METRICS_PATH = self.artifacts_dir / 'processing_metrics.json'

        self.watch_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        (
            ingestion.WATCH_DIR,
            ingestion.WATCH_PROCESSED_DIR,
            ingestion.WATCH_REJECTED_DIR,
            ingestion.UPLOAD_DIR,
            ingestion.METADATA_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.PARSED_DIR,
            ingestion.NORMALIZED_DIR,
            ingestion.VIEWER_ARTIFACTS_DIR,
            ingestion.DEAD_LETTER_DIR,
            ingestion.DEAD_LETTER_QUEUE_PATH,
            ingestion.IMAGE_DESCRIPTIONS_DIR,
            ingestion.OBSERVABILITY_LOG_PATH,
            ingestion.OBSERVABILITY_METRICS_PATH,
        ) = self.original_dirs
        self.temp_dir.cleanup()

    def test_watch_folder_moves_success_and_rejected(self):
        (self.watch_dir / "good.txt").write_text("hello", encoding="utf-8")
        (self.watch_dir / "bad.exe").write_text("nope", encoding="utf-8")

        result = ingest_watch_folder()

        self.assertEqual(result["status"], "warning")
        self.assertEqual(len(result["files"]), 2)
        self.assertTrue((self.processed_dir / "good.txt").exists())
        self.assertTrue((self.rejected_dir / "bad.exe").exists())
        self.assertTrue(any(self.upload_dir.iterdir()))
        self.assertTrue(any(self.metadata_dir.iterdir()))

    def test_watch_folder_rejects_empty_files(self):
        (self.watch_dir / "empty.txt").write_bytes(b"")

        result = ingest_watch_folder()

        self.assertEqual(result["files"][0]["status"], "error")
        self.assertTrue((self.rejected_dir / "empty.txt").exists())

    def test_watch_folder_reports_warning_status(self):
        (self.watch_dir / "bad.exe").write_text("nope", encoding="utf-8")

        result = ingest_watch_folder()

        self.assertEqual(result["status"], "warning")
        self.assertTrue(result["files"][0]["warnings"])


class TestSmtpIngestion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.upload_dir = base / "uploads"
        self.metadata_dir = self.upload_dir / "metadata"
        self.artifacts_dir = self.upload_dir / "artifacts"
        self.parsed_dir = self.artifacts_dir / "parsed"
        self.normalized_dir = self.artifacts_dir / "normalized"
        self.viewer_dir = self.artifacts_dir / "viewer"

        self.original_dirs = (
            ingestion.UPLOAD_DIR,
            ingestion.METADATA_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.PARSED_DIR,
            ingestion.NORMALIZED_DIR,
            ingestion.VIEWER_ARTIFACTS_DIR,
            ingestion.DEAD_LETTER_DIR,
            ingestion.DEAD_LETTER_QUEUE_PATH,
            ingestion.IMAGE_DESCRIPTIONS_DIR,
            ingestion.OBSERVABILITY_LOG_PATH,
            ingestion.OBSERVABILITY_METRICS_PATH,
        )

        ingestion.UPLOAD_DIR = self.upload_dir
        ingestion.METADATA_DIR = self.metadata_dir
        ingestion.ARTIFACTS_DIR = self.artifacts_dir
        ingestion.PARSED_DIR = self.parsed_dir
        ingestion.NORMALIZED_DIR = self.normalized_dir
        ingestion.VIEWER_ARTIFACTS_DIR = self.viewer_dir
        ingestion.DEAD_LETTER_DIR = self.artifacts_dir / "dead_letter"
        ingestion.DEAD_LETTER_QUEUE_PATH = ingestion.DEAD_LETTER_DIR / "queue.jsonl"
        ingestion.OBSERVABILITY_LOG_PATH = self.artifacts_dir / 'processing_logs.jsonl'
        ingestion.OBSERVABILITY_METRICS_PATH = self.artifacts_dir / 'processing_metrics.json'

    def tearDown(self):
        (
            ingestion.UPLOAD_DIR,
            ingestion.METADATA_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.PARSED_DIR,
            ingestion.NORMALIZED_DIR,
            ingestion.VIEWER_ARTIFACTS_DIR,
            ingestion.DEAD_LETTER_DIR,
            ingestion.DEAD_LETTER_QUEUE_PATH,
            ingestion.IMAGE_DESCRIPTIONS_DIR,
            ingestion.OBSERVABILITY_LOG_PATH,
            ingestion.OBSERVABILITY_METRICS_PATH,
        ) = self.original_dirs
        self.temp_dir.cleanup()

    def test_smtp_ingestion_stores_email_body(self):
        messages = [{"ID": "abc123"}]
        details = {
            "ID": "abc123",
            "Subject": "Hello",
            "From": {"Address": "sender@example.com"},
            "Text": "Hi there",
        }

        with patch(
            "backend.ingestion._fetch_mailpit_messages", return_value=messages
        ), patch("backend.ingestion._fetch_mailpit_message", return_value=details):
            result = ingest_smtp_inbox(api_url="http://mailpit", provider="mailpit")

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["files"]), 1)
        self.assertTrue(any(self.upload_dir.iterdir()))
        self.assertTrue(any(self.metadata_dir.iterdir()))
        self.assertEqual(result["files"][0]["metadata"]["sender"], "sender@example.com")

    def test_smtp_ingestion_reports_warning_status(self):
        messages = [{"ID": "abc123"}]
        details = {
            "ID": "abc123",
            "Subject": "Hello",
            "From": {"Address": "sender@example.com"},
            "Text": "",
            "HTML": "<p>Only HTML</p>",
        }

        with patch(
            "backend.ingestion._fetch_mailpit_messages", return_value=messages
        ), patch("backend.ingestion._fetch_mailpit_message", return_value=details):
            result = ingest_smtp_inbox(api_url="http://mailpit", provider="mailpit")

        self.assertEqual(result["status"], "warning")
        self.assertTrue(result["files"][0]["warnings"])


class TestDeadLetterQueue(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.upload_dir = base / "uploads"
        self.metadata_dir = self.upload_dir / "metadata"
        self.artifacts_dir = self.upload_dir / "artifacts"
        self.parsed_dir = self.artifacts_dir / "parsed"
        self.normalized_dir = self.artifacts_dir / "normalized"
        self.viewer_dir = self.artifacts_dir / "viewer"
        self.dead_letter_dir = self.artifacts_dir / "dead_letter"
        self.image_descriptions_dir = self.upload_dir / "img_descriptions"

        self.original_dirs = (
            ingestion.UPLOAD_DIR,
            ingestion.METADATA_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.PARSED_DIR,
            ingestion.NORMALIZED_DIR,
            ingestion.VIEWER_ARTIFACTS_DIR,
            ingestion.DEAD_LETTER_DIR,
            ingestion.DEAD_LETTER_QUEUE_PATH,
            ingestion.IMAGE_DESCRIPTIONS_DIR,
            ingestion.OBSERVABILITY_LOG_PATH,
            ingestion.OBSERVABILITY_METRICS_PATH,
        )

        ingestion.UPLOAD_DIR = self.upload_dir
        ingestion.METADATA_DIR = self.metadata_dir
        ingestion.ARTIFACTS_DIR = self.artifacts_dir
        ingestion.PARSED_DIR = self.parsed_dir
        ingestion.NORMALIZED_DIR = self.normalized_dir
        ingestion.VIEWER_ARTIFACTS_DIR = self.viewer_dir
        ingestion.DEAD_LETTER_DIR = self.dead_letter_dir
        ingestion.DEAD_LETTER_QUEUE_PATH = self.dead_letter_dir / "queue.jsonl"
        ingestion.IMAGE_DESCRIPTIONS_DIR = self.image_descriptions_dir
        ingestion.OBSERVABILITY_LOG_PATH = self.artifacts_dir / "processing_logs.jsonl"
        ingestion.OBSERVABILITY_METRICS_PATH = self.artifacts_dir / "processing_metrics.json"

    def tearDown(self):
        (
            ingestion.UPLOAD_DIR,
            ingestion.METADATA_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.PARSED_DIR,
            ingestion.NORMALIZED_DIR,
            ingestion.VIEWER_ARTIFACTS_DIR,
            ingestion.DEAD_LETTER_DIR,
            ingestion.DEAD_LETTER_QUEUE_PATH,
            ingestion.IMAGE_DESCRIPTIONS_DIR,
            ingestion.OBSERVABILITY_LOG_PATH,
            ingestion.OBSERVABILITY_METRICS_PATH,
        ) = self.original_dirs
        self.temp_dir.cleanup()

    def test_store_upload_writes_dead_letter_report_on_failure(self):
        with patch("backend.ingestion.parse_structured_document", side_effect=ValueError("broken parser")):
            with self.assertRaises(RuntimeError) as ctx:
                ingestion.store_upload("broken.txt", b"payload", "text/plain")

        self.assertIn("dead-letter queue", str(ctx.exception))
        reports = list(self.dead_letter_dir.glob("*.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["filename"], "broken.txt")
        self.assertEqual(report["error"]["type"], "ValueError")
        self.assertEqual(report["error"]["stage"], "store_upload")
        self.assertTrue(report["error"]["fingerprint"])
        self.assertTrue((self.dead_letter_dir / "queue.jsonl").exists())


    def test_store_upload_image_continues_when_image_pipeline_fails(self):
        with patch("backend.ingestion.process_image_for_search", side_effect=RuntimeError("broken image pipeline")):
            record = ingestion.store_upload(
                filename="scan.png",
                content_bytes=b"\x89PNG\r\n\x1a\n" + b"demo",
                content_type="image/png",
                source_type="upload",
            )

        warnings = record.get("image_pipeline_warnings") or []
        self.assertTrue(warnings)
        self.assertIn("Image search pipeline failed and was skipped.", warnings[0])
        self.assertIn("image_pipeline_error=broken image pipeline", warnings[1])
        self.assertTrue(record.get("normalized_artifact_path"))

    def test_store_upload_image_passes_llm_description_into_image_pipeline_vector_doc(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="image_ocr_parser",
            text="",
            layout=[],
            tables=[],
            media=[],
            object_structure={"ocr": {"description": {}}},
            warnings=[],
        )

        captured = {}

        def _fake_image_pipeline(**kwargs):
            raw, warnings = kwargs["vision_callable"](b"img", "prompt")
            captured["raw"] = raw
            captured["warnings"] = warnings
            return {
                "blob_uri": "blob://data/uploads/blobs/demo.bin",
                "vector_document": json.loads(raw),
                "embeddings_text": "Ein Stahlbau mit Kran.\nimage llm-analysis openai",
                "warnings": [],
            }

        analysis_result = SimpleNamespace(
            combined_text="Ein Stahlbau mit Kran.",
            description_text="Ein Stahlbau mit Kran.",
            analysis_text="Aufnahme zeigt Baustellenfortschritt.",
            open_questions=["Welches Baujahr?"],
            warnings=[],
            meta={"provider": "openai", "enabled": True},
        )

        with patch("backend.ingestion.parse_structured_document", return_value=parsed_doc):
            with patch("backend.ingestion.analyze_image_bytes_with_provider", return_value=analysis_result):
                with patch("backend.ingestion.process_image_for_search", side_effect=_fake_image_pipeline):
                    record = ingestion.store_upload(
                        filename="scan.png",
                        content_bytes=b"\x89PNG\r\n\x1a\n" + b"demo",
                        content_type="image/png",
                        source_type="upload",
                    )

        vector_doc = record.get("image_vector_document") or {}
        self.assertIn("Stahlbau", vector_doc.get("caption", ""))
        self.assertEqual((vector_doc.get("metadata") or {}).get("source"), "image_description_pipeline")
        self.assertIn("llm-analysis", vector_doc.get("tags") or [])
        self.assertTrue(captured.get("raw"))
        self.assertTrue((self.image_descriptions_dir / f"{record['stored_filename']}.txt").exists())


class TestProcessorRouting(unittest.TestCase):
    def test_route_pdf_via_magic_bytes(self):
        route = ingestion.select_processor_route(
            mime="application/octet-stream",
            ext=".bin",
            magic_bytes="application/pdf",
        )
        self.assertEqual(route, "pdf_processor")

    def test_route_docx_via_extension(self):
        route = ingestion.select_processor_route(
            mime="application/octet-stream",
            ext=".docx",
            magic_bytes=None,
        )
        self.assertEqual(route, "docx_processor")

    def test_route_unknown_fallback(self):
        route = ingestion.select_processor_route(
            mime="application/x-custom",
            ext=".custom",
            magic_bytes=None,
        )
        self.assertEqual(route, "generic_binary_processor")

    def test_route_svg_via_extension(self):
        route = ingestion.select_processor_route(
            mime="application/octet-stream",
            ext=".svg",
            magic_bytes=None,
        )
        self.assertEqual(route, "image_processor")

    def test_route_prefers_optional_converter_plugin(self):
        plugin = ingestion.OptionalConverterPlugin(
            name="legacy_dwg_plugin",
            plugin_id="legacy-dwg",
            route_name="legacy_dwg_processor",
            priority=999,
            mime_types={"application/acad"},
            extensions={".dwg"},
            magic_types=set(),
            license_name="Commercial",
            tooling="external-cad-suite",
            converter=lambda **_: ingestion.ParsedDoc(
                parser="legacy_dwg_parser",
                text="",
                layout=[],
                tables=[],
                media=[],
                object_structure={"format": "dwg"},
                warnings=[],
            ),
        )
        original_plugins = ingestion.OPTIONAL_CONVERTER_PLUGINS
        try:
            ingestion.OPTIONAL_CONVERTER_PLUGINS = (plugin,)
            route = ingestion.select_processor_route(
                mime="application/octet-stream",
                ext=".dwg",
                magic_bytes=None,
            )
        finally:
            ingestion.OPTIONAL_CONVERTER_PLUGINS = original_plugins

        self.assertEqual(route, "legacy_dwg_processor")


class TestProcessorRegistryScoring(unittest.TestCase):
    def test_registry_prefers_magic_bytes_match_over_extension_only(self):
        registry = ingestion.ProcessorRegistry(
            routes=(
                ingestion.ProcessorRoute(
                    name="ext_route",
                    priority=999,
                    mime_types=set(),
                    extensions={".pdf"},
                    magic_types=set(),
                ),
                ingestion.ProcessorRoute(
                    name="magic_route",
                    priority=1,
                    mime_types=set(),
                    extensions=set(),
                    magic_types={"application/pdf"},
                ),
            )
        )

        selected = registry.select_route(
            mime="application/octet-stream",
            ext=".pdf",
            magic_bytes="application/pdf",
        )

        self.assertEqual(selected, "magic_route")

    def test_capability_score_uses_priority_as_tiebreaker(self):
        registry = ingestion.ProcessorRegistry(
            routes=(
                ingestion.ProcessorRoute(
                    name="lower_priority",
                    priority=10,
                    mime_types={"application/pdf"},
                    extensions={".pdf"},
                    magic_types=set(),
                ),
                ingestion.ProcessorRoute(
                    name="higher_priority",
                    priority=20,
                    mime_types={"application/pdf"},
                    extensions={".pdf"},
                    magic_types=set(),
                ),
            )
        )

        selected = registry.select_route(
            mime="application/pdf",
            ext=".pdf",
            magic_bytes=None,
        )

        self.assertEqual(selected, "higher_priority")


class TestIngestMetadataBaseline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.upload_dir = base / "uploads"
        self.metadata_dir = self.upload_dir / "metadata"
        self.artifacts_dir = self.upload_dir / "artifacts"
        self.parsed_dir = self.artifacts_dir / "parsed"
        self.normalized_dir = self.artifacts_dir / "normalized"
        self.viewer_dir = self.artifacts_dir / "viewer"

        self.original_dirs = (
            ingestion.UPLOAD_DIR,
            ingestion.METADATA_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.PARSED_DIR,
            ingestion.NORMALIZED_DIR,
            ingestion.VIEWER_ARTIFACTS_DIR,
            ingestion.OBSERVABILITY_LOG_PATH,
            ingestion.OBSERVABILITY_METRICS_PATH,
        )

        ingestion.UPLOAD_DIR = self.upload_dir
        ingestion.METADATA_DIR = self.metadata_dir
        ingestion.ARTIFACTS_DIR = self.artifacts_dir
        ingestion.PARSED_DIR = self.parsed_dir
        ingestion.NORMALIZED_DIR = self.normalized_dir
        ingestion.VIEWER_ARTIFACTS_DIR = self.viewer_dir
        ingestion.OBSERVABILITY_LOG_PATH = self.artifacts_dir / 'processing_logs.jsonl'
        ingestion.OBSERVABILITY_METRICS_PATH = self.artifacts_dir / 'processing_metrics.json'

    def tearDown(self):
        (
            ingestion.UPLOAD_DIR,
            ingestion.METADATA_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.PARSED_DIR,
            ingestion.NORMALIZED_DIR,
            ingestion.VIEWER_ARTIFACTS_DIR,
            ingestion.OBSERVABILITY_LOG_PATH,
            ingestion.OBSERVABILITY_METRICS_PATH,
        ) = self.original_dirs
        self.temp_dir.cleanup()

    def test_store_upload_records_mime_magic_hash_and_versioning(self):
        payload = b"%PDF-1.7 sample"

        first = ingestion.store_upload(
            filename="sample.pdf",
            content_bytes=payload,
            content_type=None,
            source_type="upload",
        )
        second = ingestion.store_upload(
            filename="sample.pdf",
            content_bytes=payload,
            content_type=None,
            source_type="upload",
        )

        first_meta = Path(ingestion.METADATA_DIR / f"{first['stored_filename']}.json")

        import json
        first_payload = json.loads(first_meta.read_text(encoding="utf-8"))

        self.assertEqual(first_payload["detected_mime_type"], "application/pdf")
        self.assertEqual(first_payload["magic_bytes_type"], "application/pdf")
        self.assertRegex(first_payload["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(first_payload["processor_route"], "pdf_processor")
        self.assertEqual(first["source_version"], 1)
        self.assertEqual(second["source_version"], 2)

    def test_store_upload_writes_parsed_artifact_for_text(self):
        payload = b"Hello\n\nWorld"

        record = ingestion.store_upload(
            filename="note.txt",
            content_bytes=payload,
            content_type="text/plain",
            source_type="upload",
        )

        parsed_path = Path(record["parsed_artifact_path"])
        import json

        parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed_payload["parser"], "plain_text_parser")
        self.assertEqual(parsed_payload["object_structure"]["paragraph_count"], 2)
        self.assertEqual(record["parse_warnings"], [])


    def test_store_upload_writes_normalized_artifact_schema(self):
        payload = b"Hello\n\nWorld"

        record = ingestion.store_upload(
            filename="note.txt",
            content_bytes=payload,
            content_type="text/plain",
            source_type="upload",
        )

        normalized_path = Path(record["normalized_artifact_path"])
        import json

        normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
        self.assertIn("canonical_text", normalized_payload)
        self.assertIn("chunks", normalized_payload)
        self.assertIn("entities", normalized_payload)
        self.assertIn("relations", normalized_payload)
        self.assertIn("render_hints", normalized_payload)
        self.assertIn("provenance", normalized_payload)
        self.assertEqual(normalized_payload["canonical_text"], "Hello\n\nWorld")
        self.assertEqual(record["normalize_warnings"], [])


    def test_store_upload_creates_image_chunk_for_image(self):
        payload = b"\x89PNG\r\n\x1a\n" + b"demo"

        record = ingestion.store_upload(
            filename="scan.png",
            content_bytes=payload,
            content_type="image/png",
            source_type="upload",
        )

        normalized_path = Path(record["normalized_artifact_path"])
        import json

        normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
        self.assertEqual(normalized_payload["chunks"][0]["modality"], "image")
        self.assertEqual(normalized_payload["chunks"][0]["chunk_strategy"], "image_ocr_layout")
        self.assertEqual(record["normalize_warnings"], [])


    def test_store_upload_builds_embedding_inputs_and_queue_record(self):
        payload = b"Hello\n\nWorld"

        record = ingestion.store_upload(
            filename="note.txt",
            content_bytes=payload,
            content_type="text/plain",
            source_type="upload",
        )

        normalized_path = Path(record["normalized_artifact_path"])
        import json

        normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
        self.assertIn("embeddings_inputs", normalized_payload)
        self.assertGreaterEqual(len(normalized_payload["embeddings_inputs"]), 1)
        first_item = normalized_payload["embeddings_inputs"][0]
        self.assertEqual(first_item["modality"], "text")
        self.assertTrue(first_item["text"].strip())

        queue_path = Path(record["embedding_queue_path"])
        self.assertTrue(queue_path.exists())
        lines = [line for line in queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 1)
        queue_payload = json.loads(lines[-1])
        self.assertEqual(queue_payload["stored_filename"], record["stored_filename"])
        self.assertEqual(queue_payload["count"], record["embedding_inputs_count"])


    def test_store_upload_indexes_chunks_and_metadata_in_rag_index(self):
        payload = b"Hello\n\nWorld"

        record = ingestion.store_upload(
            filename="note.txt",
            content_bytes=payload,
            content_type="text/plain",
            source_type="upload",
        )

        rag_index_path = Path(record["rag_index_path"])
        self.assertTrue(rag_index_path.exists())

        import json

        rag_index_payload = json.loads(rag_index_path.read_text(encoding="utf-8"))
        doc = rag_index_payload["documents"][record["stored_filename"]]
        self.assertEqual(doc["stored_filename"], record["stored_filename"])
        self.assertEqual(doc["chunk_count"], len(doc["chunks"]))
        self.assertGreaterEqual(doc["chunk_count"], 1)
        self.assertEqual(doc["metadata"]["sha256"], record["sha256"])



    def test_store_upload_generates_viewer_artifacts_manifest(self):
        payload = b"Hello\n\nWorld"

        record = ingestion.store_upload(
            filename="note.txt",
            content_bytes=payload,
            content_type="text/plain",
            source_type="upload",
        )

        manifest_path = Path(record["viewer_artifacts_path"])
        self.assertTrue(manifest_path.exists())
        self.assertGreaterEqual(record["viewer_artifacts_count"], 1)
        self.assertIn("text", record["viewer_artifact_modalities"])

        import json

        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_payload["stored_filename"], record["stored_filename"])
        self.assertTrue(manifest_payload["artifacts"])


    def test_store_upload_writes_observability_logs_and_metrics(self):
        payload = b"Hello\n\nWorld"

        record = ingestion.store_upload(
            filename="note.txt",
            content_bytes=payload,
            content_type="text/plain",
            source_type="upload",
        )

        self.assertTrue(record["observability_log_path"].endswith("processing_logs.jsonl"))
        self.assertTrue(record["observability_metrics_path"].endswith("processing_metrics.json"))

        import json

        log_path = Path(record["observability_log_path"])
        self.assertTrue(log_path.exists())
        log_lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(log_lines), 1)
        log_payload = json.loads(log_lines[-1])
        self.assertEqual(log_payload["stored_filename"], record["stored_filename"])
        self.assertEqual(log_payload["format_key"], ".txt")
        self.assertIn("parse", log_payload["durations_ms"])
        self.assertIn("performance_budget", log_payload)
        self.assertIn("budgets_ms", log_payload["performance_budget"])
        self.assertIn("breaches", log_payload["performance_budget"])
        self.assertEqual(record["performance_budget"], log_payload["performance_budget"])

        metrics_path = Path(record["observability_metrics_path"])
        self.assertTrue(metrics_path.exists())
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertIn(".txt", metrics_payload["by_format"])
        txt_metrics = metrics_payload["by_format"][".txt"]
        self.assertGreaterEqual(txt_metrics["count"], 1)
        self.assertIn("parse", txt_metrics["durations_ms"])
        self.assertIn("performance_budget", txt_metrics)
        self.assertIn("breach_counts", txt_metrics["performance_budget"])
        self.assertIn("parse", txt_metrics["performance_budget"]["breach_counts"])

    def test_store_upload_writes_qa_artifact_with_passed_status(self):
        payload = b"Hello\n\nWorld"

        record = ingestion.store_upload(
            filename="note.txt",
            content_bytes=payload,
            content_type="text/plain",
            source_type="upload",
        )

        qa_path = Path(record["qa_artifact_path"])
        self.assertTrue(qa_path.exists())

        import json

        qa_payload = json.loads(qa_path.read_text(encoding="utf-8"))
        self.assertEqual(qa_payload["status"], "passed")
        self.assertEqual(record["qa_status"], "passed")
        self.assertEqual(record["qa_errors"], [])

    def test_pipeline_qa_marks_fatal_when_text_canonical_missing(self):
        qa_payload = ingestion.run_pipeline_qa(
            metadata={
                "filename": "note.txt",
                "stored_filename": "note.txt",
                "detected_mime_type": "text/plain",
            },
            normalized_doc=ingestion.NormalizedDoc(
                canonical_text="",
                chunks=[
                    {
                        "chunk_id": "note.txt:chunk:0",
                        "text": "fallback",
                        "modality": "text",
                    }
                ],
                entities=[],
                relations=[],
                embeddings_inputs=[
                    {
                        "chunk_id": "note.txt:chunk:0",
                        "modality": "text",
                        "text": "fallback",
                        "source": {"stored_filename": "note.txt"},
                    }
                ],
                render_hints={},
                provenance={},
                warnings=[],
            ),
        )

        self.assertEqual(qa_payload["status"], "failed")
        self.assertTrue(any(item["class"] == "fatal" for item in qa_payload["errors"]))

    def test_store_upload_records_parse_warning_for_non_text(self):
        payload = b"%PDF-1.7 sample"

        record = ingestion.store_upload(
            filename="sample.pdf",
            content_bytes=payload,
            content_type=None,
            source_type="upload",
        )

        self.assertTrue(record["parse_warnings"])


class TestStructuredParsingFormats(unittest.TestCase):
    def test_parse_uses_optional_converter_plugin_for_legacy_format(self):
        plugin = ingestion.OptionalConverterPlugin(
            name="legacy_dxf_plugin",
            plugin_id="legacy-dxf",
            route_name="legacy_dxf_processor",
            priority=400,
            mime_types={"application/dxf"},
            extensions={".dxf"},
            magic_types=set(),
            license_name="GPL-2.0-only",
            tooling="legacy-dxf-toolkit",
            converter=lambda **_: ingestion.ParsedDoc(
                parser="legacy_dxf_plugin_parser",
                text="ENTITY LINE\nENTITY CIRCLE",
                layout=[],
                tables=[],
                media=[],
                object_structure={
                    "format": "dxf",
                    "converter": {
                        "plugin_id": "legacy-dxf",
                        "license_name": "GPL-2.0-only",
                        "tooling": "legacy-dxf-toolkit",
                    },
                },
                warnings=[],
            ),
        )

        original_plugins = ingestion.OPTIONAL_CONVERTER_PLUGINS
        try:
            ingestion.OPTIONAL_CONVERTER_PLUGINS = (plugin,)
            parsed = ingestion.parse_structured_document(
                filename="drawing.dxf",
                content_bytes=b"0\nSECTION\n2\nENTITIES",
                detected_mime_type="application/octet-stream",
                magic_bytes_type=None,
            )
        finally:
            ingestion.OPTIONAL_CONVERTER_PLUGINS = original_plugins

        self.assertEqual(parsed.parser, "legacy_dxf_plugin_parser")
        self.assertEqual(parsed.object_structure["format"], "dxf")
        self.assertIn("converter", parsed.object_structure)
        self.assertIn("signature_compliance", parsed.object_structure)

    def test_parse_unknown_legacy_format_uses_best_effort_fallback(self):
        parsed = ingestion.parse_structured_document(
            filename="capture.pcap",
            content_bytes=b"PACKET START\x00src=10.0.0.1 dst=10.0.0.2",
            detected_mime_type="application/octet-stream",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "legacy_best_effort_parser")
        self.assertIn("PACKET START", parsed.text)
        self.assertTrue(any("LEGACY_FALLBACK_BEST_EFFORT_TEXT" in warning for warning in parsed.warnings))

        fallback = parsed.object_structure["fallback_strategy"]
        self.assertEqual(fallback["id"], "best_effort_text_extraction")
        self.assertEqual(fallback["legacy_format"], ".pcap")
        self.assertTrue(fallback["structured_warnings"])
        self.assertEqual(
            fallback["structured_warnings"][0]["code"],
            "LEGACY_FALLBACK_BEST_EFFORT_TEXT",
        )

    def test_parse_dxf_builds_human_readable_summary(self):
        parsed = ingestion.parse_structured_document(
            filename="drawing.dxf",
            content_bytes=(
                b"0\nSECTION\n2\nENTITIES\n"
                b"0\nLINE\n8\nWALL\n"
                b"0\nTEXT\n8\nANNOT\n1\nDoor A\n"
                b"0\nENDSEC\n0\nEOF\n"
            ),
            detected_mime_type="application/dxf",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "dxf_parser")
        self.assertEqual(parsed.object_structure["format"], "dxf")
        self.assertIn("Entities:", parsed.text)
        self.assertIn("LINE=1", parsed.text)
        self.assertIn("TEXT=1", parsed.text)
        self.assertIn("Door A", parsed.text)

    def test_legacy_regression_corpus_guards_parser_drift(self):
        fixtures_dir = Path(__file__).parent / "fixtures" / "legacy_regression"
        manifest = json.loads((fixtures_dir / "corpus_manifest.json").read_text(encoding="utf-8"))

        for case in manifest:
            payload = (fixtures_dir / case["filename"]).read_bytes()
            parsed = ingestion.parse_structured_document(
                filename=case["filename"],
                content_bytes=payload,
                detected_mime_type=case["mime_type"],
                magic_bytes_type=None,
            )

            self.assertEqual(parsed.parser, case["expected_parser"])
            self.assertIn(case["expected_text_snippet"], parsed.text)
            self.assertTrue(
                any(case["expected_warning_code"] in warning for warning in parsed.warnings),
                msg=f"missing warning for {case['filename']}",
            )

            fallback = parsed.object_structure.get("fallback_strategy") or {}
            self.assertEqual(fallback.get("legacy_format"), Path(case["filename"]).suffix)
            self.assertTrue((fallback.get("printable_ratio_pct") or 0) > 0)

    def test_parse_text_extracts_structure_sections(self):
        payload = (
            b"# Heading\n"
            b"- Item\n"
            b"| col1 | col2 |\n"
            b"| --- | --- |\n"
            b"[^1]: Footnote"
        )
        parsed = ingestion.parse_structured_document(
            filename="note.txt",
            content_bytes=payload,
            detected_mime_type="text/plain",
            magic_bytes_type=None,
        )

        sections = parsed.object_structure["structure"]["sections"]
        section_types = {item["type"] for item in sections}
        self.assertIn("heading", section_types)
        self.assertIn("list_item", section_types)
        self.assertIn("table", section_types)
        self.assertIn("footnote", section_types)

    def test_parse_html_uses_html_parser(self):
        parsed = ingestion.parse_structured_document(
            filename="page.html",
            content_bytes=b"<html><body><h1>Title</h1><p>Hello</p></body></html>",
            detected_mime_type="text/html",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "html_parser")
        self.assertIn("Title", parsed.text)

    def test_parse_html_extracts_structure_sections(self):
        parsed = ingestion.parse_structured_document(
            filename="page.html",
            content_bytes=(
                b"<html><body><h2>Kapitel</h2><ul><li>Eintrag</li></ul>"
                b"<table><tr><td>A</td></tr></table><footnote>Quelle</footnote></body></html>"
            ),
            detected_mime_type="text/html",
            magic_bytes_type=None,
        )

        sections = parsed.object_structure["structure"]["sections"]
        section_types = {item["type"] for item in sections}
        self.assertIn("heading", section_types)
        self.assertIn("list_item", section_types)
        self.assertIn("table", section_types)
        self.assertIn("footnote", section_types)

    def test_parse_email_uses_email_parser(self):
        eml = (
            b"From: alice@example.com\n"
            b"To: bob@example.com\n"
            b"Subject: Demo\n"
            b"\n"
            b"Email body"
        )
        parsed = ingestion.parse_structured_document(
            filename="mail.eml",
            content_bytes=eml,
            detected_mime_type="message/rfc822",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "email_parser")
        self.assertEqual(parsed.object_structure["headers"]["subject"], "Demo")
        self.assertIn("Email body", parsed.text)

    def test_parse_email_extracts_threading_and_relations(self):
        eml = (
            b"From: alice@example.com\n"
            b"To: bob@example.com\n"
            b"Cc: team@example.com\n"
            b"Subject: Re: Demo\n"
            b"Message-ID: <msg-2@example.com>\n"
            b"In-Reply-To: <msg-1@example.com>\n"
            b"References: <msg-0@example.com> <msg-1@example.com>\n"
            b"MIME-Version: 1.0\n"
            b"Content-Type: multipart/mixed; boundary=abc123\n"
            b"\n"
            b"--abc123\n"
            b"Content-Type: text/plain; charset=utf-8\n"
            b"\n"
            b"Antwort\n"
            b"> quoted line\n"
            b"\n"
            b"--abc123\n"
            b"Content-Type: text/plain\n"
            b"Content-Disposition: attachment; filename=notes.txt\n"
            b"\n"
            b"attachment body\n"
            b"--abc123--\n"
        )

        parsed = ingestion.parse_structured_document(
            filename="mail.eml",
            content_bytes=eml,
            detected_mime_type="message/rfc822",
            magic_bytes_type=None,
        )

        headers = parsed.object_structure["headers"]
        self.assertEqual(headers["message_id"], "<msg-2@example.com>")
        self.assertEqual(headers["in_reply_to"], "<msg-1@example.com>")
        self.assertEqual(headers["references"], ["<msg-0@example.com>", "<msg-1@example.com>"])

        threading = parsed.object_structure["threading"]
        self.assertEqual(threading["in_reply_to"], "<msg-1@example.com>")

        quoted_blocks = parsed.object_structure["quoted_blocks"]
        self.assertEqual(len(quoted_blocks), 1)
        self.assertIn("> quoted line", quoted_blocks[0]["text"])

        relation_types = {relation["type"] for relation in parsed.object_structure["relations"]}
        self.assertIn("reply_to", relation_types)
        self.assertIn("references", relation_types)
        self.assertIn("has_attachment", relation_types)
        self.assertIn("quoted_block", relation_types)
        self.assertIn("sent_by", relation_types)
        self.assertIn("sent_to", relation_types)
        self.assertIn("has_subject", relation_types)
        self.assertIn("quotes_message", relation_types)

    def test_normalize_keeps_email_relations(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="email_parser",
            text="Mail body",
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "relations": [
                    {"type": "reply_to", "from": "m2", "to": "m1"},
                ]
            },
            warnings=[],
        )
        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "mail.eml",
                "stored_filename": "abc_mail.eml",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "message/rfc822",
                "magic_bytes_type": "message/rfc822",
            },
        )
        self.assertEqual(normalized.relations, [{"type": "reply_to", "from": "m2", "to": "m1"}])


    def test_parse_image_runs_ocr_layout_analysis(self):
        parsed = ingestion.parse_structured_document(
            filename="scan.png",
            content_bytes=b"\x89PNG\r\n\x1a\nINVOICE 2026 TOTAL 199.99",
            detected_mime_type="image/png",
            magic_bytes_type="image/png",
        )

        self.assertIn(parsed.parser, {"ocr_image_parser", "image_ocr_parser"})
        self.assertTrue(parsed.text)
        self.assertTrue(parsed.layout)
        ocr_meta = parsed.object_structure["ocr"]
        if parsed.parser == "ocr_image_parser":
            self.assertGreater(ocr_meta["line_count"], 0)
            self.assertGreater(ocr_meta["word_count"], 0)
            first_word = parsed.layout[0]["lines"][0]["words"][0]
            self.assertIn("confidence", first_word)
        else:
            self.assertIn("blocks", ocr_meta)
            self.assertIn("lines", ocr_meta)
            self.assertIn("words", ocr_meta)

    def test_store_upload_image_uses_ocr_chunk_strategy_when_text_detected(self):
        record = ingestion.store_upload(
            filename="scan.png",
            content_bytes=b"\x89PNG\r\n\x1a\nINVOICE TOTAL DUE",
            content_type="image/png",
            source_type="upload",
        )

        normalized_path = Path(record["normalized_artifact_path"])
        import json

        normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
        image_chunks = [
            chunk for chunk in normalized_payload["chunks"] if chunk["modality"] == "image"
        ]
        self.assertTrue(image_chunks)
        self.assertEqual(image_chunks[0]["chunk_strategy"], "image_ocr_layout")


    def test_parse_image_detects_table_chunks(self):
        ocr_result = {
            "text": "Item,Qty,Price\nBolt,2,3.50\nNut,5,1.10",
            "layout": [],
            "blocks": [],
            "lines": [],
            "words": [],
            "warnings": [],
        }
        with patch("backend.ingestion._run_image_ocr_pipeline", return_value=ocr_result):
            parsed = ingestion.parse_structured_document(
                filename="table-scan.png",
                content_bytes=(
                    b"\x89PNG\r\n\x1a\n"
                    b"Item,Qty,Price\n"
                    b"Bolt,2,3.50\n"
                    b"Nut,5,1.10\n"
                ),
                detected_mime_type="image/png",
                magic_bytes_type="image/png",
            )

        self.assertTrue(parsed.tables)
        self.assertEqual(parsed.tables[0]["source"], "image_ocr_table_detection")
        self.assertEqual(parsed.object_structure["ocr"]["table_count"], 1)

    def test_store_upload_image_writes_table_modality_chunk_when_detected(self):
        record = ingestion.store_upload(
            filename="table-scan.png",
            content_bytes=(
                b"\x89PNG\r\n\x1a\n"
                b"Item,Qty,Price\n"
                b"Bolt,2,3.50\n"
                b"Nut,5,1.10\n"
            ),
            content_type="image/png",
            source_type="upload",
        )

        normalized_path = Path(record["normalized_artifact_path"])
        import json

        normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
        table_chunks = [chunk for chunk in normalized_payload["chunks"] if chunk["modality"] == "table"]

        self.assertTrue(table_chunks)
        self.assertEqual(table_chunks[0]["chunk_strategy"], "table_object")

    def test_store_upload_image_table_chunk_contains_tabular_text(self):
        record = ingestion.store_upload(
            filename="table-scan.png",
            content_bytes=(
                b"\x89PNG\r\n\x1a\n"
                b"Item,Qty,Price\n"
                b"Bolt,2,3.50\n"
                b"Nut,5,1.10\n"
            ),
            content_type="image/png",
            source_type="upload",
        )

        normalized_path = Path(record["normalized_artifact_path"])
        import json

        normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
        table_chunks = [chunk for chunk in normalized_payload["chunks"] if chunk["modality"] == "table"]

        self.assertTrue(table_chunks)
        self.assertIn("Item | Qty | Price", table_chunks[0]["text"])

    def test_store_upload_csv_adds_row_and_column_table_chunk_strategies(self):
        record = ingestion.store_upload(
            filename="inventory.csv",
            content_bytes=b"sku,name,qty\nA-1,Bolt,10\nA-2,Nut,15\n",
            content_type="text/csv",
            source_type="upload",
        )

        normalized_payload = json.loads(Path(record["normalized_artifact_path"]).read_text(encoding="utf-8"))
        table_chunks = [chunk for chunk in normalized_payload["chunks"] if chunk["modality"] == "table"]
        strategies = {chunk["chunk_strategy"] for chunk in table_chunks}

        self.assertIn("table_object", strategies)
        self.assertIn("table_row_as_doc", strategies)
        self.assertIn("table_column_block", strategies)

        row_chunk = next(chunk for chunk in table_chunks if chunk["chunk_strategy"] == "table_row_as_doc")
        self.assertIn("sku=A-1", row_chunk["text"])

        column_chunk = next(chunk for chunk in table_chunks if chunk["chunk_strategy"] == "table_column_block")
        self.assertIn("sku: A-1 | A-2", column_chunk["text"])

    def test_store_upload_csv_generates_table_viewer_filter_navigation_and_deeplinks(self):
        record = ingestion.store_upload(
            filename="inventory.csv",
            content_bytes=b"sku,name,qty\nA-1,Bolt,10\nA-2,Nut,15\n",
            content_type="text/csv",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        table_entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "table")
        table_payload = json.loads(Path(table_entry["path"]).read_text(encoding="utf-8"))

        self.assertEqual(table_payload["type"], "table_view")
        self.assertIn("sku", table_payload["filter_metadata"]["available_columns"])
        self.assertIn("table_row_as_doc", table_payload["filter_metadata"]["chunk_strategies"])

        self.assertTrue(table_payload["sheet_navigation"])
        self.assertEqual(table_payload["sheet_navigation"][0]["sheet_name"], "Sheet1")

        self.assertTrue(table_payload["row_deep_links"])
        self.assertIn("table://", table_payload["row_deep_links"][0]["viewer_link"])
        self.assertIn("#row=1", table_payload["row_deep_links"][0]["viewer_link"])

    def test_parse_docx_uses_docx_parser(self):
        import io
        import zipfile

        xml = b"<?xml version='1.0' encoding='UTF-8' standalone='yes'?><w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'><w:body><w:p><w:r><w:t>Hello DOCX</w:t></w:r></w:p></w:body></w:document>"

        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", xml)

        parsed = ingestion.parse_structured_document(
            filename="file.docx",
            content_bytes=stream.getvalue(),
            detected_mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            magic_bytes_type="application/zip",
        )

        self.assertEqual(parsed.parser, "docx_parser")
        self.assertIn("Hello DOCX", parsed.text)

    def test_parse_container_zip_handles_depth_dedup_and_unsafe_paths(self):
        import io
        import zipfile

        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("ok/readme.txt", "hello")
            archive.writestr("dup/a.txt", "same")
            archive.writestr("dup/b.txt", "same")
            archive.writestr("../../escape.txt", "blocked")
            archive.writestr("a/b/c/d/e.txt", "too-deep")

        parsed = ingestion.parse_structured_document(
            filename="bundle.zip",
            content_bytes=stream.getvalue(),
            detected_mime_type="application/zip",
            magic_bytes_type="application/zip",
        )

        self.assertEqual(parsed.parser, "container_parser")
        container = parsed.object_structure["container"]
        statuses = {entry["status"] for entry in container["entries"]}
        self.assertIn("parsed", statuses)
        self.assertIn("deduplicated", statuses)
        self.assertIn("blocked", statuses)
        self.assertIn("skipped", statuses)
        self.assertEqual(container["depth_limit"], ingestion.CONTAINER_DEPTH_LIMIT)
        self.assertGreaterEqual(container["deduplicated_entries"], 1)
        self.assertTrue(any(relation["type"] == "contains" for relation in parsed.object_structure["relations"]))


    def test_parse_container_zip_blocks_dangerous_filename(self):
        import io
        import zipfile

        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("CON.txt", "blocked")
            archive.writestr("ok/readme.txt", "hello")

        parsed = ingestion.parse_structured_document(
            filename="dangerous.zip",
            content_bytes=stream.getvalue(),
            detected_mime_type="application/zip",
            magic_bytes_type="application/zip",
        )

        container = parsed.object_structure["container"]
        blocked = [entry for entry in container["entries"] if entry["status"] == "blocked"]
        self.assertTrue(any(entry.get("reason") == "dangerous_filename" for entry in blocked))
        self.assertTrue(any("dangerous container entry filename" in warning for warning in parsed.warnings))

    def test_parse_container_zip_blocks_total_uncompressed_size_guard(self):
        original_limit = ingestion.CONTAINER_MAX_TOTAL_UNCOMPRESSED_BYTES
        ingestion.CONTAINER_MAX_TOTAL_UNCOMPRESSED_BYTES = 10
        try:
            import io
            import zipfile

            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("a.txt", "123456")
                archive.writestr("b.txt", "123456")

            parsed = ingestion.parse_structured_document(
                filename="size-guard.zip",
                content_bytes=stream.getvalue(),
                detected_mime_type="application/zip",
                magic_bytes_type="application/zip",
            )
        finally:
            ingestion.CONTAINER_MAX_TOTAL_UNCOMPRESSED_BYTES = original_limit

        container = parsed.object_structure["container"]
        self.assertTrue(any(entry.get("reason") == "zip_bomb_guard" for entry in container["entries"]))
        self.assertTrue(any("total uncompressed-size guard" in warning for warning in parsed.warnings))

    def test_parse_container_zip_corrupt_archive_returns_warning(self):
        parsed = ingestion.parse_structured_document(
            filename="broken.zip",
            content_bytes=b"not-a-valid-zip",
            detected_mime_type="application/zip",
            magic_bytes_type="application/zip",
        )

        self.assertEqual(parsed.parser, "container_parser")
        self.assertTrue(parsed.warnings)
        self.assertIn("corrupt ZIP archive", parsed.warnings[0])

    def test_store_upload_zip_generates_package_tree_viewer_artifact_with_source_metadata_inheritance(self):
        import io
        import zipfile

        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("docs/readme.txt", "hello")
            archive.writestr("docs/spec/plan.md", "plan")

        record = ingestion.store_upload(
            filename="bundle.zip",
            content_bytes=stream.getvalue(),
            content_type="application/zip",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        package_entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "container")
        package_payload = json.loads(Path(package_entry["path"]).read_text(encoding="utf-8"))

        self.assertEqual(package_payload["type"], "package_tree_view")
        self.assertEqual(package_payload["tree"]["root_id"], "root")
        nodes = package_payload["tree"]["nodes"]
        self.assertTrue(any(node["entry_path"] == "docs/readme.txt" for node in nodes))
        self.assertTrue(any(node["entry_path"] == "docs/spec/plan.md" for node in nodes))

        docs_node = next(node for node in nodes if node["entry_path"] == "docs")
        self.assertEqual(docs_node["kind"], "directory")
        self.assertEqual(docs_node["entry_status"], "virtual")

        leaf_node = next(node for node in nodes if node["entry_path"] == "docs/readme.txt")
        inherited = leaf_node["inherited_source_metadata"]
        self.assertEqual(inherited["filename"], "bundle.zip")
        self.assertEqual(inherited["stored_filename"], record["stored_filename"])
        self.assertEqual(inherited["source_type"], "upload")
        self.assertEqual(inherited["source_version"], record["source_version"])
        self.assertRegex(inherited["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("source_version", package_payload["source_metadata_inheritance"]["inherited_fields"])

    def test_parse_json_extracts_jsonpath_nodes_with_stable_ids(self):
        parsed = ingestion.parse_structured_document(
            filename="bundle.json",
            content_bytes=(
                b'{"patient":{"id":"p-1","active":true},"encounters":[{"id":"e-1"}]}'
            ),
            detected_mime_type="application/json",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "jsonpath_parser")
        extraction = parsed.object_structure["path_extraction"]
        self.assertEqual(extraction["engine"], "jsonpath")
        self.assertTrue(extraction["nodes"])

        first = extraction["nodes"][0]
        self.assertTrue(first["node_id"].startswith("jsonpath_node_"))
        self.assertTrue(first["path"].startswith("$"))

    def test_parse_xml_extracts_xpath_nodes_with_stable_ids(self):
        parsed = ingestion.parse_structured_document(
            filename="doc.xml",
            content_bytes=b"<root><order><id>1001</id></order></root>",
            detected_mime_type="application/xml",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "xpath_parser")
        extraction = parsed.object_structure["path_extraction"]
        self.assertEqual(extraction["engine"], "xpath")
        self.assertTrue(extraction["nodes"])
        self.assertTrue(any(node["path"] == "/root[1]/order[1]/id[1]" for node in extraction["nodes"]))
        self.assertTrue(all(node["node_id"].startswith("xpath_node_") for node in extraction["nodes"]))


    def test_parse_json_includes_container_nodes_for_stable_path_graph(self):
        parsed = ingestion.parse_structured_document(
            filename="bundle.json",
            content_bytes=b'{"patient":{"id":"p-1"},"encounters":[{"id":"e-1"}]}',
            detected_mime_type="application/json",
            magic_bytes_type=None,
        )

        nodes = parsed.object_structure["path_extraction"]["nodes"]
        by_path = {node["path"]: node for node in nodes}
        self.assertEqual(by_path["$"]["value_type"], "object")
        self.assertEqual(by_path["$.encounters"]["value_type"], "array")
        self.assertTrue(by_path["$.encounters[0].id"]["node_id"].startswith("jsonpath_node_"))

    def test_parse_xml_extracts_attributes_with_stable_xpath_ids(self):
        parsed = ingestion.parse_structured_document(
            filename="doc.xml",
            content_bytes=b'<root version="v1"><order id="1001"><name>alpha</name></order></root>',
            detected_mime_type="application/xml",
            magic_bytes_type=None,
        )

        nodes = parsed.object_structure["path_extraction"]["nodes"]
        paths = {node["path"]: node for node in nodes}
        self.assertIn("/root[1]/@version", paths)
        self.assertEqual(paths["/root[1]/@version"]["value_type"], "attribute")
        self.assertIn("/root[1]/order[1]/@id", paths)
        self.assertTrue(paths["/root[1]/order[1]/@id"]["node_id"].startswith("xpath_node_"))

    def test_parse_json_detects_fhir_schema_version(self):
        parsed = ingestion.parse_structured_document(
            filename="patient.json",
            content_bytes=(
                b'{"resourceType":"Patient","meta":{"profile":["http://hl7.org/fhir/StructureDefinition/Patient/4.0.1"]}}'
            ),
            detected_mime_type="application/json",
            magic_bytes_type=None,
        )

        schema = parsed.object_structure["schema_validation"]
        self.assertEqual(schema["schema"], "fhir")
        self.assertEqual(schema["version"], "R4")
        self.assertEqual(schema["validation_status"], "valid")

    def test_parse_xml_detects_iso20022_message_type(self):
        parsed = ingestion.parse_structured_document(
            filename="payment.xml",
            content_bytes=(
                b'<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"><CstmrCdtTrfInitn /></Document>'
            ),
            detected_mime_type="application/xml",
            magic_bytes_type=None,
        )

        schema = parsed.object_structure["schema_validation"]
        self.assertEqual(schema["schema"], "iso20022")
        self.assertEqual(schema["version"], "pain.001.001.09")
        self.assertEqual(schema["validation_status"], "valid")


    def test_parse_json_persists_reference_partof_sameas_relations(self):
        parsed = ingestion.parse_structured_document(
            filename="bundle.json",
            content_bytes=(
                b'{"id":"bundle-1","entries":[{"id":"enc-1","part_of":"bundle-1","same_as":"enc-legacy-1","references":"patient-1"}],"patient":{"id":"patient-1"}}'
            ),
            detected_mime_type="application/json",
            magic_bytes_type=None,
        )

        relations = parsed.object_structure["relations"]
        relation_types = {relation["type"] for relation in relations}
        self.assertEqual(relation_types, {"references", "part_of", "same_as"})
        self.assertTrue(any(relation.get("target_node_id") for relation in relations))

    def test_parse_xml_persists_reference_partof_sameas_relations(self):
        parsed = ingestion.parse_structured_document(
            filename="doc.xml",
            content_bytes=(
                b'<root><id>doc-1</id><part_of>doc-1</part_of><same_as>doc-legacy-1</same_as><reference>external-2</reference></root>'
            ),
            detected_mime_type="application/xml",
            magic_bytes_type=None,
        )

        relations = parsed.object_structure["relations"]
        relation_types = {relation["type"] for relation in relations}
        self.assertEqual(relation_types, {"references", "part_of", "same_as"})
        self.assertTrue(any(relation.get("target_node_id") for relation in relations))

    def test_store_upload_json_generates_tree_and_graph_viewer_artifact(self):
        record = ingestion.store_upload(
            filename="bundle.json",
            content_bytes=(
                b'{"id":"bundle-1","entries":[{"id":"enc-1","part_of":"bundle-1","same_as":"enc-legacy-1","references":"patient-1"}],"patient":{"id":"patient-1"}}'
            ),
            content_type="application/json",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        structured_entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "json_xml")
        structured_payload = json.loads(Path(structured_entry["path"]).read_text(encoding="utf-8"))

        self.assertEqual(structured_payload["type"], "structured_tree_graph_view")
        self.assertEqual(structured_payload["engine"], "jsonpath")
        self.assertTrue(structured_payload["tree_view"]["nodes"])
        self.assertTrue(structured_payload["tree_view"]["path_search_index"][0]["search_terms"])
        self.assertTrue(structured_payload["graph_view"]["nodes"])
        self.assertTrue(structured_payload["graph_view"]["edges"])
        self.assertIn(
            "resolved",
            {edge["resolution_status"] for edge in structured_payload["graph_view"]["edges"]},
        )
        self.assertIn("reference_resolution", structured_payload["graph_view"])
        self.assertGreaterEqual(structured_payload["graph_view"]["reference_resolution"]["resolved"], 1)

    def test_store_upload_xml_generates_tree_and_graph_viewer_artifact(self):
        record = ingestion.store_upload(
            filename="doc.xml",
            content_bytes=(
                b'<root><id>doc-1</id><part_of>doc-1</part_of><same_as>doc-legacy-1</same_as><reference>external-2</reference></root>'
            ),
            content_type="application/xml",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        structured_entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "json_xml")
        structured_payload = json.loads(Path(structured_entry["path"]).read_text(encoding="utf-8"))

        self.assertEqual(structured_payload["type"], "structured_tree_graph_view")
        self.assertEqual(structured_payload["engine"], "xpath")
        self.assertTrue(any(node["path"] == "/root[1]" for node in structured_payload["tree_view"]["nodes"]))
        self.assertTrue(structured_payload["graph_view"]["edges"])
        self.assertIn("reference_resolution", structured_payload["graph_view"])


    def test_validate_upload_accepts_3d_extensions(self):
        result = ingestion.validate_upload("part.stl", b"solid demo\nendsolid demo")
        self.assertEqual(result.status, "success")

    def test_store_upload_glb_generates_glb_canonical_viewer_artifact(self):
        payload = b"glTF-Binary-demo"
        record = ingestion.store_upload(
            filename="assembly.glb",
            content_bytes=payload,
            content_type="model/gltf-binary",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "3d")
        viewer_payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))

        self.assertEqual(viewer_payload["type"], "model_3d_view")
        self.assertEqual(viewer_payload["canonical_viewer_format"], "glb")
        self.assertEqual(viewer_payload["conversion_status"], "passthrough_glb")

        canonical_path = Path(viewer_payload["canonical_glb_path"])
        self.assertTrue(canonical_path.exists())
        self.assertEqual(canonical_path.read_bytes(), payload)

    def test_store_upload_stl_converts_to_canonical_glb_artifact(self):
        record = ingestion.store_upload(
            filename="mesh.stl",
            content_bytes=(
                b"solid mesh\n"
                b"facet normal 0 0 1\n"
                b" outer loop\n"
                b"  vertex 0 0 0\n"
                b"  vertex 1 0 0\n"
                b"  vertex 0 1 0\n"
                b" endloop\n"
                b"endfacet\n"
                b"endsolid mesh"
            ),
            content_type="model/stl",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "3d")
        viewer_payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))

        self.assertEqual(viewer_payload["canonical_viewer_format"], "glb")
        self.assertEqual(viewer_payload["conversion_status"], "converted_to_glb")
        canonical_path = Path(viewer_payload["canonical_glb_path"])
        self.assertTrue(canonical_path.exists())
        self.assertEqual(canonical_path.read_bytes()[:4], b"glTF")

    def test_store_upload_3d_chunks_include_summary_instead_of_placeholder(self):
        record = ingestion.store_upload(
            filename="model.obj",
            content_bytes=b"o cube\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            content_type="model/obj",
            source_type="upload",
        )

        normalized_payload = json.loads(Path(record["normalized_artifact_path"]).read_text(encoding="utf-8"))
        chunks = normalized_payload["chunks"]
        self.assertTrue(any(chunk["chunk_strategy"] == "model_3d_summary" for chunk in chunks))
        self.assertFalse(any("placeholder chunk" in (chunk.get("text") or "") for chunk in chunks))

    def test_store_upload_step_generates_tessellation_intermediate_artifact(self):
        record = ingestion.store_upload(
            filename="model.step",
            content_bytes=b"ISO-10303-21;\nEND-ISO-10303-21;",
            content_type="application/step",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "3d")
        viewer_payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))

        self.assertEqual(viewer_payload["canonical_viewer_format"], "glb")
        self.assertEqual(viewer_payload["conversion_status"], "pending_tessellation_and_conversion")
        self.assertTrue(viewer_payload["intermediate_artifact_path"])

        intermediate_path = Path(viewer_payload["intermediate_artifact_path"])
        self.assertTrue(intermediate_path.exists())
        intermediate_payload = json.loads(intermediate_path.read_text(encoding="utf-8"))
        self.assertEqual(intermediate_payload["type"], "cad_bim_tessellation_intermediate")
        self.assertEqual(intermediate_payload["source_extension"], ".step")

        canonical_path = Path(viewer_payload["canonical_glb_path"])
        self.assertTrue(canonical_path.exists())
        self.assertIn(b"tessellation intermediate", canonical_path.read_bytes())

    def test_store_upload_3d_generates_canonical_meta_with_stable_object_ids(self):
        payload = b"glTF-Binary-demo"
        record = ingestion.store_upload(
            filename="meta.glb",
            content_bytes=payload,
            content_type="model/gltf-binary",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "3d")
        viewer_payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))

        meta_path = Path(viewer_payload["canonical_meta_path"])
        self.assertTrue(meta_path.exists())
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))

        self.assertEqual(meta_payload["type"], "canonical_3d_meta")
        self.assertEqual(meta_payload["source_id"], record["sha256"])
        self.assertGreaterEqual(meta_payload["node_count"], 1)

        first_node = meta_payload["nodes"][0]
        self.assertTrue(first_node["object_id"].startswith("obj_"))
        self.assertEqual(first_node["source_id"], record["sha256"])
        self.assertIn("bbox", first_node)

    def test_store_upload_3d_generates_preview_png_and_features_json(self):
        record = ingestion.store_upload(
            filename="preview.glb",
            content_bytes=b"glTF-Binary-demo",
            content_type="model/gltf-binary",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "3d")
        viewer_payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))

        preview_path = Path(viewer_payload["preview_path"])
        features_path = Path(viewer_payload["features_path"])
        self.assertTrue(preview_path.exists())
        self.assertTrue(features_path.exists())

        self.assertEqual(preview_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

        features_payload = json.loads(features_path.read_text(encoding="utf-8"))
        self.assertEqual(features_payload["type"], "model_3d_features")
        self.assertEqual(features_payload["source_id"], record["sha256"])
        self.assertIn("vertex_count", features_payload)
        self.assertIn("surface_area", features_payload)
        self.assertIn("volume", features_payload)

    def test_store_upload_3d_viewer_payload_includes_meta_mapping_for_object_interactions(self):
        record = ingestion.store_upload(
            filename="interaction.glb",
            content_bytes=b"glTF-Binary-demo",
            content_type="model/gltf-binary",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "3d")
        viewer_payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))

        self.assertIn("meta_mapping", viewer_payload)
        mapping = viewer_payload["meta_mapping"]
        self.assertEqual(mapping["type"], "object_meta_mapping")
        self.assertEqual(mapping["source_id"], record["sha256"])
        self.assertTrue(mapping["interaction_support"]["highlight"])
        self.assertTrue(mapping["interaction_support"]["isolate"])
        self.assertTrue(mapping["interaction_support"]["fit_to_object"])
        self.assertTrue(mapping["highlight_targets"])

        first_target = mapping["highlight_targets"][0]
        self.assertIn("object_id", first_target)
        self.assertIn("bbox", first_target)
        self.assertIn("labels", first_target)

    def test_validate_upload_accepts_json_extension(self):
        result = ingestion.validate_upload("payload.json", b'{"ok":true}')
        self.assertEqual(result.status, "success")

    def test_parse_svg_extracts_semantic_regions(self):
        svg_payload = b"""
<svg xmlns='http://www.w3.org/2000/svg'>
  <title>Factory Floor</title>
  <text x='10' y='20'>Pump A</text>
  <path id='line-main' d='M10 10 L20 20' />
</svg>
"""
        parsed = ingestion.parse_structured_document(
            filename="diagram.svg",
            content_bytes=svg_payload,
            detected_mime_type="image/svg+xml",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "svg_vector_parser")
        self.assertIn("Pump A", parsed.text)
        self.assertTrue(parsed.layout)
        self.assertEqual(parsed.layout[0]["type"], "label")
        self.assertEqual(parsed.layout[1]["type"], "text")
        self.assertEqual(parsed.layout[2]["type"], "path")

    def test_store_upload_svg_writes_image_chunk(self):
        record = ingestion.store_upload(
            filename="diagram.svg",
            content_bytes=(
                b"<svg xmlns='http://www.w3.org/2000/svg'><text x='1' y='2'>Valve</text><path id='p1' d='M0 0 L1 1'/></svg>"
            ),
            content_type="image/svg+xml",
            source_type="upload",
        )

        normalized_path = Path(record["normalized_artifact_path"])
        import json

        normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
        image_chunks = [chunk for chunk in normalized_payload["chunks"] if chunk["modality"] == "image"]
        self.assertTrue(image_chunks)
        self.assertEqual(image_chunks[0]["chunk_strategy"], "image_ocr_layout")




    def test_parse_image_runs_ocr_pipeline_with_layout_confidence(self):
        with patch(
            "backend.ingestion._run_image_ocr_pipeline",
            return_value={
                "text": "Detected text",
                "layout": [{"type": "ocr_block", "block_id": "block-1-1", "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.95}],
                "blocks": [{"block_id": "block-1-1", "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.95}],
                "lines": [{"line_id": "line-1-1-1", "block_id": "block-1-1", "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.91}],
                "words": [{"text": "Detected", "line_id": "line-1-1-1", "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.9}],
                "warnings": [],
            },
        ):
            parsed = ingestion.parse_structured_document(
                filename="scan.png",
                content_bytes=b"\x89PNG\r\n\x1a\n...",
                detected_mime_type="image/png",
                magic_bytes_type="image/png",
            )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertEqual(parsed.text, "Detected text")
        self.assertEqual(parsed.layout[0]["type"], "ocr_block")
        self.assertEqual(parsed.object_structure["ocr"]["words"][0]["confidence"], 0.9)


    def test_parse_image_ocr_pipeline_extracts_table_chunks_from_ocr_text(self):
        with patch(
            "backend.ingestion._run_image_ocr_pipeline",
            return_value={
                "text": "Item,Qty,Price\nBolt,2,3.50\nNut,5,1.10",
                "layout": [{"type": "ocr_block", "block_id": "block-1", "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.95}],
                "blocks": [{"block_id": "block-1", "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.95}],
                "lines": [{"line_id": "line-1", "block_id": "block-1", "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.91}],
                "words": [{"text": "Item", "line_id": "line-1", "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "confidence": 0.9}],
                "warnings": [],
            },
        ):
            parsed = ingestion.parse_structured_document(
                filename="scan.png",
                content_bytes=b"\x89PNG\r\n\x1a\n...",
                detected_mime_type="image/png",
                magic_bytes_type="image/png",
            )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertTrue(parsed.tables)
        self.assertEqual(parsed.tables[0]["source"], "image_ocr_table_detection")
        self.assertEqual(parsed.object_structure["ocr"]["table_count"], 1)

    def test_parse_image_warns_when_ocr_dependencies_missing(self):
        parsed = ingestion.parse_structured_document(
            filename="scan.png",
            content_bytes=b"\x89PNG\r\n\x1a\n...",
            detected_mime_type="image/png",
            magic_bytes_type="image/png",
        )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertTrue(any("OCR dependencies" in warning for warning in parsed.warnings))

    def test_parse_pdf_detects_pades_markers_for_provenance(self):
        parsed = ingestion.parse_structured_document(
            filename="signed.pdf",
            content_bytes=(
                b"%PDF-1.7\n"
                b"/Type /Sig\n"
                b"/ByteRange [0 10 20 30]\n"
                b"/SubFilter /ETSI.CAdES.detached\n"
            ),
            detected_mime_type="application/pdf",
            magic_bytes_type="application/pdf",
        )

        signature = parsed.object_structure["signature_compliance"]
        self.assertTrue(signature["signature_present"])
        self.assertIn("PAdES", signature["signature_schemes"])

    def test_parse_csv_infers_schema_metadata(self):
        parsed = ingestion.parse_structured_document(
            filename="orders.csv",
            content_bytes=(
                b"order_id,customer,total\n"
                b"1001,Alice,12.50\n"
                b"1002,Bob,20.00\n"
                b"1003,Charlie,\n"
            ),
            detected_mime_type="text/csv",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "delimited_table_parser")
        schema = parsed.tables[0]["schema_inference"]
        self.assertEqual(schema["column_count"], 3)
        self.assertEqual(schema["columns"][0]["inferred_type"], "integer")
        self.assertGreater(schema["columns"][2]["null_ratio"], 0.0)
        self.assertEqual(schema["key_columns"], ["order_id", "customer"])


    def test_parse_csv_normalizes_header_units_formulas_and_pivot_ranges(self):
        parsed = ingestion.parse_structured_document(
            filename="metrics.csv",
            content_bytes=(
                b"temp_c,humidity(%),notes\n"
                b"20,55,=SUM(A2:B2)\n"
                b"22,50,PIVOT(A1:C3)\n"
            ),
            detected_mime_type="text/csv",
            magic_bytes_type=None,
        )

        table = parsed.tables[0]
        self.assertEqual(table["raw_header"], ["temp_c", "humidity(%)", "notes"])
        self.assertEqual(table["header"], ["temp", "humidity", "notes"])
        self.assertEqual(table["units"], [{"column": "temp", "unit": "c", "source": "header"}, {"column": "humidity", "unit": "%", "source": "header"}])
        self.assertEqual(table["header_rows"][0]["row"], 1)
        self.assertEqual(table["formulas"][0]["expression"], "=SUM(A2:B2)")
        self.assertEqual(table["pivot_ranges"], ["A1:C3"])

        structure = parsed.object_structure["structure"]
        self.assertEqual(structure["units"], table["units"])
        self.assertEqual(structure["pivot_ranges"], ["A1:C3"])

    def test_parse_xlsx_infers_table_structure(self):
        parsed = ingestion.parse_structured_document(
            filename="orders.xlsx",
            content_bytes=_build_sample_xlsx_bytes(),
            detected_mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            magic_bytes_type="application/zip",
        )

        self.assertEqual(parsed.parser, "xlsx_table_parser")
        self.assertTrue(parsed.tables)
        table = parsed.tables[0]
        self.assertEqual(table["sheet_name"], "Sheet1")
        self.assertEqual(table["header"], ["id", "name"])
        self.assertEqual(table["rows"], [["1", "Alice"], ["2", "Bob"]])

    def test_store_upload_csv_persists_schema_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            original_dirs = (
                ingestion.UPLOAD_DIR,
                ingestion.METADATA_DIR,
                ingestion.ARTIFACTS_DIR,
                ingestion.PARSED_DIR,
                ingestion.NORMALIZED_DIR,
                ingestion.VIEWER_ARTIFACTS_DIR,
                ingestion.OBSERVABILITY_LOG_PATH,
                ingestion.OBSERVABILITY_METRICS_PATH,
            )
            ingestion.UPLOAD_DIR = base / "uploads"
            ingestion.METADATA_DIR = ingestion.UPLOAD_DIR / "metadata"
            ingestion.ARTIFACTS_DIR = ingestion.UPLOAD_DIR / "artifacts"
            ingestion.PARSED_DIR = ingestion.ARTIFACTS_DIR / "parsed"
            ingestion.NORMALIZED_DIR = ingestion.ARTIFACTS_DIR / "normalized"
            ingestion.VIEWER_ARTIFACTS_DIR = ingestion.ARTIFACTS_DIR / "viewer"
            ingestion.OBSERVABILITY_LOG_PATH = ingestion.ARTIFACTS_DIR / "processing_logs.jsonl"
            ingestion.OBSERVABILITY_METRICS_PATH = ingestion.ARTIFACTS_DIR / "processing_metrics.json"
            try:
                result = ingestion.store_upload(
                    filename="inventory.csv",
                    content_bytes=b"sku,name,qty\nA-1,Bolt,10\nA-2,Nut,15\n",
                    content_type="text/csv",
                )
                parsed_payload = json.loads(Path(result["parsed_artifact_path"]).read_text(encoding="utf-8"))
                sheet_schema = parsed_payload["object_structure"]["structure"]["sheets"][0]
                self.assertEqual(parsed_payload["parser"], "delimited_table_parser")
                self.assertEqual(sheet_schema["sheet_name"], "Sheet1")
                self.assertEqual(sheet_schema["columns"][2]["inferred_type"], "integer")
            finally:
                (
                    ingestion.UPLOAD_DIR,
                    ingestion.METADATA_DIR,
                    ingestion.ARTIFACTS_DIR,
                    ingestion.PARSED_DIR,
                    ingestion.NORMALIZED_DIR,
                    ingestion.VIEWER_ARTIFACTS_DIR,
                    ingestion.OBSERVABILITY_LOG_PATH,
                    ingestion.OBSERVABILITY_METRICS_PATH,
                ) = original_dirs

    def test_normalize_provenance_includes_signature_compliance_defaults(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="plain_text_parser",
            text="hello",
            layout=[],
            tables=[],
            media=[],
            object_structure={},
            warnings=[],
        )
        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "note.txt",
                "stored_filename": "note_1.txt",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "text/plain",
                "magic_bytes_type": None,
            },
        )

        signature = normalized.provenance["signature_compliance"]
        self.assertFalse(signature["signature_present"])
        self.assertEqual(signature["signature_schemes"], [])



    def test_normalize_generates_viewer_hints_for_page_mapping_and_highlight_spans(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="plain_text_parser",
            text="First paragraph\n\nSecond paragraph",
            layout=[],
            tables=[],
            media=[],
            object_structure={},
            warnings=[],
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "note.txt",
                "stored_filename": "note_1.txt",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "text/plain",
                "magic_bytes_type": None,
            },
        )

        self.assertIn("page_map", normalized.render_hints)
        self.assertIn("text_highlight_spans", normalized.render_hints)
        self.assertEqual(len(normalized.render_hints["page_map"]), 2)
        self.assertEqual(len(normalized.render_hints["text_highlight_spans"]), 2)
        first_span = normalized.render_hints["text_highlight_spans"][0]
        self.assertEqual(first_span["chunk_id"], "note_1.txt:chunk:0")
        self.assertGreater(first_span["end"], first_span["start"])

    def test_normalize_image_populates_overlay_render_hints_from_ocr_layout(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="image_ocr_parser",
            text="Detected text",
            layout=[
                {
                    "type": "ocr_block",
                    "block_id": "block-1",
                    "text": "Detected text",
                    "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
                    "confidence": 0.92,
                }
            ],
            tables=[],
            media=[],
            object_structure={"format": "image"},
            warnings=[],
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "scan.png",
                "stored_filename": "scan_1.png",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "image/png",
                "magic_bytes_type": "image/png",
            },
        )

        self.assertIn("overlay", normalized.render_hints)
        self.assertEqual(len(normalized.render_hints["overlay"]), 1)
        overlay = normalized.render_hints["overlay"][0]
        self.assertEqual(overlay["type"], "ocr_block")
        self.assertEqual(overlay["id"], "block-1")
        self.assertEqual(overlay["bbox"], {"x": 10, "y": 20, "width": 30, "height": 40})
        self.assertEqual(overlay["confidence"], 0.92)

    def test_normalize_image_adds_llm_analysis_chunk_from_image_vector_document(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="image_ocr_parser",
            text="TOTAL 120 EUR",
            layout=[],
            tables=[],
            media=[],
            object_structure={"format": "image"},
            warnings=[],
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "invoice.png",
                "stored_filename": "invoice_1.png",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "image/png",
                "magic_bytes_type": "image/png",
                "image_vector_document": {
                    "caption": "Photo of an invoice on a table.",
                    "objects": ["invoice", "table"],
                    "tags": ["finance", "document"],
                    "ocr_text": "Invoice total: 120 EUR",
                    "metadata": {"analysis": "The image shows a billing document with a visible total amount."},
                },
            },
        )

        image_chunks = [chunk for chunk in normalized.chunks if chunk.get("modality") == "image"]
        self.assertEqual(len(image_chunks), 2)
        self.assertEqual(image_chunks[0]["chunk_strategy"], "image_llm_analysis")
        self.assertIn("Bildbeschreibung", image_chunks[0]["text"])
        self.assertIn("Bildanalyse", image_chunks[0]["text"])
        self.assertIn("invoice", image_chunks[0]["text"].lower())
        self.assertEqual(image_chunks[1]["chunk_strategy"], "image_ocr_layout")


    def test_normalize_image_skips_binary_like_ocr_chunk_text(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="image_ocr_parser",
            text="",
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "format": "image",
                "ocr": {"text": "IHDR IDAT IEND sRGB gAMA pHYs"},
            },
            warnings=[],
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "plot.png",
                "stored_filename": "plot_1.png",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "image/png",
                "magic_bytes_type": "image/png",
            },
        )

        image_chunks = [chunk for chunk in normalized.chunks if chunk.get("modality") == "image"]
        self.assertEqual(len(image_chunks), 1)
        self.assertEqual(image_chunks[0]["chunk_strategy"], "image_fallback")
        self.assertTrue(any("binary metadata" in warning.lower() for warning in normalized.warnings))

    def test_normalize_image_analysis_ignores_binary_like_vector_ocr_text(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="image_ocr_parser",
            text="",
            layout=[],
            tables=[],
            media=[],
            object_structure={"format": "image"},
            warnings=[],
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "plot.png",
                "stored_filename": "plot_1.png",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "image/png",
                "magic_bytes_type": "image/png",
                "image_vector_document": {
                    "caption": "3D plot with peaks",
                    "tags": ["image", "analysis"],
                    "ocr_text": "IHDR IDAT IEND sRGB gAMA pHYs",
                    "metadata": {"analysis": "Two maxima are visible."},
                },
            },
        )

        image_chunks = [chunk for chunk in normalized.chunks if chunk.get("modality") == "image"]
        self.assertEqual(len(image_chunks), 1)
        self.assertEqual(image_chunks[0]["chunk_strategy"], "image_llm_analysis")
        self.assertNotIn("Extrahierter Text (OCR)", image_chunks[0]["text"])


    def test_normalize_image_adds_llm_analysis_chunk_from_parsed_description_meta_when_vector_missing(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="image_ocr_parser",
            text="OCR TEXT",
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "format": "image",
                "ocr": {
                    "text": "OCR TEXT",
                    "description": {
                        "description_text": "Ein Schaltplan mit drei Blöcken.",
                        "analysis_text": "Signalfluss von links nach rechts.",
                        "open_questions": ["Welche Anlage ist dargestellt?"],
                    },
                },
            },
            warnings=[],
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "schema.png",
                "stored_filename": "schema_1.png",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "image/png",
                "magic_bytes_type": "image/png",
            },
        )

        image_chunks = [chunk for chunk in normalized.chunks if chunk.get("modality") == "image"]
        self.assertEqual(len(image_chunks), 2)
        self.assertEqual(image_chunks[0]["chunk_strategy"], "image_llm_analysis")
        self.assertIn("Bildbeschreibung", image_chunks[0]["text"])
        self.assertIn("Bildanalyse", image_chunks[0]["text"])
        self.assertIn("Offene Fragen", image_chunks[0]["text"])
        self.assertEqual(image_chunks[1]["chunk_strategy"], "image_ocr_layout")

    def test_normalize_image_chunks_include_image_reference_source(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="image_ocr_parser",
            text="OCR TEXT",
            layout=[],
            tables=[],
            media=[],
            object_structure={"format": "image", "ocr": {"text": "OCR TEXT"}},
            warnings=[],
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "photo.png",
                "stored_filename": "photo_1.png",
                "source_version": 1,
                "sha256": "hash123",
                "detected_mime_type": "image/png",
                "magic_bytes_type": "image/png",
                "image_blob_uri": "blob://tmp/photo.bin",
                "image_vector_document": {
                    "caption": "A street scene",
                    "metadata": {"provider": "openai", "analysis": "Cars and signs visible."},
                },
            },
        )

        image_chunks = [chunk for chunk in normalized.chunks if chunk.get("modality") == "image"]
        self.assertEqual(len(image_chunks), 2)
        self.assertEqual(image_chunks[0]["source"]["image_ref"], "blob://tmp/photo.bin")
        self.assertEqual(image_chunks[1]["source"]["image_ref"], "blob://tmp/photo.bin")
        self.assertEqual(image_chunks[1]["source"]["sha256"], "hash123")
    def test_parse_dicom_normalizes_medical_metadata_when_dependency_available(self):
        class _Dataset:
            Modality = "MR"
            StudyDate = "20260212"
            StudyTime = "090000"
            Manufacturer = "OpenAI Imaging"
            PatientID = "patient-123"
            StudyInstanceUID = "1.2.3"
            SeriesInstanceUID = "1.2.3.4"
            SOPInstanceUID = "1.2.3.4.5"
            PatientName = "Sensitive Name"

        with patch("backend.ingestion.importlib.util.find_spec", return_value=object()):
            with patch("backend.ingestion.importlib.import_module") as import_module:
                fake_pydicom = Mock()
                fake_pydicom.dcmread.return_value = _Dataset()
                import_module.return_value = fake_pydicom

                parsed = ingestion.parse_structured_document(
                    filename="scan.dcm",
                    content_bytes=b"DICM....",
                    detected_mime_type="application/dicom",
                    magic_bytes_type="application/dicom",
                )

        self.assertEqual(parsed.parser, "dicom_metadata_parser")
        medical = parsed.object_structure["medical_metadata"]
        self.assertTrue(medical["deidentified"])
        self.assertEqual(medical["tags"]["modality"], "MR")
        self.assertIn("patient_id_sha256", medical["pseudonymized"])
        self.assertIn("PatientName", medical["redacted_fields"])

    def test_parse_dicom_warns_when_pydicom_missing(self):
        with patch("backend.ingestion.importlib.util.find_spec", return_value=None):
            parsed = ingestion.parse_structured_document(
                filename="scan.dcm",
                content_bytes=b"DICM....",
                detected_mime_type="application/dicom",
                magic_bytes_type="application/dicom",
            )

        self.assertEqual(parsed.parser, "dicom_metadata_parser")
        self.assertTrue(any("pydicom" in warning for warning in parsed.warnings))
        self.assertFalse(parsed.object_structure["medical_metadata"]["available"])


    def test_parse_automation_gcode_uses_dialect_parser(self):
        parsed = ingestion.parse_structured_document(
            filename="program.gcode",
            content_bytes=b"G90\nG1 X10.0 Y5.0 F100\nM30\n",
            detected_mime_type="text/plain",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "automation_gcode_parser")
        self.assertEqual(parsed.object_structure["automation"]["dialect"], "gcode")
        self.assertTrue(parsed.object_structure["automation"]["program_blocks"])

    def test_parse_automation_models_structural_nodes_for_blocks_subroutines_parameters_comments(self):
        payload = (
            b"DEF MAIN() ; entry point\n"
            b"G1 X10.0 Y5.0 F100 ; move\n"
            b"SET TOOL=3 SPEED=250\n"
            b"END\n"
        )
        parsed = ingestion.parse_structured_document(
            filename="robot.src",
            content_bytes=payload,
            detected_mime_type="text/plain",
            magic_bytes_type=None,
        )

        automation = parsed.object_structure["automation"]
        nodes = automation["structural_nodes"]
        node_types = {node["node_type"] for node in nodes}

        self.assertIn("program_block", node_types)
        self.assertIn("subroutine", node_types)
        self.assertIn("parameter", node_types)
        self.assertIn("comment", node_types)

        parameters = [node for node in nodes if node["node_type"] == "parameter"]
        self.assertTrue(any(param["name"] == "TOOL" and param["value"] == "3" for param in parameters))

        blocks = automation["program_blocks"]
        self.assertTrue(any(block.get("comment") == "move" for block in blocks))
        self.assertTrue(any(block.get("parameters") for block in blocks))

    def test_normalize_extracts_automation_domain_entities(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="automation_gcode_parser",
            text="G1 X10.0 Y5.0 F100\nT1\nALARM200\n",
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "automation": {
                    "dialect": "gcode",
                    "program_blocks": [
                        {
                            "line": 1,
                            "raw": "G1 X10.0 Y5.0 F100",
                            "tokens": ["G1", "X10.0", "Y5.0", "F100"],
                            "parameters": [],
                        },
                        {
                            "line": 2,
                            "raw": "SET TOOL=3 SPEED=250",
                            "tokens": ["SET"],
                            "parameters": [
                                {"name": "TOOL", "value": "3"},
                                {"name": "SPEED", "value": "250"},
                            ],
                        },
                        {
                            "line": 3,
                            "raw": "ALARM200",
                            "tokens": ["ALARM200"],
                            "parameters": [],
                        },
                    ],
                }
            },
            warnings=[],
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "program.gcode",
                "stored_filename": "program_1.gcode",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "text/plain",
                "magic_bytes_type": None,
            },
        )

        entities = normalized.entities
        entity_types = {entity["entity_type"] for entity in entities}
        self.assertIn("axis", entity_types)
        self.assertIn("feed", entity_types)
        self.assertIn("tool", entity_types)
        self.assertIn("alarmcode", entity_types)
        self.assertTrue(any(entity.get("value") == "X10.0" for entity in entities))
        self.assertTrue(any(entity.get("name") == "TOOL" and entity.get("value") == "3" for entity in entities))

    def test_normalize_builds_automation_viewer_hints_for_jump_folding_and_parameter_panel(self):
        parsed_doc = ingestion.ParsedDoc(
            parser="automation_kuka_krl_parser",
            text="DEF MAIN()\nSET TOOL=3 SPEED=250\nEND\n",
            layout=[],
            tables=[],
            media=[],
            object_structure={
                "automation": {
                    "dialect": "kuka_krl",
                    "structural_nodes": [
                        {"node_type": "subroutine", "line": 1, "name": "MAIN", "raw": "DEF MAIN()"},
                        {"node_type": "program_block", "line": 2, "raw": "SET TOOL=3 SPEED=250", "token_count": 3},
                        {"node_type": "parameter", "line": 2, "name": "TOOL", "value": "3"},
                        {"node_type": "parameter", "line": 2, "name": "SPEED", "value": "250"},
                    ],
                    "program_blocks": [
                        {
                            "line": 2,
                            "raw": "SET TOOL=3 SPEED=250",
                            "tokens": ["SET"],
                            "parameters": [
                                {"name": "TOOL", "value": "3"},
                                {"name": "SPEED", "value": "250"},
                            ],
                        }
                    ],
                }
            },
            warnings=[],
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed_doc,
            metadata={
                "source_type": "upload",
                "filename": "robot.src",
                "stored_filename": "robot_1.src",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "text/plain",
                "magic_bytes_type": None,
            },
        )

        automation_hints = normalized.render_hints["automation"]
        self.assertTrue(automation_hints["jump_markers"])
        self.assertTrue(automation_hints["block_folding"])
        self.assertEqual(len(automation_hints["parameter_panel"]), 2)
        self.assertTrue(any(marker.get("label") == "MAIN" for marker in automation_hints["jump_markers"]))
        self.assertTrue(any(block.get("line") == 2 for block in automation_hints["block_folding"]))
        self.assertTrue(any(param.get("name") == "TOOL" and param.get("value") == "3" for param in automation_hints["parameter_panel"]))

    def test_parse_automation_fallback_adds_warning(self):
        parsed = ingestion.parse_structured_document(
            filename="program.nc",
            content_bytes=b"STEP_ONE start\nACTION do_something\n",
            detected_mime_type="application/octet-stream",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "automation_fallback_parser")
        self.assertTrue(any("fallback" in warning.lower() for warning in parsed.warnings))

    def test_validate_upload_accepts_machine_code_extension(self):
        result = ingestion.validate_upload("program.nc", b"G1 X1")
        self.assertEqual(result.status, "success")

    def test_parse_pdf_uses_pdf_parser_baseline(self):
        parsed = ingestion.parse_structured_document(
            filename="file.pdf",
            content_bytes=b"%PDF-1.4 fake",
            detected_mime_type="application/pdf",
            magic_bytes_type="application/pdf",
        )

        self.assertEqual(parsed.parser, "pdf_parser")
        self.assertTrue(parsed.warnings)

    def test_parse_pdf_extracts_embedded_text_when_available(self):
        mock_page = Mock()
        mock_page.extract_text.return_value = "Summary section"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]

        original_find_spec = ingestion.importlib.util.find_spec
        with patch("backend.ingestion.importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = (
                lambda name, *args, **kwargs: object() if name == "pypdf" else original_find_spec(name, *args, **kwargs)
            )
            with patch("pypdf.PdfReader", return_value=mock_reader):
                parsed = ingestion.parse_structured_document(
                    filename="file.pdf",
                    content_bytes=b"%PDF-1.4 fake",
                    detected_mime_type="application/pdf",
                    magic_bytes_type="application/pdf",
                )

        self.assertEqual(parsed.parser, "pdf_parser")
        self.assertEqual(parsed.text, "Summary section")
        self.assertEqual(parsed.object_structure["pdf"]["pages_with_text"], 1)

    def test_parse_pdf_adds_warning_when_pypdf_missing(self):
        original_find_spec = ingestion.importlib.util.find_spec
        with patch("backend.ingestion.importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = (
                lambda name, *args, **kwargs: None if name == "pypdf" else original_find_spec(name, *args, **kwargs)
            )
            parsed = ingestion.parse_structured_document(
                filename="file.pdf",
                content_bytes=b"%PDF-1.4 fake",
                detected_mime_type="application/pdf",
                magic_bytes_type="application/pdf",
            )

        self.assertEqual(parsed.parser, "pdf_parser")
        self.assertEqual(parsed.text, "")
        self.assertTrue(any("pypdf" in warning for warning in parsed.warnings))

    def test_parse_audio_generates_asr_segments_and_diarization(self):
        parsed = ingestion.parse_structured_document(
            filename="meeting.wav",
            content_bytes=b"hello team this is a short meeting transcript sample",
            detected_mime_type="audio/wav",
            magic_bytes_type=None,
        )

        self.assertEqual(parsed.parser, "asr_transcript_parser")
        asr = parsed.object_structure["asr"]
        self.assertTrue(asr["segments"])
        self.assertTrue(asr["words"])
        self.assertTrue(asr["diarization"]["enabled"])

    def test_normalize_audio_creates_audio_modality_chunk(self):
        parsed = ingestion.parse_structured_document(
            filename="meeting.wav",
            content_bytes=b"hello team this is a short meeting transcript sample",
            detected_mime_type="audio/wav",
            magic_bytes_type=None,
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed,
            metadata={
                "source_type": "upload",
                "filename": "meeting.wav",
                "stored_filename": "meeting_1.wav",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "audio/wav",
                "magic_bytes_type": None,
            },
        )

        audio_chunks = [chunk for chunk in normalized.chunks if chunk["modality"] == "audio"]
        self.assertTrue(audio_chunks)
        self.assertEqual(audio_chunks[0]["chunk_strategy"], "asr_chapter_semantic_timeline")


    def test_parse_audio_detects_chapters_from_semantic_markers(self):
        parsed = ingestion.parse_structured_document(
            filename="meeting.wav",
            content_bytes=(
                b"intro project update agenda budget planning "
                b"next roadmap risks summary"
            ),
            detected_mime_type="audio/wav",
            magic_bytes_type=None,
        )

        chapters = parsed.object_structure["asr"]["chapters"]
        self.assertGreaterEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["boundary"], "semantic")

    def test_normalize_audio_prefers_chapter_semantic_timeline_chunks(self):
        parsed = ingestion.parse_structured_document(
            filename="meeting.wav",
            content_bytes=(
                b"kickoff status update agenda blockers next actions summary"
            ),
            detected_mime_type="audio/wav",
            magic_bytes_type=None,
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed,
            metadata={
                "source_type": "upload",
                "filename": "meeting.wav",
                "stored_filename": "meeting_2.wav",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "audio/wav",
                "magic_bytes_type": None,
            },
        )

        audio_chunks = [chunk for chunk in normalized.chunks if chunk["modality"] == "audio"]
        self.assertTrue(audio_chunks)
        self.assertTrue(all(chunk["chunk_strategy"] == "asr_chapter_semantic_timeline" for chunk in audio_chunks))
        self.assertTrue(all(chunk["source"].get("start_sec") is not None for chunk in audio_chunks))

    def test_normalize_audio_adds_transcript_timeline_alignment_hints(self):
        parsed = ingestion.parse_structured_document(
            filename="meeting.wav",
            content_bytes=b"intro update budget next steps summary",
            detected_mime_type="audio/wav",
            magic_bytes_type=None,
        )

        normalized = ingestion.normalize_parsed_document(
            parsed_doc=parsed,
            metadata={
                "source_type": "upload",
                "filename": "meeting.wav",
                "stored_filename": "meeting_3.wav",
                "source_version": 1,
                "sha256": "hash",
                "detected_mime_type": "audio/wav",
                "magic_bytes_type": None,
            },
        )

        alignment = normalized.render_hints["transcript_timeline_alignment"]
        self.assertTrue(alignment)
        self.assertTrue(all(entry["timeline"].get("start_sec") is not None for entry in alignment))

    def test_store_upload_audio_generates_timeline_deeplink_viewer_artifact(self):
        record = ingestion.store_upload(
            filename="meeting.wav",
            content_bytes=b"intro update budget next steps summary",
            content_type="audio/wav",
            source_type="upload",
        )

        manifest_payload = json.loads(Path(record["viewer_artifacts_path"]).read_text(encoding="utf-8"))
        audio_entry = next(item for item in manifest_payload["artifacts"] if item["modality"] == "audio")
        audio_payload = json.loads(Path(audio_entry["path"]).read_text(encoding="utf-8"))

        self.assertEqual(audio_payload["type"], "transcript_timeline_view")
        self.assertTrue(audio_payload["timeline_entries"])
        self.assertTrue(audio_payload["deep_link_support"]["click_to_seek"])
        self.assertIn("media://", audio_payload["timeline_entries"][0]["viewer_link"])
        self.assertTrue(audio_payload["search_hit_jump_targets"])
        self.assertEqual(
            audio_payload["search_hit_jump_targets"][0]["viewer_link"],
            audio_payload["timeline_entries"][0]["viewer_link"],
        )

    def test_validate_upload_accepts_powerpoint_extensions(self):
        self.assertEqual(ingestion.validate_upload("slides.pptx", b"PK\x03\x04").status, "success")
        self.assertEqual(ingestion.validate_upload("legacy.ppt", b"D0CF11E0").status, "success")

    def test_parse_pptx_extracts_slide_text(self):
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "ppt/slides/slide1.xml",
                """<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Titel Folie</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>""",
            )

        parsed = ingestion.parse_structured_document(
            filename="slides.pptx",
            content_bytes=buffer.getvalue(),
            detected_mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            magic_bytes_type="application/zip",
        )

        self.assertEqual(parsed.parser, "pptx_parser")
        self.assertIn("Titel Folie", parsed.text)
        self.assertEqual(parsed.object_structure["presentation"]["slide_count"], 1)

    def test_parse_image_uses_llm_description_when_ocr_empty(self):
        with patch("backend.ingestion._run_image_ocr_pipeline", return_value={"text": "", "layout": [], "blocks": [], "lines": [], "words": [], "warnings": []}):
            with patch("backend.ingestion._run_ocr_layout_analysis", return_value=("", [], [])):
                with patch("backend.ingestion._generate_image_description_with_llm", return_value=("Ein Diagramm mit drei Balken.", [], {"enabled": True, "provider": "openai"})):
                    parsed = ingestion.parse_structured_document(
                        filename="chart.png",
                        content_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 20,
                        detected_mime_type="image/png",
                        magic_bytes_type="image/png",
                    )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertIn("Diagramm", parsed.text)
        self.assertEqual(parsed.object_structure["ocr"]["description"]["provider"], "openai")

    def test_parse_image_uses_llm_description_when_ocr_is_binary_metadata_noise(self):
        noisy_text = "PNG IHDR IDAT sRGB gAMA pHYs IEND"
        with patch("backend.ingestion._run_image_ocr_pipeline", return_value={"text": noisy_text, "layout": [], "blocks": [], "lines": [], "words": [], "warnings": []}):
            with patch("backend.ingestion._generate_image_description_with_llm", return_value=("A bar chart showing quarterly sales.", [], {"enabled": True, "provider": "openai"})):
                parsed = ingestion.parse_structured_document(
                    filename="chart.png",
                    content_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 20,
                    detected_mime_type="image/png",
                    magic_bytes_type="image/png",
                )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertIn("bar chart", parsed.text)
        self.assertEqual(parsed.object_structure["ocr"]["description"]["provider"], "openai")

    def test_parse_image_with_binary_like_payload_does_not_use_embedded_text_branch(self):
        content_bytes = b"\xff\xd8\xff" + b"\x00" * 80 + b"JFIFmeta" + b"\x00" * 80
        with patch(
            "backend.ingestion._run_image_ocr_pipeline",
            return_value={"text": "", "layout": [], "blocks": [], "lines": [], "words": [], "warnings": []},
        ) as ocr_pipeline:
            with patch("backend.ingestion._run_ocr_layout_analysis", return_value=("", [], [])) as heuristic_ocr:
                with patch(
                    "backend.ingestion._generate_image_description_with_llm",
                    return_value=("A product photo with multiple items.", [], {"enabled": True, "provider": "openai"}),
                ):
                    parsed = ingestion.parse_structured_document(
                        filename="photo.jpg",
                        content_bytes=content_bytes,
                        detected_mime_type="image/jpeg",
                        magic_bytes_type="image/jpeg",
                    )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertTrue(ocr_pipeline.called)
        self.assertTrue(heuristic_ocr.called)
        self.assertIn("product photo", parsed.text)

    def test_parse_image_appends_llm_description_even_when_ocr_has_text(self):
        with patch(
            "backend.ingestion._run_image_ocr_pipeline",
            return_value={"text": "Invoice total 120 EUR", "layout": [], "blocks": [], "lines": [], "words": [], "warnings": []},
        ):
            with patch(
                "backend.ingestion._generate_image_description_with_llm",
                return_value=("A photographed invoice on a desk.", [], {"enabled": True, "provider": "gemini"}),
            ):
                parsed = ingestion.parse_structured_document(
                    filename="invoice.png",
                    content_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 20,
                    detected_mime_type="image/png",
                    magic_bytes_type="image/png",
                )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertIn("Invoice total 120 EUR", parsed.text)
        self.assertIn("A photographed invoice on a desk.", parsed.text)
        self.assertEqual(parsed.object_structure["ocr"]["text"], "Invoice total 120 EUR")
        self.assertEqual(parsed.object_structure["ocr"]["description"]["provider"], "gemini")

    def test_parse_image_with_embedded_text_still_appends_llm_description(self):
        with patch(
            "backend.ingestion._generate_image_description_with_llm",
            return_value=("A label image with serial number.", [], {"enabled": True, "provider": "openai"}),
        ):
            parsed = ingestion.parse_structured_document(
                filename="label.png",
                content_bytes=b"\x89PNG\r\n\x1a\nSERIAL NUMBER SN12345",
                detected_mime_type="image/png",
                magic_bytes_type="image/png",
            )

        self.assertIn(parsed.parser, {"ocr_image_parser", "image_ocr_parser"})
        self.assertIn("A label image with serial number.", parsed.text)
        self.assertEqual(parsed.object_structure["ocr"]["description"]["provider"], "openai")

    def test_generate_image_description_prompt_requests_full_image_analysis(self):
        captured = {}

        class _Result:
            def __init__(self):
                self.status = "success"
                self.raw_response = "Beschreibung"
                self.warnings = []

        def _fake_describe_image_with_openai(*, api_key, model, image_bytes, prompt):
            captured["prompt"] = prompt
            return _Result()

        with patch("backend.vision_analyze.describe_image_with_openai", side_effect=_fake_describe_image_with_openai):
            with patch.dict(
                "os.environ",
                {
                    "RAG_IMAGE_DESCRIPTION_PROVIDER": "openai",
                    "OPENAI_API_KEY": "test-key",
                    "RAG_IMAGE_DESCRIPTION_OPENAI_MODEL": "gpt-4.1-mini",
                },
                clear=False,
            ):
                text, warnings, meta = ingestion._generate_image_description_with_llm(
                    filename="scene.png",
                    content_bytes=b"\x89PNG\r\n\x1a\n" + b"abc",
                )

        self.assertEqual(text, "Beschreibung")
        self.assertEqual(warnings, [])
        self.assertEqual(meta["provider"], "openai")
        self.assertIn("präziser Bildanalyst", captured["prompt"])
        self.assertIn("Bitte beschreibe und analysiere", captured["prompt"])
        self.assertIn("offener Fragen", captured["prompt"])


    def test_parse_image_ignores_binary_like_heuristic_ocr_fallback(self):
        with patch("backend.ingestion._run_image_ocr_pipeline", return_value={"text": "", "layout": [], "blocks": [], "lines": [], "words": [], "warnings": []}):
            with patch("backend.ingestion._run_ocr_layout_analysis", return_value=("PNG IHDR IDAT sRGB gAMA pHYs IEND", [], [])):
                with patch("backend.ingestion._generate_image_description_with_llm", return_value=("", [], {"enabled": False})):
                    parsed = ingestion.parse_structured_document(
                        filename="blob.png",
                        content_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
                        detected_mime_type="image/png",
                        magic_bytes_type="image/png",
                    )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertEqual(parsed.text, "")
        self.assertEqual(parsed.object_structure["ocr"]["text"], "")
        self.assertIn(
            "Heuristic OCR output resembled binary image metadata; ignoring fallback text.",
            parsed.warnings,
        )

    def test_binary_like_raw_bytes_do_not_skip_ocr_pipeline(self):
        with patch(
            "backend.ingestion._run_image_ocr_pipeline",
            return_value={"text": "Recognized process diagram", "layout": [], "blocks": [], "lines": [], "words": [], "warnings": []},
        ) as ocr_pipeline:
            with patch(
                "backend.ingestion._generate_image_description_with_llm",
                return_value=("", [], {"enabled": False}),
            ):
                parsed = ingestion.parse_structured_document(
                    filename="blob.png",
                    content_bytes=b"\x89PNG\r\n\x1a\nIHDRIDATsRGBgAMApHYsIEND",
                    detected_mime_type="image/png",
                    magic_bytes_type="image/png",
                )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertTrue(ocr_pipeline.called)
        self.assertIn("Recognized process diagram", parsed.text)

    def test_generate_image_description_prints_warnings_when_no_llm_text(self):
        with patch(
            "backend.ingestion.analyze_image_bytes_with_provider",
            return_value=SimpleNamespace(
                combined_text=None,
                description_text=None,
                analysis_text=None,
                open_questions=[],
                warnings=["Image description skipped: OPENAI_API_KEY not configured."],
                meta={"enabled": True, "provider": "openai"},
            ),
        ):
            with patch("builtins.print") as mocked_print:
                ingestion._generate_image_description_with_llm(
                    filename="missing-key.png",
                    content_bytes=b"test-bytes",
                )

        printed_lines = [call.args[0] for call in mocked_print.call_args_list if call.args]
        self.assertIn("[image-vision] Datei: missing-key.png", printed_lines)
        self.assertIn("[image-vision] Keine LLM-Bildbeschreibung erzeugt.", printed_lines)
        self.assertIn(
            "[image-vision][warning] Image description skipped: OPENAI_API_KEY not configured.",
            printed_lines,
        )

    def test_generate_image_description_prints_split_sections_to_terminal(self):
        with patch(
            "backend.ingestion.analyze_image_bytes_with_provider",
            return_value=SimpleNamespace(
                combined_text="Beschreibung\nAnalyse",
                description_text="Ein Flussdiagramm mit fünf Knoten.",
                analysis_text="Die Pfeile zeigen einen linearen Prozess mit Feedback.",
                open_questions=["Welche Domäne zeigt das Diagramm?"],
                warnings=[],
                meta={"enabled": True, "provider": "openai"},
            ),
        ):
            with patch("builtins.print") as mocked_print:
                ingestion._generate_image_description_with_llm(
                    filename="flow.png",
                    content_bytes=b"\x89PNG\r\n\x1a\n",
                )

        printed_lines = [call.args[0] for call in mocked_print.call_args_list if call.args]
        self.assertIn("[image-vision] Datei: flow.png", printed_lines)
        self.assertIn("[image-vision] Beschreibung:", printed_lines)
        self.assertIn("Ein Flussdiagramm mit fünf Knoten.", printed_lines)
        self.assertIn("[image-vision] Analyse:", printed_lines)
        self.assertIn("Die Pfeile zeigen einen linearen Prozess mit Feedback.", printed_lines)
        self.assertIn("[image-vision] Offene Fragen:", printed_lines)
        self.assertIn("- Welche Domäne zeigt das Diagramm?", printed_lines)

    def test_image_table_extraction_uses_ocr_text_not_binary_payload(self):
        with patch(
            "backend.ingestion._run_image_ocr_pipeline",
            return_value={"text": "name | value\nfoo | 10\nbar | 20", "layout": [], "blocks": [], "lines": [], "words": [], "warnings": []},
        ):
            with patch(
                "backend.ingestion._generate_image_description_with_llm",
                return_value=("", [], {"enabled": False}),
            ):
                parsed = ingestion.parse_structured_document(
                    filename="table.png",
                    content_bytes=b"\x89PNG\r\n\x1a\nIDATBINARYBYTES",
                    detected_mime_type="image/png",
                    magic_bytes_type="image/png",
                )

        self.assertEqual(parsed.parser, "image_ocr_parser")
        self.assertEqual(len(parsed.tables), 1)
        self.assertEqual(parsed.tables[0]["headers"], ["name", "value"])



if __name__ == "__main__":
    unittest.main()
