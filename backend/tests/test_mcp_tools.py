import json
import tempfile
import unittest
from pathlib import Path

from backend import chunking, ingestion, mcp_tools


class TestMcpTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.upload_dir = base / "uploads"
        self.artifacts_dir = self.upload_dir / "artifacts"
        self.metadata_dir = self.upload_dir / "metadata"

        self.original_dirs = (
            ingestion.UPLOAD_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.METADATA_DIR,
        )

        ingestion.UPLOAD_DIR = self.upload_dir
        ingestion.ARTIFACTS_DIR = self.artifacts_dir
        ingestion.METADATA_DIR = self.metadata_dir

    def tearDown(self):
        (
            ingestion.UPLOAD_DIR,
            ingestion.ARTIFACTS_DIR,
            ingestion.METADATA_DIR,
        ) = self.original_dirs
        self.temp_dir.cleanup()

    def test_tools_list_exposes_document_stats_schema(self):
        payload = mcp_tools.tools_list()
        tools = payload.get("tools") or []
        names = {tool.get("name") for tool in tools}
        self.assertIn("rag_document_stats", names)
        self.assertIn("rag_format_histogram", names)

    def test_tools_call_returns_structured_content(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        for name in ("a.txt", "b.txt"):
            (self.artifacts_dir / f"{name}.md").write_text("Inhalt", encoding="utf-8")
            (self.metadata_dir / f"{name}.json").write_text(
                json.dumps({"stored_filename": name}),
                encoding="utf-8",
            )
            chunk_result = chunking.chunk_stored_markdown(name, chunk_size=200, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        payload = mcp_tools.tools_call("rag_document_stats", {"stored_filenames": ["a.txt"]})

        self.assertFalse(payload.get("isError"))
        content = payload.get("content") or []
        self.assertTrue(content)
        self.assertEqual(content[0].get("type"), "json")
        self.assertEqual(content[0].get("json", {}).get("document_count"), 1)

    def test_tools_call_filters_by_extension(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        files = ("a.pdf", "b.docx", "c.pdf")
        for name in files:
            (self.artifacts_dir / f"{name}.md").write_text("Inhalt", encoding="utf-8")
            (self.metadata_dir / f"{name}.json").write_text(
                json.dumps({"stored_filename": name, "filename": name}),
                encoding="utf-8",
            )
            chunk_result = chunking.chunk_stored_markdown(name, chunk_size=200, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        payload = mcp_tools.tools_call("rag_document_stats", {"file_extensions": ["pdf"]})
        self.assertFalse(payload.get("isError"))
        stats = (payload.get("content") or [{}])[0].get("json") or {}
        self.assertEqual(stats.get("document_count"), 2)


    def test_tools_call_filters_presentations_semantically(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        fixtures = [
            ("deck.pptx", {"stored_filename": "deck.pptx", "filename": "deck.pptx"}),
            ("slides.pdf", {"stored_filename": "slides.pdf", "filename": "team_presentation.pdf"}),
            ("report.pdf", {"stored_filename": "report.pdf", "filename": "quarterly_report.pdf"}),
        ]

        for name, metadata in fixtures:
            (self.artifacts_dir / f"{name}.md").write_text("Inhalt", encoding="utf-8")
            (self.metadata_dir / f"{name}.json").write_text(json.dumps(metadata), encoding="utf-8")
            chunk_result = chunking.chunk_stored_markdown(name, chunk_size=200, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        payload = mcp_tools.tools_call("rag_document_stats", {"document_category": "presentation"})
        self.assertFalse(payload.get("isError"))
        stats = (payload.get("content") or [{}])[0].get("json") or {}
        self.assertEqual(stats.get("document_count"), 2)

    def test_tools_call_histogram_contains_svg(self):
        payload = mcp_tools.tools_call("rag_format_histogram", {})
        self.assertFalse(payload.get("isError"))
        data = (payload.get("content") or [{}])[0].get("json") or {}
        self.assertIn("chart", data)
        self.assertIn("svg", data.get("chart", {}))

    def test_document_stats_includes_all_supported_extensions(self):
        payload = mcp_tools.tools_call("rag_document_stats", {})
        self.assertFalse(payload.get("isError"))
        stats = (payload.get("content") or [{}])[0].get("json") or {}
        by_extension = stats.get("by_extension") or {}
        self.assertIn(".pdf", by_extension)
        self.assertIn(".png", by_extension)
        self.assertIn(".json", by_extension)

    def test_tools_call_rejects_unknown_arguments(self):
        payload = mcp_tools.tools_call("rag_document_stats", {"invalid": True})
        self.assertTrue(payload.get("isError"))



if __name__ == "__main__":
    unittest.main()
