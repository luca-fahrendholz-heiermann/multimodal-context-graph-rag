import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from backend import chunking, ingestion, rag


class TestRagQueryPipeline(unittest.TestCase):
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

    def test_query_returns_ranked_chunks(self):
        stored_filename = "sample.txt"
        markdown = "Hello world from the RAG pipeline."

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifacts_dir / f"{stored_filename}.md"
        artifact_path.write_text(markdown, encoding="utf-8")

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        source_metadata_path = self.metadata_dir / f"{stored_filename}.json"
        source_metadata_path.write_text(
            json.dumps({"stored_filename": stored_filename}),
            encoding="utf-8",
        )

        chunk_result = chunking.chunk_stored_markdown(
            stored_filename,
            chunk_size=12,
            overlap=0,
        )
        self.assertEqual(chunk_result.status, "success")

        query_result = rag.query_rag("Hello world", top_k=2)
        self.assertEqual(query_result.status, "success")
        self.assertEqual(query_result.used_tool, "rag_similarity_search")
        self.assertTrue(query_result.results)
        self.assertEqual(query_result.results[0]["stored_filename"], stored_filename)
        self.assertIn(query_result.results[0]["text"], markdown)
        self.assertIn(
            query_result.results[0]["confidence_hint"],
            {"high", "medium", "low"},
        )

    def test_query_diversifies_results_across_documents(self):
        docs = {
            "email.txt": "Hallo zusammen, anbei meine Bewerbung damit ihr mich ablehnen koennt.",
            "cv.txt": "Technische Skills: Python FastAPI Docker Kubernetes.",
        }

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        for stored_filename, markdown in docs.items():
            (self.artifacts_dir / f"{stored_filename}.md").write_text(markdown, encoding="utf-8")
            (self.metadata_dir / f"{stored_filename}.json").write_text(
                json.dumps({"stored_filename": stored_filename}),
                encoding="utf-8",
            )
            chunk_result = chunking.chunk_stored_markdown(stored_filename, chunk_size=500, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        query_result = rag.query_rag("technische skills", top_k=2)

        self.assertEqual(query_result.status, "success")
        filenames = {entry["stored_filename"] for entry in query_result.results}
        self.assertIn("cv.txt", filenames)
        self.assertIn("email.txt", filenames)


    def test_query_returns_local_answer_text_when_no_api_key(self):
        docs = {
            "email.txt": "Hallo zusammen, anbei meine Bewerbung damit ihr mich ablehnen koennt.",
            "cv.txt": "Technische Kompetenzen: Python, FastAPI, Docker und Kubernetes.",
        }

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        for stored_filename, markdown in docs.items():
            (self.artifacts_dir / f"{stored_filename}.md").write_text(markdown, encoding="utf-8")
            (self.metadata_dir / f"{stored_filename}.json").write_text(
                json.dumps({"stored_filename": stored_filename}),
                encoding="utf-8",
            )
            chunk_result = chunking.chunk_stored_markdown(stored_filename, chunk_size=500, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        query_result = rag.query_rag("Welche technischen Kompetenzen stehen im CV?", top_k=2)

        self.assertEqual(query_result.status, "success")
        self.assertIsInstance(query_result.answer, str)
        self.assertIn("Relevanteste Referenz: [", query_result.answer)



    def test_query_falls_back_when_model_denies_present_information(self):
        docs = {
            "cv.txt": "Kurzprofil: Senior AI Specialist mit Fokus auf RAG-Systeme.",
        }

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        for stored_filename, markdown in docs.items():
            (self.artifacts_dir / f"{stored_filename}.md").write_text(markdown, encoding="utf-8")
            (self.metadata_dir / f"{stored_filename}.json").write_text(
                json.dumps({"stored_filename": stored_filename}),
                encoding="utf-8",
            )
            chunk_result = chunking.chunk_stored_markdown(stored_filename, chunk_size=500, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        with patch("backend.rag.generate_text_with_openai") as mocked_openai:
            mocked_openai.return_value = type("Resp", (), {
                "status": "success",
                "raw_response": "In den bereitgestellten Belegstellen ist kein Kurzprofil enthalten.",
                "warnings": [],
            })()

            query_result = rag.query_rag(
                "Gib mir ein Kurzprofil des CVs",
                top_k=1,
                provider="chatgpt",
                api_key="test-key",
            )

        self.assertEqual(query_result.status, "success")
        self.assertIsInstance(query_result.answer, str)
        self.assertIn("Relevanteste Referenz: [1]", query_result.answer)
        self.assertTrue(any("contradicted retrieved evidence" in warning for warning in query_result.warnings))


    def test_query_falls_back_when_model_has_no_citation(self):
        docs = {
            "cv.txt": "Kurzprofil: Data Engineer mit Fokus auf ETL und Python.",
        }

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        for stored_filename, markdown in docs.items():
            (self.artifacts_dir / f"{stored_filename}.md").write_text(markdown, encoding="utf-8")
            (self.metadata_dir / f"{stored_filename}.json").write_text(
                json.dumps({"stored_filename": stored_filename}),
                encoding="utf-8",
            )
            chunk_result = chunking.chunk_stored_markdown(stored_filename, chunk_size=500, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        with patch("backend.rag.generate_text_with_openai") as mocked_openai:
            mocked_openai.return_value = type("Resp", (), {
                "status": "success",
                "raw_response": "Das Kurzprofil beschreibt einen Data Engineer.",
                "warnings": [],
            })()

            query_result = rag.query_rag(
                "Gib mir ein Kurzprofil des CVs",
                top_k=1,
                provider="chatgpt",
                api_key="test-key",
            )

        self.assertEqual(query_result.status, "success")
        self.assertIn("Relevanteste Referenz: [1]", query_result.answer)
        self.assertTrue(any("missed evidence citations" in warning for warning in query_result.warnings))


    def test_query_can_be_scoped_to_graph_documents(self):
        docs = {
            "include.txt": "Python und FastAPI Projekterfahrung.",
            "exclude.txt": "Vertrauliche Vertragsklauseln ohne Skills.",
        }

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        for stored_filename, markdown in docs.items():
            (self.artifacts_dir / f"{stored_filename}.md").write_text(markdown, encoding="utf-8")
            (self.metadata_dir / f"{stored_filename}.json").write_text(
                json.dumps({"stored_filename": stored_filename}),
                encoding="utf-8",
            )
            chunk_result = chunking.chunk_stored_markdown(stored_filename, chunk_size=500, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        query_result = rag.query_rag("Python", top_k=2, stored_filenames=["include.txt"])

        self.assertEqual(query_result.status, "success")
        self.assertTrue(query_result.results)
        self.assertTrue(all(item["stored_filename"] == "include.txt" for item in query_result.results))

    def test_query_rejects_empty_text(self):
        result = rag.query_rag("   ")
        self.assertEqual(result.status, "error")


    def test_query_answers_document_count_uses_mcp_tools_list_and_call(self):
        with patch("backend.rag.mcp_tools.tools_list") as mock_tools_list, patch(
            "backend.rag.mcp_tools.tools_call"
        ) as mock_tools_call:
            mock_tools_list.return_value = {
                "tools": [
                    {
                        "name": "rag_document_stats",
                        "description": "stats",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }
            mock_tools_call.return_value = {
                "isError": False,
                "content": [
                    {
                        "type": "json",
                        "json": {"document_count": 7, "total_chunks": 10},
                    }
                ],
            }

            result = rag.query_rag("Wie viele Dokumente gibt es im System?")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.answer, "Im aktuellen Index sind 7 Dokumente vorhanden.")
        self.assertEqual(result.used_tool, "rag_document_stats")
        mock_tools_list.assert_called_once()
        mock_tools_call.assert_called_once_with(
            "rag_document_stats",
            {"stored_filename": None, "stored_filenames": None, "file_extensions": [], "document_category": None},
        )

    def test_query_answers_document_count_without_retrieval(self):
        docs = {
            "one.txt": "Erster Inhalt.",
            "two.txt": "Zweiter Inhalt.",
        }

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        for stored_filename, markdown in docs.items():
            (self.artifacts_dir / f"{stored_filename}.md").write_text(markdown, encoding="utf-8")
            (self.metadata_dir / f"{stored_filename}.json").write_text(
                json.dumps({"stored_filename": stored_filename}),
                encoding="utf-8",
            )
            chunk_result = chunking.chunk_stored_markdown(stored_filename, chunk_size=500, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        result = rag.query_rag("Wie viele Dokumente gibt es im System?")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.used_tool, "rag_document_stats")
        self.assertEqual(result.results, [])
        self.assertEqual(result.query_embedding, [])
        self.assertEqual(result.answer, "Im aktuellen Index sind 2 Dokumente vorhanden.")

    def test_query_answers_document_count_with_scope_filter(self):
        docs = {
            "one.txt": "Erster Inhalt.",
            "two.txt": "Zweiter Inhalt.",
        }

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        for stored_filename, markdown in docs.items():
            (self.artifacts_dir / f"{stored_filename}.md").write_text(markdown, encoding="utf-8")
            (self.metadata_dir / f"{stored_filename}.json").write_text(
                json.dumps({"stored_filename": stored_filename}),
                encoding="utf-8",
            )
            chunk_result = chunking.chunk_stored_markdown(stored_filename, chunk_size=500, overlap=0)
            self.assertEqual(chunk_result.status, "success")

        result = rag.query_rag(
            "Wie viele Dokumente gibt es im System?",
            stored_filenames=["one.txt"],
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.answer, "Im aktuellen Index sind 1 Dokumente vorhanden.")
        self.assertEqual(result.used_tool, "rag_document_stats")

    def test_query_answers_pdf_document_count_via_mcp(self):
        with patch("backend.rag.mcp_tools.tools_list") as mock_tools_list, patch(
            "backend.rag.mcp_tools.tools_call"
        ) as mock_tools_call:
            mock_tools_list.return_value = {
                "tools": [
                    {"name": "rag_document_stats", "description": "stats", "inputSchema": {"type": "object"}}
                ]
            }
            mock_tools_call.return_value = {
                "isError": False,
                "content": [{"type": "json", "json": {"document_count": 3, "by_extension": {"pdf": 3}}}],
            }

            result = rag.query_rag("Wie viele PDF-Dateien gibt es?")

        self.assertEqual(result.status, "success")
        self.assertIn("Format pdf", result.answer)
        self.assertEqual(result.used_tool, "rag_document_stats")
        mock_tools_call.assert_called_once_with(
            "rag_document_stats",
            {"stored_filename": None, "stored_filenames": None, "file_extensions": ["pdf"], "document_category": None},
        )


    def test_query_answers_powerpoint_count_via_semantic_category(self):
        with patch("backend.rag.mcp_tools.tools_list") as mock_tools_list, patch(
            "backend.rag.mcp_tools.tools_call"
        ) as mock_tools_call:
            mock_tools_list.return_value = {
                "tools": [
                    {"name": "rag_document_stats", "description": "stats", "inputSchema": {"type": "object"}}
                ]
            }
            mock_tools_call.return_value = {
                "isError": False,
                "content": [{"type": "json", "json": {"document_count": 4, "by_extension": {"pptx": 2, "pdf": 2}}}],
            }

            result = rag.query_rag("Wie viele PowerPoint Präsentationen gibt es?")

        self.assertEqual(result.status, "success")
        self.assertIn("Präsentationsdokumente", result.answer)
        self.assertEqual(result.used_tool, "rag_document_stats")
        mock_tools_call.assert_called_once_with(
            "rag_document_stats",
            {"stored_filename": None, "stored_filenames": None, "file_extensions": [], "document_category": "presentation"},
        )

    def test_query_returns_format_histogram_via_mcp_tool(self):
        with patch("backend.rag.mcp_tools.tools_list") as mock_tools_list, patch(
            "backend.rag.mcp_tools.tools_call"
        ) as mock_tools_call:
            mock_tools_list.return_value = {
                "tools": [
                    {"name": "rag_format_histogram", "description": "chart", "inputSchema": {"type": "object"}}
                ]
            }
            mock_tools_call.return_value = {
                "isError": False,
                "content": [{"type": "json", "json": {"chart": {"svg": "<svg></svg>"}}}],
            }

            result = rag.query_rag("Mach mir ein Häufigkeitsdiagramm über alle Dateiformate im RAG System")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.used_tool, "rag_format_histogram")
        self.assertIsInstance(result.tool_payload, dict)
        self.assertIn("chart", result.tool_payload)



if __name__ == "__main__":
    unittest.main()
