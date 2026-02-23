import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import ingestion, vector_store


class TestVectorStoreOverviewFilters(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.upload_dir = base / "uploads"
        self.metadata_dir = self.upload_dir / "metadata"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.original_upload_dir = ingestion.UPLOAD_DIR
        self.original_metadata_dir = ingestion.METADATA_DIR

        ingestion.UPLOAD_DIR = self.upload_dir
        ingestion.METADATA_DIR = self.metadata_dir

    def tearDown(self):
        ingestion.UPLOAD_DIR = self.original_upload_dir
        ingestion.METADATA_DIR = self.original_metadata_dir
        self.temp_dir.cleanup()

    def test_filter_overview_documents_by_label_and_date(self):
        overview = {
            "documents": [
                {
                    "stored_filename": "invoice_jan.pdf",
                    "source_filename": "invoice_jan.pdf",
                    "source_type": "upload",
                    "source_timestamp": "2025-01-15T09:00:00+00:00",
                    "classification": {"label": "invoice", "confidence": 0.91},
                    "chunk_count": 2,
                },
                {
                    "stored_filename": "contract_feb.pdf",
                    "source_filename": "contract_feb.pdf",
                    "source_type": "upload",
                    "source_timestamp": "2025-02-15T09:00:00+00:00",
                    "classification": {"label": "contract", "confidence": 0.82},
                    "chunk_count": 3,
                },
            ]
        }

        filtered = vector_store.filter_overview_documents(
            overview,
            text_query="invoice",
            class_label="invoice",
            date_from="2025-01-01T00:00:00+00:00",
            date_to="2025-01-31T23:59:59+00:00",
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["stored_filename"], "invoice_jan.pdf")


    def test_filter_overview_documents_supports_month_without_explicit_year(self):
        overview = {
            "documents": [
                {
                    "stored_filename": "report_feb_2026.pdf",
                    "source_filename": "report_feb_2026.pdf",
                    "source_type": "upload",
                    "source_timestamp": "2026-02-11T09:00:00+00:00",
                    "classification": {"label": "report", "confidence": 0.91},
                    "chunk_count": 2,
                    "embedding_dimensions": 8,
                    "index_status": "indexed",
                },
                {
                    "stored_filename": "report_mar_2026.pdf",
                    "source_filename": "report_mar_2026.pdf",
                    "source_type": "upload",
                    "source_timestamp": "2026-03-02T09:00:00+00:00",
                    "classification": {"label": "report", "confidence": 0.90},
                    "chunk_count": 1,
                    "embedding_dimensions": 8,
                    "index_status": "indexed",
                },
            ]
        }

        filtered = vector_store.filter_overview_documents(
            overview,
            text_query="report",
            month=2,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["stored_filename"], "report_feb_2026.pdf")


    def test_filter_overview_documents_supports_month_range(self):
        overview = {
            "documents": [
                {
                    "stored_filename": "report_feb.pdf",
                    "source_filename": "report_feb.pdf",
                    "source_timestamp": "2026-02-11T09:00:00+00:00",
                    "classification": {"label": "report", "confidence": 0.91},
                },
                {
                    "stored_filename": "report_mar.pdf",
                    "source_filename": "report_mar.pdf",
                    "source_timestamp": "2026-03-11T09:00:00+00:00",
                    "classification": {"label": "report", "confidence": 0.91},
                },
                {
                    "stored_filename": "report_apr.pdf",
                    "source_filename": "report_apr.pdf",
                    "source_timestamp": "2026-04-11T09:00:00+00:00",
                    "classification": {"label": "report", "confidence": 0.91},
                },
            ]
        }

        filtered = vector_store.filter_overview_documents(
            overview,
            month_from=2,
            month_to=3,
        )

        self.assertEqual(len(filtered), 2)
        self.assertEqual({doc["stored_filename"] for doc in filtered}, {"report_feb.pdf", "report_mar.pdf"})

    def test_filter_overview_documents_supports_semantic_doc_ids(self):
        overview = {
            "documents": [
                {"stored_filename": "bagger_1.pdf", "source_filename": "bagger_1.pdf"},
                {"stored_filename": "other.pdf", "source_filename": "other.pdf"},
            ]
        }

        filtered = vector_store.filter_overview_documents(
            overview,
            semantic_doc_ids={"bagger_1.pdf"},
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["stored_filename"], "bagger_1.pdf")

    def test_filter_overview_documents_by_file_extension(self):
        overview = {
            "documents": [
                {
                    "stored_filename": "slides.pptx",
                    "source_filename": "slides.pptx",
                    "source_timestamp": "2026-02-11T09:00:00+00:00",
                    "classification": {"label": "report", "confidence": 0.91},
                },
                {
                    "stored_filename": "report.pdf",
                    "source_filename": "report.pdf",
                    "source_timestamp": "2026-02-11T09:00:00+00:00",
                    "classification": {"label": "report", "confidence": 0.91},
                },
            ]
        }

        filtered = vector_store.filter_overview_documents(overview, file_extensions=["pptx"])

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["stored_filename"], "slides.pptx")

    def test_filter_overview_documents_by_presentation_category(self):
        overview = {
            "documents": [
                {
                    "stored_filename": "team_presentation.pdf",
                    "source_filename": "team_presentation.pdf",
                    "source_timestamp": "2026-02-11T09:00:00+00:00",
                    "classification": {"label": "report", "confidence": 0.91},
                },
                {
                    "stored_filename": "general_report.pdf",
                    "source_filename": "general_report.pdf",
                    "source_timestamp": "2026-02-11T09:00:00+00:00",
                    "classification": {"label": "report", "confidence": 0.91},
                },
            ]
        }

        filtered = vector_store.filter_overview_documents(overview, document_category="presentation")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["stored_filename"], "team_presentation.pdf")

    def test_upsert_rag_document_persists_container_provenance_graph(self):
        first = vector_store.upsert_rag_document(
            stored_filename="bundle.zip",
            metadata={"filename": "bundle.zip"},
            chunks=[{"chunk_id": "bundle.zip:chunk:0", "text": "content"}],
            relations=[
                {
                    "type": "contains",
                    "source": "bundle.zip",
                    "target": "docs/a.txt",
                    "target_sha256": "abc123",
                }
            ],
        )
        self.assertTrue(Path(first["path"]).exists())

        vector_store.upsert_rag_document(
            stored_filename="child.json",
            metadata={"filename": "child.json"},
            chunks=[{"chunk_id": "child.json:chunk:0", "text": "child"}],
            relations=[
                {
                    "type": "derived_from",
                    "source": "child.json",
                    "target": "bundle.zip",
                }
            ],
        )

        rag_index = Path(first["path"]).read_text(encoding="utf-8")
        import json

        payload = json.loads(rag_index)
        provenance_graph = payload.get("provenance_graph") or {}
        self.assertGreaterEqual(provenance_graph.get("node_count", 0), 3)
        self.assertGreaterEqual(provenance_graph.get("edge_count", 0), 2)

        edge_types = {edge["type"] for edge in provenance_graph.get("edges", [])}
        self.assertIn("contains", edge_types)
        self.assertIn("derived_from", edge_types)

        contains_edge = next(
            edge for edge in provenance_graph.get("edges", []) if edge["type"] == "contains"
        )
        self.assertEqual(contains_edge["target_sha256"], "abc123")


class TestVectorStoreIdempotentUpserts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.upload_dir = base / "uploads"
        self.metadata_dir = self.upload_dir / "metadata"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.original_upload_dir = ingestion.UPLOAD_DIR
        self.original_metadata_dir = ingestion.METADATA_DIR

        ingestion.UPLOAD_DIR = self.upload_dir
        ingestion.METADATA_DIR = self.metadata_dir

    def tearDown(self):
        ingestion.UPLOAD_DIR = self.original_upload_dir
        ingestion.METADATA_DIR = self.original_metadata_dir
        self.temp_dir.cleanup()

    def test_upsert_rag_document_replaces_entries_with_same_sha256(self):
        shared_hash = "deadbeef" * 8

        vector_store.upsert_rag_document(
            stored_filename="doc_v1.txt",
            metadata={"filename": "doc.txt", "sha256": shared_hash},
            chunks=[{"chunk_id": "doc_v1:0", "text": "v1"}],
            relations=[],
        )
        vector_store.upsert_rag_document(
            stored_filename="doc_v2.txt",
            metadata={"filename": "doc.txt", "sha256": shared_hash},
            chunks=[{"chunk_id": "doc_v2:0", "text": "v2"}],
            relations=[],
        )

        payload = vector_store._load_rag_index()
        self.assertEqual(set(payload["documents"].keys()), {"doc_v2.txt"})

    def test_upsert_embeddings_replaces_entries_with_same_sha256(self):
        shared_hash = "feedface" * 8
        first_metadata = self.metadata_dir / "doc_v1.txt.json"
        second_metadata = self.metadata_dir / "doc_v2.txt.json"
        first_metadata.write_text('{"sha256": "%s"}' % shared_hash, encoding="utf-8")
        second_metadata.write_text('{"sha256": "%s"}' % shared_hash, encoding="utf-8")

        vector_store.upsert_embeddings(
            {
                "stored_filename": "doc_v1.txt",
                "embedding_dimensions": 8,
                "chunk_count": 1,
                "embeddings": [{"index": 0, "embedding": [0.1] * 8}],
                "chunk_metadata_path": "chunks-v1.json",
                "embeddings_path": "embeddings-v1.json",
            }
        )
        vector_store.upsert_embeddings(
            {
                "stored_filename": "doc_v2.txt",
                "embedding_dimensions": 8,
                "chunk_count": 1,
                "embeddings": [{"index": 0, "embedding": [0.2] * 8}],
                "chunk_metadata_path": "chunks-v2.json",
                "embeddings_path": "embeddings-v2.json",
            }
        )

        payload = vector_store._load_vector_store()
        self.assertEqual(set(payload["documents"].keys()), {"doc_v2.txt"})



class TestVectorStoreSearchRobustness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.upload_dir = base / "uploads"
        self.metadata_dir = self.upload_dir / "metadata"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.original_upload_dir = ingestion.UPLOAD_DIR
        self.original_metadata_dir = ingestion.METADATA_DIR

        ingestion.UPLOAD_DIR = self.upload_dir
        ingestion.METADATA_DIR = self.metadata_dir

    def tearDown(self):
        ingestion.UPLOAD_DIR = self.original_upload_dir
        ingestion.METADATA_DIR = self.original_metadata_dir
        self.temp_dir.cleanup()

    def test_search_embeddings_boosts_lexical_match(self):
        with patch("backend.vector_store.chroma_store.chroma_search_embeddings", return_value=None):
            vector_store.upsert_embeddings(
                {
                    "stored_filename": "cv.txt",
                    "embedding_dimensions": 8,
                    "chunk_count": 1,
                    "embeddings": [{"index": 0, "text": "Kurzprofil Senior Data Engineer", "embedding": [0.1] * 8}],
                    "chunk_metadata_path": "chunks-cv.json",
                    "embeddings_path": "embeddings-cv.json",
                }
            )
            vector_store.upsert_embeddings(
                {
                    "stored_filename": "noise.txt",
                    "embedding_dimensions": 8,
                    "chunk_count": 1,
                    "embeddings": [{"index": 0, "text": "Random unrelated text", "embedding": [0.1] * 8}],
                    "chunk_metadata_path": "chunks-noise.json",
                    "embeddings_path": "embeddings-noise.json",
                }
            )

            results = vector_store.search_embeddings(
                query_embedding=[0.1] * 8,
                top_k=1,
                query_text="kurzprofil",
            )

            self.assertEqual(results[0]["stored_filename"], "cv.txt")
            self.assertGreater(results[0]["lexical_score"], 0)



class TestVectorStoreChromaBridge(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.upload_dir = base / "uploads"
        self.metadata_dir = self.upload_dir / "metadata"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.original_upload_dir = ingestion.UPLOAD_DIR
        self.original_metadata_dir = ingestion.METADATA_DIR

        ingestion.UPLOAD_DIR = self.upload_dir
        ingestion.METADATA_DIR = self.metadata_dir

    def tearDown(self):
        ingestion.UPLOAD_DIR = self.original_upload_dir
        ingestion.METADATA_DIR = self.original_metadata_dir
        self.temp_dir.cleanup()

    def test_upsert_embeddings_writes_json_and_calls_chroma_bridge(self):
        with patch("backend.vector_store.chroma_store.chroma_upsert_embeddings") as chroma_upsert:
            chroma_upsert.return_value = {"backend": "chroma", "written": 1}
            result = vector_store.upsert_embeddings(
                {
                    "stored_filename": "demo.txt",
                    "embedding_dimensions": 4,
                    "chunk_count": 1,
                    "embeddings": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4], "text": "demo"}],
                    "chunk_metadata_path": "chunks-demo.json",
                    "embeddings_path": "embeddings-demo.json",
                }
            )

        self.assertTrue(Path(result["path"]).exists())
        chroma_upsert.assert_called_once()
        self.assertEqual(result["chroma"], {"backend": "chroma", "written": 1})

    def test_search_embeddings_prefers_chroma_result_when_available(self):
        with patch("backend.vector_store.chroma_store.chroma_search_embeddings") as chroma_search:
            chroma_search.return_value = [
                {
                    "stored_filename": "demo.txt",
                    "chunk_index": 0,
                    "score": 0.9,
                    "vector_score": 0.9,
                    "lexical_score": 0.0,
                    "chunk_metadata_path": "chunks-demo.json",
                    "embedding_dimensions": 4,
                }
            ]
            result = vector_store.search_embeddings(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=1)

        chroma_search.assert_called_once()
        self.assertEqual(result[0]["stored_filename"], "demo.txt")

if __name__ == "__main__":
    unittest.main()
