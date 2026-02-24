import json
import os
import unittest
from importlib import reload
from unittest.mock import patch

import backend.app as app_module
from backend.llm_provider import LlmJsonResult


class TestFrontendConfig(unittest.TestCase):
    def setUp(self):
        self.original_recipient = os.environ.get("RAG_DEFAULT_EMAIL_RECIPIENT")
        self.original_cors_origins = os.environ.get("RAG_CORS_ALLOWED_ORIGINS")

    def tearDown(self):
        if self.original_recipient is None:
            os.environ.pop("RAG_DEFAULT_EMAIL_RECIPIENT", None)
        else:
            os.environ["RAG_DEFAULT_EMAIL_RECIPIENT"] = self.original_recipient

        if self.original_cors_origins is None:
            os.environ.pop("RAG_CORS_ALLOWED_ORIGINS", None)
        else:
            os.environ["RAG_CORS_ALLOWED_ORIGINS"] = self.original_cors_origins

        reload(app_module)

    def test_frontend_config_returns_default_email_recipient(self):
        os.environ["RAG_DEFAULT_EMAIL_RECIPIENT"] = "support@example.com"
        module = reload(app_module)

        payload = module.frontend_config()

        self.assertEqual(payload.get("default_email_recipient"), "support@example.com")

    def test_default_cors_allows_local_frontend_origin(self):
        os.environ.pop("RAG_CORS_ALLOWED_ORIGINS", None)
        module = reload(app_module)

        self.assertIn("http://localhost:3000", module.CORS_ALLOWED_ORIGINS)
        self.assertIn("http://127.0.0.1:3000", module.CORS_ALLOWED_ORIGINS)

        cors_middleware_entries = [
            entry
            for entry in module.app.user_middleware
            if entry.cls.__name__ == "CORSMiddleware"
        ]
        self.assertTrue(cors_middleware_entries)


class TestLlmConnectionCheck(unittest.TestCase):
    def test_connection_check_success_openai(self):
        module = reload(app_module)
        request = module.LlmConnectionCheckRequest(api_key="test-key", provider="chatgpt")

        with patch("backend.app.generate_text_with_openai") as mock_openai:
            mock_openai.return_value = LlmJsonResult(status="success", raw_response="CONNECTED", warnings=[])
            payload = module.check_llm_connection(request)

        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["connected"])
        self.assertEqual(payload["response_preview"], "CONNECTED")

    def test_connection_check_failure_propagates_warnings(self):
        module = reload(app_module)
        request = module.LlmConnectionCheckRequest(api_key="bad-key", provider="gemini")

        with patch("backend.app.generate_text_with_gemini") as mock_gemini:
            mock_gemini.return_value = LlmJsonResult(
                status="error",
                raw_response=None,
                warnings=["Gemini request failed with HTTP 401."],
            )
            response = module.check_llm_connection(request)

        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.body.decode("utf-8"))
        self.assertFalse(payload["connected"])
        self.assertIn("Gemini request failed with HTTP 401.", payload["warnings"])


class TestRagStoreOverview(unittest.TestCase):
    def test_rag_store_overview_uses_safe_limit_and_returns_payload(self):
        module = reload(app_module)

        with patch("backend.app.get_store_overview") as mock_overview:
            mock_overview.return_value = {
                "updated_at": "2026-02-01T10:00:00+00:00",
                "document_count": 1,
                "total_chunks": 3,
                "documents": [{"stored_filename": "demo.txt", "chunk_count": 3}],
            }
            payload = module.rag_store_overview(max_chunks_per_document=999)

        mock_overview.assert_called_once_with(max_chunks_per_document=10)
        self.assertEqual(payload["status"], "success")
        self.assertIn("overview", payload)
        self.assertEqual(payload["overview"]["document_count"], 1)


class TestRagStoreOverviewFilter(unittest.TestCase):
    def test_filter_endpoint_applies_filters(self):
        module = reload(app_module)
        request = module.RagTableFilterRequest(
            query="zeige rechnungen",
            classification="invoice",
            date_from="2025-01-01T00:00:00+00:00",
            date_to="2025-12-31T23:59:59+00:00",
        )

        with patch("backend.app.get_store_overview") as mock_overview, patch(
            "backend.app.filter_overview_documents"
        ) as mock_filter:
            mock_overview.return_value = {
                "updated_at": "2026-02-01T10:00:00+00:00",
                "document_count": 2,
                "total_chunks": 5,
                "documents": [{"stored_filename": "a.txt", "chunk_count": 2}],
            }
            mock_filter.return_value = [{"stored_filename": "invoice_1.pdf", "chunk_count": 2}]

            payload = module.rag_store_overview_filter(request)

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["overview"]["document_count"], 1)
        mock_filter.assert_called_once_with(
            mock_overview.return_value,
            text_query="zeige rechnungen",
            class_label="invoice",
            date_from="2025-01-01T00:00:00+00:00",
            date_to="2025-12-31T23:59:59+00:00",
            month=None,
            month_from=None,
            month_to=None,
            year=None,
            file_extensions=[],
            document_category=None,
            semantic_doc_ids=None,
        )

    def test_filter_query_heuristics_detects_dates_and_rechnung(self):
        module = reload(app_module)

        parsed = module._parse_filter_query("zeige alle rechnungen von 2025-01-01 bis 2025-12-31")

        self.assertEqual(parsed["classification"], "invoice")
        self.assertTrue(parsed["date_from"].startswith("2025-01-01"))
        self.assertTrue(parsed["date_to"].startswith("2025-12-31"))

    def test_filter_query_heuristics_detects_month_without_year(self):
        module = reload(app_module)

        parsed = module._parse_filter_query("alle dokumente von februar")

        self.assertEqual(parsed["month"], 2)
        self.assertIsNone(parsed["year"])
        self.assertIsNone(parsed["date_from"])
        self.assertIsNone(parsed["date_to"])
        self.assertIsNone(parsed["text_query"])

    def test_filter_query_heuristics_detects_german_month_typo(self):
        module = reload(app_module)

        parsed = module._parse_filter_query("alle dokumente von februrar 2026")

        self.assertTrue(parsed["date_from"].startswith("2026-02-01"))
        self.assertTrue(parsed["date_to"].startswith("2026-02-28"))

    def test_filter_query_heuristics_detects_presentation_and_extension(self):
        module = reload(app_module)

        parsed = module._parse_filter_query("zeige präsentationen als pdf")

        self.assertEqual(parsed["document_category"], "presentation")
        self.assertIn("pdf", parsed["file_extensions"])

    def test_filter_endpoint_month_query_does_not_apply_full_text_constraint(self):
        module = reload(app_module)
        request = module.RagTableFilterRequest(query="alle dokumente im februar")

        with patch("backend.app.get_store_overview") as mock_overview, patch(
            "backend.app.filter_overview_documents"
        ) as mock_filter:
            mock_overview.return_value = {
                "updated_at": "2026-02-01T10:00:00+00:00",
                "document_count": 1,
                "total_chunks": 2,
                "documents": [{"stored_filename": "report_feb.pdf", "chunk_count": 2}],
            }
            mock_filter.return_value = [{"stored_filename": "report_feb.pdf", "chunk_count": 2}]

            module.rag_store_overview_filter(request)

        mock_filter.assert_called_once_with(
            mock_overview.return_value,
            text_query=None,
            class_label=None,
            date_from=None,
            date_to=None,
            month=2,
            month_from=None,
            month_to=None,
            year=None,
            file_extensions=[],
            document_category=None,
            semantic_doc_ids=None,
        )


    def test_filter_query_heuristics_detects_month_range(self):
        module = reload(app_module)

        parsed = module._parse_filter_query("alle dokumente zwischen februar und märz")

        self.assertEqual(parsed["month_from"], 2)
        self.assertEqual(parsed["month_to"], 3)
        self.assertIsNone(parsed["month"])



    def test_semantic_text_match_document_ids_uses_relative_threshold(self):
        module = reload(app_module)

        with patch("backend.app.embed_text") as mock_embed, patch(
            "backend.app.search_embeddings"
        ) as mock_search:
            mock_embed.return_value = [0.1, 0.2]
            mock_search.return_value = [
                {"stored_filename": "bagger_a.pdf", "score": 0.91},
                {"stored_filename": "bagger_a.pdf", "score": 0.76},
                {"stored_filename": "bagger_b.pdf", "score": 0.84},
                {"stored_filename": "wolke.png", "score": 0.55},
            ]

            result = module._semantic_text_match_document_ids("bagger")

        self.assertEqual(result, {"bagger_a.pdf", "bagger_b.pdf"})


    def test_semantic_text_match_document_ids_keeps_best_on_low_scores(self):
        module = reload(app_module)

        with patch("backend.app.embed_text") as mock_embed, patch(
            "backend.app.search_embeddings"
        ) as mock_search:
            mock_embed.return_value = [0.1, 0.2]
            mock_search.return_value = [
                {"stored_filename": "bagger.png", "score": 0.21},
                {"stored_filename": "wolke.png", "score": 0.17},
                {"stored_filename": "kurve.png", "score": 0.11},
            ]

            result = module._semantic_text_match_document_ids("alle dateien die etwas mit bagger beinhalten filtern")

        self.assertEqual(result, {"bagger.png"})

    def test_filter_endpoint_uses_semantic_doc_ids_for_natural_text(self):
        module = reload(app_module)
        request = module.RagTableFilterRequest(query="zeige mir alle dokumente über einen bagger")

        with patch("backend.app.get_store_overview") as mock_overview, patch(
            "backend.app.search_embeddings"
        ) as mock_search, patch("backend.app.filter_overview_documents") as mock_filter:
            mock_overview.return_value = {
                "updated_at": "2026-02-01T10:00:00+00:00",
                "document_count": 1,
                "total_chunks": 2,
                "documents": [{"stored_filename": "bagger_1.pdf", "chunk_count": 2}],
            }
            mock_search.return_value = [{"stored_filename": "bagger_1.pdf"}]
            mock_filter.return_value = [{"stored_filename": "bagger_1.pdf", "chunk_count": 2}]

            payload = module.rag_store_overview_filter(request)

        self.assertEqual(payload["overview"]["document_count"], 1)
        mock_filter.assert_called_once()
        kwargs = mock_filter.call_args.kwargs
        self.assertIsNone(kwargs["text_query"])
        self.assertEqual(kwargs["semantic_doc_ids"], {"bagger_1.pdf"})


    def test_filter_endpoint_uses_metadata_matches_when_semantic_returns_empty(self):
        module = reload(app_module)
        request = module.RagTableFilterRequest(query="zeige bagger 2026")

        with patch("backend.app.get_store_overview") as mock_overview, patch(
            "backend.app.search_embeddings"
        ) as mock_search, patch("backend.app.filter_overview_documents") as mock_filter:
            mock_overview.return_value = {
                "updated_at": "2026-02-01T10:00:00+00:00",
                "document_count": 2,
                "total_chunks": 3,
                "documents": [
                    {"stored_filename": "bagger_2026.pdf", "source_filename": "bagger_2026.pdf", "chunk_count": 2},
                    {"stored_filename": "bericht.pdf", "source_filename": "bericht.pdf", "chunk_count": 1},
                ],
            }
            mock_search.return_value = []
            mock_filter.return_value = [{"stored_filename": "bagger_2026.pdf", "chunk_count": 2}]

            module.rag_store_overview_filter(request)

        kwargs = mock_filter.call_args.kwargs
        self.assertIsNone(kwargs["text_query"])
        self.assertEqual(kwargs["semantic_doc_ids"], {"bagger_2026.pdf"})


class TestDocumentChunksEndpoint(unittest.TestCase):
    def test_document_chunks_includes_chunk_id(self):
        module = reload(app_module)

        with patch.object(module.ingestion, "METADATA_DIR") as mock_metadata_dir:
            from pathlib import Path
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp_dir:
                metadata_path = Path(tmp_dir) / "demo.txt.json"
                metadata_path.write_text(
                    json.dumps(
                        {
                            "chunks": [
                                {
                                    "chunk_id": "demo.txt:chunk:0",
                                    "index": 0,
                                    "text": "Chunk content",
                                    "start": 0,
                                    "end": 13,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                mock_metadata_dir.__truediv__.side_effect = lambda name: Path(tmp_dir) / name

                payload = module.document_chunks("demo.txt")

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["results"][0]["chunk_id"], "demo.txt:chunk:0")

    def test_document_chunks_handles_null_chunks(self):
        module = reload(app_module)

        with patch.object(module.ingestion, "METADATA_DIR") as mock_metadata_dir:
            from pathlib import Path
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp_dir:
                metadata_path = Path(tmp_dir) / "demo.txt.json"
                metadata_path.write_text(
                    json.dumps(
                        {
                            "chunks": None,
                        }
                    ),
                    encoding="utf-8",
                )
                mock_metadata_dir.__truediv__.side_effect = lambda name: Path(tmp_dir) / name

                payload = module.document_chunks("demo.txt")

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["results"], [])

    def test_document_chunks_falls_back_to_normalized_artifact_chunks(self):
        module = reload(app_module)

        with patch.object(module.ingestion, "METADATA_DIR") as mock_metadata_dir:
            from pathlib import Path
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                normalized_path = tmp_path / "demo.normalized.json"
                normalized_path.write_text(
                    json.dumps(
                        {
                            "chunks": [
                                {
                                    "chunk_id": "demo.png:chunk:0",
                                    "text": "Bildbeschreibung:\nBagger und zwei Rahmen.",
                                    "modality": "image",
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

                metadata_path = tmp_path / "demo.png.json"
                metadata_path.write_text(
                    json.dumps(
                        {
                            "chunks": None,
                            "normalized_artifact_path": str(normalized_path),
                        }
                    ),
                    encoding="utf-8",
                )
                mock_metadata_dir.__truediv__.side_effect = lambda name: tmp_path / name

                payload = module.document_chunks("demo.png")

        self.assertEqual(payload["status"], "success")
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["chunk_id"], "demo.png:chunk:0")
        self.assertEqual(payload["results"][0]["chunk_index"], 0)
        self.assertIn("Bildbeschreibung", payload["results"][0]["text"])


class TestRagQueryEndpoint(unittest.TestCase):
    def test_rag_query_uses_graph_document_scope(self):
        module = reload(app_module)
        request = module.RagQueryRequest(
            query_text="frage",
            top_k=3,
            graph_id="g_1",
            graph_version_id="v_1",
        )

        mock_result = type("R", (), {"status": "success", "to_dict": lambda self: {"status": "success", "results": []}})()

        with patch("backend.app.graph_store.resolve_graph_document_ids") as mock_resolve, patch("backend.app.query_rag") as mock_query:
            mock_resolve.return_value = ["doc-a", "doc-b"]
            mock_query.return_value = mock_result

            payload = module.rag_query(request)

        self.assertEqual(payload["status"], "success")
        mock_resolve.assert_called_once_with("g_1", "v_1")
        mock_query.assert_called_once_with(
            "frage",
            top_k=3,
            stored_filename=None,
            stored_filenames=["doc-a", "doc-b"],
            provider=None,
            api_key=None,
        )



class TestGraphEndpoints(unittest.TestCase):
    def test_list_graphs_returns_payload(self):
        module = reload(app_module)

        with patch("backend.app.graph_store.list_graphs") as mock_list:
            mock_list.return_value = {
                "active_graph_id": "g_1",
                "graphs": [{"graph_id": "g_1", "name": "Default", "version_count": 1}],
            }
            payload = module.list_graphs()

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["active_graph_id"], "g_1")

    def test_get_graph_view_not_found(self):
        module = reload(app_module)

        with patch("backend.app.graph_store.get_graph_view", side_effect=KeyError("graph_not_found")):
            response = module.get_graph_view("missing", "v1")

        self.assertEqual(response.status_code, 404)
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["status"], "warning")

    def test_add_edge_success(self):
        module = reload(app_module)
        request = module.AddGraphEdgeRequest(from_doc_id="a", to_doc_id="b", type="references", note="demo")

        with patch("backend.app.graph_store.add_edge") as mock_add:
            mock_add.return_value = {
                "graph_id": "g_1",
                "version_id": "v_1",
                "layout_algo": "spring_layout",
                "layout_seed": 42,
                "nodes": [],
                "edges": [],
                "layout_positions": {},
                "document_options": [],
            }
            payload = module.add_graph_edge("g_1", "v_1", request)

        self.assertEqual(payload["status"], "success")
        mock_add.assert_called_once()


    def test_rebuild_3d_viewer_action_returns_warning_when_glb_not_ready(self):
        module = reload(app_module)
        request = module.Rag3dActionRequest(stored_filename="building.ifc", provider="chatgpt")

        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            upload_dir = base / "uploads"
            metadata_dir = base / "metadata"
            upload_dir.mkdir()
            metadata_dir.mkdir()

            (upload_dir / "building.ifc").write_text("ISO-10303-21;", encoding="utf-8")

            def _fake_prepare(*, metadata, provider, api_key):
                metadata["model_3d_ifc_obj_status"] = "converter_missing"
                metadata["model_3d_conversion_status"] = "pending_tessellation_and_conversion"
                metadata["model_3d_conversion_warnings"] = ["converter missing"]
                metadata["model_3d_ifc_obj_warnings"] = ["converter missing"]

            with patch.object(module.ingestion, "UPLOAD_DIR", upload_dir), patch.object(
                module.ingestion, "METADATA_DIR", metadata_dir
            ), patch("backend.app.ingestion._prepare_3d_pipeline_artifacts", side_effect=_fake_prepare):
                response = module.rag_action_rebuild_3d_viewer(request)

            self.assertEqual(response.status_code, 409)
            payload = json.loads(response.body.decode("utf-8"))
            self.assertEqual(payload["status"], "warning")


class TestOpen3dAction(unittest.TestCase):
    def test_open3d_view_document_for_ply_triggers_launcher(self):
        module = reload(app_module)

        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            upload_dir = Path(tmp_dir)
            (upload_dir / "cloud.ply").write_text("ply\nformat ascii 1.0\n", encoding="utf-8")

            with patch.object(module.ingestion, "UPLOAD_DIR", upload_dir), patch(
                "backend.app._launch_open3d_ply_viewer", return_value=(True, "launched")
            ) as mock_launch:
                payload = module.open3d_view_document("cloud.ply")

            self.assertEqual(payload["status"], "success")
            mock_launch.assert_called_once()


    def test_open3d_view_document_returns_error_when_launcher_fails(self):
        module = reload(app_module)

        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            upload_dir = Path(tmp_dir)
            (upload_dir / "cloud.ply").write_text("ply\nformat ascii 1.0\n", encoding="utf-8")

            with patch.object(module.ingestion, "UPLOAD_DIR", upload_dir), patch(
                "backend.app._launch_open3d_ply_viewer", return_value=(False, "No graphical display detected")
            ):
                response = module.open3d_view_document("cloud.ply")

            self.assertEqual(response.status_code, 500)
            payload = json.loads(response.body.decode("utf-8"))
            self.assertEqual(payload["status"], "error")
    def test_open3d_view_document_rejects_non_ply(self):
        module = reload(app_module)

        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            upload_dir = Path(tmp_dir)
            (upload_dir / "mesh.obj").write_text("v 0 0 0\n", encoding="utf-8")

            with patch.object(module.ingestion, "UPLOAD_DIR", upload_dir):
                response = module.open3d_view_document("mesh.obj")

            self.assertEqual(response.status_code, 400)


class TestRagIfcActions(unittest.TestCase):
    def test_rebuild_3d_viewer_action_rejects_non_ifc(self):
        module = reload(app_module)
        request = module.Rag3dActionRequest(stored_filename="demo.obj")

        with patch.object(module.ingestion, "UPLOAD_DIR") as mock_upload_dir:
            from pathlib import Path
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp_dir:
                source_path = Path(tmp_dir) / "demo.obj"
                source_path.write_text("v 0 0 0", encoding="utf-8")
                mock_upload_dir.__truediv__.side_effect = lambda name: Path(tmp_dir) / name

                response = module.rag_action_rebuild_3d_viewer(request)

        self.assertEqual(response.status_code, 400)

    def test_rebuild_3d_viewer_action_rebuilds_ifc_and_persists_metadata(self):
        module = reload(app_module)
        request = module.Rag3dActionRequest(stored_filename="building.ifc", provider="chatgpt")

        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            upload_dir = base / "uploads"
            metadata_dir = base / "metadata"
            upload_dir.mkdir()
            metadata_dir.mkdir()

            (upload_dir / "building.ifc").write_text("ISO-10303-21;", encoding="utf-8")

            def _fake_prepare(*, metadata, provider, api_key):
                metadata["model_3d_ifc_obj_status"] = "converted"
                metadata["model_3d_ifc_obj_path"] = str(upload_dir / "building.obj")
                metadata["model_3d_conversion_status"] = "converted_to_glb"
                metadata["model_3d_canonical_glb_path"] = str(base / "viewer" / "building.ifc.canonical.glb")
                metadata["model_3d_conversion_warnings"] = []
                metadata["model_3d_ifc_obj_warnings"] = []

            with patch.object(module.ingestion, "UPLOAD_DIR", upload_dir), patch.object(
                module.ingestion, "METADATA_DIR", metadata_dir
            ), patch("backend.app.ingestion._prepare_3d_pipeline_artifacts", side_effect=_fake_prepare):
                payload = module.rag_action_rebuild_3d_viewer(request)

            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["conversion_status"], "converted_to_glb")

            persisted = json.loads((metadata_dir / "building.ifc.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["model_3d_ifc_obj_status"], "converted")
            self.assertEqual(persisted["model_3d_conversion_status"], "converted_to_glb")


if __name__ == "__main__":
    unittest.main()


class TestRagIfcActions(unittest.TestCase):
    def test_rebuild_3d_viewer_action_rejects_non_ifc(self):
        module = reload(app_module)
        request = module.Rag3dActionRequest(stored_filename="demo.obj")

        with patch.object(module.ingestion, "UPLOAD_DIR") as mock_upload_dir:
            from pathlib import Path
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp_dir:
                source_path = Path(tmp_dir) / "demo.obj"
                source_path.write_text("v 0 0 0", encoding="utf-8")
                mock_upload_dir.__truediv__.side_effect = lambda name: Path(tmp_dir) / name

                response = module.rag_action_rebuild_3d_viewer(request)

        self.assertEqual(response.status_code, 400)

    def test_rebuild_3d_viewer_action_rebuilds_ifc_and_persists_metadata(self):
        module = reload(app_module)
        request = module.Rag3dActionRequest(stored_filename="building.ifc", provider="chatgpt")

        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            upload_dir = base / "uploads"
            metadata_dir = base / "metadata"
            upload_dir.mkdir()
            metadata_dir.mkdir()

            (upload_dir / "building.ifc").write_text("ISO-10303-21;", encoding="utf-8")

            def _fake_prepare(*, metadata, provider, api_key):
                metadata["model_3d_ifc_obj_status"] = "converted"
                metadata["model_3d_ifc_obj_path"] = str(upload_dir / "building.obj")
                metadata["model_3d_conversion_status"] = "converted_to_glb"
                metadata["model_3d_canonical_glb_path"] = str(base / "viewer" / "building.ifc.canonical.glb")
                metadata["model_3d_conversion_warnings"] = []
                metadata["model_3d_ifc_obj_warnings"] = []

            with patch.object(module.ingestion, "UPLOAD_DIR", upload_dir), patch.object(
                module.ingestion, "METADATA_DIR", metadata_dir
            ), patch("backend.app.ingestion._prepare_3d_pipeline_artifacts", side_effect=_fake_prepare):
                payload = module.rag_action_rebuild_3d_viewer(request)

            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["conversion_status"], "converted_to_glb")

            persisted = json.loads((metadata_dir / "building.ifc.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["model_3d_ifc_obj_status"], "converted")
            self.assertEqual(persisted["model_3d_conversion_status"], "converted_to_glb")
