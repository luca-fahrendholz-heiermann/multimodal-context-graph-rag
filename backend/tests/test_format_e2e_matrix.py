import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from backend import ingestion


class TestFormatFamilyE2EMatrix(unittest.TestCase):
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
            ingestion.OBSERVABILITY_LOG_PATH,
            ingestion.OBSERVABILITY_METRICS_PATH,
        ) = self.original_dirs
        self.temp_dir.cleanup()

    def _zip_bytes(self, entries: dict[str, bytes]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in entries.items():
                archive.writestr(name, payload)
        return buffer.getvalue()

    def test_e2e_matrix_per_format_family(self):
        scenario_payloads = {
            "happy": {
                "text": ("sample.txt", b"Hello pipeline\nThis is a valid text document."),
                "image": ("scan.png", b"\x89PNG\r\n\x1a\n" + b"PNGDATA"),
                "table": ("sheet.csv", b"id,name\n1,Alice\n2,Bob\n"),
                "automation": ("program.gcode", b"G0 X0 Y0\nG1 X1 Y1 F300\n"),
                "structured": ("payload.json", b'{"resourceType":"Observation","status":"final"}'),
                "3d": ("model.stl", b"solid cube\nfacet normal 0 0 1\nendsolid cube\n"),
                "audio_video": ("clip.wav", b"RIFF\x24\x00\x00\x00WAVEfmt "),
                "container": ("bundle.zip", self._zip_bytes({"inner.txt": b"inside zip"})),
                "legacy": ("capture.log", b"LEGACY_EVENT 2026-02-14 ok\n"),
            },
            "malformed": {
                "text": ("bad.txt", b"\x00\x00\x00\x01broken"),
                "image": ("broken.png", b"not-a-real-image"),
                "table": ("bad.csv", b"col1,col2\n\"unterminated\n"),
                "automation": ("broken.gcode", b"GOTO ???\n@@@\n"),
                "structured": ("broken.json", b'{"a": 1,, }'),
                "3d": ("broken.stl", b"not really stl"),
                "audio_video": ("broken.wav", b"WAVE??"),
                "container": ("broken.zip", b"PK\x03\x04broken"),
                "legacy": ("broken.log", b"\x81\x8D\x8F\x90"),
            },
            "large_file": {
                "text": ("large.txt", ("line with umlaut ä\n" * 6000).encode("utf-8")),
                "image": ("large.png", b"\x89PNG\r\n\x1a\n" + (b"A" * 50000)),
                "table": ("large.csv", ("id,name\n" + "\n".join(f"{i},row-{i}" for i in range(3000))).encode("utf-8")),
                "automation": ("large.gcode", ("\n".join(f"G1 X{i} Y{i} F1200" for i in range(4000))).encode("utf-8")),
                "structured": ("large.json", json.dumps({"items": [{"id": i, "value": f"v-{i}"} for i in range(2000)]}).encode("utf-8")),
                "3d": ("large.stl", ("solid many\n" + ("facet normal 0 0 1\n" * 3000) + "endsolid many\n").encode("utf-8")),
                "audio_video": ("large.wav", b"RIFF" + (b"\x00" * 120000) + b"WAVE"),
                "container": ("large.zip", self._zip_bytes({"bulk.txt": b"x" * 80000, "nested/info.txt": b"meta"})),
                "legacy": ("large.log", ("legacy-stream-entry\n" * 10000).encode("utf-8")),
            },
            "encoding_edge": {
                "text": ("encoding.txt", "Grüße aus München – mañana".encode("cp1252", errors="replace")),
                "image": ("encoding.png", b"\x89PNG\r\n\x1a\n" + "äöü".encode("cp1252")),
                "table": ("encoding.csv", "id;name\n1;Jörg\n2;Beyoncé\n".encode("cp1252", errors="replace")),
                "automation": ("encoding.gcode", "N10 ; Kühlung an\nG1 X1 Y1\n".encode("cp1252", errors="replace")),
                "structured": ("encoding.yaml", "title: Café\ncity: São Paulo\n".encode("cp1252", errors="replace")),
                "3d": ("encoding.stl", "solid grüße\nendsolid grüße\n".encode("cp1252", errors="replace")),
                "audio_video": ("encoding.wav", b"RIFF\xff\xfe\x00\x00WAVE"),
                "container": ("encoding.zip", self._zip_bytes({"ümlaut.txt": "Übergrößenträger".encode("cp1252", errors="replace")})),
                "legacy": ("encoding.log", "Legacy äöüß entries".encode("cp1252", errors="replace")),
            },
        }

        for scenario, family_payloads in scenario_payloads.items():
            for family, (filename, payload) in family_payloads.items():
                with self.subTest(scenario=scenario, family=family, filename=filename):
                    metadata = ingestion.store_upload(
                        filename=filename,
                        content_bytes=payload,
                        content_type=None,
                        source_type=f"matrix-{scenario}",
                    )

                    self.assertEqual(metadata["filename"], filename)
                    self.assertTrue(metadata.get("sha256"))
                    self.assertIn(metadata.get("qa_status"), {"passed", "warning"})

                    normalized_path = Path(metadata["normalized_artifact_path"])
                    parsed_path = Path(metadata["parsed_artifact_path"])
                    viewer_path = Path(metadata["viewer_artifacts_path"])
                    self.assertTrue(parsed_path.exists())
                    self.assertTrue(normalized_path.exists())
                    self.assertTrue(viewer_path.exists())

                    normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
                    self.assertTrue(normalized_payload["chunks"])
                    self.assertTrue(normalized_payload["embeddings_inputs"])

                    rag_index_path = Path(metadata["rag_index_path"])
                    rag_index = json.loads(rag_index_path.read_text(encoding="utf-8"))
                    self.assertIn(metadata["stored_filename"], rag_index["documents"])
