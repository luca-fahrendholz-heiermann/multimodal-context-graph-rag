import json
import tempfile
import unittest
from pathlib import Path

from backend.image_search_pipeline import (
    detect_image_mime_type,
    generate_structured_description,
    process_image_for_search,
)


class TestImageMimeDetection(unittest.TestCase):
    def test_png_magic_header_detection(self):
        mime, source = detect_image_mime_type(
            filename="sample.bin",
            content_type="application/octet-stream",
            content_bytes=b"\x89PNG\r\n\x1a\nrest",
        )
        self.assertEqual(mime, "image/png")
        self.assertEqual(source, "magic_bytes")

    def test_content_type_mismatch_prefers_magic_bytes(self):
        mime, source = detect_image_mime_type(
            filename="photo.txt",
            content_type="text/plain",
            content_bytes=b"\xff\xd8\xff\xe0rest",
        )
        self.assertEqual(mime, "image/jpeg")
        self.assertEqual(source, "magic_bytes")


class TestSchemaAndFallback(unittest.TestCase):
    def test_json_schema_validation_with_repair(self):
        calls = []

        def fake_vision(_image: bytes, _prompt: str):
            calls.append(1)
            if len(calls) == 1:
                return '{"caption": 123}', []
            return json.dumps(
                {
                    "caption": "A factory floor",
                    "tags": ["factory", "machine"],
                    "objects": ["robot"],
                    "ocr_text": "line A",
                    "metadata": {"quality": "high"},
                }
            ), []

        payload, warnings = generate_structured_description(
            image_bytes=b"img",
            source_uri="blob://x",
            sha256="abc",
            vision_callable=fake_vision,
        )
        self.assertEqual(payload["caption"], "A factory floor")
        self.assertEqual(len(calls), 2)
        self.assertEqual(warnings, [])

    def test_fallback_when_json_remains_invalid(self):
        def fake_vision(_image: bytes, _prompt: str):
            return "not-json", []

        payload, _warnings = generate_structured_description(
            image_bytes=b"img",
            source_uri="blob://x",
            sha256="abc",
            vision_callable=fake_vision,
        )
        self.assertIn("fallback", payload["tags"])
        self.assertEqual(payload["sha256"], "abc")


class TestIdempotency(unittest.TestCase):
    def test_same_sha256_returns_existing_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            blob_dir = base / "blobs"
            index_path = base / "idempotency.json"

            image_bytes = b"\x89PNG\r\n\x1a\nhello"
            def fake_vision(_image: bytes, _prompt: str):
                return (
                    json.dumps(
                        {
                            "caption": "An assembly line",
                            "tags": ["assembly"],
                            "objects": ["conveyor"],
                            "ocr_text": "station 4",
                            "metadata": {},
                        }
                    ),
                    [],
                )

            image_bytes = b"\x89PNG\r\n\x1a\nhello"
            first = process_image_for_search(
                filename="a.png",
                content_type="image/png",
                upload_bytes=image_bytes,
                blob_dir=blob_dir,
                idempotency_index_path=index_path,
                vision_callable=fake_vision,
            )
            second = process_image_for_search(
                filename="a-again.png",
                content_type="image/png",
                upload_bytes=image_bytes,
                blob_dir=blob_dir,
                idempotency_index_path=index_path,
                vision_callable=fake_vision,
            )

            self.assertEqual(first["sha256"], second["sha256"])
            self.assertTrue(second["idempotent_hit"])
            self.assertEqual(first["blob_uri"], second["blob_uri"])

    def test_existing_fallback_record_is_refreshed_when_vision_callable_is_provided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            blob_dir = base / "blobs"
            index_path = base / "idempotency.json"

            image_bytes = b"\x89PNG\r\n\x1a\nhello"
            sha = __import__("hashlib").sha256(image_bytes).hexdigest()
            index_path.write_text(
                json.dumps(
                    {
                        "records": {
                            sha: {
                                "blob_uri": f"blob://{blob_dir / (sha + '.bin')}",
                                "vector_document": {
                                    "caption": "Image upload with unavailable structured description.",
                                    "tags": ["image", "fallback"],
                                    "objects": [],
                                    "ocr_text": "",
                                    "metadata": {"fallback_reason": "invalid_json"},
                                    "source_uri": f"blob://{blob_dir / (sha + '.bin')}",
                                    "sha256": sha,
                                },
                                "derived_text": "Image upload with unavailable structured description.",
                                "embeddings_text": "Image upload with unavailable structured description.",
                                "warnings": [],
                                "sha256": sha,
                                "detected_mime_type": "image/png",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            def fake_vision(_image: bytes, _prompt: str):
                return (
                    json.dumps(
                        {
                            "caption": "A construction machine in a scanned point cloud",
                            "tags": ["construction", "pointcloud"],
                            "objects": ["excavator"],
                            "ocr_text": "",
                            "metadata": {},
                        }
                    ),
                    [],
                )

            refreshed = process_image_for_search(
                filename="img.png",
                content_type="image/png",
                upload_bytes=image_bytes,
                blob_dir=blob_dir,
                idempotency_index_path=index_path,
                vision_callable=fake_vision,
            )

            self.assertIn("construction machine", refreshed["vector_document"]["caption"])
            self.assertNotIn("fallback_reason", refreshed["vector_document"].get("metadata") or {})
            self.assertIn("Idempotency cache refreshed", " ".join(refreshed.get("warnings") or []))


if __name__ == "__main__":
    unittest.main()
