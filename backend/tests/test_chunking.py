import json
import tempfile
import unittest
from pathlib import Path

from backend import chunking, ingestion


class TestChunkingPipeline(unittest.TestCase):
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

    def test_chunking_creates_chunk_metadata(self):
        stored_filename = "sample.txt"
        markdown = "0123456789ABCDEFGHIJ"

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifacts_dir / f"{stored_filename}.md"
        artifact_path.write_text(markdown, encoding="utf-8")

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        source_metadata_path = self.metadata_dir / f"{stored_filename}.json"
        source_metadata_path.write_text(
            json.dumps({"stored_filename": stored_filename}),
            encoding="utf-8",
        )

        result = chunking.chunk_stored_markdown(
            stored_filename,
            chunk_size=8,
            overlap=2,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.chunks), 3)
        self.assertTrue(result.metadata_path)
        self.assertTrue(result.embeddings_path)
        self.assertTrue(result.vector_store_path)
        self.assertTrue(Path(result.metadata_path).exists())
        self.assertTrue(Path(result.embeddings_path).exists())
        self.assertTrue(Path(result.vector_store_path).exists())

        metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
        self.assertEqual(metadata["stored_filename"], stored_filename)
        self.assertEqual(metadata["chunk_count"], 3)
        self.assertIn("source_metadata", metadata)
        self.assertEqual(metadata["chunks"][0]["text"], "01234567")

        embeddings = json.loads(
            Path(result.embeddings_path).read_text(encoding="utf-8")
        )
        self.assertEqual(embeddings["chunk_count"], 3)
        self.assertEqual(embeddings["embedding_dimensions"], 8)
        self.assertEqual(embeddings["embeddings"][0]["index"], 0)

        vector_store_payload = json.loads(
            Path(result.vector_store_path).read_text(encoding="utf-8")
        )
        self.assertIn(stored_filename, vector_store_payload["documents"])

    def test_chunking_rejects_invalid_overlap(self):
        result = chunking.chunk_stored_markdown("missing.txt", chunk_size=5, overlap=5)

        self.assertEqual(result.status, "error")
        self.assertIn("overlap", result.message)


if __name__ == "__main__":
    unittest.main()
