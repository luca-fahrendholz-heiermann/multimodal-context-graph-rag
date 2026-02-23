import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend import chroma_store


class TestChromaStoreModuleLoading(unittest.TestCase):
    def test_load_chromadb_module_returns_none_when_import_fails(self):
        with patch("backend.chroma_store.chroma_available", return_value=True), patch(
            "backend.chroma_store.importlib.import_module", side_effect=AttributeError("np.float_ removed")
        ):
            self.assertIsNone(chroma_store._load_chromadb_module())


class _FakeCollection:
    def __init__(self):
        self.deleted_where: list[dict] = []
        self.upsert_payload: dict | None = None
        self.query_payload: dict | None = None

    def delete(self, where: dict):
        self.deleted_where.append(where)

    def upsert(self, **kwargs):
        self.upsert_payload = kwargs

    def query(self, **kwargs):
        self.query_payload = kwargs
        return {
            "documents": [["Chunk Inhalt"]],
            "metadatas": [[{"stored_filename": "demo.txt", "chunk_index": 0, "embedding_dimensions": 3, "chunk_metadata_path": "chunks-demo.json"}]],
            "distances": [[0.2]],
        }


class _FakeClient:
    def __init__(self, collection: _FakeCollection):
        self.collection = collection
        self.collection_calls: list[dict] = []

    def get_or_create_collection(self, **kwargs):
        self.collection_calls.append(kwargs)
        return self.collection


class _FakeChromaModule:
    def __init__(self):
        self.collection = _FakeCollection()
        self.client = _FakeClient(self.collection)

    def PersistentClient(self, path: str):
        return self.client


class TestChromaStoreOperations(unittest.TestCase):
    def test_upsert_uses_cosine_collection_and_filters_invalid_dimensions(self):
        fake_chroma = _FakeChromaModule()
        payload = {
            "stored_filename": "demo.txt",
            "embedding_dimensions": 3,
            "embeddings": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3], "text": "ok"},
                {"index": 1, "embedding": [0.1, 0.2], "text": "skip"},
            ],
            "chunk_metadata_path": "chunks-demo.json",
            "embeddings_path": "embeddings-demo.json",
        }

        with patch("backend.chroma_store._load_chromadb_module", return_value=fake_chroma):
            with TemporaryDirectory() as temp_dir:
                result = chroma_store.chroma_upsert_embeddings(
                    persist_dir=Path(temp_dir),
                    embedding_payload=payload,
                    remove_stored_filenames=[],
                )

        self.assertEqual(result, {"backend": "chroma", "written": 1})
        self.assertEqual(fake_chroma.client.collection_calls[0]["metadata"], {"hnsw:space": "cosine"})
        assert fake_chroma.collection.upsert_payload is not None
        self.assertEqual(fake_chroma.collection.upsert_payload["ids"], ["demo.txt:0"])

    def test_query_returns_documents_scores_and_metric(self):
        fake_chroma = _FakeChromaModule()

        with patch("backend.chroma_store._load_chromadb_module", return_value=fake_chroma):
            with TemporaryDirectory() as temp_dir:
                result = chroma_store.chroma_search_embeddings(
                    persist_dir=Path(temp_dir),
                    query_embedding=[0.1, 0.2, 0.3],
                    top_k=1,
                    stored_filename="demo.txt",
                )

        self.assertEqual(len(result or []), 1)
        assert result is not None
        self.assertEqual(result[0]["document"], "Chunk Inhalt")
        self.assertEqual(result[0]["metric"], "cosine")
        self.assertAlmostEqual(result[0]["score"], 0.8)


if __name__ == "__main__":
    unittest.main()
