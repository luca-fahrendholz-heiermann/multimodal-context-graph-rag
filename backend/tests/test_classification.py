import json
import unittest

from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from backend import ingestion
from backend.classification import (
    classify_text,
    select_classification_labels,
    store_classification_metadata,
)
from backend.classification_config import LabelWhitelist


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.whitelist = LabelWhitelist(labels=["finance", "legal", "hr"], source="default")

    def test_selects_labels_from_whitelist(self):
        result = select_classification_labels(["finance", "hr"], self.whitelist)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.labels, ["finance", "hr"])

    def test_rejects_unknown_labels(self):
        result = select_classification_labels(["unknown"], self.whitelist)
        self.assertEqual(result.status, "error")
        self.assertIn("Unsupported labels", result.warnings[0])

    def test_classifies_text_with_confidence(self):
        labels = ["finance", "legal", "hr"]
        result = classify_text("Invoice payment received", labels)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.label, "finance")
        self.assertIsInstance(result.confidence, float)

    def test_stores_classification_metadata(self):
        labels = ["finance", "legal", "hr"]
        result = classify_text("Invoice payment received", labels)
        stored_filename = "example.txt"

        with TemporaryDirectory() as temp_dir:
            metadata_dir = Path(temp_dir) / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            source_metadata_path = metadata_dir / f"{stored_filename}.json"
            source_metadata_path.write_text(
                json.dumps({"stored_filename": stored_filename, "source_type": "upload"}),
                encoding="utf-8",
            )

            original_metadata_dir = ingestion.METADATA_DIR
            ingestion.METADATA_DIR = metadata_dir
            try:
                store_result = store_classification_metadata(stored_filename, result)
                self.assertEqual(store_result["status"], "success")

                classification_path = metadata_dir / f"{stored_filename}.classification.json"
                self.assertTrue(classification_path.exists())
                stored_metadata = json.loads(
                    classification_path.read_text(encoding="utf-8")
                )
                self.assertEqual(stored_metadata["label"], "finance")
                self.assertIn("source_metadata", stored_metadata)

                updated_source = json.loads(
                    source_metadata_path.read_text(encoding="utf-8")
                )
                self.assertEqual(updated_source["classification"]["label"], "finance")
            finally:
                ingestion.METADATA_DIR = original_metadata_dir


    def test_uses_openai_provider_when_available(self):
        labels = ["finance", "legal", "hr"]
        with patch("backend.classification.classify_with_openai") as mock_openai:
            mock_openai.return_value.status = "success"
            mock_openai.return_value.raw_response = '{"label":"hr","confidence":0.88}'
            mock_openai.return_value.warnings = []

            result = classify_text(
                "Hiring onboarding plan",
                labels,
                provider="openai",
                api_key="test-key",
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.label, "hr")

    def test_falls_back_to_local_when_openai_fails(self):
        labels = ["finance", "legal", "hr"]
        with patch("backend.classification.classify_with_openai") as mock_openai:
            mock_openai.return_value.status = "error"
            mock_openai.return_value.raw_response = None
            mock_openai.return_value.warnings = ["OpenAI request failed"]

            result = classify_text(
                "Invoice payment received",
                labels,
                provider="openai",
                api_key="test-key",
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.label, "finance")
        self.assertIn("Falling back to local demo classifier.", result.warnings)


if __name__ == "__main__":
    unittest.main()
