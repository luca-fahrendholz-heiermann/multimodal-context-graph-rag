import unittest
from unittest.mock import patch

from backend.vision_analyze import _split_vision_sections, analyze_image_bytes_with_provider


class TestVisionAnalyze(unittest.TestCase):
    def test_split_vision_sections_extracts_description_analysis_questions(self):
        sample = (
            "Beschreibung:\n"
            "- Ein Diagramm mit Umsatzkurve.\n"
            "Analyse:\n"
            "- Die Werte steigen über drei Monate.\n"
            "Offene Fragen:\n"
            "- Welche Datenquelle wurde genutzt?\n"
        )
        description, analysis, questions = _split_vision_sections(sample)
        self.assertIn("Diagramm", description or "")
        self.assertIn("Werte steigen", analysis or "")
        self.assertEqual(questions, ["Welche Datenquelle wurde genutzt?"])

    def test_analyze_defaults_to_openai_when_api_key_provided(self):
        class _Result:
            status = "success"
            raw_response = (
                "Beschreibung:\n"
                "Ein Baustellenfoto.\n"
                "Analyse:\n"
                "Kran und Stahlträger sichtbar.\n"
                "Offene Fragen:\n"
                "- Welches Projekt?"
            )
            warnings = []

        with patch("backend.vision_analyze.describe_image_with_openai", return_value=_Result()) as mocked_openai:
            result = analyze_image_bytes_with_provider(
                filename="img.jpg",
                content_bytes=b"\xff\xd8\xff",
                provider=None,
                api_key="test-key",
            )

        self.assertIsNotNone(result.combined_text)
        self.assertEqual(result.meta.get("provider"), "openai")
        self.assertTrue(mocked_openai.called)

    def test_analyze_warns_when_no_provider_or_key_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            result = analyze_image_bytes_with_provider(
                filename="img.jpg",
                content_bytes=b"\xff\xd8\xff",
                provider=None,
                api_key=None,
            )

        self.assertIsNone(result.combined_text)
        self.assertIn("no vision provider or API key configured", " ".join(result.warnings))


if __name__ == "__main__":
    unittest.main()
