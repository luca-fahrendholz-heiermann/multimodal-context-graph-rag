import unittest
from unittest.mock import patch

from backend import llm_provider


class TestOpenAiResponseParsing(unittest.TestCase):
    def test_extracts_top_level_output_text(self):
        payload = {"output_text": "CONNECTED"}

        text = llm_provider._extract_openai_text(payload)

        self.assertEqual(text, "CONNECTED")

    def test_extracts_text_from_output_content(self):
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "CONNECTED"},
                    ],
                }
            ]
        }

        text = llm_provider._extract_openai_text(payload)

        self.assertEqual(text, "CONNECTED")

    def test_generate_text_with_openai_uses_output_fallback(self):
        payload = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "CONNECTED"},
                    ]
                }
            ]
        }

        with patch("backend.llm_provider._post_json", return_value=payload):
            result = llm_provider.generate_text_with_openai(
                api_key="test-key",
                model="gpt-4.1-mini",
                prompt="Reply with CONNECTED",
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.raw_response, "CONNECTED")

    def test_generate_text_with_openai_returns_error_for_missing_text(self):
        with patch("backend.llm_provider._post_json", return_value={"output": []}):
            result = llm_provider.generate_text_with_openai(
                api_key="test-key",
                model="gpt-4.1-mini",
                prompt="Reply with CONNECTED",
            )

        self.assertEqual(result.status, "error")
        self.assertIn("extractable text content", result.warnings[0])


class TestGeminiResponseParsing(unittest.TestCase):
    def test_generate_text_with_gemini_collects_all_parts(self):
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Teil 1"},
                            {"text": "Teil 2"},
                        ]
                    }
                }
            ]
        }

        with patch("backend.llm_provider._post_json", return_value=payload):
            result = llm_provider.generate_text_with_gemini(
                api_key="test-key",
                model="gemini-1.5-flash",
                prompt="Beschreibe das Bild",
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.raw_response, "Teil 1\nTeil 2")


class TestImageDescriptionMimeHandling(unittest.TestCase):
    def test_describe_image_with_openai_uses_detected_jpeg_mime(self):
        captured = {}

        def _fake_post_json(_url, payload, _headers):
            captured['payload'] = payload
            return {'output_text': 'ok'}

        with patch('backend.llm_provider._post_json', side_effect=_fake_post_json):
            result = llm_provider.describe_image_with_openai(
                api_key='test-key',
                model='gpt-4.1-mini',
                image_bytes=b'\xff\xd8\xff' + b'1234',
                prompt='describe',
            )

        self.assertEqual(result.status, 'success')
        image_url = captured['payload']['input'][0]['content'][1]['image_url']
        self.assertTrue(image_url.startswith('data:image/jpeg;base64,'))

    def test_describe_image_with_gemini_uses_detected_jpeg_mime(self):
        captured = {}

        def _fake_post_json(_url, payload, _headers):
            captured['payload'] = payload
            return {'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]}

        with patch('backend.llm_provider._post_json', side_effect=_fake_post_json):
            result = llm_provider.describe_image_with_gemini(
                api_key='test-key',
                model='gemini-1.5-flash',
                image_bytes=b'\xff\xd8\xff' + b'1234',
                prompt='describe',
            )

        self.assertEqual(result.status, 'success')
        mime_type = captured['payload']['contents'][0]['parts'][1]['inline_data']['mime_type']
        self.assertEqual(mime_type, 'image/jpeg')

if __name__ == "__main__":
    unittest.main()
