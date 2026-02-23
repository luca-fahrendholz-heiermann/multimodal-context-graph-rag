from fastapi.testclient import TestClient

from backend.app import app


def test_api_prefixed_llm_connection_route_supported(monkeypatch):
    def _fake_generate_text_with_openai(**_kwargs):
        class _Result:
            status = 'success'
            raw_response = 'CONNECTED'
            warnings = []

        return _Result()

    monkeypatch.setattr('backend.app.generate_text_with_openai', _fake_generate_text_with_openai)

    client = TestClient(app)
    response = client.post(
        '/api/llm/check-connection',
        json={'provider': 'chatgpt', 'api_key': 'test-key'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'success'
    assert payload['connected'] is True
