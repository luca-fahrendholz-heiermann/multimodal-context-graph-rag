from fastapi.testclient import TestClient

from backend.app import app


def test_api_prefixed_health_route_supported():
    client = TestClient(app)

    response = client.get('/api/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
