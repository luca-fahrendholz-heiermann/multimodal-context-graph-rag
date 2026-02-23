from fastapi.testclient import TestClient

from backend.app import app


def test_health_endpoint_ok():
    client = TestClient(app)
    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
