import os
import pytest

# Ensure LLM fallback disabled for tests to avoid external calls
os.environ.setdefault("ENABLE_LLM_FALLBACK", "False")

from app import create_app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_health_endpoints(client):
    for path in ["/health", "/api/v1/health", "/api/v1/ocr/health", "/api/v1/nlp/health"]:
        resp = client.get(path)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and body.get("code") == 200


def test_process_text_mode(client):
    payload = {
        "text": "学生张三，学号20201234，软件工程专业，请假三天。",
        "options": {"disable_ai": True},
    }
    resp = client.post("/api/v1/ai/process", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["code"] == 200
    data = body.get("data") or {}
    assert "classification" in data
    assert "fields" in data
