from app import create_app


class ProbeOKService:
    def probe_connectivity(self):
        return {
            "provider": "nvidia",
            "enabled": True,
            "model": "mock-text",
            "vision_model": "mock-vision",
            "endpoint": "https://integrate.api.nvidia.com/v1",
            "connected": True,
            "sample": "ok",
        }


class ProbeFailService:
    def probe_connectivity(self):
        return {
            "provider": "nvidia",
            "enabled": True,
            "model": "mock-text",
            "vision_model": "mock-vision",
            "endpoint": "https://integrate.api.nvidia.com/v1",
            "connected": False,
            "reason": "404 Client Error",
        }


def test_llm_health_reports_connected(monkeypatch):
    import app.api.health as health_module

    monkeypatch.setattr(health_module, "get_llm_service", lambda: ProbeOKService())
    app = create_app()
    client = app.test_client()

    resp = client.get("/api/v1/llm/health")
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["code"] == 200
    assert payload["data"]["status"] == "healthy"
    assert payload["data"]["connected"] is True
    assert payload["data"]["provider"] == "nvidia"


def test_llm_health_reports_degraded_when_probe_fails(monkeypatch):
    import app.api.health as health_module

    monkeypatch.setattr(health_module, "get_llm_service", lambda: ProbeFailService())
    app = create_app()
    client = app.test_client()

    resp = client.get("/api/v1/llm/health")
    payload = resp.get_json()

    assert resp.status_code == 200
    assert payload["code"] == 200
    assert payload["data"]["status"] == "degraded"
    assert payload["data"]["connected"] is False
    assert payload["data"]["reason"] == "404 Client Error"

