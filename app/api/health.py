from flask import Blueprint, jsonify
from app.services.factory import get_llm_service
from app.utils.response import ok

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
@health_bp.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify(ok({"status": "healthy"}))


@health_bp.route("/api/v1/ocr/health", methods=["GET"])
def ocr_health():
    try:
        from app.services.paddle_ocr_service import get_paddle_ocr_service

        svc = get_paddle_ocr_service()
        ready = bool(getattr(svc, "is_ready", False))
    except Exception:
        ready = False
    return jsonify(ok({"status": "healthy", "service": "ocr", "simple_ocr_ready": ready}))


@health_bp.route("/api/v1/nlp/health", methods=["GET"])
def nlp_health():
    return jsonify(ok({"status": "healthy", "service": "nlp"}))


@health_bp.route("/api/v1/llm/health", methods=["GET"])
def llm_health():
    try:
        svc = get_llm_service()
        probe = svc.probe_connectivity()
        status = "healthy" if probe.get("connected") else "degraded"
        return jsonify(ok({"status": status, "service": "llm", **probe}))
    except Exception as exc:
        return jsonify(ok({"status": "degraded", "service": "llm", "connected": False, "reason": str(exc)}))

