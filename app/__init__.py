"""
Application factory for AI Service
Provides global error handling, logging, and API registration.
"""
from flask import Flask, jsonify
from flask_cors import CORS

from app.config import settings
from app.utils.logger import setup_logger
from app.utils.errors import register_error_handlers
from app.api.ai import ai_bp
from app.api.health import health_bp
from app.utils.response import ok


def create_app(config_name: str = None) -> Flask:
    """Create and configure the Flask application."""
    _ = config_name  # reserved for future multi-env selection

    app = Flask(__name__)

    # Logging
    setup_logger()

    # CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Blueprints
    app.register_blueprint(ai_bp, url_prefix="/api/v1/ai")
    app.register_blueprint(health_bp)

    # Root route for quick smoke testing
    @app.route("/", methods=["GET"])
    def index():
        return jsonify(ok({"service": "document-classification-ai", "version": "1.0.0"}))

    # Error handlers
    register_error_handlers(app)

    # Optional warm-up for WSGI deployments
    try:
        if getattr(settings, "PRELOAD_SERVICES", False):
            from app.services.factory import preload_all_services
            with app.app_context():
                preload_all_services()
    except Exception:
        # 预热失败不影响服务启动，后续仍可懒加载
        pass

    return app
