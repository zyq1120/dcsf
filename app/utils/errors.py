"""
Custom API errors and global error handling.
"""
from dataclasses import dataclass
from typing import Any, Dict
from flask import jsonify
from loguru import logger


@dataclass
class APIError(Exception):
    code: int = 500
    message: str = "Internal Server Error"
    detail: Any = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ValidationError(APIError):
    def __init__(self, message: str = "参数校验失败", detail: Any = None):
        # 语义化使用 400 表示请求参数或体校验未通过
        super().__init__(code=400, message=message, detail=detail)


class ServiceError(APIError):
    def __init__(self, message: str = "服务处理失败", detail: Any = None):
        # 语义化使用 500 表示服务内部处理异常
        super().__init__(code=500, message=message, detail=detail)


def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(err: APIError):
        logger.error(f"APIError: {err.message} | detail={err.detail}")
        return jsonify(err.to_dict()), err.code

    @app.errorhandler(404)
    def handle_404(err):
        logger.warning(f"Route not found: {err}")
        return jsonify({"code": 404, "message": "接口不存在"}), 404

    @app.errorhandler(Exception)
    def handle_exception(err: Exception):
        logger.exception(f"Unhandled exception: {err}")
        return jsonify({"code": 500, "message": "服务器内部错误"}), 500
