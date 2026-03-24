"""
Unified AI API endpoints.
"""
import uuid
from flask import Blueprint, request, jsonify
from loguru import logger
from app.services.factory import get_final_ai_service
from app.utils.errors import ValidationError, ServiceError
from app.utils.response import ok

ai_bp = Blueprint("ai", __name__)


@ai_bp.route("/process", methods=["POST"])
def process():
    trace_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    data = request.get_json(silent=True) or {}
    file_id = data.get("file_id") or str(uuid.uuid4())

    with logger.contextualize(trace_id=trace_id, file_id=file_id):
        file_path = data.get("file_path")
        text = data.get("text")
        file_content = data.get("file_content")
        template_config = data.get("template_config") or {}

        # Basic input validation to prevent overlarge payloads and unsupported formats
        max_text_len = 50000
        if text and len(text) > max_text_len:
            raise ValidationError(
                f"text 过长（>{max_text_len} 字符）",
                detail="请截断文本或改用文件上传",
            )

        if file_content:
            approx_size = len(file_content) * 0.75  # base64 粗略估计
            max_bytes = 8 * 1024 * 1024  # 8MB
            if approx_size > max_bytes:
                raise ValidationError(
                    "file_content 过大",
                    detail="最大支持约 8MB，请压缩或使用更小文件",
                )

        logger.info(
            "API /process called",
            file_path=file_path,
            has_text=bool(text),
            template_fields=len(template_config.get("fields", [])),
        )

        try:
            service = get_final_ai_service()
            result = service.process(
                {**data, "file_id": file_id, "template_config": template_config})
            logger.info(
                "API /process finished",
                file_path=file_path,
                has_text=bool(text),
                processing_time=result.get("processing_time"),
            )
            return jsonify(ok(result))
        except ValidationError as ve:
            raise ve
        except ServiceError as se:
            raise se
        except Exception as exc:
            raise ServiceError("智能解析失败", detail=str(exc))
