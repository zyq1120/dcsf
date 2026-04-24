import pytest
from typing import Any, cast

from app.config import settings
from app.services.final_ai_service import FinalAIService
from app.services.llm_service import LLMService
from app.utils.errors import ServiceError


class FakeOCRService:
    def __init__(self):
        self.calls = 0

    def recognize(self, file_path, options):
        self.calls += 1
        return {
            "text": "这是一段足够长的 OCR 识别文本，用于避免后续再次触发多模态增强。",
            "confidence": 0.93,
            "total_boxes": 10,
        }


class FakeNLPService:
    def analyze_ocr_text(self, full_text, ocr_result):
        return {
            "document_type": {"doc_type": "测试文档"},
            "reconstructed": {"text": full_text, "table_lines": []},
            "key_value_pairs": [],
            "extracted_fields": [],
        }

    def extract_fields(self, nlp_text, template_config, file_id=None):
        return {
            "extract_main": {
                "total_fields": len(template_config.get("fields", [])),
                "extracted_fields": len(template_config.get("fields", [])),
                "confidence": 0.95,
                "status": "success",
            },
            "extract_details": [
                {
                    "field_name": "name",
                    "field_value": "Alice",
                    "field_type": "TEXT",
                    "confidence": 0.95,
                }
            ],
            "processing_time": 0,
        }

    def extract_entities(self, nlp_text):
        return []

    def extract_relations(self, nlp_text, entities):
        return []

    def map_entities_to_slots(self, entities, nlp_text):
        return []

    def classify_document(self, nlp_text):
        return {"document_type": "测试文档", "confidence": 0.91, "source": "nlp"}


class FakeDocTypeService:
    def detect(self, *args, **kwargs):
        return {
            "doc_type": "测试文档",
            "candidates": {"测试文档": 1.0},
            "primary_type": "测试文档",
        }


class DummyFinalAIService(FinalAIService):
    def _load_builtin_templates(self):
        return {}

    def _validate_file_path(self, file_path):
        return file_path

    def _file_to_base64(self, file_path):
        return "ZmFrZV9pbWFnZV9kYXRh"

    def _build_llm_fallback_ocr_context(self, *args, **kwargs):
        return {}

    def _build_llm_input_text(self, *args, **kwargs):
        return args[0] if args else ""

    def _load_cached_candidates(self, *args, **kwargs):
        return []

    def _extract_courses_table(self, *args, **kwargs):
        return []

    def _merge_fields(self, base, overlay):
        merged = dict(base or {})
        if overlay:
            merged.update(overlay)
        return merged

    def _finalize_response(self, raw, template_config, **kwargs):
        return {
            "final_doc_type": kwargs.get("final_doc_type"),
            "classification": raw.get("classification"),
            "fields": raw.get("fields"),
            "text": raw.get("text"),
        }

    def _inject_extract_details_from_llm(self, data, template_config):
        return data

    def _validate_and_fix_fields(self, fields):
        return fields

    def _should_log_debug_snapshot(self, resp):
        return False

    def _sample_extract_details(self, details, limit=10):
        return []

    def _clean_llm_courses(self, courses):
        return courses

    def _is_noisy_courses(self, courses):
        return False

    def _log_llm_result(self, *args, **kwargs):
        return None


class FakeLLMService:
    def __init__(self):
        self.enabled = True
        self.provider = "nvidia"
        self.model = "mock-model"
        self.image_calls = 0

    def enhance_ocr(self, ocr_text, confidence, override=None):
        return {
            "text": ocr_text,
            "confidence": confidence,
            "source": "ocr",
            "enhanced": False,
        }

    def extract_with_llm_image(self, file_content_b64, file_name, template_config, override=None):
        self.image_calls += 1
        raise ServiceError(
            "LLM 调用失败",
            detail="404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/chat/completions; body=404 page not found",
        )

    def extract_with_llm(self, *args, **kwargs):
        return None

    def classify_with_llm(self, *args, **kwargs):
        return None


@pytest.fixture()
def sample_image_file(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(b"fake image bytes")
    return path


def test_llm_image_failure_falls_back_to_ocr_nlp(monkeypatch, sample_image_file):
    monkeypatch.setattr(settings, "OCR_USE_SIMPLE_MODE", True, raising=False)
    monkeypatch.setattr(FinalAIService, "_load_builtin_templates", lambda self: {})

    ocr_service = FakeOCRService()
    nlp_service = FakeNLPService()
    llm_service = FakeLLMService()
    service = DummyFinalAIService(
        ocr_service=cast(Any, ocr_service),
        nlp_service=cast(Any, nlp_service),
        llm_service=cast(Any, llm_service),
        doc_type_service=cast(Any, FakeDocTypeService()),
    )

    result = service.process(
        {
            "file_path": str(sample_image_file),
            "file_id": "file-1",
            "template_config": {"fields": [{"name": "name", "type": "TEXT", "description": "姓名"}]},
            "options": {
                "llm_image": True,
                "llm_fallback_with_image": True,
                "vision_grounding_check": False,
            },
        }
    )

    assert ocr_service.calls == 1
    assert llm_service.image_calls == 1
    assert result["final_doc_type"] == "测试文档"
    assert result["fields"]["extract_main"]["extracted_fields"] == 1
    assert result["classification"]["document_type"] == "测试文档"


def test_llm_call_stops_retrying_on_404(monkeypatch):
    service = LLMService()
    service.enabled = True
    service.api_key = "test-key"
    service.retry = 3

    calls = []

    def fake_call_text(prompt, model=None, api_key=None, timeout=None):
        calls.append(1)
        raise ServiceError(
            "LLM 调用失败",
            detail="404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/chat/completions; body=404 page not found",
        )

    monkeypatch.setattr(service, "_call_text", fake_call_text)

    with pytest.raises(ServiceError):
        service._call_llm("hello")

    assert len(calls) == 1

