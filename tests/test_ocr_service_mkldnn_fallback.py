import types
import sys

from app.services.ocr_service import OCRService


class _FakePaddleOCR:
    calls = []

    def __init__(self, **kwargs):
        self.__class__.calls.append(kwargs)
        if kwargs.get("enable_mkldnn"):
            raise AttributeError(
                "'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_mkldnn_cache_capacity'"
            )



def test_init_engine_falls_back_when_mkldnn_cache_api_missing(monkeypatch):
    fake_module = types.SimpleNamespace(PaddleOCR=_FakePaddleOCR)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    _FakePaddleOCR.calls = []
    svc = OCRService()

    svc._init_engine()

    assert svc._engine_ready is True
    assert svc._ocr is not None
    assert len(_FakePaddleOCR.calls) == 2
    assert _FakePaddleOCR.calls[0]["enable_mkldnn"] is True
    assert _FakePaddleOCR.calls[1]["enable_mkldnn"] is False

