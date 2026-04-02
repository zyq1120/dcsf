"""
Thread-safe singleton factory for services.

Note on thread safety:
- Uses double-checked locking pattern with RLock for thread-safe lazy initialization.
- In CPython, the GIL (Global Interpreter Lock) provides additional safety for simple
  attribute reads/writes, but we still use explicit locking for correctness and
  compatibility with alternative Python implementations.
- RLock is used instead of Lock to allow recursive calls within the same thread.
"""
import threading
from typing import Optional
from loguru import logger

from app.services.ocr_service import OCRService
from app.services.nlp_service import NLPService
from app.services.llm_service import LLMService
from app.services.final_ai_service import FinalAIService
from app.services.paddle_ocr_service import PaddleOCRService, get_paddle_ocr_service


class _NoopLLMService:
    """Simple OCR mode stub: keeps FinalAIService wiring but never calls external LLM."""
    enabled = False
    provider = "nvidia"
    model = "disabled"

    def enhance_ocr(self, ocr_text, confidence, override=None):
        return {
            "enhanced": False,
            "text": ocr_text,
            "confidence": confidence,
            "source": "ocr",
        }

    def extract_with_llm(self, text, template_config, override=None, ocr_context=None, file_content_b64=None, file_name=None):
        return None

    def classify_with_llm(self, text, override=None):
        return None

    def extract_with_llm_image(self, file_content_b64, file_name, template_config, override=None):
        return None

# Module-level lock for thread-safe singleton creation
# Using RLock to allow recursive calls from the same thread
_lock = threading.RLock()

# Singleton instances - access should be through getter functions
_ocr_instance: Optional[OCRService] = None
_nlp_instance: Optional[NLPService] = None
_llm_instance: Optional[LLMService] = None
_final_ai_instance: Optional[FinalAIService] = None
_noop_llm_instance: Optional[_NoopLLMService] = None


def get_ocr_service() -> OCRService:
    """复杂 OCRService（包含重预处理/表格分支）按需创建，避免启动期开销。"""
    global _ocr_instance
    if _ocr_instance is None:
        with _lock:
            if _ocr_instance is None:
                logger.info("Creating OCRService singleton (lazy)")
                _ocr_instance = OCRService()
    return _ocr_instance


def preload_all_services():
    """
    预热所有服务：在应用启动时显式初始化单例。

    策略：
    - 默认仅预热简洁版 PaddleOCR（识别率高且路径最常用）
    - 复杂 OCRService（带表格分支/重预处理）按需惰性加载，避免启动时双份 OCR 引擎占用
    """
    logger.info("Starting services warm-up (Pre-loading models)...")

    # 1. 预加载 NLP 服务
    try:
        get_nlp_service()
        logger.info("NLP Service ready.")
    except Exception as e:
        logger.error(f"NLP Service warm-up failed: {e}")

    from app.config import settings

    # 2. 预加载 LLM 服务（Simple OCR 模式下跳过）
    try:
        if getattr(settings, "OCR_USE_SIMPLE_MODE", True):
            logger.info("Skip LLM Service warm-up (OCR_USE_SIMPLE_MODE=true)")
        else:
            get_llm_service()
            logger.info("LLM Service ready.")
    except Exception as e:
        logger.error(f"LLM Service warm-up failed: {e}")

    # 3. 预加载简洁版 PaddleOCR 服务（与 test_ocr.py 逻辑一致）
    try:
        if getattr(settings, "WARMUP_SIMPLE_OCR", True):
            logger.info("Warming up Simple PaddleOCR engine...")
            get_paddle_ocr_service()
            logger.info("Simple PaddleOCR Service ready.")
        else:
            logger.info("Skip Simple PaddleOCR warm-up (WARMUP_SIMPLE_OCR=false)")
    except Exception as e:
        logger.error(f"Simple PaddleOCR Service warm-up failed: {e}")

    # 4. 组装 FinalAIService
    try:
        get_final_ai_service()
        logger.info("FinalAIService wired.")
    except Exception as e:
        logger.error(f"Final AI Service warm-up failed: {e}")

    logger.info("All services warmed up! Server is ready for requests.")

def get_nlp_service() -> NLPService:
    global _nlp_instance
    if _nlp_instance is None:
        with _lock:
            if _nlp_instance is None:
                logger.info("Creating NLPService singleton")
                _nlp_instance = NLPService()
    return _nlp_instance


def get_llm_service() -> LLMService:
    global _llm_instance
    if _llm_instance is None:
        with _lock:
            if _llm_instance is None:
                logger.info("Creating LLMService singleton")
                _llm_instance = LLMService()
    return _llm_instance


def get_final_ai_service() -> FinalAIService:
    global _final_ai_instance
    if _final_ai_instance is None:
        with _lock:
            if _final_ai_instance is None:
                logger.info("Creating FinalAIService singleton")
                llm_service = get_llm_service()
                # OCR 初始化开销大、且可能未安装 paddleocr。延迟到真正需要 file_path 时再创建。
                # 注意：OCRService 现在是“轻量外壳”，默认会走 Simple OCRService（已预热），只有表格/强制复杂时才加载重引擎。
                _final_ai_instance = FinalAIService(
                    ocr_service=None,
                    nlp_service=get_nlp_service(),
                    llm_service=llm_service,
                    ocr_loader=OCRService,
                )

                # Apply compatibility patch if needed
                try:
                    from app.services.final_ai_service_compat_patch import process as _patched_process
                    # Only patch if current bound method can raise NameError 'is_loan_doc'
                    if not getattr(_final_ai_instance.process, "__name__", "") == _patched_process.__name__:
                        _final_ai_instance.process = _patched_process.__get__(_final_ai_instance, FinalAIService)  # type: ignore[assignment]
                except Exception:
                    pass
    return _final_ai_instance
