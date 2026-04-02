"""
Centralized configuration management.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    """Read env var and normalize quotes/trailing commas from .env values."""
    value = os.getenv(name, default)
    if not isinstance(value, str):
        return str(value)
    return value.strip().strip('"').strip("'").rstrip(",")


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, str(default)).lower() == "true"


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


class Settings:
    # Flask
    ENV = _env("FLASK_ENV", "development")
    DEBUG = _env_bool("FLASK_DEBUG", True)
    PORT = _env_int("OCR_SERVICE_PORT", 5006)

    # Paths
    MODEL_PATH = _env("MODEL_PATH", "./models")
    LOG_PATH = _env("LOG_PATH", "./logs")

    # OCR
    OCR_LANGUAGE = _env("OCR_LANGUAGE", "ch")
    OCR_USE_ANGLE_CLS = _env_bool("OCR_USE_ANGLE_CLS", True)
    OCR_CONFIDENCE_THRESHOLD = _env_float("OCR_CONFIDENCE_THRESHOLD", 0.7)
    USE_GPU = _env_bool("USE_GPU", False)
    # 是否默认使用简洁版 OCR（与 test_ocr.py 逻辑一致，识别率更高）
    OCR_USE_SIMPLE_MODE = _env_bool("OCR_USE_SIMPLE_MODE", True)

    # LLM (Nvidia only)
    ENABLE_LLM_FALLBACK = _env_bool("ENABLE_LLM_FALLBACK", False)
    LLM_PROVIDER = "nvidia"

    # Nvidia keys from .env.
    LLM_API_KEY_NVIDIA = _env("NVIDIA_API_KEY")
    LLM_API_KEY = LLM_API_KEY_NVIDIA or _env("LLM_API_KEY")
    LLM_MODEL_NVIDIA = _env("NVIDIA_MODEL") or _env("MODEL") or _env("LLM_MODEL")
    LLM_MODEL_NVIDIA_VISION = _env("NVIDIA_VISION_MODEL") or _env("LLM_MODEL_NVIDIA_VISION")
    LLM_MODEL_NVIDIA_VISION_FALLBACK = _env("NVIDIA_VISION_MODEL_FALLBACK") or _env("LLM_MODEL_NVIDIA_VISION_FALLBACK")
    LLM_MODEL = LLM_MODEL_NVIDIA
    LLM_ENDPOINT_NVIDIA = _env("NVIDIA_BASE_URL") or _env("BASE_URL") or _env("LLM_ENDPOINT")
    LLM_ENDPOINT = LLM_ENDPOINT_NVIDIA or "https://integrate.api.nvidia.com/v1"


    LLM_TIMEOUT = _env_float("LLM_TIMEOUT", 30)
    LLM_RETRY = _env_int("LLM_RETRY", 1)  # 减少重试次数，避免频率过高
    LLM_TRUST_ENV = _env_bool("LLM_TRUST_ENV", False)  # 是否使用系统代理/证书环境变量
    LLM_TEMPERATURE = _env_float("TEMPERATURE", 1.0)
    LLM_TOP_P = _env_float("TOP_P", 0.95)
    LLM_MAX_TOKENS = _env_int("MAX_TOKENS", 2048)
    LLM_VISION_TIMEOUT = _env_float("LLM_VISION_TIMEOUT", 45)
    LLM_VISION_MAX_TOKENS = _env_int("LLM_VISION_MAX_TOKENS", 1024)
    LLM_VISION_MAX_SIDE = _env_int("LLM_VISION_MAX_SIDE", 1600)
    LLM_RAW_LOG_LIMIT = _env_int("LLM_RAW_LOG_LIMIT", 3000)
    LLM_STARTUP_CONNECTIVITY_CHECK = _env_bool("LLM_STARTUP_CONNECTIVITY_CHECK", False)
    LLM_CONNECTIVITY_TIMEOUT = _env_float("LLM_CONNECTIVITY_TIMEOUT", 5)
    LLM_CONNECTIVITY_STRICT = _env_bool("LLM_CONNECTIVITY_STRICT", False)

    # Performance
    MAX_WORKERS = _env_int("MAX_WORKERS", 4)
    REQUEST_TIMEOUT = _env_int("REQUEST_TIMEOUT", 60)

    # File Upload Limits
    MAX_FILE_SIZE_MB = _env_int("MAX_FILE_SIZE_MB", 50)  # 最大文件大小 (MB)
    MAX_TEXT_LENGTH = _env_int("MAX_TEXT_LENGTH", 50000)  # 最大文本长度 (字符)
    MAX_BASE64_SIZE_MB = _env_int("MAX_BASE64_SIZE_MB", 50)  # Base64 最大大小 (MB)

    # Cache Settings
    MAX_CACHE_ENTRIES = _env_int("MAX_CACHE_ENTRIES", 100)  # 最大缓存条目数
    MIN_CACHE_CONFIDENCE = _env_float("MIN_CACHE_CONFIDENCE", 0.7)  # 缓存最小置信度

    # Logging
    LOG_LEVEL = _env("LOG_LEVEL", "INFO")

    # Warmup / Preload
    # 是否在应用启动时预热服务（PaddleOCR simple / NLP / LLM / FinalAIService wiring）
    PRELOAD_SERVICES = _env_bool("PRELOAD_SERVICES", False)
    # 是否预热简洁版 PaddleOCR（建议开启，可避免首个 OCR 请求卡顿）
    WARMUP_SIMPLE_OCR = _env_bool("WARMUP_SIMPLE_OCR", True)


settings = Settings()
