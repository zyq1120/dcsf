"""
Centralized configuration management.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Flask
    ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("OCR_SERVICE_PORT", 5000))

    # Paths
    MODEL_PATH = os.getenv("MODEL_PATH", "./models")
    LOG_PATH = os.getenv("LOG_PATH", "./logs")

    # OCR
    OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "ch")
    OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "True").lower() == "true"
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", 0.7))
    USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
    OCR_USE_SIMPLE_MODE = os.getenv("OCR_USE_SIMPLE_MODE", "True").lower() == "true"

    # LLM
    ENABLE_LLM_FALLBACK = os.getenv("ENABLE_LLM_FALLBACK", "False").lower() == "true"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia")

    # 通用 key
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")

    # 旧字段保留（兼容）
    LLM_API_KEY_GEMINI = os.getenv("LLM_API_KEY_GEMINI", "")
    LLM_API_KEY_OPENAI = os.getenv("LLM_API_KEY_OPENAI", "")
    LLM_API_KEY_QWEN = os.getenv("LLM_API_KEY_QWEN", "")

    # 新增：NVIDIA
    LLM_API_KEY_NVIDIA = os.getenv("LLM_API_KEY_NVIDIA", "")

    # 通用模型
    LLM_MODEL = os.getenv("LLM_MODEL", "")

    # 旧字段保留（兼容）
    LLM_MODEL_GEMINI = os.getenv("LLM_MODEL_GEMINI", "")
    LLM_MODEL_OPENAI = os.getenv("LLM_MODEL_OPENAI", "")
    LLM_MODEL_QWEN = os.getenv("LLM_MODEL_QWEN", "")

    # 新增：NVIDIA
    LLM_MODEL_NVIDIA_TEXT = os.getenv(
        "LLM_MODEL_NVIDIA_TEXT",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    )
    LLM_MODEL_NVIDIA_VISION = os.getenv(
        "LLM_MODEL_NVIDIA_VISION",
        "nvidia/nemotron-nano-12b-v2-vl",
    )

    # 接口地址

    # 新增：NVIDIA
    LLM_ENDPOINT_NVIDIA = os.getenv(
        "LLM_ENDPOINT_NVIDIA",
        "https://integrate.api.nvidia.com/v1",
    )

    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", 120.0))
    LLM_RETRY = int(os.getenv("LLM_RETRY", 1))
    LLM_TRUST_ENV = os.getenv("LLM_TRUST_ENV", "False").lower() == "true"

    # Performance
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 60))

    # File Upload Limits
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 50000))
    MAX_BASE64_SIZE_MB = int(os.getenv("MAX_BASE64_SIZE_MB", 50))

    # Cache Settings
    MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", 100))
    MIN_CACHE_CONFIDENCE = float(os.getenv("MIN_CACHE_CONFIDENCE", 0.7))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Warmup / Preload
    PRELOAD_SERVICES = os.getenv("PRELOAD_SERVICES", "False").lower() == "true"
    WARMUP_SIMPLE_OCR = os.getenv("WARMUP_SIMPLE_OCR", "True").lower() == "true"


settings = Settings()