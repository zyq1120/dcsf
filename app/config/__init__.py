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
    PORT = int(os.getenv("OCR_SERVICE_PORT", 5005))

    # Paths
    MODEL_PATH = os.getenv("MODEL_PATH", "./models")
    LOG_PATH = os.getenv("LOG_PATH", "./logs")

    # OCR
    OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "ch")
    OCR_USE_ANGLE_CLS = os.getenv(
        "OCR_USE_ANGLE_CLS", "True").lower() == "true"
    OCR_CONFIDENCE_THRESHOLD = float(
        os.getenv("OCR_CONFIDENCE_THRESHOLD", 0.7))
    USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
    # 是否默认使用简洁版 OCR（与 test_ocr.py 逻辑一致，识别率更高）
    OCR_USE_SIMPLE_MODE = os.getenv(
        "OCR_USE_SIMPLE_MODE", "True").lower() == "true"

    # LLM
    ENABLE_LLM_FALLBACK = os.getenv(
        "ENABLE_LLM_FALLBACK", "False").lower() == "true"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    # 通用密钥占位；也支持按供应商分别配置，便于前端切换 provider 时自动选取
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_API_KEY_GEMINI = os.getenv("LLM_API_KEY_GEMINI", "")
    LLM_API_KEY_OPENAI = os.getenv("LLM_API_KEY_OPENAI", "")
    LLM_API_KEY_QWEN = os.getenv("LLM_API_KEY_QWEN", "")
    # 模型占位：通用 + 各厂商可独立配置
    LLM_MODEL = os.getenv("LLM_MODEL", "")
    LLM_MODEL_GEMINI = os.getenv("LLM_MODEL_GEMINI", "")
    LLM_MODEL_OPENAI = os.getenv("LLM_MODEL_OPENAI", "")
    LLM_MODEL_QWEN = os.getenv("LLM_MODEL_QWEN", "")
    # LLM 接口基础 URL，可在 .env 中统一管理
    LLM_ENDPOINT_GEMINI = os.getenv(
        "LLM_ENDPOINT_GEMINI", "https://generativelanguage.googleapis.com")
    LLM_ENDPOINT_OPENAI = os.getenv(
        "LLM_ENDPOINT_OPENAI", "https://api.openai.com")
    LLM_ENDPOINT_QWEN = os.getenv(
        "LLM_ENDPOINT_QWEN", "https://dashscope.aliyuncs.com")
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", 30))
    LLM_RETRY = int(os.getenv("LLM_RETRY", 1))  # 减少重试次数，避免频率过高
    LLM_TRUST_ENV = os.getenv(
        "LLM_TRUST_ENV", "False").lower() == "true"  # 是否使用系统代理/证书环境变量

    # Performance
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 60))

    # File Upload Limits
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))  # 最大文件大小 (MB)
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 50000))  # 最大文本长度 (字符)
    MAX_BASE64_SIZE_MB = int(
        os.getenv("MAX_BASE64_SIZE_MB", 50))  # Base64 最大大小 (MB)

    # Cache Settings
    MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", 100))  # 最大缓存条目数
    MIN_CACHE_CONFIDENCE = float(
        os.getenv("MIN_CACHE_CONFIDENCE", 0.7))  # 缓存最小置信度

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Warmup / Preload
    # 是否在应用启动时预热服务（PaddleOCR simple / NLP / LLM / FinalAIService wiring）
    PRELOAD_SERVICES = os.getenv("PRELOAD_SERVICES", "False").lower() == "true"
    # 是否预热简洁版 PaddleOCR（建议开启，可避免首个 OCR 请求卡顿）
    WARMUP_SIMPLE_OCR = os.getenv("WARMUP_SIMPLE_OCR", "True").lower() == "true"


settings = Settings()
