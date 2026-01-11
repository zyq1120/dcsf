"""
Logging configuration.
"""
import sys
from pathlib import Path
from loguru import logger
from app.config import settings


_LOGGER_CONFIGURED = False


def setup_logger():
    """Configure loguru logger with console/file sinks and default tracing context."""
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return logger

    # Default contextual fields to avoid KeyError when未注入 trace/file 信息
    # logger.configure(extra={"trace_id": "-", "file_id": "-"})

    logger.configure(extra={"trace_id": "-"})
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "trace={extra[trace_id]}| "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=fmt,
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    log_dir = Path(settings.LOG_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        # "trace={extra[trace_id]} file={extra[file_id]} | "
        "trace={extra[trace_id]}| "
        "{name}:{function}:{line} - {message}"
    )

    logger.add(
        log_dir / "ai_service_{time:YYYY-MM-DD}.log",
        rotation="200 MB",
        retention="30 days",
        compression="zip",
        format=file_fmt,
        level=settings.LOG_LEVEL,
    )

    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        format=file_fmt,
        level="ERROR",
    )

    _LOGGER_CONFIGURED = True
    return logger
