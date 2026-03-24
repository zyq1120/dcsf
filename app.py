"""
Entry point for the AI Service (unified orchestrator).
"""
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_mkldnn_disable_memory_cache"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from app import create_app
from app.config import settings
from app.utils.logger import setup_logger
# [新增] 导入我们在 factory.py 中定义的预热函数
from app.services.factory import preload_all_services

log = setup_logger()

app = create_app()


if __name__ == "__main__":
    log.info(f"Starting AI Service on port {settings.PORT}")
    log.info(f"Environment: {settings.ENV}")
    log.info(f"Debug mode: {settings.DEBUG}")

    # [关键修改] 在启动服务器之前，进行服务预热
    # 这一步会卡住约 5-15 秒，用于加载 PaddleOCR 和 NLP 模型
    # 但换来的是：服务启动后，第一个请求将瞬间完成，无需等待初始化
    with app.app_context():
        try:
            preload_all_services()
        except Exception as e:
            log.warning(f"Warning: Service pre-loading failed, services will lazy-load instead. Error: {e}")

    app.run(host="0.0.0.0", port=settings.PORT, debug=settings.DEBUG)