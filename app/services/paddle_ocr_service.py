"""
简洁版 PaddleOCR Service - 基于 test_ocr.py 的逻辑
当没有开启强制表格 OCR 时使用此服务
"""
import os
import time
import threading
import numpy as np
from typing import Dict, Optional
from loguru import logger

from app.utils.errors import ServiceError
from app.config import settings


class PaddleOCRService:
    """
    简洁版 OCR 服务，直接调用 PaddleOCR 引擎，
    不进行复杂的表格检测和预处理流程。
    适用于普通文档识别场景。
    """

    def __init__(self):
        self._ocr = None

    @property
    def is_ready(self) -> bool:
        return self._ocr is not None

    def _init_engine(self):
        """
        初始化 PaddleOCR 引擎（简洁模式）。
        """
        if self._ocr is not None:
            return

        try:
            os.environ.setdefault(
                "PADDLEOCR_HOME",
                os.path.join(os.path.expanduser("~"), ".paddleocr")
            )
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ServiceError("未安装 PaddleOCR，请先安装依赖", detail=str(exc))

        init_kwargs = dict(
            use_angle_cls=bool(getattr(settings, "OCR_USE_ANGLE_CLS", False)),
            lang=str(getattr(settings, "OCR_LANGUAGE", "ch") or "ch"),
            use_gpu=bool(getattr(settings, "USE_GPU", False)),
            show_log=False,
        )

        # 记录关键环境信息（排查偶发 primitive 错误很关键）
        logger.info(
            "Initializing PaddleOCR (Simple Mode)",
            **init_kwargs,
            pid=os.getpid(),
            thread=threading.current_thread().name,
        )
        self._ocr = PaddleOCR(**init_kwargs)
        logger.info("PaddleOCR (Simple Mode) initialized")

    def recognize(self, file_path: str, options: Optional[Dict] = None) -> Dict:
        """
        按 test_ocr.py 逻辑做 OCR（带一次重试与降级）。

        Args:
            file_path: 图片或 PDF 文件路径
            options: 可选参数（此简洁版暂不使用）

        Returns:
            包含 text, confidence, boxes 等字段的字典
        """
        self._init_engine()

        if not os.path.exists(file_path):
            raise ServiceError("文件不存在", detail=file_path)

        logger.info(
            "OCR (Simple Mode) | 开始处理文件",
            file_path=file_path,
            pid=os.getpid(),
            thread=threading.current_thread().name,
        )

        def _run_ocr(cls_enabled: bool):
            # 直接调用核心 OCR 引擎
            return self._ocr.ocr(file_path, cls=cls_enabled)

        try:
            # 第一次：正常路径（cls=True）
            result_pages = _run_ocr(cls_enabled=True)
        except RuntimeError as e:
            msg = str(e)
            # Paddle/PaddleOCR 偶发会在 predictor.run 抛：could not execute a primitive
            # 常见诱因：线程并发/oneDNN 算子选择/瞬时内存/图片解码异常等。
            if "could not execute a primitive" in msg:
                logger.warning(
                    "OCR (Simple Mode) | primitive 执行失败，触发重试降级",
                    error=msg,
                    file_path=file_path,
                )
                time.sleep(0.2)
                # 第二次：关闭 cls（减少算子路径），通常能绕开该偶发错误
                try:
                    result_pages = _run_ocr(cls_enabled=False)
                except Exception as e2:
                    logger.error(
                        "OCR (Simple Mode) | 重试仍失败",
                        error=str(e2),
                        file_path=file_path,
                    )
                    # 第三次终极兜底：强制降采样 + 最简模式
                    try:
                        logger.warning(
                            "OCR (Simple Mode) | 触发终极兜底：降采样 + 最简模式",
                            file_path=file_path,
                        )
                        result_pages = self._last_resort_ocr(file_path)
                    except Exception as e3:
                        logger.error(
                            "OCR (Simple Mode) | 终极兜底仍失败",
                            error=str(e3),
                            file_path=file_path,
                        )
                        raise ServiceError("OCR 识别失败", detail=f"已尝试所有降级策略仍失败: {str(e3)}")
            else:
                raise ServiceError("OCR 识别失败", detail=msg)
        except Exception as e:
            raise ServiceError("OCR 识别失败", detail=str(e))

        try:
            if not result_pages or not result_pages[0]:
                logger.warning("OCR (Simple Mode) | 未识别出任何文本。")
                return {
                    "text": "",
                    "confidence": 0.0,
                    "boxes": [],
                    "total_boxes": 0,
                    "source": "simple_ocr"
                }

            all_texts = []
            all_scores = []
            all_boxes = []

            for page_idx, page_result in enumerate(result_pages):
                if not page_result:
                    continue

                for line in page_result:
                    if not line:
                        continue

                    position = line[0]
                    text = line[1][0]
                    confidence = line[1][1]

                    all_texts.append(text)
                    all_scores.append(confidence)

                    x_coords = [p[0] for p in position]
                    y_coords = [p[1] for p in position]
                    bbox = [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords)),
                    ]
                    all_boxes.append({
                        "text": text,
                        "position": bbox,
                        "confidence": round(float(confidence), 4),
                        "page": page_idx
                    })

            full_text = "\n".join(all_texts)
            avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0

            logger.info(
                "OCR (Simple Mode) | 处理成功",
                lines=len(all_boxes),
                avg_conf=round(float(avg_confidence), 4),
            )
            logger.info(f"OCR (Simple Mode) | 识别文本内容:\n{full_text}")

            return {
                "text": full_text,
                "confidence": round(float(avg_confidence), 4),
                "boxes": all_boxes,
                "total_boxes": len(all_boxes),
                "source": "simple_ocr"
            }
        except Exception as e:
            logger.error(f"OCR (Simple Mode) | 结果解析失败: {e}")
            raise ServiceError("OCR 识别失败", detail=str(e))

    def recognize_image(self, image, options: Optional[Dict] = None) -> Dict:
        """
        识别内存中的图像（numpy 数组）

        Args:
            image: numpy.ndarray 格式的图像
            options: 可选参数

        Returns:
            包含 text, confidence, boxes 等字段的字典
        """
        self._init_engine()

        logger.info(
            "OCR (Simple Mode) | 开始处理内存图像",
            pid=os.getpid(),
            thread=threading.current_thread().name,
        )

        def _run_ocr(cls_enabled: bool):
            return self._ocr.ocr(image, cls=cls_enabled)

        try:
            result = _run_ocr(cls_enabled=True)
        except RuntimeError as e:
            msg = str(e)
            # Paddle/PaddleOCR 偶发会在 predictor.run 抛：could not execute a primitive
            # 常见诱因：线程并发/oneDNN 算子选择/瞬时内存/图片解码异常等。
            if "could not execute a primitive" in msg:
                logger.warning(
                    "OCR (Simple Mode) | primitive 执行失败（内存图像），触发重试降级",
                    error=msg,
                )
                time.sleep(0.2)
                # 第二次：关闭 cls（减少算子路径），通常能绕开该偶发错误
                try:
                    result = _run_ocr(cls_enabled=False)
                except Exception as e2:
                    logger.error(
                        "OCR (Simple Mode) | 内存图像重试仍失败",
                        error=str(e2),
                    )
                    # 第三次终极兜底：降采样内存图像
                    try:
                        logger.warning("OCR (Simple Mode) | 触发内存图像终极兜底：降采样")
                        downsampled = self._downsample_image(image)
                        result = self._ocr.ocr(downsampled, cls=False)
                    except Exception as e3:
                        logger.error(
                            "OCR (Simple Mode) | 内存图像终极兜底仍失败",
                            error=str(e3),
                        )
                        raise ServiceError("OCR 识别失败", detail=f"内存图像已尝试所有降级策略仍失败: {str(e3)}")
            else:
                raise ServiceError("OCR 识别失败", detail=msg)
        except Exception as e:
            raise ServiceError("OCR 识别失败", detail=str(e))

        try:
            if not result or not result[0]:
                logger.warning("OCR (Simple Mode) | 未识别出任何文本。")
                return {
                    "text": "",
                    "confidence": 0.0,
                    "boxes": [],
                    "total_boxes": 0,
                    "source": "simple_ocr"
                }

            # 解析结果
            texts = []
            scores = []
            boxes = []

            for line in result[0]:
                if not line:
                    continue

                position = line[0]
                text = line[1][0]
                confidence = line[1][1]

                texts.append(text)
                scores.append(confidence)

                x_coords = [p[0] for p in position]
                y_coords = [p[1] for p in position]
                bbox = [
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords)),
                ]
                boxes.append({
                    "text": text,
                    "position": bbox,
                    "confidence": round(confidence, 4),
                })

            full_text = "\n".join(texts)
            avg_confidence = sum(scores) / len(scores) if scores else 0.0

            logger.info(
                f"OCR (Simple Mode) | 处理成功，识别出 {len(boxes)} 行文本"
            )

            # 打印识别出的完整文本到日志
            logger.info(f"OCR (Simple Mode) | 识别文本内容:\n{full_text}")

            return {
                "text": full_text,
                "confidence": round(avg_confidence, 4),
                "boxes": boxes,
                "total_boxes": len(boxes),
                "source": "simple_ocr"
            }

        except Exception as e:
            logger.error(f"OCR (Simple Mode) | 处理失败: {e}")
            raise ServiceError("OCR 识别失败", detail=str(e))

    def _last_resort_ocr(self, file_path: str):
        """
        终极兜底：当所有重试都失败时，强制降采样图片到极小尺寸，并用最简配置再试一次。

        策略：
        - 用 PIL/cv2 加载图片并缩小到短边 800px
        - 转为最简单的灰度/RGB 格式
        - 用 cls=False 调用 OCR
        """
        try:
            import cv2

            img = cv2.imread(file_path)
            if img is None:
                raise ServiceError("无法加载图片", detail=file_path)

            h, w = img.shape[:2]
            short_edge = min(h, w)
            if short_edge > 800:
                scale = 800.0 / short_edge
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.info(
                    "OCR (Simple Mode) | 终极兜底降采样",
                    original=(w, h),
                    downsampled=(new_w, new_h),
                )

            # 转为 RGB（避免 RGBA/灰度等兼容问题）
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 用最简配置：cls=False
            result = self._ocr.ocr(img, cls=False)
            logger.info("OCR (Simple Mode) | 终极兜底成功")
            return result
        except Exception as e:
            logger.error(f"OCR (Simple Mode) | 终极兜底失败: {e}")
            raise

    def _downsample_image(self, image):
        """
        降采样内存图像（numpy.ndarray）到短边 800px。
        """
        try:
            import cv2

            if not isinstance(image, np.ndarray):
                return image

            h, w = image.shape[:2]
            short_edge = min(h, w)
            if short_edge <= 800:
                return image

            scale = 800.0 / short_edge
            new_w = int(w * scale)
            new_h = int(h * scale)
            downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(
                "OCR (Simple Mode) | 内存图像降采样",
                original=(w, h),
                downsampled=(new_w, new_h),
            )
            return downsampled
        except Exception:
            return image


# 单例实例，方便直接导入使用
_paddle_ocr_service = None


def get_paddle_ocr_service() -> PaddleOCRService:
    """获取 PaddleOCRService 单例"""
    global _paddle_ocr_service
    if _paddle_ocr_service is None:
        _paddle_ocr_service = PaddleOCRService()
    return _paddle_ocr_service
