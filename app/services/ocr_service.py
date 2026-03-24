"""
OCR Service using PaddleOCR with configurable preprocessing pipeline.
"""
import time
import os
import cv2
import numpy as np
from loguru import logger
from typing import Dict, Optional, Any, Tuple, List

from app.config import settings
from app.utils.errors import ServiceError


class OCRService:
    def __init__(self):
        # 复杂引擎改为惰性初始化：避免启动/首次 import 时占用大量资源
        self._ocr = None
        self._table_ocr = None

        # 标记：复杂引擎是否已经初始化
        self._engine_ready = False

    def _init_engine(self):
        """惰性初始化 PaddleOCR 引擎（复杂模式）。"""
        if self._engine_ready and self._ocr is not None:
            return
        try:
            os.environ.setdefault(
                "PADDLEOCR_HOME",
                os.path.join(os.path.expanduser("~"), ".paddleocr")
            )
            from paddleocr import PaddleOCR
        except ImportError as exc:
            # 明确提示安装依赖，避免启动时静默失败
            raise ServiceError("未安装 PaddleOCR，请先安装依赖", detail=str(exc))

        init_kwargs = dict(
            use_angle_cls=settings.OCR_USE_ANGLE_CLS,
            lang=settings.OCR_LANGUAGE,
            use_gpu=settings.USE_GPU,
            show_log=False,
            enable_mkldnn=True,
            use_tensorrt=False,
        )

        # ==== 精度增强：可选自定义字符集 / 模型目录 ====
        rec_char_dict_path = getattr(settings, "OCR_REC_CHAR_DICT_PATH", None)
        if rec_char_dict_path:
            init_kwargs["rec_char_dict_path"] = rec_char_dict_path

        rec_model_dir = getattr(settings, "OCR_REC_MODEL_DIR", None)
        if rec_model_dir:
            init_kwargs["rec_model_dir"] = rec_model_dir

        det_model_dir = getattr(settings, "OCR_DET_MODEL_DIR", None)
        if det_model_dir:
            init_kwargs["det_model_dir"] = det_model_dir

        logger.info(
            "Initializing PaddleOCR engine",
            **{k: v for k, v in init_kwargs.items() if k != "rec_char_dict_path"},
        )
        self._ocr = PaddleOCR(**init_kwargs)
        self._engine_ready = True
        logger.info("PaddleOCR initialized")

    def _init_table_engine(self):
        """
        初始化 PP-Structure 表格引擎。
        修复：显式添加 ocr=True，防止因 layout=False 导致文字识别被自动关闭。
        """
        if self._table_ocr is not None:
            return

        try:
            from paddleocr import PPStructure
        except ImportError as exc:
            logger.warning(
                "PP-Structure 未安装，禁用表格 OCR 功能: {}", str(exc)
            )
            self._table_ocr = None
            return

        if not self._has_local_table_model():
            logger.warning(
                "PP-Structure 表格模型未在本地准备，禁用表格 OCR 功能"
            )
            self._table_ocr = None
            return

        self._table_ocr = PPStructure(
            layout=False,
            ocr=True,  # <--- FIX: 强制开启文字识别，消除 warning 并确保能提取文字
            show_log=False,
            image_orientation=False,
            return_ocr_result_in_table=True,
        )
        logger.info("PP-Structure table OCR initialized (ocr=True forced)")

    def recognize(self, image_path: str, options: Optional[Dict] = None) -> Dict:
        """
        OCR 入口：
        - 默认走简洁 PaddleOCRService（识别率更高，且已预热）
        - 若开启 table_force / detect_table 或指定 use_complex_mode，则走复杂 OCRService
        - 若开启 handwriting_mode / enhance_image，则先对图像做预处理，再交给简洁 OCR 识别
        """
        options = options or {}
        start = time.time()

        # 开关约定：
        # - use_simple_mode: 走简洁 OCR（默认取 settings.OCR_USE_SIMPLE_MODE）
        # - use_complex_mode: 强制走复杂 OCR（用于表格、特殊场景）
        use_simple_mode = options.get("use_simple_mode", settings.OCR_USE_SIMPLE_MODE)
        use_complex_mode = bool(options.get("use_complex_mode", False))
        table_force = bool(options.get("table_force", False))
        detect_table = bool(options.get("detect_table", False))

        # 只要手写/增强开启，就让简洁 OCR 也吃到预处理后的图
        enhance = bool(options.get("enhance_image", True))
        handwriting = bool(options.get("handwriting_mode", False))

        # ===== 默认路径：简洁 OCR（推荐/预热） =====
        if use_simple_mode and not (use_complex_mode or table_force or detect_table):
            from app.services.paddle_ocr_service import get_paddle_ocr_service

            # 若不开增强/手写，直接走文件路径识别（最快，最接近 test_ocr.py）
            if not enhance and not handwriting:
                try:
                    result = get_paddle_ocr_service().recognize(image_path, options)
                    result["processing_time"] = round(time.time() - start, 2)
                    return result
                except ServiceError as exc:
                    # 对偶发 primitive 执行失败做一次“预处理 + 内存识别”兜底
                    detail = str(getattr(exc, "detail", "") or str(exc))
                    if "could not execute a primitive" not in detail:
                        raise
                    logger.warning(
                        "Simple OCR failed with primitive error, fallback to preprocessed memory OCR",
                        image_path=image_path,
                        detail=detail,
                    )

            # 否则先加载图片并做预处理，再用简洁 OCR 的 numpy 入口
            img = cv2.imread(image_path)
            if img is None:
                raise ServiceError("图像加载失败", detail=image_path)
            preprocess_cfg = self._build_preprocess_config(options)
            processed = self._preprocess_image(img, preprocess_cfg)
            result = get_paddle_ocr_service().recognize_image(processed, options)
            result["processing_time"] = round(time.time() - start, 2)
            result["source"] = "simple_ocr_preprocessed"
            return result

        # ===== 复杂路径：需要表格/强制复杂时才加载引擎 =====
        self._init_engine()

        logger.info(
            "OCR recognize start",
            image_path=image_path,
            enhance=options.get("enhance_image", True),
        )

        # 1. 读取图片
        t0 = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise ServiceError("图像加载失败", detail=image_path)
        # 原始图像用于表格检测 / PP-Structure，避免预处理破坏细节
        orig_image = image.copy()
        t1 = time.time()
        logger.info(
            "OCR load image done",
            cost=round(t1 - t0, 3),
            shape=image.shape,
        )

        logger.debug(
            "OCR image loaded",
            shape=image.shape if hasattr(image, "shape") else None,
            dtype=image.dtype if hasattr(image, "dtype") else None,
        )

        # 2. 预处理（主要服务于通用文本 OCR）
        if options.get("enhance_image", True):
            preprocess_cfg = self._build_preprocess_config(options)
            image = self._preprocess_image(image, preprocess_cfg)
        t2 = time.time()
        logger.info(
            "OCR preprocess done",
            cost=round(t2 - t1, 3),
        )

        # 3. 表格检测 & 表格 OCR（可选）
        detect_table = bool(options.get("detect_table", True))
        table_model_ready = self._has_local_table_model()
        table_res = None
        is_table = False
        page_type = "text"

        # [修复后代码]
        # 即使是表格识别，也需要限制最大分辨率，否则 CPU 推理会超时
        # 复用 _ensure_min_resolution 逻辑，限制长边不超过 2000
        table_image = self._ensure_min_resolution(
            orig_image,
            min_short_edge=1000,
            max_long_edge=2000,  # <--- 关键限制，防止超大图进入表格引擎
            long_ratio_threshold=3.0
        )
        logger.debug(
            "Table image resized",
            original=orig_image.shape[:2],
            resized=table_image.shape[:2]
        )

        table_branch_start = time.time()
        if detect_table:
            if not table_model_ready:
                logger.warning(
                    "Table OCR model missing locally, skip PP-Structure table OCR"
                )
            else:
                try:
                    # 强制表格模式（如前端明确指定 table_force）
                    if options.get("table_force"):
                        is_table = True
                        logger.info("Table force enabled, treat page as table")
                    else:
                        is_table = self._detect_table_by_image(
                            table_image,
                            threshold=options.get("table_detect_threshold"),
                        )
                    page_type = "table" if is_table else "text"
                    if is_table:
                        table_res = self._run_table_ocr(table_image)
                except Exception as exc:
                    logger.warning(
                        f"Table detection/PP-Structure failed, skip table OCR: {exc}"
                    )
        table_branch_end = time.time()
        logger.info(
            "OCR table branch done",
            cost=round(table_branch_end - table_branch_start, 3),
            detect_table=detect_table,
            is_table=is_table,
        )

        # 4. 主 OCR（通用文本识别）
        ocr_start = time.time()
        result = self._ocr.ocr(image, cls=True)
        ocr_end = time.time()
        logger.info(
            "OCR engine ocr done",
            cost=round(ocr_end - ocr_start, 3),
        )

        parsed = self._parse_result(result, image.shape)
        parsed["source"] = "ocr"
        self._log_text("ocr_primary_text", parsed.get("text", ""))
        parsed["processing_time"] = round(time.time() - start, 2)
        parsed["table_detected"] = is_table
        parsed["page_type"] = page_type

        if table_res is not None:
            parsed["table"] = table_res
            # 若表格识别有结果，提取纯文本兜底
            table_plain = self._table_plain_text(table_res)
            if table_plain and not parsed.get("text"):
                parsed["text"] = table_plain
            parsed["table_plain"] = table_plain
            if table_plain:
                self._log_text("ocr_table_plain", table_plain)

        # 若文本为空且存在表格模型，尝试一次强制表格 OCR 兜底（使用原图）
        if not parsed.get("text") and table_model_ready and not table_res:
            try:
                logger.info(
                    "Primary text empty, run forced table OCR fallback"
                )
                forced = self._run_table_ocr(table_image)
                if forced:
                    parsed["table"] = forced
                    parsed["table_detected"] = True
                    parsed["page_type"] = "table"
                    table_plain = self._table_plain_text(forced)
                    if table_plain:
                        parsed["text"] = table_plain
                        parsed["table_plain"] = table_plain
                        self._log_text("ocr_forced_table_plain", table_plain)
                    parsed["processing_time"] = round(time.time() - start, 2)
            except Exception as exc:
                logger.warning(f"Force table OCR fallback failed: {exc}")

        # 若文本仍为空，尝试一次“去表格线”预处理后重跑 OCR，缓解表格网格遮挡
        if not parsed.get("text"):
            try:
                logger.info(
                    "Primary text still empty, run remove_table_lines fallback OCR"
                )
                fallback_cfg = self._build_preprocess_config(
                    {**options, "remove_table_lines": True}
                )
                fallback_img = self._preprocess_image(orig_image, fallback_cfg)
                fb_res = self._ocr.ocr(fallback_img, cls=True)
                fb_parsed = self._parse_result(fb_res, fallback_img.shape)
                if fb_parsed.get("text"):
                    parsed = fb_parsed
                    parsed["source"] = "ocr_fallback"
                    parsed["preprocess_fallback"] = "remove_table_lines"
                    parsed["processing_time"] = round(time.time() - start, 2)
                    self._log_text("ocr_fallback_text", parsed.get("text", ""))
                    logger.info(
                        "OCR fallback (remove_table_lines) succeeded",
                        total_boxes=parsed.get("total_boxes", 0),
                    )
                else:
                    logger.warning(
                        "OCR fallback (remove_table_lines) still empty text"
                    )
            except Exception as exc:
                logger.warning(
                    f"OCR fallback (remove_table_lines) failed: {exc}"
                )

        # 长图/宽图切片兜底：避免超长票据/表格整体识别为空
        if not parsed.get("text"):
            try:
                h, w = orig_image.shape[:2]
                ratio = max(h, w) / max(1, min(h, w))
                if ratio >= 2.2 or max(h, w) > 2600:
                    logger.info(
                        "OCR fallback (tiling) start",
                        ratio=round(ratio, 3),
                        size=(h, w),
                    )
                    tiled = self._ocr_on_tiles(orig_image)
                    if tiled.get("text"):
                        parsed = tiled
                        parsed["source"] = "ocr_tile"
                        parsed["processing_time"] = round(
                            time.time() - start, 2)
                        self._log_text("ocr_tile_text", parsed.get("text", ""))
                        logger.info(
                            "OCR fallback (tiling) succeeded",
                            total_boxes=parsed.get("total_boxes", 0),
                        )
                    else:
                        logger.warning("OCR tiling produced empty text")
            except Exception as exc:
                logger.warning(f"OCR fallback (tiling) failed: {exc}")

        # 若文本极短/置信度低/检测框过少，尝试一次质量增强兜底（不依赖 LLM）
        if parsed.get("text"):
            text_len = len(parsed.get("text", "").strip())
            avg_conf = float(parsed.get("confidence") or 0.0)
            box_cnt = int(parsed.get("total_boxes") or 0)
            weak_text = (text_len < 25) or (avg_conf < 0.35) or (box_cnt <= 4)
            if weak_text:
                try:
                    logger.info(
                        "OCR weak text detected, run quality fallback",
                        text_len=text_len,
                        confidence=avg_conf,
                        total_boxes=box_cnt,
                    )
                    q_cfg = self._build_preprocess_config(
                        {**options, "remove_table_lines": True,
                            "binarize": "adaptive"}
                    )
                    q_img = self._preprocess_image(orig_image, q_cfg)
                    q_res = self._ocr.ocr(q_img, cls=True)
                    q_parsed = self._parse_result(q_res, q_img.shape)

                    def _better(new, old):
                        if not new or not new.get("text"):
                            return False
                        if len(new.get("text", "")) > len(old.get("text", "")) * 1.1:
                            return True
                        if int(new.get("total_boxes") or 0) > int(old.get("total_boxes") or 0):
                            return True
                        if float(new.get("confidence") or 0.0) > float(old.get("confidence") or 0.0) + 0.05:
                            return True
                        return False

                    if _better(q_parsed, parsed):
                        parsed = q_parsed
                        parsed["source"] = "ocr_quality_fallback"
                        parsed["processing_time"] = round(
                            time.time() - start, 2)
                        self._log_text("ocr_quality_fallback_text",
                                       parsed.get("text", ""))
                        logger.info(
                            "OCR quality fallback improved result",
                            total_boxes=parsed.get("total_boxes", 0),
                            confidence=parsed.get("confidence"),
                        )
                    else:
                        # 若仍弱文本，再尝试切片
                        h, w = orig_image.shape[:2]
                        ratio = max(h, w) / max(1, min(h, w))
                        if ratio >= 2.0 or max(h, w) > 2200:
                            logger.info(
                                "OCR quality fallback still weak, run tiling",
                                ratio=round(ratio, 3),
                                size=(h, w),
                            )
                            tiled = self._ocr_on_tiles(orig_image)
                            if tiled.get("text") and _better(tiled, parsed):
                                parsed = tiled
                                parsed["source"] = "ocr_tile_quality"
                                parsed["processing_time"] = round(
                                    time.time() - start, 2)
                                self._log_text(
                                    "ocr_tile_quality_text", parsed.get("text", ""))
                                logger.info(
                                    "OCR tiling quality fallback improved result",
                                    total_boxes=parsed.get("total_boxes", 0),
                                    confidence=parsed.get("confidence"),
                                )
                except Exception as exc:
                    logger.warning(f"OCR quality fallback failed: {exc}")

        logger.info(
            "OCR recognize done",
            image_path=image_path,
            total_boxes=parsed.get("total_boxes", 0),
            confidence=parsed.get("confidence"),
            processing_time=parsed["processing_time"],
        )
        return parsed

    def _ocr_on_tiles(self, image: np.ndarray, num_tiles: int = 3, overlap: float = 0.05) -> Dict:
        """Slice long/ wide images into tiles, run OCR per tile, and merge results."""
        h, w = image.shape[:2]
        vertical = h >= w
        long_len = h if vertical else w
        step = max(int(long_len / num_tiles), 1)
        boxes = []

        for i in range(num_tiles):
            start = int(max(i * step - overlap * step, 0))
            end = int(min((i + 1) * step + overlap * step, long_len))
            if end - start < 16:
                continue
            tile = image[start:end, :] if vertical else image[:, start:end]
            res = self._ocr.ocr(tile, cls=True)
            if not res or not res[0]:
                continue
            for line in res[0]:
                if not line:
                    continue
                position = line[0]
                text = line[1][0]
                confidence = line[1][1]
                x_coords = [p[0] for p in position]
                y_coords = [p[1] for p in position]
                bbox = [
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords)),
                ]
                # 将局部坐标平移回原图坐标
                if vertical:
                    bbox[1] += start
                    bbox[3] += start
                else:
                    bbox[0] += start
                    bbox[2] += start
                boxes.append(
                    {
                        "text": text,
                        "position": bbox,
                        "confidence": round(confidence, 4),
                    }
                )

        if not boxes:
            return {
                "text": "",
                "confidence": 0.0,
                "boxes": [],
                "total_boxes": 0,
                "image_size": {"height": h, "width": w},
            }

        total_conf = sum(b.get("confidence", 0.0) for b in boxes)
        avg_conf = total_conf / len(boxes)
        text = "\n".join([b["text"] for b in boxes])

        return {
            "text": text,
            "confidence": round(avg_conf, 4),
            "boxes": boxes,
            "total_boxes": len(boxes),
            "image_size": {"height": h, "width": w},
        }

    def _build_preprocess_config(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并预处理配置。
        优化：降低 max_long_edge 默认值，大幅提升 CPU 推理速度。
        """
        # 从 settings 读取，如果没配则使用更保守的默认值 2000 (原 2600)
        default_max_long = getattr(settings, "OCR_MAX_LONG_EDGE", 2000)

        default_cfg = {
            "enabled": True,
            "handwriting": bool(options.get("handwriting_mode", False)),
            "denoise": True,
            "denoise_method": "auto",
            "binarize": "auto",
            "deskew": True,
            "deskew_angle_threshold": 1.5,
            "remove_table_lines": False,
            "crop_text_regions": False,
            # [性能优化] 降低分辨率上限
            "min_short_edge": getattr(settings, "OCR_MIN_SHORT_EDGE", 960),  # 稍微调小短边要求
            "max_long_edge": default_max_long,  # 关键：2000px 大约比 2600px 快 40-50%
            "long_ratio_threshold": getattr(
                settings, "OCR_LONG_RATIO_THRESHOLD", 3.0
            ),
        }
        user_cfg: Dict[str, Any] = {}
        if isinstance(options.get("preprocess"), dict):
            user_cfg.update(options["preprocess"])

        for key in default_cfg.keys():
            if key in options:
                user_cfg[key] = options[key]
        cfg = {**default_cfg, **user_cfg}
        return cfg

    def _preprocess_image(self, image: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        """
        预处理链路：灰度/缩放 -> 去噪 -> 二值化 -> 倾斜校正 -> 去表格线 -> 文本区域裁剪。
        增加“长文档”特殊处理，优先保证短边分辨率，避免宽/长表格被压得太小。
        """
        try:
            handwriting = bool(cfg.get("handwriting", False))

            # 灰度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
                image.shape
            ) == 3 else image.copy()

            # 控制分辨率：对长宽比大的长图不再死压长边，优先保证短边清晰
            min_short_edge = int(cfg.get("min_short_edge", 1400))
            max_long_edge = int(cfg.get("max_long_edge", 2600))
            long_ratio_threshold = float(
                cfg.get("long_ratio_threshold", 3.0)
            )
            gray = self._ensure_min_resolution(
                gray,
                min_short_edge=min_short_edge,
                max_long_edge=max_long_edge,
                long_ratio_threshold=long_ratio_threshold,
            )

            # 放大有助于手写笔迹（注意这一步之后尺寸可能会再次放大）
            if handwriting:
                gray = cv2.resize(
                    gray,
                    None,
                    fx=1.6,
                    fy=1.6,
                    interpolation=cv2.INTER_CUBIC,
                )

            # 去噪
            if cfg.get("denoise", True):
                gray = self._apply_denoise(
                    gray, cfg.get("denoise_method", "auto")
                )

            # CLAHE 提升对比度，便于后续阈值与检测
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # 二值化（自动选择 Otsu，自适应兜底）
            bin_img = None
            bin_mode = str(cfg.get("binarize", "auto")).lower()
            if bin_mode != "none":
                bin_img = self._auto_binarize(gray, bin_mode)

            # 倾斜校正（基于二值图估计角度）
            if cfg.get("deskew", True) and bin_img is not None:
                rotated_bin, rotated_color, angle = self._deskew(
                    bin_img,
                    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                    cfg.get("deskew_angle_threshold", 1.5),
                )
                if rotated_bin is not None and rotated_color is not None:
                    bin_img, image = rotated_bin, rotated_color

            # 去表格线，减少干扰（配置开启时使用）
            if cfg.get("remove_table_lines", False) and bin_img is not None:
                bin_img = self._remove_table_lines(bin_img)

            # 若有二值化结果，则作为最终输入；否则使用增强后的灰度 BGR
            final_img = (
                cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
                if bin_img is not None
                else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            )

            # 文本区域裁剪（检测模式），仅在开启时调用，避免额外开销
            if cfg.get("crop_text_regions", False):
                final_img = self._crop_text_region(final_img)

            logger.debug(
                "OCR preprocess success",
                original_shape=image.shape,
                output_shape=final_img.shape,
                handwriting=handwriting,
                denoise=cfg.get("denoise", True),
                binarize=bin_mode,
                deskew=cfg.get("deskew", True),
                remove_table_lines=cfg.get("remove_table_lines", False),
                crop_text_regions=cfg.get("crop_text_regions", False),
                min_short_edge=min_short_edge,
                max_long_edge=max_long_edge,
                long_ratio_threshold=long_ratio_threshold,
            )
            return final_img
        except Exception as exc:
            logger.warning(f"Preprocess failed, use original image: {exc}")
            return image

    def _apply_denoise(self, gray: np.ndarray, method: str) -> np.ndarray:
        """
        去噪策略：大图避免使用双边滤波，防止时间爆炸。
        """
        h, w = gray.shape[:2]
        short_edge = min(h, w)
        method = (method or "auto").lower()

        # 显式指定
        if method == "median":
            return cv2.medianBlur(gray, 3)
        if method == "gaussian":
            return cv2.GaussianBlur(gray, (3, 3), 0)
        if method == "bilateral":
            # 大图禁用双边滤波，改用高斯
            if short_edge > 2500:
                return cv2.GaussianBlur(gray, (3, 3), 0)
            return cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)

        # auto 策略
        if short_edge <= 1200:
            return cv2.medianBlur(gray, 3)
        if short_edge <= 2500:
            return cv2.GaussianBlur(gray, (3, 3), 0)
        # 特别大的图，一律用高斯
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _ensure_min_resolution(
        self,
        gray: np.ndarray,
        min_short_edge: int = 1000,
        max_long_edge: int = 2200,
        long_ratio_threshold: float = 3.0,
    ) -> np.ndarray:
        """
        控制图像分辨率在一个合理范围：
        - 对普通文档：短边 < min_short_edge 时放大；长边 > max_long_edge 时缩小
        - 对“超长图”（长宽比 >= long_ratio_threshold）：
          优先保证短边 >= min_short_edge，不再强行压长边，避免宽/长表格被压得太小。
        """
        h, w = gray.shape[:2]
        short_edge = min(h, w)
        long_edge = max(h, w)

        ratio = long_edge / max(short_edge, 1)
        scale = 1.0

        # 长文档：优先保证短边分辨率
        if ratio >= long_ratio_threshold:
            if short_edge < min_short_edge:
                scale = min_short_edge / short_edge
            else:
                scale = 1.0

            if abs(scale - 1.0) < 1e-3:
                return gray

            new_w = int(w * scale)
            new_h = int(h * scale)
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            resized = cv2.resize(gray, (new_w, new_h), interpolation=interp)
            logger.debug(
                "Resize for OCR (long doc)",
                original=(w, h),
                scaled=(new_w, new_h),
                scale=round(scale, 3),
                ratio=round(ratio, 3),
            )
            return resized

        # 普通文档：兼顾 min_short_edge 和 max_long_edge
        if short_edge < min_short_edge:
            scale = min_short_edge / short_edge

        if long_edge * scale > max_long_edge:
            scale = max_long_edge / long_edge

        if abs(scale - 1.0) < 1e-3:
            return gray

        new_w = int(w * scale)
        new_h = int(h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(gray, (new_w, new_h), interpolation=interp)
        logger.debug(
            "Resize for OCR",
            original=(w, h),
            scaled=(new_w, new_h),
            scale=round(scale, 3),
            ratio=round(ratio, 3),
        )
        return resized

    def _is_low_contrast(self, gray: np.ndarray, std_thresh: float = 28.0) -> bool:
        _, stddev = cv2.meanStdDev(gray)
        return float(stddev[0][0]) < std_thresh

    def _auto_binarize(self, gray: np.ndarray, mode: str) -> np.ndarray:
        mode = (mode or "auto").lower()
        # Otsu 首选
        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        if mode == "otsu":
            return otsu
        # 自适应兜底：低对比度或显式 adaptive
        if mode == "adaptive" or self._is_low_contrast(gray):
            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                35,
                10,
            )
            return adaptive
        return otsu

    def _detect_table_by_image(
        self, image: np.ndarray, threshold: Optional[float] = None
    ) -> bool:
        """
        通过简单线段统计判断是否为表格页：大量水平/垂直线条则认为存在表格。
        为避免大图运算量过大，这里先下采样到 max_side ≈ 1600。
        """
        try:
            h, w = image.shape[:2]
            max_side = max(h, w)
            if max_side > 1600:
                scale = 1600 / max_side
                image = cv2.resize(
                    image,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
                logger.debug(
                    "Table detect downscale",
                    original=(w, h),
                    scaled=image.shape[:2],
                )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
                image.shape
            ) == 3 else image
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh_img = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                15,
                10,
            )
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
            horizontal = cv2.morphologyEx(
                thresh_img, cv2.MORPH_OPEN, h_kernel, iterations=1
            )
            vertical = cv2.morphologyEx(
                thresh_img, cv2.MORPH_OPEN, v_kernel, iterations=1
            )
            h_count = cv2.countNonZero(horizontal)
            v_count = cv2.countNonZero(vertical)
            pixels = image.shape[0] * image.shape[1]
            ratio = (h_count + v_count) / max(pixels, 1)
            logger.debug(
                "Table heuristic",
                h=h_count,
                v=v_count,
                ratio=round(ratio, 5),
            )
            # 默认阈值略微放宽，方便宽/长表格被识别为表格
            thresh = threshold if isinstance(
                threshold, (int, float)
            ) else 0.0003
            return ratio > thresh
        except Exception as exc:
            logger.warning(f"Table detection heuristic failed: {exc}")
            return False

    def _run_table_ocr(self, image: np.ndarray):
        """
        调用 PP-Structure 表格识别，返回 HTML 与原始结构。
        """
        self._init_table_engine()
        if self._table_ocr is None:
            logger.warning(
                "Table OCR engine not available, skip table OCR"
            )
            return None
        try:
            res = self._table_ocr(img=image)
            # PP-Structure 返回 list，每个元素包含 'html' 等字段
            tables = []
            for item in res:
                tables.append(
                    {
                        "html": item.get("html"),
                        "excel": item.get("excel"),
                        "boxes": item.get("boxes"),
                    }
                )
            return tables
        except Exception as exc:
            logger.warning(f"PP-Structure table OCR failed: {exc}")
            return None

    @staticmethod
    def _table_plain_text(tables: List[Dict]) -> str:
        """
        将表格 OCR 结果粗略转为纯文本，供兜底使用。
        """
        if not tables:
            return ""
        lines = []
        for tbl in tables:
            html = tbl.get("html")
            if html:
                try:
                    import re

                    plain = re.sub(r"<[^>]+>", " ", html)
                    plain = re.sub(r"\s{2,}", " ", plain).strip()
                    if plain:
                        lines.append(plain)
                except Exception:
                    continue
        return "\n".join(lines).strip()

    @staticmethod
    def _log_text(label: str, text: str, max_len: int = 1000) -> None:
        """
        将 OCR 识别到的原始文本打印到日志中，便于排查问题。
        默认最多打印 max_len 字符，超出部分会在末尾标记被截断。
        """
        if text is None:
            return

        total_len = len(text)
        if total_len <= max_len:
            snippet = text
            suffix = ""
        else:
            snippet = text[:max_len]
            suffix = f"... (+{total_len - max_len} chars truncated)"

        logger.info(
            "{} | length={} | preview={}{}",
            label,
            total_len,
            snippet,
            suffix,
        )

    @staticmethod
    def _has_local_table_model() -> bool:
        base_dir = os.path.join(
            os.path.expanduser("~"),
            ".paddleocr",
            "whl",
            "table",
        )
        candidates = [
            os.path.join(
                base_dir,
                "ch_ppstructure_mobile_v2.0_SLANet_infer",
                "ch_ppstructure_mobile_v2.0_SLANet_infer",
            ),
            os.path.join(
                base_dir,
                "ch_ppstructure_table_structure_mobile_v2.0_SLANet_infer",
                "ch_ppstructure_table_structure_mobile_v2.0_SLANet_infer",
            ),
        ]
        for path in candidates:
            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "inference.pdmodel")
            ):
                return True
        return False

    def _deskew(
        self,
        bin_img: np.ndarray,
        color_img: np.ndarray,
        angle_threshold: float = 1.5,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        coords = cv2.findNonZero(255 - bin_img)  # 取文本区域
        if coords is None or len(coords) < 10:
            return None, None, 0.0
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < angle_threshold:
            return bin_img, color_img, angle
        (h, w) = bin_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_bin = cv2.warpAffine(
            bin_img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rotated_color = cv2.warpAffine(
            color_img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("Deskew applied", angle=round(angle, 2))
        return rotated_bin, rotated_color, angle

    def _remove_table_lines(self, bin_img: np.ndarray) -> np.ndarray:
        """
        简单去表格线：横竖开运算提取线条后进行反掩膜。
        """
        h, w = bin_img.shape
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(10, w // 30), 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(10, h // 30))
        )
        horizontal_lines = cv2.morphologyEx(
            bin_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
        )
        vertical_lines = cv2.morphologyEx(
            bin_img, cv2.MORPH_OPEN, vertical_kernel, iterations=1
        )
        lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
        mask = cv2.bitwise_not(lines)
        cleaned = cv2.bitwise_and(bin_img, mask)
        return cleaned

    def _crop_text_region(self, image: np.ndarray) -> np.ndarray:
        """
        使用 PaddleOCR 检测框裁剪主要文本区域，减少背景干扰。
        这个操作本身也比较重，所以默认关闭，仅在需要时开启。
        """
        try:
            det_result = self._ocr.ocr(image, det=True, rec=False, cls=False)
            if not det_result or not det_result[0]:
                return image
            boxes = det_result[0]
            xs = [int(pt[0]) for box in boxes for pt in box]
            ys = [int(pt[1]) for box in boxes for pt in box]
            if not xs or not ys:
                return image
            x_min, x_max = max(min(xs), 0), max(xs)
            y_min, y_max = max(min(ys), 0), max(ys)
            h, w = image.shape[:2]
            margin = int(0.02 * min(h, w))
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w - 1)
            y_max = min(y_max + margin, h - 1)
            if x_max <= x_min or y_max <= y_min:
                return image
            cropped = image[y_min:y_max, x_min:x_max]
            logger.debug(
                "Cropped text region",
                bbox=(x_min, y_min, x_max, y_max),
            )
            return cropped if cropped.size else image
        except Exception as exc:
            logger.warning(
                f"Crop text region failed, fallback to full image: {exc}"
            )
            return image

    def _parse_result(self, ocr_result, image_shape) -> Dict:
        """
        将 PaddleOCR 原始结果转为简洁结构：文本、bbox、平均置信度等。
        """
        if not ocr_result or not ocr_result[0]:
            return {
                "text": "",
                "confidence": 0.0,
                "boxes": [],
                "total_boxes": 0,
                "image_size": {
                    "height": image_shape[0],
                    "width": image_shape[1],
                },
            }

        boxes = []
        total_conf = 0.0
        for line in ocr_result[0]:
            if not line:
                continue
            position = line[0]
            text = line[1][0]
            confidence = line[1][1]
            x_coords = [p[0] for p in position]
            y_coords = [p[1] for p in position]
            bbox = [
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords)),
            ]
            boxes.append(
                {
                    "text": text,
                    "position": bbox,
                    "confidence": round(confidence, 4),
                }
            )
            total_conf += confidence

        avg_conf = total_conf / len(boxes) if boxes else 0.0

        return {
            "text": "\n".join([b["text"] for b in boxes]),
            "confidence": round(avg_conf, 4),
            "boxes": boxes,
            "total_boxes": len(boxes),
            "image_size": {"height": image_shape[0], "width": image_shape[1]},
        }

    def recognize_simple(self, file_path: str) -> dict:
        """
        兼容旧接口：保留方法名，但内部改为调用简洁 PaddleOCRService。
        说明：原实现会触发复杂引擎初始化，违背“默认简洁 + 复杂按需”的策略。
        """
        from app.services.paddle_ocr_service import get_paddle_ocr_service
        return get_paddle_ocr_service().recognize(file_path, options={})
