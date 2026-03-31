"""
Final orchestrator that composes OCR, LLM, and NLP services.
当 OCR 与 NLP 规则无法覆盖时，自动切换到 LLM进行兜底处理。
支持 file_path 或 base64 file_content（方便前端直接上传图片）。
"""
import base64
import json
from pathlib import Path
import uuid
import os
import tempfile
import time
import re
import hashlib
from typing import Dict, Optional, List
from loguru import logger

from app.config import settings
from app.utils.errors import ValidationError, ServiceError
from app.services.ocr_service import OCRService
from app.services.nlp_service import NLPService
from app.services.llm_service import LLMService
from app.services.llm_prompts import LLM_NORMALIZE_PROMPT as DEFAULT_LLM_NORMALIZE_PROMPT
from app.services.document_type_service import DocumentTypeService


class FinalAIService:
    # LLM 最终归一化指导提示，确保输出固定 JSON 结构
    LLM_NORMALIZE_PROMPT = DEFAULT_LLM_NORMALIZE_PROMPT
    LABEL_TOKENS = {
        "开户行",
        "账户名称",
        "名称",
        "学生姓名",
        "学院",
        "专业",
        "学制",
        "身份证号码",
        "用款期限",
        "就读高校",
        "户籍地址",
    }

    def __init__(
        self,
        ocr_service: Optional[OCRService],
        nlp_service: NLPService,
        llm_service: LLMService,
        ocr_loader=None,
        doc_type_service: Optional[DocumentTypeService] = None,
    ):
        # OCR 可选懒加载，避免启动时即依赖 paddleocr
        self.ocr_service = ocr_service
        self._ocr_loader = ocr_loader
        self.nlp_service = nlp_service
        self.llm_service = llm_service
        # 文档类型统一服务：如果外部没有注入，则在这里创建一个默认的
        self.doc_type_service = doc_type_service or DocumentTypeService()
        self._builtin_templates = self._load_builtin_templates()
        logger.info(
            "FinalAIService wired",
            ocr_enabled=bool(self.ocr_service),
            nlp_enabled=bool(self.nlp_service),
            llm_enabled=self.llm_service.enabled,
            llm_provider=self.llm_service.provider,
        )

    def _cleanup_temp_files(self, cleanup_paths: List[str]) -> None:
        """
        安全清理临时文件列表。

        无论主流程是否抛出异常，都应确保调用此方法清理临时文件。
        建议在 process() 的各个返回点或异常处理中调用此方法。

        Args:
            cleanup_paths: 需要清理的临时文件路径列表
        """
        for tmp in cleanup_paths:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                    logger.debug("临时文件已删除", temp_file=tmp)
                except Exception as exc:
                    logger.warning(f"删除临时文件失败: {exc}")

    def process(self, payload: Dict) -> Dict:
        """
        统一处理入口：校验输入 -> OCR -> NLP -> LLM 兜底 -> 合并结果。
        """
        start = time.time()
        file_path = payload.get("file_path")
        text = payload.get("text")
        template_config = payload.get("template_config", {})
        file_id = payload.get("file_id")
        options = payload.get("options", {})
        auto_infer_fields = bool(options.get("auto_infer_fields")) if isinstance(
            options, dict) else False
        llm_only = bool(options.get("llm_only")) if isinstance(
            options, dict) else False
        llm_disabled = bool(options.get("disable_ai")) if isinstance(
            options, dict) else False
        analysis_enabled = bool(options.get("analyze_ocr")) if isinstance(
            options, dict) else False
        llm_image_opt = bool(options.get("llm_image")) if isinstance(
            options, dict) else False
        ocr_fast_return = bool(options.get("ocr_fast_return")) if isinstance(
            options, dict) else False
        llm_override = {}
        if isinstance(options, dict):
            if options.get("llm_provider"):
                llm_override["provider"] = str(
                    options.get("llm_provider")).lower()
            if options.get("llm_model"):
                llm_override["model"] = str(options.get("llm_model"))
            if options.get("llm_vision_model"):
                llm_override["vision_model"] = str(options.get("llm_vision_model"))
            if options.get("llm_api_key"):
                llm_override["api_key"] = str(options.get("llm_api_key"))
        # 默认开启 LLM 兜底（只要未 disable_ai）；仍保留 llm_only 选项
        llm_allowed = not llm_disabled
        llm_available = self.llm_service.enabled and llm_allowed
        explicit_llm_opt_in = llm_only or llm_image_opt
        # 显式 llm_only/llm_image 优先：即使 Simple OCR 开启，也强制走大模型路径
        if explicit_llm_opt_in and llm_allowed:
            if not self.llm_service.enabled:
                logger.warning("LLM forced on by explicit option (llm_only/llm_image)")
                self.llm_service.enabled = True
            llm_available = True
            logger.info("Explicit LLM mode enabled", llm_only=llm_only, llm_image=llm_image_opt)
        # Simple OCR 默认模式：禁用大模型
        elif getattr(settings, "OCR_USE_SIMPLE_MODE", True):
            llm_available = False
        file_content = payload.get("file_content")  # base64 字符串（前端上传图片时使用）

        temp_file_path = None
        cleanup_paths = []
        parsed_courses = []
        llm_img_for_merge = None
        llm_cls_from_img = None

        if not file_path and file_content:
            temp_file_path = self._save_temp_file(
                file_content, payload.get("file_name"))
            file_path = temp_file_path
            cleanup_paths.append(temp_file_path)

        # 显式 llm_only + 文件输入时，自动走多模态直读路径（不进入 OCR）
        if llm_only and file_path:
            llm_image_opt = True

        # 如果显式选择“直接读图”且 LLM 可用，优先走多模态直读，直接返回标准 JSON，跳过 OCR/NLP
        if llm_image_opt and llm_available and file_path:
            llm_img = None
            try:
                _, ext = os.path.splitext(file_path)
                if ext.lower() == ".pdf":
                    file_path = self._convert_pdf_to_image(file_path)
                    cleanup_paths.append(file_path)
                img_b64 = file_content or self._file_to_base64(file_path)
                if not img_b64:
                    raise ValidationError("无法读取文件生成 base64")
                llm_img = self.llm_service.extract_with_llm_image(
                    img_b64,
                    os.path.basename(file_path),
                    template_config,
                    override=llm_override,
                )
                if not llm_img:
                    raise ServiceError("LLM 多模态解析失败", detail="未返回有效结果")

                base_fields = {
                    "file_id": file_id,
                    "template_id": template_config.get("template_id"),
                    "extract_main": {
                        "total_fields": len(template_config.get("fields", [])),
                        "extracted_fields": 0,
                        "confidence": 0,
                        "status": "failed",
                    },
                    "extract_details": [],
                    "processing_time": 0,
                }
                llm_img = self._inject_extract_details_from_llm(llm_img, template_config)
                merged_fields = self._merge_fields(base_fields, llm_img)

                # 将多模态 LLM 的 doc_type 也统一通过 DocumentTypeService 走一遍
                llm_cls_for_doc = None
                if llm_img and llm_img.get("document_type"):
                    llm_cls_for_doc = {
                        "document_type": llm_img.get("document_type"),
                        "confidence": self._normalize_confidence(llm_img.get("confidence_overall")),
                        "probabilities": llm_img.get("document_type_candidates") or {},
                        "source": "llm_image",
                    }

                doc_type_result = self.doc_type_service.detect(
                    text=llm_img.get("text") or "",
                    ocr_analysis=None,
                    nlp_cls=None,
                    llm_cls=llm_cls_for_doc,
                )
                final_doc_type = doc_type_result.get(
                    "doc_type") or llm_img.get("document_type") or "generic_document"
                doc_type_candidates = doc_type_result.get(
                    "candidates") or (llm_img.get("document_type_candidates") or {})
                primary_type = doc_type_result.get("primary_type")

                classification = {
                    "document_type": final_doc_type,
                    "confidence": self._normalize_confidence(llm_img.get("confidence_overall")),
                    "probabilities": doc_type_candidates,
                    "source": "llm_image",
                }

                raw = {
                    "file_id": file_id,
                    "ocr": None,
                    "fields": merged_fields,
                    "generated_template": None,
                    "courses": (llm_img or {}).get("courses") or [],
                    "text": (llm_img or {}).get("text"),
                    "entities": [],
                    "relations": [],
                    "classification": classification,
                    "processing_time": 0,
                    "text_sections": {"body_text": "", "table_text": ""},
                    "ocr_analysis": None,
                }
                resp = self._finalize_response(
                    raw,
                    template_config,
                    steps=["llm_image_only"],
                    fail_reason=None,
                    ocr_analysis=None,
                    final_doc_type=final_doc_type,
                    doc_type_candidates=doc_type_candidates,
                    primary_type=primary_type,
                )
                # 直读图像场景下，若结果为空壳或类型异常，打印返回内容片段便于调试
                try:
                    if resp and self._should_log_debug_snapshot(resp):
                        logger.warning(
                            "LLM image response debug snapshot",
                            file_id=file_id,
                            document_type=resp.get("document_type"),
                            classification=resp.get("classification"),
                            summary=resp.get("summary"),
                            text_preview=(resp.get("text") or "")[:500],
                            extract_main=resp.get("fields", {}).get("extract_main"),
                            extract_details_sample=self._sample_extract_details(resp.get("fields", {}).get("extract_details"), 10),
                        )
                except Exception:
                    pass
                # 直接清理临时文件并返回
                for tmp in cleanup_paths:
                    if tmp and os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                return resp
            except Exception as exc:
                logger.error(f"LLM image path failed: {exc}")
                raise ServiceError("LLM 多模态解析失败", detail=str(exc))

        if not file_path and not text:
            raise ValidationError("必须提供 file_path 或 text 作为输入")
        logger.info(
            "FinalAIService start",
            file_path=file_path,
            has_text=bool(text),
            file_id=file_id,
            template_fields=len(template_config.get("fields", [])),
            llm_enabled=self.llm_service.enabled,
            llm_disabled=llm_disabled,
            llm_only=llm_only,
            llm_user_opt_in=llm_allowed,
            auto_infer_fields=auto_infer_fields,
            analyze_ocr=analysis_enabled,
        )

        ocr_result = None
        ocr_text_available = False

        # ===== OCR 阶段：不再在这里删除临时文件 =====
        if file_path:
            # 懒加载 OCR，避免仅文本请求时强行加载 paddleocr 依赖
            if self.ocr_service is None:
                try:
                    self.ocr_service = self._ocr_loader() if self._ocr_loader else OCRService()
                    logger.info("OCRService loaded lazily")
                except Exception as exc:
                    logger.error(f"初始化 OCR 失败: {exc}")
                    raise ServiceError("OCR 初始化失败", detail=str(exc))

            # 1) 路径校验：防目录穿越、格式/大小限制
            file_path = self._validate_file_path(file_path)

            # PDF 转图片（取第一页）
            _, ext = os.path.splitext(file_path)
            if ext.lower() == ".pdf":
                file_path = self._convert_pdf_to_image(file_path)
                cleanup_paths.append(file_path)

            if not os.path.exists(file_path):
                raise ValidationError(f"文件不存在: {file_path}")

            logger.debug("OCR stage: start",
                         normalized_path=file_path, options=options)
            try:
                ocr_result = self.ocr_service.recognize(file_path, options)
                text = ocr_result.get("text", "")
                ocr_text_available = bool(text)
                logger.debug(
                    "OCR raw output",
                    characters=len(text),
                    avg_confidence=ocr_result.get("confidence"),
                    total_boxes=ocr_result.get("total_boxes", 0),
                )
            except Exception as exc:
                detail = getattr(exc, 'detail', None) or str(exc)
                logger.error(f"OCR 识别失败（已尝试所有降级策略）: {detail}")
                raise ServiceError("OCR 识别失败", detail=detail)

            # LLM 纠错提升 OCR 结果
            if llm_available and ocr_text_available:
                enhanced = self.llm_service.enhance_ocr(
                    text,
                    ocr_result.get("confidence", 0),
                    override=llm_override,
                )
                # 保留原 OCR 元信息（bbox/processing_time），仅覆盖文本和置信度
                ocr_result.update(enhanced)
                text = ocr_result.get("text", text)
                logger.info(
                    "OCR finished (with optional LLM enhance)",
                    file_path=file_path,
                    confidence=ocr_result.get("confidence"),
                    total_boxes=ocr_result.get("total_boxes", 0),
                    processing_time=ocr_result.get("processing_time"),
                    source=ocr_result.get("source", "ocr"),
                )
            else:
                logger.info(
                    "OCR finished",
                    file_path=file_path,
                    confidence=ocr_result.get(
                        "confidence") if ocr_result else None,
                    total_boxes=ocr_result.get(
                        "total_boxes", 0) if ocr_result else 0,
                    processing_time=ocr_result.get(
                        "processing_time") if ocr_result else None,
                    source=ocr_result.get(
                        "source", "ocr") if ocr_result else "ocr",
                )

        if not text and ocr_result and ocr_result.get("table_plain"):
            text = ocr_result.get("table_plain")
        if not text:
            # OCR 和手工文本都为空，尝试从表格/HTML 萃取纯文本再兜底
            text = self._fallback_text_from_ocr(ocr_result)
            if not text and llm_available and file_path:
                try:
                    logger.info("OCR 文本为空，触发 LLM 多模态兜底")
                    llm_img_path = file_path
                    if file_path.lower().endswith(".pdf"):
                        llm_img_path = self._convert_pdf_to_image(file_path)
                        cleanup_paths.append(llm_img_path)

                    img_b64 = file_content or self._file_to_base64(
                        llm_img_path)
                    llm_img = self.llm_service.extract_with_llm_image(
                        img_b64,
                        os.path.basename(file_path),
                        template_config,
                        override=llm_override,
                    )

                    if llm_img:
                        llm_img = self._inject_extract_details_from_llm(llm_img, template_config)
                        base_fields = {
                            "file_id": file_id,
                            "template_id": template_config.get("template_id"),
                            "extract_main": {
                                "total_fields": len(template_config.get("fields", [])),
                                "extracted_fields": 0,
                                "confidence": 0,
                                "status": "failed",
                            },
                            "extract_details": [],
                            "processing_time": 0,
                        }
                        merged_fields = self._merge_fields(
                            base_fields, llm_img)

                        llm_cls_for_doc = None
                        if llm_img.get("document_type"):
                            llm_cls_for_doc = {
                                "document_type": llm_img.get("document_type"),
                                "confidence": self._normalize_confidence(llm_img.get("confidence_overall")),
                                "probabilities": llm_img.get("document_type_candidates") or {},
                                "source": "llm_image",
                            }

                        doc_type_result = self.doc_type_service.detect(
                            text=llm_img.get("text") or "",
                            ocr_analysis=None,
                            nlp_cls=None,
                            llm_cls=llm_cls_for_doc,
                        )
                        final_doc_type = doc_type_result.get("doc_type") or llm_img.get(
                            "document_type") or "generic_document"
                        doc_type_candidates = doc_type_result.get("candidates") or (
                            llm_img.get("document_type_candidates") or {})
                        primary_type = doc_type_result.get("primary_type")

                        classification = {
                            "document_type": final_doc_type,
                            "confidence": self._normalize_confidence(llm_img.get("confidence_overall")),
                            "probabilities": doc_type_candidates,
                            "source": "llm_image",
                        }

                        raw = {
                            "file_id": file_id,
                            "ocr": None,
                            "fields": merged_fields,
                            "generated_template": None,
                            "courses": (llm_img or {}).get("courses") or [],
                            "text": (llm_img or {}).get("text"),
                            "entities": [],
                            "relations": [],
                            "classification": classification,
                            "processing_time": 0,
                            "text_sections": {"body_text": "", "table_text": ""},
                            "ocr_analysis": None,
                        }
                        resp = self._finalize_response(
                            raw,
                            template_config,
                            steps=["ocr", "llm_image_fallback"],
                            fail_reason=None,
                            ocr_analysis=None,
                            final_doc_type=final_doc_type,
                            doc_type_candidates=doc_type_candidates,
                            primary_type=primary_type,
                        )
                        for tmp in cleanup_paths:
                            if tmp and os.path.exists(tmp):
                                try:
                                    os.remove(tmp)
                                except Exception:
                                    pass
                        return resp
                    else:
                        logger.warning("LLM 多模态兜底未返回有效结果")
                except Exception as exc:
                    logger.warning(f"LLM 多模态兜底失败: {exc}")

            if not text:
                raise ValidationError("无法获取文本内容进行解析，请检查文件或原文输入")

        # 若 OCR 文本极短或置信度过低，自动触发一次 LLM 多模态增强（不中断主流程）
        weak_text = False
        if text:
            weak_text = len(text.strip()) < 25
            if ocr_result:
                weak_text = weak_text or (
                    float(ocr_result.get("confidence") or 0) < 0.35
                ) or (int(ocr_result.get("total_boxes") or 0) <= 4)

        if weak_text and llm_available and file_path:
            try:
                logger.info(
                    "OCR 文本疑似无效，触发 LLM 多模态增强",
                    length=len(text.strip()) if text else 0,
                    confidence=ocr_result.get(
                        "confidence") if ocr_result else None,
                    boxes=ocr_result.get(
                        "total_boxes") if ocr_result else None,
                )

                llm_img_b64 = file_content or self._file_to_base64(file_path)
                llm_img_for_merge = self.llm_service.extract_with_llm_image(
                    llm_img_b64,
                    os.path.basename(file_path),
                    template_config,
                    override=llm_override,
                )
                if llm_img_for_merge:
                    llm_img_for_merge = self._inject_extract_details_from_llm(
                        llm_img_for_merge, template_config
                    )

                if llm_img_for_merge:
                    if llm_img_for_merge.get("text"):
                        text = llm_img_for_merge.get("text")
                        if ocr_result is not None:
                            ocr_result["text"] = text
                    if llm_img_for_merge.get("document_type"):
                        llm_cls_from_img = {
                            "document_type": llm_img_for_merge.get("document_type"),
                            "confidence": self._normalize_confidence(llm_img_for_merge.get("confidence_overall")),
                            "probabilities": llm_img_for_merge.get("document_type_candidates") or {},
                            "source": "llm_image",
                        }
                    logger.info("LLM 多模态增强完成，将与后续字段合并并更新文本")
                else:
                    logger.warning("LLM 多模态增强未返回有效结果")
            except Exception as exc:
                logger.warning(f"LLM 多模态增强失败: {exc}")

        # 全文文本（OCR 或输入）
        full_text = text

        self._log_text_preview("ocr_text", full_text)

        # 如果显式启用 OCR 快速返回，且 OCR 质量良好，则直接返回（默认关闭）
        ocr_conf = float(ocr_result.get("confidence") or 0) if ocr_result else 0
        text_len = len(full_text.strip()) if full_text else 0
        if (
            ocr_fast_return
            and
            not llm_disabled 
            and not llm_only 
            and ocr_result 
            and ocr_conf >= 0.7  # OCR 置信度足够高（70%）
            and text_len >= 100  # OCR 文本足够长（100 字）
            and ocr_result.get("total_boxes", 0) >= 5  # 至少识别到 5 个文本框
        ):
            logger.info(
                "OCR quality sufficient, skipping NLP/LLM and returning directly",
                ocr_confidence=ocr_conf,
                text_length=text_len,
                total_boxes=ocr_result.get("total_boxes"),
            )
            # 构造最终返回结果（仅使用 OCR 结果）
            resp = {
                "file_id": file_id,
                "text": full_text,
                "confidence_overall": ocr_conf,
                "document_type": ocr_result.get("document_type") or "generic_document",
                "classification": {
                    "document_type": ocr_result.get("document_type") or "generic_document",
                    "confidence": ocr_conf,
                    "probabilities": {},
                    "source": "ocr",
                },
                "fields": {
                    "file_id": file_id,
                    "extract_main": {
                        "total_fields": len(template_config.get("fields", [])),
                        "extracted_fields": 0,
                        "confidence": 0,
                        "status": "ocr_only",
                    },
                    "extract_details": [],
                    "processing_time": ocr_result.get("processing_time", 0),
                    "template_id": template_config.get("template_id"),
                },
                "basic_info": {},
                "academic_info": {},
                "certificate_info": {},
                "financial_info": {},
                "leave_info": {},
                "courses": [],
                "tables": [],
                "summary": f"类型 {ocr_result.get('document_type') or '其他'}，置信度 {ocr_conf:.2f}，仅使用 OCR 识别结果",
                "meta": {
                    "file_id": file_id,
                    "ocr_quality": "good",
                    "stat_date": None,
                },
            }
            logger.info(
                "FinalAIService complete (OCR only, fast path)",
                file_id=file_id,
                confidence=ocr_conf,
                processing_time=ocr_result.get("processing_time", 0),
            )
            return resp

        # 切分正文/表格区域供后续处理（仍保留全文用作 LLM 兜底）
        sections = self._split_text_sections(full_text)
        nlp_text = sections["body_text"] + \
            ("\n" + sections["table_text"] if sections["table_text"] else "")
        nlp_text = nlp_text.strip() or full_text

        # 自动字段推断与模板生成（无字段或显式开启时）
        auto_template_generated = False
        auto_fail_reason = None
        nlp_fields = None
        candidate_fields = None
        if auto_infer_fields or not template_config.get("fields"):
            auto_result = self.nlp_service.infer_fields_auto(nlp_text)
            template_config = auto_result["template_config"]
            nlp_fields = auto_result["extract_result"]
            auto_template_generated = True
            auto_fail_reason = auto_result.get("fail_reason")
            candidate_fields = nlp_fields.get(
                "extract_details") if nlp_fields else None
        else:
            # 自动匹配内置模板：若前端未提供字段则尝试选择最优模板
            if not template_config.get("fields"):
                doc_cls = self.nlp_service.classify_document(text)
                matched = self._select_best_template(text, doc_cls)
                template_config = matched or self._build_default_template()

            # required_fields -> fields 转换，确保后续 extract_fields 有字段可用
            if template_config and not template_config.get("fields") and template_config.get("required_fields"):
                fields = []
                for f in template_config.get("required_fields", []):
                    fields.append(
                        {
                            "name": f.get("field_name") or f.get("name"),
                            "type": f.get("field_type") or f.get("type", "TEXT"),
                            "description": f.get("description", ""),
                            "required": True,
                        }
                    )
                template_config["fields"] = fields

            # 如果选择仅使用 LLM，则跳过规则 NLP 抽取
            if not llm_only:
                logger.debug("NLP stage: start", file_id=file_id)
                nlp_fields = self.nlp_service.extract_fields(
                    nlp_text, template_config, file_id=file_id)
                candidate_fields = nlp_fields.get(
                    "extract_details") if nlp_fields else None

        # 课程表规则解析（简单分块法，适用于成绩单表格结构）
        parsed_courses = self._extract_courses_table(
            sections["table_text"] or full_text)

        # 实体与关系（只要非 llm_only 都执行）
        entities = []
        relations = []
        classification = None
        nlp_cls = None
        if not llm_only:
            entities = self.nlp_service.extract_entities(nlp_text)
            relations = self.nlp_service.extract_relations(nlp_text, entities)
            entity_slots = self.nlp_service.map_entities_to_slots(
                entities, nlp_text)
            if entity_slots:
                candidate_fields = (candidate_fields or []) + entity_slots
            classification = self.nlp_service.classify_document(nlp_text)
            nlp_cls = classification
            if entities:
                logger.info("NLP entities extracted", count=len(
                    entities), sample=entities[:10])
            if nlp_fields:
                logger.info(
                    "NLP extracted fields",
                    extracted=nlp_fields["extract_main"]["extracted_fields"],
                    total=nlp_fields["extract_main"]["total_fields"],
                    sample=self._sample_extract_details(
                        nlp_fields.get("extract_details")),
                )
                logger.info(
                    "NLP finished",
                    extracted_fields=nlp_fields["extract_main"]["extracted_fields"],
                    total_fields=nlp_fields["extract_main"]["total_fields"],
                    entities=len(entities),
                    relations=len(relations),
                )

        # OCR 文本综合分析（类型/质量/预期字段/模板）
        ocr_analysis = None
        if full_text:
            ocr_analysis = self.nlp_service.analyze_ocr_text(
                full_text, ocr_result)
            if not candidate_fields and ocr_analysis.get("extracted_fields"):
                candidate_fields = ocr_analysis.get("extracted_fields")

        # 构造传入 LLM 的综合文本（清洗文本 + 表格行 + doc hint + candidate_fields + kv）
        llm_input_text = text
        if ocr_analysis and ocr_analysis.get("reconstructed"):
            recon_text = ocr_analysis["reconstructed"].get("text") or text
            table_lines = ocr_analysis["reconstructed"].get(
                "table_lines") or []
            doc_hint = ocr_analysis.get("document_type", {}) or {}
            kv_pairs = ocr_analysis.get("key_value_pairs") or []
            llm_input_text = self._build_llm_input_text(
                recon_text, table_lines, doc_hint, candidate_fields, kv_pairs)
        cached_candidates = self._load_cached_candidates(llm_input_text)
        if cached_candidates:
            candidate_fields = (candidate_fields or []) + cached_candidates

        # 确保 nlp_fields 至少有基本结构，避免空指针
        if nlp_fields is None:
            nlp_fields = {
                "file_id": file_id,
                "template_id": template_config.get("template_id"),
                "extract_main": {
                    "total_fields": len(template_config.get("fields", [])),
                    "extracted_fields": 0,
                    "confidence": 0,
                    "status": "failed",
                },
                "extract_details": [],
                "processing_time": 0,
            }

        # 3) LLM 兜底（当 OCR+NLP 未能提取字段时触发）
        llm_fallback = None
        fail_reason = None
        if llm_only:
            if not llm_available:
                raise ServiceError(
                    "LLM 未启用", detail="请开启 ENABLE_LLM_FALLBACK 或检查密钥，或取消禁用 AI")
            logger.info("LLM only mode enabled",
                        provider=self.llm_service.provider, model=self.llm_service.model)
            llm_fallback = self.llm_service.extract_with_llm(
                llm_input_text, template_config, override=llm_override)
            merged_fields = self._merge_fields(
                {
                    "file_id": file_id,
                    "template_id": template_config.get("template_id"),
                    "extract_main": {
                        "total_fields": len(template_config.get("fields", [])),
                        "extracted_fields": 0,
                        "confidence": 0,
                        "status": "failed",
                    },
                    "extract_details": [],
                    "processing_time": 0,
                },
                llm_fallback,
            )
            if llm_fallback:
                self._log_llm_result(llm_fallback, stage="llm_only")
            if not llm_fallback:
                fail_reason = "LLM 调用失败或未返回结果"
        else:
            nlp_extract_main = (nlp_fields.get("extract_main") or {})
            extracted_count = int(
                nlp_extract_main.get("extracted_fields") or 0)
            llm_triggered = llm_available and extracted_count == 0

            if llm_triggered:
                logger.info(
                    "LLM fallback triggered because NLP extracted 0 fields",
                    provider=self.llm_service.provider,
                    model=self.llm_service.model,
                )
                llm_fallback = self.llm_service.extract_with_llm(
                    llm_input_text, template_config, override=llm_override)
                if llm_fallback:
                    logger.info("LLM fallback used", model=self.llm_service.model,
                                provider=self.llm_service.provider)
                    self._log_llm_result(llm_fallback, stage="llm_fallback")
                else:
                    logger.warning(
                        "LLM fallback skipped or failed to return extract_details")
                    fail_reason = "LLM 调用失败或未返回结果"
            elif not self.llm_service.enabled:
                logger.debug("LLM fallback disabled, skip LLM stage")
                fail_reason = "LLM 未启用或已禁用"

            merged_fields = self._merge_fields(nlp_fields, llm_fallback)
            if auto_fail_reason and not merged_fields.get("extract_main", {}).get("extracted_fields"):
                fail_reason = auto_fail_reason
            if merged_fields and merged_fields.get("extract_details"):
                merged_fields["extract_details"] = self._validate_and_fix_fields(
                    merged_fields["extract_details"])

        # 若前面多模态增强返回了结果，这里与字段结果合并
        if llm_img_for_merge:
            merged_fields = self._merge_fields(
                merged_fields, llm_img_for_merge)
            merged_fields["llm_image_used"] = True

            # 补充课程信息：如果规则/文本结果为空或明显噪声，则采用多模态返回的课程
            if llm_img_for_merge.get("courses") and (not parsed_courses or self._is_noisy_courses(parsed_courses)):
                parsed_courses = self._clean_llm_courses(
                    llm_img_for_merge.get("courses"))

        # 可选：在文本兜底后，再追加一次「图像直读」的 LLM 兜底，以缓解 OCR 误差
        if llm_available and llm_image_opt and file_path and (not llm_only):
            try:
                img_b64 = file_content or self._file_to_base64(file_path)
                if img_b64:
                    logger.info("LLM image fallback triggered",
                                provider=self.llm_service.provider)
                    llm_img = self.llm_service.extract_with_llm_image(
                        img_b64,
                        os.path.basename(file_path),
                        template_config,
                        override=llm_override,
                    )
                    if llm_img:
                        llm_img = self._inject_extract_details_from_llm(llm_img, template_config)
                        merged_fields = self._merge_fields(
                            merged_fields, llm_img)
                        # 如果图片直读带回课程且当前课程为空/噪声，尝试采用
                        if llm_img.get("courses") and (not parsed_courses or self._is_noisy_courses(parsed_courses)):
                            parsed_courses = self._clean_llm_courses(
                                llm_img.get("courses"))
                        logger.info("LLM image fallback applied")
            except Exception as exc:
                logger.warning(f"LLM image fallback failed: {exc}")

        total_time = round(time.time() - start, 2)
        logger.info(
            "FinalAIService complete",
            file_id=file_id,
            processing_time=total_time,
            llm_fallback_used=bool(merged_fields.get("llm_fallback_used")),
        )

        # LLM 文档类型分类（优先使用 LLM 结果覆盖 NLP），然后统一通过 DocumentTypeService 再决策一次
        llm_cls = llm_cls_from_img
        if llm_available:
            text_cls = self.llm_service.classify_with_llm(
                full_text, override=llm_override)
            if text_cls:
                # 若已有多模态分类，选择置信度更高者
                if llm_cls is None or self._normalize_confidence(text_cls.get("confidence")) > self._normalize_confidence(llm_cls.get("confidence")):
                    llm_cls = text_cls
            if llm_cls:
                classification = llm_cls

        # 统一文档类型检测：只在这里做一次，后面都只读 doc_type，不再随意修改
        doc_type_result = self.doc_type_service.detect(
            text=full_text,
            ocr_analysis=ocr_analysis,
            nlp_cls=nlp_cls,
            llm_cls=llm_cls,
        )
        final_doc_type = doc_type_result.get("doc_type") or "generic_document"
        doc_type_candidates = doc_type_result.get("candidates") or {}
        primary_type = doc_type_result.get("primary_type")

        # 将统一后的类型信息回填到 classification
        if classification is None:
            classification = {
                "document_type": final_doc_type,
                "confidence": doc_type_candidates.get(final_doc_type, 0.0),
                "probabilities": doc_type_candidates,
                "source": "doc_type_service",
            }
        else:
            classification["document_type"] = final_doc_type
            existing_probs = classification.get("probabilities") or {}
            # 以 DocumentTypeService 的 candidates 为主
            merged_probs = dict(existing_probs)
            merged_probs.update(doc_type_candidates)
            classification["probabilities"] = merged_probs

        if classification:
            logger.info(
                "Document classified",
                cls=classification.get("document_type"),
                source=classification.get("source", "nlp"),
            )

        raw = {
            "file_id": file_id,
            "ocr": ocr_result,
            "fields": merged_fields,
            "generated_template": template_config if auto_template_generated else None,
            "courses": parsed_courses,
            "entities": entities,
            "relations": relations,
            "classification": classification,
            "processing_time": total_time,
            "text_sections": sections,
            "ocr_analysis": ocr_analysis,
        }
        resp = self._finalize_response(
            raw,
            template_config,
            steps=["ocr", "nlp_auto" if auto_template_generated else "nlp",
                   "llm" if llm_fallback else "nlp_only"],
            fail_reason=fail_reason,
            ocr_analysis=ocr_analysis,
            final_doc_type=final_doc_type,
            doc_type_candidates=doc_type_candidates,
            primary_type=primary_type,
        )
        # 常规流程下若出现“无有效类型/无字段”，打印返回内容用于线上调试
        try:
            if resp and self._should_log_debug_snapshot(resp):
                logger.warning(
                    "Final response debug snapshot",
                    file_id=file_id,
                    document_type=resp.get("document_type"),
                    classification=resp.get("classification"),
                    summary=resp.get("summary"),
                    text_preview=(resp.get("text") or "")[:500],
                    extract_main=resp.get("fields", {}).get("extract_main"),
                    extract_details_sample=self._sample_extract_details(resp.get("fields", {}).get("extract_details"), 10),
                )
        except Exception:
            pass
        # 缓存高置信度字段供后续相似文档复用
        if resp and resp.get("fields", {}).get("extract_details"):
            self._save_cache_entry(
                llm_input_text, resp["fields"]["extract_details"])

        # ===== 在流程结束时统一清理临时文件 =====
        for tmp in cleanup_paths:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                    logger.debug("临时文件已删除", temp_file=tmp)
                except Exception as exc:
                    logger.warning(f"删除临时文件失败: {exc}")

        return resp

    @staticmethod
    def _sample_extract_details(details: Optional[List[Dict]], limit: int = 8) -> List[Dict]:
        if not details:
            return []
        sample = []
        for d in details:
            sample.append(
                {
                    "name": d.get("field_name"),
                    "type": d.get("field_type"),
                    "value": d.get("field_value"),
                    "status": d.get("status"),
                }
            )
            if len(sample) >= limit:
                break
        return sample

    @staticmethod
    def _has_structured_values(resp: Dict) -> bool:
        for section in ("basic_info", "academic_info", "certificate_info", "financial_info", "leave_info"):
            data = resp.get(section)
            if isinstance(data, dict) and any(v not in (None, "", [], {}) for v in data.values()):
                return True
        return False

    @classmethod
    def _should_log_debug_snapshot(cls, resp: Dict) -> bool:
        details = resp.get("fields", {}).get("extract_details") or []
        has_details = bool(details)
        text_len = len(str(resp.get("text") or "").strip())
        summary_len = len(str(resp.get("summary") or "").strip())
        has_structured = cls._has_structured_values(resp)
        doc_type = str(resp.get("document_type") or "").strip().lower()
        invalid_type = doc_type in {"", "generic_document", "unknown", "其他"}

        # Only log snapshot when response is truly empty/invalid.
        no_content = (not has_details) and text_len < 10 and summary_len < 5 and (not has_structured)
        return no_content or (invalid_type and not has_details and not has_structured and text_len < 10)

    @staticmethod
    def _log_text_preview(label: str, text: Optional[str], limit: int = 400) -> None:
        if text is None:
            return
        snippet = text[:limit]
        more = len(text) - len(snippet)
        suffix = f"... (+{more} chars)" if more > 0 else ""
        logger.info(f"{label} preview", length=len(
            text), preview=snippet + suffix)

    def _log_llm_result(self, llm_result: Dict, stage: str) -> None:
        sample = self._sample_extract_details(
            llm_result.get("extract_details"))
        logger.info(
            "LLM output",
            stage=stage,
            extracted=llm_result.get(
                "extract_main", {}).get("extracted_fields"),
            total=llm_result.get("extract_main", {}).get("total_fields"),
            sample=sample,
            doc_type=llm_result.get("document_type") or (
                llm_result.get("classification") or {}).get("document_type"),
        )

    def _validate_file_path(self, file_path: str) -> str:
        """
        Validate file path for security:
        1. Check file extension is allowed
        2. Normalize path to prevent directory traversal
        3. Check file size is reasonable
        """
        allowed_extensions = {".jpg", ".jpeg", ".png",
                              ".bmp", ".tiff", ".tif", ".webp", ".pdf"}

        normalized_path = os.path.normpath(file_path)
        _, ext = os.path.splitext(normalized_path)
        if ext.lower() not in allowed_extensions:
            raise ValidationError(
                f"不支持的文件格式: {ext}",
                detail=f"允许的格式: {', '.join(sorted(allowed_extensions))}",
            )

        if os.path.exists(normalized_path):
            if not os.path.isfile(normalized_path):
                raise ValidationError(f"路径不是文件: {normalized_path}")

            file_size = os.path.getsize(normalized_path)
            max_size = 10 * 1024 * 1024  # 10MB
            if file_size > max_size:
                raise ValidationError(
                    f"文件过大: {file_size / 1024 / 1024:.2f}MB",
                    detail=f"最大允许 {max_size / 1024 / 1024}MB",
                )

        return normalized_path

    def _save_temp_file(self, file_content_b64: str, file_name: Optional[str] = None) -> str:
        """
        将 base64 编码的图片保存到临时文件，返回路径。
        """
        try:
            data = base64.b64decode(file_content_b64.split(",")[-1])
        except Exception as exc:
            raise ValidationError(
                "file_content 不是有效的 base64 字符串", detail=str(exc))

        suffix = ".png"
        if file_name:
            _, ext = os.path.splitext(file_name)
            if ext:
                suffix = ext if ext.startswith(".") else f".{ext}"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        logger.info("临时文件已创建", temp_file=tmp.name, size=len(data))
        return tmp.name

    def _convert_pdf_to_image(self, pdf_path: str) -> str:
        """
        将 PDF 第一页转换为 PNG，返回临时图片路径。
        依赖 poppler；若容器未安装，将提示安装 poppler-utils。
        """
        try:
            from pdf2image import convert_from_path
            from pdf2image.exceptions import PDFInfoNotInstalledError
        except ImportError as exc:
            raise ServiceError("缺少 pdf2image 依赖，无法处理 PDF", detail=str(exc))
        try:
            # 降低 DPI 到 200，减少图片尺寸与内存占用
            images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
            if not images:
                raise ServiceError("PDF 转图片失败", detail="未生成任何页面")
            img = images[0]

            # 检查图片尺寸，如果过大则立即降采样
            w, h = img.size
            max_dim = 2400  # 限制最大边，避免后续 OCR primitive 执行失败
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                from PIL import Image
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.warning(
                    "PDF 转图片后尺寸过大，已降采样",
                    original=(w, h),
                    downsampled=(new_w, new_h),
                    pdf=pdf_path,
                )
            else:
                logger.info(
                    "PDF 转图片尺寸正常",
                    size=(w, h),
                    pdf=pdf_path,
                )

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img.save(tmp.name, "PNG")
            logger.info("PDF 已转换为图片", pdf=pdf_path, image=tmp.name, final_size=(img.size))
            return tmp.name
        except PDFInfoNotInstalledError as exc:
            raise ServiceError(
                "PDF 转图片失败",
                detail="缺少 poppler，可在容器中安装 poppler-utils 后重试。",
            ) from exc
        except Exception as exc:
            raise ServiceError("PDF 转图片失败", detail=str(exc))

    def _merge_fields(self, nlp_fields: Dict, llm_fallback: Optional[Dict]) -> Dict:
        """
        合并规则抽取与 LLM 兜底的结果，避免重复字段。
        """
        if not llm_fallback:
            return nlp_fields

        merged = nlp_fields.copy()
        merged_details: List[Dict] = nlp_fields.get("extract_details", [])[:]
        llm_details = llm_fallback.get("extract_details", [])

        # Only add missing fields from LLM result
        existing_names = {item.get("field_name") for item in merged_details}
        for item in llm_details:
            if item.get("field_name") not in existing_names:
                merged_details.append(item)
                logger.debug(
                    "Merged LLM field",
                    field_name=item.get("field_name"),
                    confidence=item.get("confidence"),
                    source="llm",
                )

        merged["extract_details"] = merged_details
        merged["llm_fallback_used"] = True

        # 重算主摘要指标：抽取数量、平均置信度与状态
        extracted_fields = sum(
            1 for d in merged_details if d.get("field_value") is not None)
        total_conf = sum(d.get("confidence", 0)
                         for d in merged_details if d.get("field_value") is not None)
        avg_conf = total_conf / extracted_fields if extracted_fields else 0.0
        if "extract_main" not in merged:
            merged["extract_main"] = {}
        merged["extract_main"].update(
            {
                "extracted_fields": extracted_fields,
                "confidence": round(avg_conf, 4),
                "status": "success" if extracted_fields > 0 else "failed",
            }
        )

        return merged

    @staticmethod
    def _inject_extract_details_from_llm(llm_result: Dict, template_config: Dict) -> Dict:
        """Convert LLM structured blocks into extract_details for unified field pipeline."""
        if not isinstance(llm_result, dict):
            return llm_result

        existing = llm_result.get("extract_details") or []
        if existing:
            return llm_result

        def _iter_pairs(section_name: str):
            section = llm_result.get(section_name)
            if isinstance(section, dict):
                for k, v in section.items():
                    yield k, v

        details: List[Dict] = []
        seen = set()

        # Prefer template fields to keep output aligned with selected schema.
        for f in (template_config or {}).get("fields", []):
            name = f.get("name")
            if not name:
                continue
            value = None
            for sec in ("basic_info", "academic_info", "certificate_info", "financial_info", "leave_info"):
                block = llm_result.get(sec)
                if isinstance(block, dict) and name in block:
                    value = block.get(name)
                    break
            details.append(
                {
                    "field_name": name,
                    "field_type": f.get("type", "TEXT"),
                    "field_value": value,
                    "confidence": FinalAIService._normalize_confidence(llm_result.get("confidence_overall"), 0.8 if value is not None else 0.0),
                    "status": "ok" if value is not None else "missing",
                    "source": "llm_image",
                }
            )
            seen.add(name)

        # If no template or template misses keys, include non-null structured keys from LLM.
        for sec in ("basic_info", "academic_info", "certificate_info", "financial_info", "leave_info"):
            for key, value in _iter_pairs(sec):
                if key in seen:
                    continue
                if value is None:
                    continue
                details.append(
                    {
                        "field_name": key,
                        "field_type": "TEXT",
                        "field_value": value,
                        "confidence": FinalAIService._normalize_confidence(llm_result.get("confidence_overall"), 0.8),
                        "status": "ok",
                        "source": "llm_image",
                    }
                )
                seen.add(key)

        if details:
            llm_result = dict(llm_result)
            llm_result["extract_details"] = details
            llm_result["extract_main"] = {
                "total_fields": len((template_config or {}).get("fields", [])) if (template_config or {}).get("fields") else len(details),
                "extracted_fields": sum(1 for d in details if d.get("field_value") is not None),
                "confidence": FinalAIService._normalize_confidence(llm_result.get("confidence_overall")),
                "status": "success" if any(d.get("field_value") is not None for d in details) else "failed",
            }
        return llm_result

    @staticmethod
    def _normalize_confidence(value, default: float = 0.0) -> float:
        """Normalize confidence to 0~1, supporting '97%' / 97 / '0.97'."""
        try:
            if value is None:
                return float(default)
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return float(default)
                if s.endswith("%"):
                    num = float(s[:-1].strip()) / 100.0
                else:
                    num = float(s)
            else:
                num = float(value)

            # If value is like 97 (not 0.97), map to percentage.
            if num > 1.0 and num <= 100.0:
                num = num / 100.0
            if num < 0:
                return 0.0
            if num > 1:
                return 1.0
            return num
        except Exception:
            return float(default)

    def _build_default_template(self) -> Dict:
        """
        默认模板：无须前端提供时，也能抽取常见字段。
        """
        fields = [
            {
                "name": "student_name",
                "type": "PERSON",
                "required": False,
                "patterns": [r"姓名[:：]\s*([^\s\n]+)"],
                "description": "姓名",
            },
            {
                "name": "student_id",
                "type": "STUDENT_ID",
                "required": False,
                "patterns": [r"学号[:：]\s*([\dA-Za-z]+)"],
                "description": "学号/学籍号",
            },
            {
                "name": "average_gpa",
                "type": "NUMBER",
                "required": False,
                "patterns": [r"绩点[:：]?\s*([\d\.]+)", r"平均绩点[:：]?\s*([\d\.]+)"],
                "description": "平均绩点",
            },
            {
                "name": "total_credit",
                "type": "NUMBER",
                "required": False,
                "patterns": [r"总学分[:：]?\s*([\d\.]+)"],
                "description": "总学分",
            },
            {
                "name": "stat_date",
                "type": "DATE",
                "required": False,
                "patterns": [r"统计时间[:：]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"],
                "description": "统计日期",
            },
        ]
        return {"fields": fields, "template_id": "auto-generated"}

    def _load_builtin_templates(self) -> List[Dict]:
        """
        读取内置模板 JSON，一次加载缓存。
        """
        try:
            tpl_path = Path(__file__).resolve().parent.parent / \
                "templates" / "builtin_templates.json"
            if tpl_path.exists():
                import json
                data = json.loads(tpl_path.read_text(encoding="utf-8"))
                logger.info("Builtin templates loaded", total=len(data))
                return data
            logger.warning(f"Builtin templates file not found: {tpl_path}")
            return []
        except Exception as exc:
            logger.error(f"加载内置模板失败: {exc}")
            return []

    def _select_best_template(self, text: str, classification: Optional[Dict]) -> Optional[Dict]:
        """
        简单匹配：基于 document_type 一致性 + 关键词命中数量评分，返回最高分模板。
        """
        if not self._builtin_templates:
            return None
        doc_type = (classification or {}).get("document_type", "").strip()
        best = None
        best_score = 0.0
        text_lower = text.lower()
        for tpl in self._builtin_templates:
            score = 0.0
            if doc_type and tpl.get("document_type") == doc_type:
                score += 5
            hits = 0
            keywords = tpl.get("match_keywords") or []
            for kw in keywords:
                if kw and kw.lower() in text_lower:
                    hits += 1
            if keywords:
                score += hits
            norm = min(score / 15.0, 1.0)
            if norm > best_score:
                best_score = norm
                best = tpl
        if best:
            logger.info("Builtin template matched", template_id=best.get(
                "template_id"), score=round(best_score, 3))
        return best

    def _finalize_response(
        self,
        data: Dict,
        template_config: Dict,
        steps: List[str],
        fail_reason: Optional[str] = None,
        ocr_analysis: Optional[Dict] = None,
        final_doc_type: Optional[str] = None,
        doc_type_candidates: Optional[Dict] = None,
        primary_type: Optional[str] = None,
    ) -> Dict:
        """
        输出统一的稳定 JSON 结构，包含类型、字段、表格、清洗后的文本等。

        文档类型在前面的 DocumentTypeService 中已经统一决策，这里只负责使用，不再重新“猜类型”。
        """
        trace_id = str(uuid.uuid4())

        # 文档类型：以 DocumentTypeService 结果为准
        doc_type = final_doc_type or data.get("classification", {}).get(
            "document_type") or "generic_document"
        doc_type_candidates = doc_type_candidates or data.get(
            "classification", {}).get("probabilities") or {}

        fields_block = data.get("fields") or {}
        main = fields_block.get("extract_main") or {}
        extracted = main.get("extracted_fields") or 0
        total_fields = main.get("total_fields") or (len(
            template_config.get("fields", [])) if template_config else 0)
        if not total_fields:
            # In pure-LLM mode template may be empty; derive from available details.
            total_fields = max(
                int(extracted or 0),
                len(fields_block.get("extract_details") or []),
            )
        conf = float(main.get("confidence") or 0.0)
        status = "failed"
        if extracted and total_fields <= 0:
            status = "success"
        elif extracted and extracted == total_fields:
            status = "success"
        elif extracted and extracted < total_fields:
            status = "partial"
        main_normalized = {
            "status": status,
            "confidence": round(conf, 4),
        }

        # 规范化字段详情
        norm_details = []
        for d in fields_block.get("extract_details") or []:
            if not d:
                continue
            item = {
                "field_name": d.get("field_name"),
                "field_type": d.get("field_type") or d.get("type"),
                "field_value": d.get("field_value") if d.get("field_value") is not None else None,
                "confidence": float(d.get("confidence") or 0.0),
            }
            norm_details.append(item)
        norm_details = self._normalize_fields(norm_details, doc_type)

        # 实体清洗
        entities = []
        for e in self._clean_entities(data.get("entities") or []):
            entities.append(e)

        # 表格构建：优先使用 OCR 分析中的 parsed_tables/transcript_rows
        transcript_rows = ocr_analysis.get(
            "transcript_rows") if ocr_analysis else None
        tables = self._format_tables(
            ocr_analysis.get("reconstructed", {}).get(
                "parsed_tables") if ocr_analysis else None,
            transcript_rows,
            doc_type,
        )
        key_values = ocr_analysis.get(
            "key_value_pairs") if ocr_analysis else []

        cleaned_text = None
        if ocr_analysis and ocr_analysis.get("reconstructed", {}).get("text"):
            cleaned_text = self._clean_text_for_output(
                ocr_analysis["reconstructed"]["text"])
        if not cleaned_text:
            cleaned_text = self._clean_text_for_output(
                data.get("text") or (data.get("ocr") or {}).get("text") or ""
            )

        meta_block = {
            "file_id": data.get("file_id"),
            "ocr_quality": ocr_analysis.get("ocr_quality") if ocr_analysis else None,
            "stat_date": None,
        }

        # 默认摘要（后续合同场景会自适应）
        summary = f"类型 {doc_type}，字段 {extracted}/{total_fields}，置信度 {conf:.2f}"
        if fail_reason and extracted == 0:
            summary += f"，原因：{fail_reason}"

        tpl_id = template_config.get("template_id") if isinstance(
            template_config, dict) else None
        # 展平字段为通用字段 dict（空用 None）
        flat_fields = self._flatten_fields(norm_details)
        # 合同类助学贷款字段专项映射
        contract_info = None
        loan_keywords = ("助学贷款", "贷款合同", "贷款合同编号")
        if doc_type == "合同" and any(k in (cleaned_text or "") for k in loan_keywords):
            contract_info = self._map_contract_loan_fields(
                entities=entities,
                key_values=key_values,
                text=cleaned_text or (data.get("ocr") or {}).get("text") or "",
                flat_fields=flat_fields,
            )

        # courses：汇总规则解析 + transcript_rows 恢复
        courses = data.get("courses") or []
        # 先对规则解析得到的课程做一次去噪，避免将纯数字等噪声返回前端
        if courses:
            courses = self._clean_rule_courses(courses)
        if transcript_rows:
            courses.extend(self._courses_from_transcript(transcript_rows))
        # 成绩单/课程表场景：若课程缺失或明显噪声/条目过少，则强制走 LLM 兜底重建课程表
        doc_is_course = doc_type in ("成绩单", "成绩单/学业成绩表", "课程表", "课表/选课单")
        llm_courses_failed = False
        if doc_is_course and self.llm_service.enabled:
            need_llm_courses = (not courses) or len(
                courses) < 3 or self._is_noisy_courses(courses)
            if need_llm_courses:
                llm_courses = self._llm_courses_fallback(
                    cleaned_text or (data.get("ocr") or {}).get("text") or "", ocr_analysis)
                if llm_courses:
                    courses = llm_courses
                else:
                    # LLM 兜底未返回有效课程，清空疑似噪声
                    courses = []
                    llm_courses_failed = True

        analysis_notes = []
        if fail_reason:
            analysis_notes.append(f"字段提取失败原因：{fail_reason}")
        if ocr_analysis and ocr_analysis.get("ocr_problems"):
            analysis_notes.append(
                f"OCR 问题：{'；'.join(ocr_analysis.get('ocr_problems'))}")
        if not courses and doc_type in ("成绩单", "课程表"):
            analysis_notes.append("未恢复课程表，可能 OCR 表格被拆散或缺失表格模型")
        if llm_courses_failed:
            analysis_notes.append("课程列表疑似噪声，LLM 兜底未返回有效课程，已清空")

        llm_status = {
            "enabled": self.llm_service.enabled,
            "provider": getattr(self.llm_service, "provider", None),
            "model_text": getattr(self.llm_service, "model_text", None),
            "model_vision": getattr(self.llm_service, "model_vision", None),
        }

        fields_payload = {
            "extract_main": main_normalized,
            "extract_details": norm_details,
            "file_id": data.get("file_id"),
            "template_id": tpl_id,
            "processing_time": data.get("processing_time"),
        }

        normalized = self._build_normalized_blocks(
            doc_type=doc_type,
            doc_type_candidates=doc_type_candidates,
            confidence=round(conf, 4),
            flat_fields=flat_fields,
            entities=entities,
            key_values=key_values,
            tables=tables,
            cleaned_text=cleaned_text or "",
            summary=summary,
            meta_block=meta_block,
            fields_block=fields_payload,
            courses=courses,
            ocr_analysis=ocr_analysis,
            llm_status=llm_status,
            trace_id=trace_id,
            analysis_notes=analysis_notes,
            contract_info=contract_info,
            fail_reason=fail_reason,
        )
        # 合同/贷款合同后处理，纠正核心字段与摘要
        # 这里不再依赖候选概率强行把其它文档当合同处理
        if doc_type in ("合同", "贷款合同", "生源地助学贷款合同"):
            normalized = self._post_process_contract(
                normalized, doc_type_candidates)
        # 最终输出为前端友好结构
        simplified = self._simplify_for_frontend(normalized)
        # 终端兜底修正
        return self._post_fix_normalized_data(simplified)

    def _simplify_for_frontend(self, normalized: Dict) -> Dict:
        """将内部 normalized 结构裁剪为前端/接口稳定输出。

        目前默认透传核心字段并去掉 debug，避免返回体过大。
        """
        if not isinstance(normalized, dict):
            return {}
        out = dict(normalized)
        out.pop("debug", None)
        return out

    def _post_fix_normalized_data(self, data: Dict) -> Dict:
        """最终兜底修复：保证关键字段存在。"""
        if not isinstance(data, dict):
            return {}
        if "classification" not in data:
            doc_type = data.get("document_type") or "generic_document"
            conf = data.get("confidence_overall")
            data["classification"] = {
                "document_type": doc_type,
                "confidence": conf,
                "probabilities": data.get("document_type_candidates") or {},
                "source": "final_ai_service",
            }
        return data

    def _fallback_text_from_ocr(self, ocr_result: Optional[Dict]) -> str:
        """
        当 OCR 主文本为空时，尝试从表格 HTML/box 列表中提取纯文本。
        """
        if not ocr_result:
            return ""
        texts = []
        for b in ocr_result.get("boxes", []):
            if b.get("text"):
                texts.append(str(b["text"]))
        for t in ocr_result.get("table") or []:
            html = t.get("html")
            if html:
                try:
                    plain = re.sub(r"<[^>]+>", " ", html)
                    plain = re.sub(r"\s{2,}", " ", plain).strip()
                    if plain:
                        texts.append(plain)
                except Exception:
                    continue
        return "\n".join(texts).strip()

    @staticmethod
    def _file_to_base64(file_path: str) -> Optional[str]:
        """
        读取本地图像文件并编码为 base64（统一转为 PNG），避免格式不兼容。
        """
        try:
            from PIL import Image
            import base64
            from io import BytesIO

            img = Image.open(file_path)
            # 部分模式（如 RGBA/CMYK）转为 RGB，提升兼容性
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            try:
                # 兜底：直接读文件
                import base64
                with open(file_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None

    @staticmethod
    def _cache_path() -> str:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.abspath(os.path.join(cache_dir, "doc_cache.json"))

    def _load_cached_candidates(self, text: str) -> Optional[List[Dict]]:
        """
        简单缓存：按文本哈希存储高置信度字段，便于相似文档复用候选。
        """
        try:
            cache_file = self._cache_path()
            if not os.path.exists(cache_file):
                return None
            raw = Path(cache_file).read_text(encoding="utf-8")
            if not raw:
                return None
            data = json.loads(raw)
            h = hashlib.md5(text[:2000].encode(
                "utf-8", errors="ignore")).hexdigest()
            entry = data.get(h)
            if entry and entry.get("extract_details"):
                return entry["extract_details"]
        except Exception:
            return None
        return None

    def _save_cache_entry(self, text: str, details: List[Dict]):
        """
        将高置信度字段写入缓存，限制缓存大小。
        """
        try:
            cache_file = self._cache_path()
            h = hashlib.md5(text[:2000].encode(
                "utf-8", errors="ignore")).hexdigest()
            payload = {
                "extract_details": [
                    d for d in details if d.get("confidence", 0) >= 0.7 and d.get("field_value")
                ]
            }
            if not payload["extract_details"]:
                return
            data = {}
            if os.path.exists(cache_file):
                raw = Path(cache_file).read_text(encoding="utf-8") or "{}"
                data = json.loads(raw)
            data[h] = payload
            # 限制最大条目数
            if len(data) > 100:
                keys = list(data.keys())
                for k in keys[: len(data) - 100]:
                    data.pop(k, None)
            Path(cache_file).write_text(json.dumps(
                data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _build_llm_input_text(
        self,
        text: str,
        table_lines: List[str],
        doc_hint: Dict,
        candidate_fields: Optional[List[Dict]],
        kv_pairs: List[Dict],
    ) -> str:
        """
        组合清洗文本、表格、类型 hint、候选字段与 KV 对，提升 LLM 解析的稳定性。
        """
        parts = []
        parts.append("【清洗文本】\n" + (text or ""))
        if table_lines:
            parts.append("【表格行】\n" + "\n".join(table_lines))
        if doc_hint:
            parts.append("【文档类型提示】\n" +
                         json.dumps(doc_hint, ensure_ascii=False))
        if candidate_fields:
            try:
                cand_compact = [
                    {k: v for k, v in d.items() if k in {"field_name",
                                                         "field_value", "confidence", "field_type"}}
                    for d in candidate_fields
                ]
                parts.append(
                    "【候选字段】\n" + json.dumps(cand_compact, ensure_ascii=False))
            except Exception:
                pass
        if kv_pairs:
            parts.append(
                "【键值对对齐】\n" + json.dumps(kv_pairs, ensure_ascii=False))
        # 指导 LLM 输出统一归一化 JSON
        parts.append("【NORMALIZE_GUIDE】\n" + self.LLM_NORMALIZE_PROMPT.strip())
        return "\n\n".join(parts)

    def _validate_and_fix_fields(self, details: List[Dict]) -> List[Dict]:
        """
        规则校验/修正：身份证/日期/手机号/学号等规范化，生日与身份证一致性检查。
        """
        header_blacklist = {
            "姓名", "性别", "班级", "班主任", "联系电话", "联系方式", "学校", "学院",
            "专业", "学号", "学籍号", "证件号", "身份证号", "出生日期", "地址",
            "年级", "班级姓名", "姓名性别", "姓名性别班级", "姓名性别班级班主任",
        }
        header_blacklist_compact = {
            re.sub(r"\s+", "", h) for h in header_blacklist}

        id_number = None
        birth_from_id = None
        for d in details:
            if d.get("field_type") == "ID_NUMBER" and d.get("field_value"):
                id_number = d.get("field_value")
                if len(id_number) == 18 and id_number[:14].isdigit():
                    birth_from_id = f"{id_number[6:10]}-{id_number[10:12]}-{id_number[12:14]}"
        fixed = []
        for d in details:
            ftype = d.get("field_type")
            val = d.get("field_value")
            conf = float(d.get("confidence") or 0.0)

            # 表头/列标题过滤：若值疑似表头则置空
            if isinstance(val, str):
                norm_val = val.strip()
                compact = re.sub(r"\s+", "", norm_val)
                if compact in header_blacklist_compact:
                    val = None
                    conf = 0.0

            if ftype in ("ID_NUMBER", "PHONE", "STUDENT_ID") and val:
                val = self._normalize_identity(ftype, val)
            if ftype == "DATE" and val:
                val = self._normalize_date(val) or val
            if ftype == "DATE" and val is None and birth_from_id:
                val = birth_from_id
                conf = min(conf, 0.6) or 0.6
            fixed.append(
                {
                    "field_name": d.get("field_name"),
                    "field_type": ftype,
                    "field_value": val if val not in ("", " ") else None,
                    "confidence": conf,
                }
            )
        return fixed

    @staticmethod
    def _format_tables(parsed_tables: Optional[List], transcript_rows: Optional[List[Dict]], doc_type: str) -> List[Dict]:
        if doc_type not in ("成绩单", "课程表", "班级成员表"):
            return []
        tables: List[Dict] = []
        if parsed_tables:
            for idx, tbl in enumerate(parsed_tables):
                rows_out = []
                for row in tbl:
                    cells = row.get("cells") or []
                    if doc_type == "班级成员表" and len(cells) >= 4:
                        row_dict = {
                            "number": cells[0],
                            "class": cells[1],
                            "name": cells[2],
                            "gender": cells[3],
                        }
                    else:
                        row_dict = {
                            f"col{c+1}": cells[c] if c < len(cells) else None for c in range(len(cells))}
                    rows_out.append(row_dict)
                tables.append(
                    {"name": f"auto_table_{idx+1}", "rows": rows_out})
        if doc_type in ("成绩单", "课程表") and transcript_rows:
            rows_out = []
            for r in transcript_rows:
                rows_out.append(
                    {
                        "semester": r.get("semester"),
                        "course_name": r.get("course_name"),
                        "credit": float(r["credit"])
                        if r.get("credit") and str(r.get("credit")).replace(".", "", 1).isdigit()
                        else None,
                        "score": float(r["score"])
                        if r.get("score") and str(r.get("score")).replace(".", "", 1).isdigit()
                        else None,
                        "grade": r.get("grade"),
                    }
                )
            tables.append({"name": "transcript", "rows": rows_out})
        return tables

    @staticmethod
    def _flatten_fields(details: List[Dict]) -> Dict[str, Optional[str]]:
        """
        将 extract_details 映射为常见字段键值，缺失填 None。
        """
        flat_keys = {
            "name",
            "gender",
            "birth_date",
            "id_number",
            "student_id",
            "ticket_no",
            "class",
            "major",
            "college",
            "department",
            "school_name",
            "address",
            "phone",
            "contact",
            "issuer",
            "admission_school",
            "date",
            "reason",
            "certificate_id",
            "verify_code",
            "verify_url",
            "enroll_date",
            "expected_grad_date",
            "status",
            "degree_level",
            "education_type",
            "study_mode",
            "loan_amount",
            "bank_name",
            "account_number",
            "loan_years",
            "contract_id",
            "scholarship_name",
            "scholarship_amount",
            "start_date",
            "end_date",
            "days",
            "approve_status",
            "issue_date",
            "total_score",
            "listening_score",
            "reading_score",
            "comprehensive_score",
            "writing_translation_score",
            "reason",
        }
        flat = {k: None for k in flat_keys}
        for d in details:
            k = d.get("field_name")
            if k in flat:
                v = d.get("field_value")
                # Keep the latest non-empty value; do not override with None/empty noise.
                if v is None:
                    continue
                sv = str(v).strip() if isinstance(v, str) else v
                if sv == "":
                    continue
                flat[k] = v
        # 补充可能遗漏的贷款/奖助学金字段名称别名
        alias_map = {
            "loan_amount": ["loan_amount", "贷款金额", "借款金额"],
            "bank_name": ["bank_name", "loan_bank_name", "贷款银行"],
            "contract_id": ["contract_id", "loan_contract_no", "合同编号"],
            "loan_years": ["loan_years", "loan_term_month"],
            "scholarship_name": ["scholarship_name", "奖学金"],
            "scholarship_amount": ["scholarship_amount", "资助金额"],
            "certificate_id": ["certificate_id", "number", "certificate_number", "证书编号", "成绩单编号"],
            "issuer": ["issuer", "directed_by", "issued_by", "signing_authority", "签发单位", "委托发布单位"],
            "issue_date": ["issue_date", "date", "exam_date", "考试时间"],
            "ticket_no": ["ticket_no", "exam_id", "准考证号", "准考证"],
            "total_score": ["total_score", "score", "总分"],
            "listening_score": ["listening_score", "listening", "听力"],
            "reading_score": ["reading_score", "reading", "阅读"],
            "comprehensive_score": ["comprehensive_score", "comprehensive", "综合"],
            "writing_translation_score": ["writing_translation_score", "writing", "写作和翻译"],
        }
        for target, aliases in alias_map.items():
            if flat.get(target):
                continue
            for alias in aliases:
                val = next((d.get("field_value")
                           for d in details if d.get("field_name") == alias), None)
                if val:
                    flat[target] = val
                    break
        return flat

    def _map_contract_loan_fields(
        self,
        entities: List[Dict],
        key_values: List[Dict],
        text: str,
        flat_fields: Dict[str, Optional[str]],
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """
        针对 doc_type=合同 且包含助学贷款信息的专项映射。
        """
        id_pattern = re.compile(r"\b\d{17}[\dXx]\b")
        amount_pattern = re.compile(
            r"(\d+(?:\.\d+)?)(?=\s*(?:元|（?元）?|\\(元\\)?|\\(元\\)))")
        year_pattern = re.compile(r"(\d+(?:\.\d+)?)[\s]*年")
        month_pattern = re.compile(r"(\d+(?:\.\d+)?)[\s]*月")
        digit_pattern = re.compile(r"\b\d{8,20}\b")
        invalid_labels = {"开户行", "学生姓名", "名称", "账户名称", "开户银行", "专业名称", "学院名称"}

        def clean_label_value(val: Optional[str]) -> Optional[str]:
            if not val:
                return None
            stripped = str(val).strip()
            if self._is_label_token(stripped):
                return None
            return stripped

        def pick_entity(predicate):
            filtered = [e for e in entities if predicate(e)]
            if not filtered:
                return None
            # 优先置信度高、文本更长的
            filtered.sort(
                key=lambda e: (float(e.get("confidence") or 0.0),
                               len(str(e.get("text") or ""))),
                reverse=True,
            )
            return filtered[0]

        # name
        name_entity = pick_entity(
            lambda e: e.get("type") == "PERSON"
            and 2 <= len(str(e.get("text") or "").strip()) <= 4
            and all(lbl not in str(e.get("text") or "") for lbl in invalid_labels)
            and "贷款信息" not in str(e.get("text") or "")
        )
        name = clean_label_value(name_entity.get(
            "text") if name_entity else None)

        # id_number
        id_number = None
        for e in entities:
            if e.get("type") in ("ID_NUMBER", "VERIFY_CODE"):
                m = id_pattern.search(str(e.get("text") or ""))
                if m:
                    id_number = m.group(0)
                    break

        # school_name
        def org_candidates():
            for e in entities:
                if e.get("type") in ("ORG", "COLLEGE_NAME"):
                    txt = str(e.get("text") or "")
                    if any(k in txt for k in ("学院", "大学", "学校")):
                        yield txt.strip()

        school_name = None
        org_list = list(org_candidates())
        if org_list:
            org_list.sort(key=len, reverse=True)
            school_name = org_list[0]

        # college from kv or entities
        college = flat_fields.get("college") if flat_fields else None
        if college in invalid_labels:
            college = None
        if not college:
            for kv in key_values or []:
                key = str(kv.get("key") or "")
                if "学院" in key:
                    val = clean_label_value(kv.get("value"))
                    if val:
                        college = val
                        break
        if not college:
            for e in entities:
                if e.get("type") in ("ORG", "COLLEGE_NAME"):
                    txt = str(e.get("text") or "")
                    if "学院" in txt:
                        college = clean_label_value(txt)
                        if college:
                            break

        # major
        major = flat_fields.get("major") if flat_fields else None
        if major in invalid_labels:
            major = None
        if not major:
            for kv in key_values or []:
                key = str(kv.get("key") or "")
                if "专业" in key:
                    val = clean_label_value(kv.get("value"))
                    if val:
                        # 去掉“名称”等标签
                        major = re.sub(r"(名称|专业名称)", "", val).strip() or val
                        break

        # address
        address_candidates = [e for e in entities if e.get(
            "type") == "ADDRESS_FULL" and e.get("text")]
        address = None
        if address_candidates:
            address_candidates.sort(key=lambda e: len(
                str(e.get("text") or "")), reverse=True)
            address = clean_label_value(address_candidates[0].get("text"))

        # bank name
        bank_entity = pick_entity(lambda e: e.get(
            "type") == "LOAN_BANK_NAME" and e.get("text"))
        bank_name = clean_label_value(
            bank_entity.get("text") if bank_entity else None)

        # account_number
        account_number = None
        for e in entities:
            val = str(e.get("text") or "")
            if e.get("type") in ("ACCOUNT_NUMBER", "VERIFY_CODE", "NUMBER", "CARD_ID"):
                if id_number and id_number == val:
                    continue
                if digit_pattern.fullmatch(val) and not id_pattern.fullmatch(val):
                    account_number = val
                    break
        if not account_number:
            for kv in key_values or []:
                val = str(kv.get("value") or "")
                if digit_pattern.fullmatch(val) and not id_pattern.fullmatch(val):
                    account_number = val
                    break

        # loan amount
        loan_amount = None
        amt_match = amount_pattern.search(text or "")
        if amt_match:
            try:
                loan_amount = float(amt_match.group(1))
            except Exception:
                loan_amount = None

        # loan years
        loan_years = None
        ym_text = text or ""
        ym_match = year_pattern.search(ym_text)
        if ym_match:
            loan_years = float(ym_match.group(1))
        else:
            m_match = month_pattern.search(ym_text)
            if m_match:
                try:
                    months = float(m_match.group(1))
                    loan_years = round(months / 12.0, 2)
                except Exception:
                    loan_years = m_match.group(1)

        # certificate / verify_code
        verify_code = None
        vc_candidates = []
        for e in entities:
            if e.get("type") == "VERIFY_CODE":
                txt = str(e.get("text") or "").strip()
                if digit_pattern.fullmatch(txt):
                    vc_candidates.append(
                        (len(txt), float(e.get("confidence") or 0.0), txt))
        if vc_candidates:
            vc_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            verify_code = vc_candidates[0][2]
        if not verify_code:
            for kv in key_values or []:
                val = str(kv.get("value") or "").strip()
                if digit_pattern.fullmatch(val) and not id_pattern.fullmatch(val):
                    verify_code = val
                    break

        basic_info = {
            "name": name,
            "id_number": id_number,
            "school_name": school_name,
            "college": college,
            "major": major,
            "address": address,
        }
        financial_info = {
            "bank_name": bank_name,
            "account_number": account_number,
            "loan_amount": loan_amount,
            "loan_years": loan_years,
        }
        certificate_info = {"certificate_id": verify_code,
                            "verify_code": verify_code}
        return {
            "basic_info": basic_info,
            "financial_info": financial_info,
            "certificate_info": certificate_info,
        }

    # ---------------------------------------------------------------------
    # Helpers used across the pipeline (must exist to avoid runtime 500)
    # ---------------------------------------------------------------------

    @staticmethod
    def _clean_text_for_output(text: str) -> str:
        """输出前轻量清洗，避免返回过多噪声。"""
        if not text:
            return ""
        s = str(text)
        for t in ("糖点", "硬是炎到", "心糖译"):
            s = s.replace(t, "")
        s = s.replace("\u3000", " ")
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    @staticmethod
    def _clean_entities(entities: List[Dict]) -> List[Dict]:
        """清洗实体，剔除明显表头/噪声，保证输出结构统一。"""
        if not entities:
            return []
        person_blacklist = {
            "姓名", "性别", "学号", "专业", "班级", "学院", "学校",
            "学习形式", "全日制", "本科", "必修", "选修", "课程", "学分", "成绩", "绩点",
        }
        cleaned: List[Dict] = []
        for e in entities:
            if not isinstance(e, dict):
                continue
            txt = str(e.get("text") or "").strip()
            if not txt:
                continue
            etype = e.get("type")
            conf = float(e.get("confidence") or 0.0)
            if etype == "PERSON":
                if txt in person_blacklist:
                    continue
                if any(ch.isdigit() for ch in txt):
                    continue
                if len(txt) < 2 or len(txt) > 10:
                    continue
            cleaned.append(
                {
                    "type": etype,
                    "text": txt,
                    "start": int(e.get("start") or 0),
                    "end": int(e.get("end") or 0),
                    "confidence": conf,
                }
            )
        return cleaned

    def _normalize_date(self, val: str) -> Optional[str]:
        """将常见中文日期/短日期格式归一化为 YYYY-MM-DD。"""
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        m = re.search(r"(\d{4})\s*[.\-/年]\s*(\d{1,2})\s*[.\-/月]\s*(\d{1,2})?", s)
        if m:
            try:
                y = int(m.group(1))
                mo = int(m.group(2))
                d = int(m.group(3)) if m.group(3) else 1
                if 1 <= mo <= 12 and 1 <= d <= 31:
                    return f"{y:04d}-{mo:02d}-{d:02d}"
            except Exception:
                return None
        m2 = re.search(r"(\d{4})\s*[.\-/年]\s*(\d{1,2})\s*月?", s)
        if m2:
            try:
                y = int(m2.group(1))
                mo = int(m2.group(2))
                if 1 <= mo <= 12:
                    return f"{y:04d}-{mo:02d}-01"
            except Exception:
                return None
        return None

    def _normalize_identity(self, ftype: str, val: str) -> Optional[str]:
        """标准化证件号/手机号/学号。"""
        if val is None:
            return None
        raw = str(val).strip()
        if not raw:
            return None
        digits = re.sub(r"\D", "", raw)

        if ftype == "ID_NUMBER":
            if re.fullmatch(r"\d{17}[\dXx]", raw):
                return raw.upper()
            if re.fullmatch(r"\d{18}", digits):
                return digits
            return None

        if ftype == "PHONE":
            return digits if len(digits) == 11 else None

        if ftype == "STUDENT_ID":
            return digits if 6 <= len(digits) <= 20 else None

        return raw

    def _normalize_fields(self, details: List[Dict], doc_type: str) -> List[Dict]:
        """归一化 extract_details，补齐 status 并对关键字段做格式修正。"""
        normalized: List[Dict] = []
        allow_course = doc_type in ("成绩单", "课程表", "课表/选课单", "成绩单/学业成绩表")

        for d in details or []:
            if not isinstance(d, dict):
                continue
            name = d.get("field_name")
            ftype = d.get("field_type") or d.get("type")
            val = d.get("field_value")
            conf = float(d.get("confidence") or 0.0)

            if ftype in ("COURSE", "CREDIT", "SCORE", "GPA") and not allow_course:
                val = None

            if ftype == "DATE" and val:
                val = self._normalize_date(val) or val

            if ftype in ("ID_NUMBER", "PHONE", "STUDENT_ID") and val:
                val = self._normalize_identity(ftype, val)

            val = val if val not in ("", " ") else None
            status = "ok" if val is not None else "missing"
            if status == "missing":
                conf = 0.0

            normalized.append(
                {
                    "field_name": name,
                    "field_type": ftype,
                    "field_value": val,
                    "confidence": conf,
                    "status": status,
                }
            )

        return normalized

    @staticmethod
    def _clean_llm_courses(courses: List[Dict]) -> List[Dict]:
        """清洗 LLM courses，去掉空/纯数字课程名，并把数值字段转 float。"""
        if not courses:
            return []

        def num(v):
            try:
                if v is None or v == "":
                    return None
                return float(str(v).replace(",", ""))
            except Exception:
                return None

        cleaned: List[Dict] = []
        for c in courses:
            if not isinstance(c, dict):
                continue
            name = str(c.get("course_name") or "").strip()
            if not name or name.isdigit() or len(name) < 2:
                continue
            cleaned.append(
                {
                    "academic_year": c.get("academic_year"),
                    "semester": c.get("semester"),
                    "course_code": c.get("course_code"),
                    "course_seq": c.get("course_seq"),
                    "course_name": name,
                    "course_type": c.get("course_type"),
                    "credit": num(c.get("credit")),
                    "score": num(c.get("score")),
                    "final_score": num(c.get("final_score")),
                    "gpa": num(c.get("gpa")),
                    "remark": c.get("remark"),
                }
            )
        return cleaned

    @staticmethod
    def _is_noisy_courses(courses: List[Dict]) -> bool:
        """判断课程列表是否明显噪声（例如大量空名/纯数字名/过短名）。"""
        if not courses:
            return True
        bad = 0
        for c in courses:
            if not isinstance(c, dict):
                bad += 1
                continue
            name = str(c.get("course_name") or "").strip()
            if not name or name.isdigit() or len(name) <= 2:
                bad += 1
                continue
            digit_ratio = sum(ch.isdigit() for ch in name) / max(1, len(name))
            if digit_ratio > 0.5:
                bad += 1
        return bad >= max(1, len(courses) // 2)

    def _clean_rule_courses(self, courses: List[Dict]) -> List[Dict]:
        return self._clean_llm_courses(courses)

    @staticmethod
    def _courses_from_transcript(transcript_rows: List[Dict]) -> List[Dict]:
        if not transcript_rows:
            return []
        out: List[Dict] = []
        for row in transcript_rows:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "academic_year": row.get("academic_year"),
                    "semester": row.get("semester"),
                    "course_code": row.get("course_code"),
                    "course_seq": row.get("course_seq"),
                    "course_name": row.get("course_name"),
                    "course_type": row.get("category") or row.get("course_type"),
                    "credit": row.get("credit"),
                    "score": row.get("score") or row.get("final_score") or row.get("score_total"),
                    "final_score": row.get("final_score"),
                    "gpa": row.get("gpa"),
                    "remark": row.get("remark"),
                }
            )
        return out

    def _llm_courses_fallback(self, text: str, ocr_analysis: Optional[Dict]) -> List[Dict]:
        if not getattr(self.llm_service, "enabled", False):
            return []
        try:
            llm_result = self.llm_service.extract_with_llm(text, {"fields": []})
            courses = (llm_result or {}).get("courses")
            return self._clean_llm_courses(courses or [])
        except Exception:
            return []

    def _post_process_contract(self, data: Dict, doc_type_candidates: Dict) -> Dict:
        """
        合同/贷款合同专用后处理：修正 basic_info / certificate_info / financial_info / summary，避免标签词和错位值。
        """
        entities = data.get("debug", {}).get("entities") or []
        kv_pairs = data.get("debug", {}).get("key_value_pairs") or []
        text = data.get("text") or ""

        # 贷款类判断：doc_type 优先，其次看候选概率（避免 doc_type=合同 时漏掉贷款合同）
        doc_type = (data.get("document_type") or data.get("basic_info", {}).get("document_type") or data.get("classification", {}).get("document_type") or "").strip() or data.get("document_type")
        is_loan_doc = (doc_type in ("生源地助学贷款合同", "贷款合同")) or (
            float((doc_type_candidates or {}).get("生源地助学贷款", 0) or 0) > 0
            or float((doc_type_candidates or {}).get("贷款合同", 0) or 0) > float((doc_type_candidates or {}).get("合同", 0) or 0)
        )

        def sanitize(val: Optional[str]) -> Optional[str]:
            if val is None:
                return None
            if self._is_label_token(val):
                return None
            return str(val).strip()

        id_re = re.compile(r"\b\d{17}[\dXx]\b")
        digit_re = re.compile(r"\b\d{8,20}\b")
        amount_re = re.compile(r"(\d+(?:\.\d+)?)(?=\s*（?元）?|\s*元|\s*\(元\)?)")
        year_re = re.compile(r"(\d+(?:\.\d+)?)[\s]*年")
        month_re = re.compile(r"(\d+)[\s]*（?月）?")
        address_keywords = ("省", "市", "县", "区", "镇", "乡", "村")

        # doc_subtype 提示（保留，仅作为 debug）
        loan_contract_prob = doc_type_candidates.get("贷款合同", 0)
        if loan_contract_prob >= max(doc_type_candidates.get("合同", 0), doc_type_candidates.get("在校证明", 0)):
            data.setdefault("debug", {})["doc_subtype"] = "贷款合同"

        # name
        name = data.get("basic_info", {}).get("name")
        if name in {"姓名", "开户行", "账户名称", "名称"} or self._is_label_token(name or ""):
            name = None
        if not name:
            for ent in entities:
                if ent.get("type") in ("PERSON", "COLLEGE_NAME"):
                    txt = str(ent.get("text") or "").strip()
                    if any(bad in txt for bad in ("贷款信息", "开户行", "学生姓名")):
                        continue
                    # 截断“的”左侧
                    if "的" in txt:
                        txt = txt.split("的", 1)[0]
                    if 2 <= len(txt) <= 4 and all("\u4e00" <= ch <= "\u9fff" for ch in txt) and not any(
                        kw in txt for kw in ("学院", "大学")
                    ):
                        name = txt
                        break
        data.setdefault("basic_info", {})["name"] = name

        # id_number
        id_number = data.get("basic_info", {}).get("id_number")
        if self._is_label_token(id_number or ""):
            id_number = None
        if not id_number:
            for ent in entities:
                if ent.get("type") in ("ID_NUMBER", "VERIFY_CODE"):
                    m = id_re.search(str(ent.get("text") or ""))
                    if m:
                        id_number = m.group(0)
                        break
        data["basic_info"]["id_number"] = id_number

        # school_name
        school = data.get("basic_info", {}).get("school_name")
        if self._is_label_token(school or ""):
            school = None
        ff_school = (data.get("debug", {}).get(
            "flat_fields") or {}).get("school_name")
        if ff_school:
            school = ff_school
        if not school:
            orgs = []
            for ent in entities:
                if ent.get("type") in ("ORG", "COLLEGE_NAME"):
                    txt = str(ent.get("text") or "")
                    if any(k in txt for k in ("学院", "大学", "学校")) and not self._is_label_token(txt):
                        orgs.append(txt.strip())
            if orgs:
                orgs.sort(key=len, reverse=True)
                school = orgs[0]
        data["basic_info"]["school_name"] = school

        # college from kv key contains 学院
        college = data.get("basic_info", {}).get("college")
        if self._is_label_token(college or ""):
            college = None
        if college and "账户名称 院系名称 就读高校" in college:
            college = None
        if not college:
            for kv in kv_pairs:
                if "学院" in str(kv.get("key") or ""):
                    val = str(kv.get("value") or "")
                    m = re.search(
                        r"[\u4e00-\u9fffA-Za-z0-9（）()·]{2,40}学院", val)
                    if m and not self._is_label_token(m.group(0)):
                        college = m.group(0)
                        break
        if not college:
            for ent in entities:
                if ent.get("type") == "ORG":
                    txt = str(ent.get("text") or "")
                    if "学院" in txt and txt != school and not self._is_label_token(txt):
                        college = txt.strip()
                        break
        data["basic_info"]["college"] = college

        # major from kv key contains 专业
        major = data.get("basic_info", {}).get("major")
        if self._is_label_token(major or ""):
            major = None
        if not major:
            for kv in kv_pairs:
                if "专业" in str(kv.get("key") or ""):
                    val = str(kv.get("value") or "")
                    val = re.sub(r"(名称|专业名称)", "", val).strip()
                    if val and not self._is_label_token(val):
                        major = val
                        break
        data["basic_info"]["major"] = major

        # address from ADDRESS_FULL
        address = data.get("basic_info", {}).get("address")
        if address:
            address = address.splitlines()[0].strip()
            if self._is_label_token(address):
                address = None
        if not address:
            addr_candidates = []
            for ent in entities:
                if ent.get("type") == "ADDRESS_FULL":
                    txt = str(ent.get("text") or "").replace("\n", " ").strip()
                    if any(k in txt for k in address_keywords):
                        # 截断到出现“银行”前
                        if "银行" in txt:
                            txt = txt.split("银行")[0]
                        parts = re.split(r"\s+", txt)
                        cleaned = parts[0] if parts else txt
                        addr_candidates.append(cleaned)
            if addr_candidates:
                addr_candidates.sort(key=len, reverse=True)
                address = addr_candidates[0]
        data["basic_info"]["address"] = address

        # certificate_id / verify_code from VERIFY_CODE (16-20 digits, exclude ID)
        cert_val = None
        vc_list = []
        for ent in entities:
            if ent.get("type") == "VERIFY_CODE":
                txt = str(ent.get("text") or "").strip()
                if id_re.fullmatch(txt):
                    continue
                if digit_re.fullmatch(txt):
                    vc_list.append(txt)
        if vc_list:
            vc_list.sort(key=len, reverse=True)
            for v in vc_list:
                if 16 <= len(v) <= 20:
                    cert_val = v
                    break
            if not cert_val:
                cert_val = vc_list[0]
        if cert_val:
            data.setdefault("certificate_info", {})[
                "certificate_id"] = cert_val
            data["certificate_info"]["verify_code"] = cert_val

        # bank_name
        bank_name = data.get("financial_info", {}).get("bank_name")
        if self._is_label_token(bank_name or ""):
            bank_name = None
        if not bank_name:
            for ent in entities:
                if ent.get("type") == "LOAN_BANK_NAME":
                    txt = str(ent.get("text") or "")
                    if "银行" in txt and not self._is_label_token(txt):
                        bank_name = txt.strip()
                        break
        if not bank_name:
            for ent in entities:
                if ent.get("type") == "ADDRESS_FULL":
                    txt = str(ent.get("text") or "")
                    if "银行" in txt:
                        bank_name = txt.strip()
                        break
        data.setdefault("financial_info", {})["bank_name"] = bank_name

        # loan_amount float from text
        loan_amount = data.get("financial_info", {}).get("loan_amount")
        if is_loan_doc:
            if isinstance(loan_amount, str):
                try:
                    loan_amount = float(loan_amount)
                except Exception:
                    loan_amount = None
            if loan_amount is None:
                m = amount_re.search(text)
                if m:
                    try:
                        loan_amount = float(m.group(1))
                    except Exception:
                        loan_amount = None
        else:
            # 非贷款文档：不解析金额，避免乱识别
            loan_amount = None

        # loan_years float
        loan_years = data.get("financial_info", {}).get("loan_years")
        if is_loan_doc:
            if isinstance(loan_years, str):
                try:
                    loan_years = float(loan_years)
                except Exception:
                    loan_years = None
            if loan_years is None:
                m = re.search(
                    r"(?:期限|用款期限|贷款期限)[^\n\r]{0,10}?(\d+(?:\.\d+)?)\s*年", text)
                if m:
                    try:
                        loan_years = float(m.group(1))
                    except Exception:
                        loan_years = None
            if loan_years is None:
                m = year_re.search(text)
                if m:
                    try:
                        loan_years = float(m.group(1))
                    except Exception:
                        loan_years = None
                if loan_years is None:
                    mm = month_re.search(text)
                    if mm:
                        try:
                            loan_years = round(float(mm.group(1)) / 12.0, 2)
                        except Exception:
                            loan_years = None
        else:
            # 非贷款文档：年限一律置空，避免把“2002年出生”当贷款 2002 年
            loan_years = None

        # account_number: 8-20 digits excluding id and certificate
        account_number = None
        exclude_set = set()
        if id_number:
            exclude_set.add(id_number)
        if cert_val:
            exclude_set.add(cert_val)

        def scan_digits(seq):
            candidates = []
            for obj in seq:
                val = None
                if isinstance(obj, dict):
                    val = obj.get("text") or obj.get("value")
                else:
                    val = obj
                val = str(val or "").strip()
                if digit_re.fullmatch(val):
                    if id_re.fullmatch(val):
                        continue
                    if val in exclude_set:
                        continue
                    candidates.append(val)
            return candidates

        digit_candidates = scan_digits(
            entities) + scan_digits([kv.get("value") for kv in kv_pairs])
        if digit_candidates:
            digit_candidates = [
                d for d in digit_candidates if 8 <= len(d) <= 20]
            if digit_candidates:
                # 优先 8-12 长度
                preferred = [d for d in digit_candidates if 8 <= len(d) <= 12]
                chosen_list = preferred or digit_candidates
                chosen_list.sort(key=len)
                account_number = chosen_list[0]
        data["financial_info"]["account_number"] = account_number

        # ---------- 摘要 summary ----------
        if is_loan_doc:
            parts = [f"类型 {doc_type}"]
            if name:
                parts.append(f"借款人 {name}")
            if loan_amount is not None:
                parts.append(f"金额 {loan_amount:.1f} 元")
            if loan_years is not None:
                parts.append(f"期限 {loan_years:.1f} 年")
            if school:
                parts.append(f"学校 {school}")
            summary = "，".join(parts)
        else:
            # 学籍卡 / 成绩单 / 在校证明 / 其他
            parts = [f"类型 {doc_type}"]
            if name:
                parts.append(f"姓名 {name}")
            if school:
                parts.append(f"学校 {school}")
            if college:
                parts.append(f"学院 {college}")
            if major:
                parts.append(f"专业 {major}")
            summary = "，".join(parts)

        # 回填 financial 结果
        data.setdefault("financial_info", {})["loan_amount"] = loan_amount
        data["financial_info"]["loan_years"] = loan_years

        # 回写 summary（上层会用它输出）
        data["summary"] = summary

        # 维持原结构：返回同一个 data（只做修正）
        return data

    @staticmethod
    def _split_text_sections(text: str) -> Dict[str, str]:
        """将全文粗分为正文与表格区。

        说明：这里不追求完美，只为让课程/成绩表解析有一个更高命中率的输入区。
        """
        if not text:
            return {"body_text": "", "table_text": ""}
        lines = [ln.rstrip() for ln in str(text).splitlines()]
        body_lines: List[str] = []
        table_lines: List[str] = []
        for ln in lines:
            s = (ln or "").strip()
            if not s:
                continue
            digit_ratio = sum(ch.isdigit() for ch in s) / max(1, len(s))
            sep_cnt = s.count(" ") + s.count("\t") + s.count("|") + s.count(",")
            # 常见表格行特征：数字多 / 分隔符多 / “学分/成绩/课程”字样
            if digit_ratio >= 0.35 or sep_cnt >= 3 or any(k in s for k in ("学分", "绩点", "成绩", "课程")):
                table_lines.append(s)
            else:
                body_lines.append(s)
        return {"body_text": "\n".join(body_lines), "table_text": "\n".join(table_lines)}

    @staticmethod
    def _safe_number(val):
        """把常见数字字符串安全转为 int/float；失败则返回 None。"""
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        s = s.replace(",", "")
        if re.fullmatch(r"-?\d+", s):
            try:
                return int(s)
            except Exception:
                return None
        if re.fullmatch(r"-?\d+\.\d+", s):
            try:
                return float(s)
            except Exception:
                return None
        return None

    def _extract_courses_table(self, text: str) -> List[Dict]:
        """从 OCR 文本中尽可能恢复课程/成绩表（规则兜底）。

        输出结构尽量贴近 LLM courses：
        - semester / academic_year / course_code / course_name / credit / score / gpa / remark

        支持两类常见输入：
        1) 逐行 token（部分 OCR 会把表格拆成一列一个 token）：
           2024-2025-1\n 123456\n 01\n 高等数学\n 必修\n 4\n 95\n 95\n 4.0 ...
        2) 单行多列（空格/制表符/竖线分隔）：
           2024-2025-1 123456 01 高等数学 必修 4 95 4.0
        """
        if not text:
            return []

        raw = str(text)
        sections = self._split_text_sections(raw)
        table_text = sections.get("table_text") or raw

        # 先按行解析（多列行）
        lines = [ln.strip() for ln in table_text.splitlines() if ln.strip()]
        courses: List[Dict] = []

        def parse_semester_token(token: str):
            t = (token or "").strip()
            # 2024-2025 1 / 2024-2025-1 / 2024/2025 2
            m = re.search(r"(20\d{2})\s*[-/]\s*(20\d{2})\s*[-]?\s*([12一二])", t)
            if not m:
                return None
            ay = f"{m.group(1)}-{m.group(2)}"
            sem = m.group(3)
            sem = "1" if sem in ("1", "一") else "2"
            return ay, sem

        def num(v):
            n = self._safe_number(v)
            if n is None:
                # 处理如 "95.0分"/"4.0绩点"
                m = re.search(r"-?\d+(?:\.\d+)?", str(v))
                if m:
                    return self._safe_number(m.group(0))
            return n

        # A) 行内多列
        for ln in lines:
            # 统一分隔符
            ln2 = re.sub(r"[|]+", " ", ln)
            ln2 = re.sub(r"\s{2,}", " ", ln2).strip()
            parts = ln2.split(" ")
            if len(parts) < 6:
                continue

            sem_info = parse_semester_token(parts[0])
            if not sem_info:
                continue

            academic_year, sem = sem_info
            semester = f"{academic_year}-{sem}"

            # 尝试定位列：code/seq/name/credit/score/gpa
            course_code = parts[1]
            course_seq = parts[2]

            # 找到第一个“看起来像学分”的位置（通常 0-10）
            credit_idx = None
            for idx in range(3, min(len(parts), 10)):
                c = num(parts[idx])
                if c is not None and 0 < float(c) <= 20:
                    credit_idx = idx
                    break
            if credit_idx is None or credit_idx <= 3:
                continue

            course_name = " ".join(parts[3:credit_idx]).strip()
            if not course_name or course_name.isdigit():
                continue

            credit = num(parts[credit_idx])
            score = num(parts[credit_idx + 1]) if credit_idx + 1 < len(parts) else None
            gpa = None
            for idx in range(credit_idx + 2, min(len(parts), credit_idx + 6)):
                maybe = num(parts[idx])
                if maybe is None:
                    continue
                if 0 <= float(maybe) <= 5:
                    gpa = float(maybe)
                    break

            courses.append(
                {
                    "academic_year": academic_year,
                    "semester": semester,
                    "course_code": course_code,
                    "course_seq": course_seq,
                    "course_name": course_name,
                    "course_type": None,
                    "credit": float(credit) if isinstance(credit, (int, float)) else None,
                    "score": float(score) if isinstance(score, (int, float)) else None,
                    "final_score": None,
                    "gpa": gpa,
                    "remark": None,
                }
            )

        if courses:
            return courses

        # B) token 流（逐行拆散）
        tokens = []
        for ln in lines:
            # 仍然可能是一行多个 token（用空格/\t）
            parts = re.split(r"\s+", ln)
            tokens.extend([p for p in parts if p])

        def is_course_name(tok: str) -> bool:
            t = (tok or "").strip()
            if not t or t.isdigit():
                return False
            # 至少包含一个中文或字母
            return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", t)) and len(t) >= 2

        i = 0
        while i < len(tokens):
            sem_info = parse_semester_token(tokens[i])
            if not sem_info:
                i += 1
                continue
            academic_year, sem = sem_info
            semester = f"{academic_year}-{sem}"
            if i + 5 >= len(tokens):
                break

            course_code = tokens[i + 1]
            course_seq = tokens[i + 2]
            course_name = tokens[i + 3]

            # 课程名可能被拆成多 token：向后合并直到遇到学分
            j = i + 3
            name_parts = []
            while j < len(tokens):
                if num(tokens[j]) is not None and 0 < float(num(tokens[j])) <= 20:
                    break
                if is_course_name(tokens[j]):
                    name_parts.append(tokens[j])
                j += 1
            if not name_parts:
                i += 1
                continue
            course_name = "".join(name_parts)

            if j >= len(tokens):
                break
            credit = num(tokens[j])
            score = num(tokens[j + 1]) if j + 1 < len(tokens) else None

            gpa = None
            for k in range(j + 2, min(len(tokens), j + 6)):
                maybe = num(tokens[k])
                if maybe is None:
                    continue
                if 0 <= float(maybe) <= 5:
                    gpa = float(maybe)
                    break

            courses.append(
                {
                    "academic_year": academic_year,
                    "semester": semester,
                    "course_code": course_code,
                    "course_seq": course_seq,
                    "course_name": course_name,
                    "course_type": None,
                    "credit": float(credit) if isinstance(credit, (int, float)) else None,
                    "score": float(score) if isinstance(score, (int, float)) else None,
                    "final_score": None,
                    "gpa": gpa,
                    "remark": None,
                }
            )

            # 步进：至少跳过到 score 后
            i = j + 2

        return courses

    @classmethod
    def _is_label_token(cls, val: str) -> bool:
        """判断一个值是否是表头/标签词（避免把“姓名/专业/学院”等当成真实字段值）。"""
        if val is None:
            return True
        s = str(val).strip()
        if not s:
            return True
        if s in cls.LABEL_TOKENS:
            return True
        # 一些常见的“冒号+标签”残留
        s2 = re.sub(r"[：:]$", "", s).strip()
        if s2 in cls.LABEL_TOKENS:
            return True
        # 仅当短中文明显是字段名时才判定为标签，避免把“周宇清”这类姓名误判为标签。
        if 2 <= len(s2) <= 4 and all("\u4e00" <= ch <= "\u9fff" for ch in s2):
            if any(k in s2 for k in ("姓名", "性别", "学号", "班级", "专业", "学院", "学校", "地址", "电话", "日期", "时间", "成绩", "备注")):
                return True
        return False

    def _build_normalized_blocks(
        self,
        doc_type: str,
        doc_type_candidates: Dict,
        confidence: float,
        flat_fields: Dict[str, Optional[str]],
        entities: List[Dict],
        key_values: List[Dict],
        tables: List[Dict],
        cleaned_text: str,
        summary: str,
        meta_block: Dict,
        fields_block: Dict,
        courses: List[Dict],
        ocr_analysis: Optional[Dict],
        llm_status: Dict,
        trace_id: str,
        analysis_notes: List[str],
        contract_info: Optional[Dict],
        fail_reason: Optional[str],
    ) -> Dict:
        """将抽取结果整理为统一输出结构。

        该函数是 _finalize_response 的核心组装器，必须稳定输出：
        - document_type / candidates / confidence
        - basic_info / academic_info / leave_info / certificate_info / financial_info
        - courses / tables / fields / text / summary / meta
        """
        invalid_labels = self.LABEL_TOKENS
        tables = tables or []
        courses = courses or []

        def sanitize(v: Optional[str]) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            if self._is_label_token(s):
                return None
            # 把“标签:值”形式中标签去掉
            if "：" in s and len(s) <= 30:
                left, right = s.split("：", 1)
                if left.strip() in invalid_labels and right.strip():
                    s = right.strip()
            if ":" in s and len(s) <= 30:
                left, right = s.split(":", 1)
                if left.strip() in invalid_labels and right.strip():
                    s = right.strip()
            return s

        def block(keys: List[str]) -> Dict[str, Optional[str]]:
            return {k: sanitize(flat_fields.get(k)) for k in keys}

        basic_info = block(
            [
                "name",
                "gender",
                "birth_date",
                "id_number",
                "student_id",
                "ticket_no",
                "class",
                "major",
                "college",
                "department",
                "school_name",
                "address",
                "phone",
                "contact",
            ]
        )
        academic_info = block(
            [
                "degree_level",
                "education_type",
                "enroll_date",
                "expected_grad_date",
                "status",
                "study_mode",
            ]
        )
        certificate_info = {
            "certificate_id": sanitize(flat_fields.get("certificate_id")),
            "issue_date": sanitize(flat_fields.get("issue_date")),
            "issuer": sanitize(flat_fields.get("issuer")),
            "verify_code": sanitize(flat_fields.get("verify_code")),
            "verify_url": sanitize(flat_fields.get("verify_url")),
        }
        leave_info = {
            "reason": sanitize(flat_fields.get("reason")),
            "start_date": sanitize(flat_fields.get("start_date")),
            "end_date": sanitize(flat_fields.get("end_date")),
            "days": sanitize(flat_fields.get("days")),
            "issuer": sanitize(flat_fields.get("issuer")),
            "approve_status": sanitize(flat_fields.get("approve_status")),
        }
        financial_info = {
            "loan_amount": self._safe_number(flat_fields.get("loan_amount")) if flat_fields.get("loan_amount") is not None else None,
            "loan_years": self._safe_number(flat_fields.get("loan_years")) if flat_fields.get("loan_years") is not None else None,
            "bank_name": sanitize(flat_fields.get("bank_name")),
            "account_number": sanitize(flat_fields.get("account_number")),
            "scholarship_name": sanitize(flat_fields.get("scholarship_name")),
            "scholarship_amount": self._safe_number(flat_fields.get("scholarship_amount")) if flat_fields.get("scholarship_amount") is not None else None,
        }

        # 合同专项覆盖（如有）
        if contract_info:
            for k, v in (contract_info.get("basic_info") or {}).items():
                if v:
                    basic_info[k] = sanitize(v)
            for k, v in (contract_info.get("certificate_info") or {}).items():
                if v:
                    certificate_info[k] = sanitize(v)
            for k, v in (contract_info.get("financial_info") or {}).items():
                if v is not None:
                    if k in ("loan_amount", "scholarship_amount", "loan_years"):
                        financial_info[k] = v
                    else:
                        financial_info[k] = sanitize(v)

        # summary 兜底增强
        summary_final = summary
        if fail_reason and (not summary_final or summary_final.startswith("类型")):
            summary_final = (summary_final or "") + f"，原因：{fail_reason}"

        debug_block = {
            "flat_fields": flat_fields,
            "entities": entities,
            "key_value_pairs": key_values,
            "tables_raw": (ocr_analysis.get("reconstructed", {}).get("parsed_tables") if ocr_analysis else None) or tables,
            "ocr_analysis": ocr_analysis,
            "llm_status": llm_status,
            "trace_id": trace_id,
            "analysis_notes": analysis_notes,
        }

        return {
            "document_type": doc_type,
            "document_type_candidates": doc_type_candidates,
            "confidence_overall": confidence,
            "basic_info": basic_info,
            "academic_info": academic_info,
            "certificate_info": certificate_info,
            "leave_info": leave_info,
            "financial_info": financial_info,
            "courses": courses,
            "tables": tables,
            "fields": fields_block,
            "text": cleaned_text,
            "summary": summary_final,
            "meta": meta_block,
            "debug": debug_block,
        }
