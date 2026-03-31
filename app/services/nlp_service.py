"""
Rule-based NLP service for field extraction, entity recognition, and relation extraction.
包含大量日志与注释，便于排查规则覆盖不到时的行为。
"""
import re
import time
from typing import Dict, List, Optional, Any
from loguru import logger

from .nlp_utils import (
    normalize_date,
    normalize_phone,
    normalize_id_number,
    normalize_student_id,
    parse_money_amount,
    months_to_years,
    extract_date_range,
)


class NLPService:
    """NLP Service for information extraction from text"""

    def __init__(self):
        # 简单实体缓存：同一份 text 多次调用 extract_entities 时避免重复正则扫描
        self._entity_cache_text: Optional[str] = None
        self._entity_cache_result: List[Dict] = []
        logger.info("NLP Service initialized")

    # ===== 通用标签/表头过滤（避免“名称/学院/专业/账号”等被当成值） =====
    LABEL_TOKENS = {
        "名称", "姓名", "性别", "学号", "班级", "学院", "院系", "系所", "专业", "专业名称",
        "学校", "学校名称", "就读高校", "院系名称",
        "身份证", "身份证号", "身份证号码",
        "账号", "账户", "账户名称", "开户行", "开户银行", "银行", "卡号",
        "贷款信息", "贷款合同编号", "合同编号", "用款期限", "本次用款金额", "回执验证码", "验证码",
        "姓名及手机", "姓名及手机：", "家长姓名及手机", "本人手机号", "微信号", "QQ号", "E-mail",
    }

    @classmethod
    def _is_label_like(cls, val: Optional[str]) -> bool:
        if not val:
            return False
        s = str(val).strip().replace("：", "").replace(":", "")
        if not s:
            return False
        if s in cls.LABEL_TOKENS:
            return True
        # 短表头：无数字且包含关键字
        if len(s) <= 6 and not any(ch.isdigit() for ch in s) and any(k in s for k in ("姓名", "学院", "专业", "班级", "账号", "手机", "联系方式")):
            return True
        return False

    # ---------------------
    # OCR 文本综合分析（类型判定、质量、字段推断与模板建议）
    # ---------------------
    def analyze_ocr_text(self, text: str, ocr_raw: Optional[Dict] = None) -> Dict:
        """
        输入一段 OCR 文本（可能包含噪声/表格乱码），输出：
        - 文档类型判定
        - OCR 质量与问题列表
        - 预期字段列表
        - 已抽取字段（尽力而为）
        - 失败分析
        - 动态生成的模板
        - 修复建议
        """
        text = text or ""
        reconstructed = self._reconstruct_text(ocr_raw, text)
        recon_text = reconstructed.get("text") or text

        segments = self.segment_document(recon_text)
        doc_pattern = self.infer_doc_pattern(recon_text, segments)
        doc_type_info = self._classify_doc_type_free(recon_text)
        quality = self._assess_ocr_quality(recon_text)
        expected_fields = self._expected_fields_for_type(
            doc_type_info["document_type"]
        )
        extracted_fields = self._extract_loose_fields(
            recon_text, expected_fields
        )
        key_value_pairs = self.align_key_values(
            recon_text, reconstructed.get("lines")
        )
        fail_analysis = self._build_fail_analysis(
            extracted_fields, expected_fields, quality, doc_type_info
        )
        suggested_template = self._template_from_expected(
            doc_type_info["document_type"], expected_fields
        )
        repair_plan = self._suggest_repair(quality, doc_type_info, recon_text)
        parsed_tables = reconstructed.get("parsed_tables") or []
        # 班级成员表表格行提取
        roster_rows = self._extract_roster_rows(recon_text)
        if roster_rows:
            roster_table = [
                {"row": idx, "cells": [
                    str(r.get("number") or idx + 1),
                    str(r.get("class") or ""),
                    str(r.get("name") or ""),
                    str(r.get("gender") or ""),
                ]}
                for idx, r in enumerate(roster_rows)
            ]
            parsed_tables.append(roster_table)
        if segments.get("table"):
            parsed_tables.extend(self.parse_table_plain(segments.get("table")))
        # 将补充的表格回填到重建结果，便于后续统一输出
        reconstructed["parsed_tables"] = parsed_tables
        if roster_rows:
            reconstructed["roster_rows"] = roster_rows
        transcript_rows = None
        if doc_type_info.get("document_type") in ("成绩单", "课程表"):
            if parsed_tables:
                transcript_rows = self._extract_transcript_from_tables(
                    parsed_tables
                )

        return {
            "document_type": doc_type_info,
            "ocr_quality": quality["level"],
            "ocr_problems": quality["problems"],
            "fields_expected": expected_fields,
            "extracted_fields": extracted_fields,
            "fail_analysis": fail_analysis,
            "suggested_template": suggested_template,
            "repair_plan": repair_plan,
            "reconstructed": reconstructed,
            "transcript_rows": transcript_rows,
            "doc_pattern": doc_pattern,
            "segments": segments,
            "key_value_pairs": key_value_pairs,
            "roster_rows": roster_rows,
        }

    # ---------------------
    # 自动字段推断与模板生成
    # ---------------------
    def infer_fields_auto(self, text: str) -> Dict:
        """
        不依赖预设模板，基于规则/关键词/NER 自动推断应抽取的字段并给出结果与可复用模板。
        返回：
        - extract_result: 仿 extract_fields 的结构，包含提取详情
        - template_config: 自动生成的模板（可复用）
        - fail_reason: 若提取为空，给出简单原因
        """
        start_time = time.time()
        doc_hint = self._classify_doc_type_free(text).get("document_type")

        # 考试证书（如全国计算机等级考试合格证书）
        if doc_hint == "考试证书":
            def _clean_cert_value(field_name: str, value: Optional[str]) -> Optional[str]:
                if value is None:
                    return None
                v = str(value).strip()
                if not v:
                    return None
                if field_name == "name":
                    m = re.search(r"[\u4e00-\u9fa5]{2,8}", v)
                    return m.group(0) if m else None
                if field_name == "id_number":
                    m = re.search(r"\d{17}[\dXx]", v)
                    return m.group(0) if m else None
                if field_name == "certificate_id":
                    # 证书编号通常是较长数字串，避免把 MINISTRY 这类 OCR 英文噪声当作证书号
                    if sum(ch.isdigit() for ch in v) < 6:
                        return None
                    return v
                if field_name == "verify_url":
                    if v.startswith("www.") or v.startswith("http://") or v.startswith("https://"):
                        return v
                    return None
                return v

            template_fields = [
                {"name": "name", "type": "PERSON", "required": False,
                 "patterns": [r"(?:姓\s*名|名)[:：]?\s*([A-Za-z\u4e00-\u9fa5]{2,20})"],
                 "description": "姓名"},
                {"name": "id_number", "type": "ID_NUMBER", "required": False,
                 "patterns": [r"(?:身份证件号|证件号码|身份证号码|身份证号)[:：]?\s*(\d{17}[\dXx])"],
                 "description": "身份证号"},
                {"name": "certificate_id", "type": "CERT_ID", "required": False,
                 "patterns": [r"(?:证书编号|Certificate\s*Number)[:：]?\s*([A-Za-z0-9]{8,30})"],
                 "description": "证书编号"},
                {"name": "verify_code", "type": "TEXT", "required": False,
                 "patterns": [r"(?:校验码|验证码)[:：]?\s*([A-Za-z0-9]{6,40})"],
                 "description": "校验码"},
                {"name": "verify_url", "type": "TEXT", "required": False,
                 "patterns": [r"(?:查询网址|查询地址|网址)[:：]?\s*(https?://[^\s]+|www\.[^\s]+)"],
                 "description": "查询网址"},
            ]

            extract_details = []
            extracted_fields = 0
            total_confidence = 0.0
            for field in template_fields:
                value, conf, pos = self._extract_by_patterns(text, field.get("patterns", []), field["type"])
                value = _clean_cert_value(field["name"], value)
                if value is not None:
                    extracted_fields += 1
                    total_confidence += conf
                extract_details.append(
                    {
                        "field_name": field["name"],
                        "field_value": value,
                        "field_type": field["type"],
                        "confidence": round(conf, 4),
                        "source_position": pos,
                        "source": "exam_certificate",
                    }
                )

            avg_confidence = total_confidence / extracted_fields if extracted_fields else 0.0
            processing_time = round(time.time() - start_time, 2)
            extract_result = {
                "file_id": None,
                "template_id": "auto-exam-certificate",
                "extract_main": {
                    "total_fields": len(template_fields),
                    "extracted_fields": extracted_fields,
                    "confidence": round(avg_confidence, 4),
                    "status": "success" if extracted_fields > 0 else "failed",
                },
                "extract_details": extract_details,
                "processing_time": processing_time,
            }
            return {
                "extract_result": extract_result,
                "template_config": {"template_id": "auto-exam-certificate", "fields": template_fields},
                "fail_reason": None if extracted_fields else "未识别到考试证书关键字段",
            }

        # 学籍信息卡：走专用模板，避免通用字段把标签词误当值（如 class=姓名、college=层次）
        if doc_hint == "学籍信息卡":
            invalid_tokens = {
                "姓名", "性别", "出生日期", "民族", "学校名称", "层次", "专业", "学制", "学历类别",
                "学习形式", "分院", "系所", "入学日期", "学籍状态", "预计毕业日期", "在线验证码",
                "姓名：", "性别：", "学校名称：", "层次：", "专业：", "学制：", "学历类别：", "学习形式：",
                "分院：", "系所：", "入学日期：", "学籍状态：", "预计毕业日期：", "在线验证码：",
            }

            def _clean_card_value(field_name: str, value: Optional[str]) -> Optional[str]:
                if value is None:
                    return None
                cleaned = self._clean_kv_value(value)
                if not cleaned:
                    return None
                if cleaned in invalid_tokens:
                    return None
                if cleaned.endswith("：") and len(cleaned) <= 12:
                    return None

                if field_name in ("name",):
                    m = re.search(r"[\u4e00-\u9fa5]{2,8}", cleaned)
                    return m.group(0) if m else None
                if field_name in ("gender",):
                    return cleaned if cleaned in ("男", "女") else None
                if field_name in ("student_id",):
                    m = re.search(r"\d{6,12}", cleaned)
                    return m.group(0) if m else None
                if field_name in ("verify_code",):
                    m = re.search(r"[A-Za-z0-9]{6,30}", cleaned)
                    return m.group(0) if m else None
                if field_name in ("birth_date", "enroll_date", "expected_grad_date"):
                    norm = normalize_date(cleaned)
                    return norm if norm else None
                if field_name in ("verify_url",):
                    if cleaned.startswith("http://") or cleaned.startswith("https://") or cleaned.startswith("www."):
                        return cleaned
                    return None
                if field_name in ("school_name", "college"):
                    # 学校/学院名常为短中文，避免被通用标签过滤误杀。
                    if any(k in cleaned for k in ("大学", "学院", "学校")):
                        return cleaned
                if self._is_bad_field_value(cleaned):
                    return None
                return cleaned

            template_fields = [
                {"name": "name", "type": "PERSON", "required": False, "patterns": [r"姓名[:：]?\s*([^\n]{1,20})"], "description": "姓名"},
                {"name": "gender", "type": "TEXT", "required": False, "patterns": [r"性别[:：]?\s*([^\n]{1,8})"], "description": "性别"},
                {"name": "birth_date", "type": "DATE", "required": False, "patterns": [r"出生日期[:：]?\s*([^\n]{6,20})"], "description": "出生日期"},
                {"name": "school_name", "type": "ORG", "required": False, "patterns": [r"(?:学校名称|校名称|学校名)[:：]?\s*([^\n]{2,40})"], "description": "学校名称"},
                {"name": "degree_level", "type": "TEXT", "required": False, "patterns": [r"层次[:：]?\s*([^\n]{1,20})"], "description": "层次"},
                {"name": "major", "type": "TEXT", "required": False, "patterns": [r"专业[:：]?\s*([^\n]{2,40})"], "description": "专业"},
                {"name": "study_mode", "type": "TEXT", "required": False, "patterns": [r"(?:学习形式|习形式)[:：]?\s*([^\n]{2,20})"], "description": "学习形式"},
                {"name": "education_type", "type": "TEXT", "required": False, "patterns": [r"学历类别[:：]?\s*([^\n]{2,20})"], "description": "学历类别"},
                {"name": "college", "type": "ORG", "required": False, "patterns": [r"(?:分院|系所)[:：]?\s*([^\n]{2,40})"], "description": "分院/系所"},
                {"name": "enroll_date", "type": "DATE", "required": False, "patterns": [r"入学日期[:：]?\s*([^\n]{6,20})"], "description": "入学日期"},
                {"name": "status", "type": "TEXT", "required": False, "patterns": [r"学籍状态[:：]?\s*([^\n]{2,30})"], "description": "学籍状态"},
                {"name": "expected_grad_date", "type": "DATE", "required": False, "patterns": [r"预计毕业日期[:：]?\s*([^\n]{6,20})"], "description": "预计毕业日期"},
                {"name": "verify_code", "type": "TEXT", "required": False, "patterns": [r"(?:在线验证码|在.?证码)[:：]?\s*([A-Za-z0-9]{6,30})"], "description": "在线验证码"},
                {"name": "verify_url", "type": "TEXT", "required": False, "patterns": [r"(?:在线查验网址|查询网址|网址)[:：]?\s*(https?://[^\s]+|www\.[^\s]+)"], "description": "在线查验网址"},
            ]

            extract_details = []
            extracted_fields = 0
            total_confidence = 0.0
            for field in template_fields:
                value, conf, pos = self._extract_by_patterns(text, field.get("patterns", []), field["type"])
                value = _clean_card_value(field["name"], value)
                if value is not None:
                    extracted_fields += 1
                    total_confidence += conf
                extract_details.append(
                    {
                        "field_name": field["name"],
                        "field_value": value,
                        "field_type": field["type"],
                        "confidence": round(conf if value is not None else 0.0, 4),
                        "source_position": pos,
                        "source": "student_card",
                    }
                )

            avg_confidence = total_confidence / extracted_fields if extracted_fields else 0.0
            processing_time = round(time.time() - start_time, 2)
            extract_result = {
                "file_id": None,
                "template_id": "auto-student-card",
                "extract_main": {
                    "total_fields": len(template_fields),
                    "extracted_fields": extracted_fields,
                    "confidence": round(avg_confidence, 4),
                    "status": "success" if extracted_fields > 0 else "failed",
                },
                "extract_details": extract_details,
                "processing_time": processing_time,
            }
            return {
                "extract_result": extract_result,
                "template_config": {"template_id": "auto-student-card", "fields": template_fields},
                "fail_reason": None if extracted_fields else "未识别到学籍信息卡关键字段",
            }

        # 准考证：优先走专用模板，避免回落到通用字段产生脏值
        if doc_hint == "准考证":
            invalid_tokens = {
                "所属学校", "所属学校：", "院系班级", "院系班级：", "证件号码", "证件号码：",
                "姓名", "姓名：", "性别", "性别：", "考试地点", "考试地点：", "报到时间", "报到时间：",
            }

            def _clean_ticket_value(field_name: str, value: Optional[str]) -> Optional[str]:
                if value is None:
                    return None
                cleaned = self._clean_kv_value(value)
                if not cleaned:
                    return None
                if cleaned in invalid_tokens:
                    return None
                if cleaned.endswith("：") and len(cleaned) <= 12:
                    return None
                if field_name in ("school_name", "class", "exam_site"):
                    if any(k in cleaned for k in ("考生须知", "考生须听从", "按违规处理", "证件不全", "不得参加")):
                        return None
                if self._is_bad_field_value(cleaned):
                    return None
                return cleaned

            template_fields = [
                {"name": "ticket_no", "type": "TEXT", "required": False, "patterns": [r"(?:准考证号|准考证)[:：]?\s*([A-Za-z0-9]{8,20})"], "description": "准考证号"},
                {"name": "name", "type": "PERSON", "required": False, "patterns": [r"(?:姓\s*名|姓名)[:：]?\s*([A-Za-z\u4e00-\u9fa5]{2,8})"], "description": "姓名"},
                {"name": "gender", "type": "TEXT", "required": False, "patterns": [r"性别[:：]?\s*(男|女)"], "description": "性别"},
                {"name": "id_number", "type": "ID_NUMBER", "required": False, "patterns": [r"(?:证件号码|身份证号|身份证号码)[:：]?\s*(\d{17}[\dXx])"], "description": "证件号码"},
                {"name": "school_name", "type": "ORG", "required": False, "patterns": [r"(?:所属学校|学校名称|学\s*校)[:：]?\s*([^\n]{2,40})"], "description": "所属学校"},
                {"name": "class", "type": "CLASS", "required": False, "patterns": [r"(?:院系班级|班级|院\s*[（(]?\s*系\s*[）)]?)[:：]?\s*([^\n]{2,40})"], "description": "院系班级"},
                {"name": "student_id", "type": "STUDENT_ID", "required": False, "patterns": [r"学号[:：]?\s*([0-9]{6,12})"], "description": "学号"},
                {"name": "exam_date", "type": "DATE", "required": False, "patterns": [r"(?:考试日期|考试时间)[:：]?\s*([0-9]{4}[-/.年][0-9]{1,2}(?:[-/.月][0-9]{1,2}日?)?)"], "description": "考试日期"},
                {"name": "report_time", "type": "TEXT", "required": False, "patterns": [r"报到时间[:：]?\s*([0-9]{1,2}:[0-9]{2})"], "description": "报到时间"},
                {"name": "exam_time", "type": "TEXT", "required": False, "patterns": [r"考试时间[:：]?\s*([0-9]{1,2}:[0-9]{2}(?:\s*[-~]\s*[0-9]{1,2}:[0-9]{2})?)"], "description": "考试时间"},
                {"name": "exam_site", "type": "TEXT", "required": False, "patterns": [r"考试地点[:：]?\s*([^\n]{2,80})"], "description": "考试地点"},
                {"name": "room_no", "type": "TEXT", "required": False, "patterns": [r"考场号[:：]?\s*([A-Za-z0-9]{1,10})"], "description": "考场号"},
                {"name": "seat_no", "type": "TEXT", "required": False, "patterns": [r"座位号[:：]?\s*([A-Za-z0-9]{1,10})"], "description": "座位号"},
            ]

            extract_details = []
            extracted_fields = 0
            total_confidence = 0.0
            for field in template_fields:
                value, conf, pos = self._extract_by_patterns(text, field.get("patterns", []), field["type"])
                value = _clean_ticket_value(field["name"], value)
                if value is not None:
                    extracted_fields += 1
                    total_confidence += conf
                extract_details.append({
                    "field_name": field["name"],
                    "field_value": value,
                    "field_type": field["type"],
                    "confidence": round(conf, 4),
                    "source_position": pos,
                    "source": "admission_ticket",
                })

            avg_confidence = total_confidence / extracted_fields if extracted_fields else 0.0
            processing_time = round(time.time() - start_time, 2)
            extract_result = {
                "file_id": None,
                "template_id": "auto-admission-ticket",
                "extract_main": {
                    "total_fields": len(template_fields),
                    "extracted_fields": extracted_fields,
                    "confidence": round(avg_confidence, 4),
                    "status": "success" if extracted_fields > 0 else "failed",
                },
                "extract_details": extract_details,
                "processing_time": processing_time,
            }
            return {
                "extract_result": extract_result,
                "template_config": {"template_id": "auto-admission-ticket", "fields": template_fields},
                "fail_reason": None if extracted_fields else "未识别到准考证关键字段",
            }

        # 住宿表：优先走专用模板，避免回落通用字段集合
        if doc_hint == "住宿表":
            dorm_sample = self._extract_dorm_row_sample(text)
            has_dorm_hint = bool(
                re.search(r"住宿表|宿舍表|宿舍名单|住宿名单|楼栋|宿舍号|床位", text)
            )
            if not dorm_sample and not has_dorm_hint:
                pass
            else:
                template_fields = [
                    {
                        "name": "building",
                        "type": "TEXT",
                        "required": False,
                        "patterns": [
                            r"(?:楼栋|宿舍楼|楼号)[:：]?\s*([0-9A-Za-z一二三四五六七八九十]{1,6}(?:栋|号楼)?)",
                            r"(?:^|\n)\s*\d{0,3}\s*([0-9A-Za-z一二三四五六七八九十]{1,6}(?:栋|号楼))\s+",
                        ],
                        "description": "楼栋",
                    },
                    {
                        "name": "dorm_no",
                        "type": "TEXT",
                        "required": False,
                        "patterns": [
                            r"(?:宿舍号|寝室号|房间号)[:：]?\s*([A-Za-z0-9\-]{2,10})",
                            r"(?:^|\n)\s*\d{0,3}\s*[0-9A-Za-z一二三四五六七八九十]{1,6}(?:栋|号楼)\s+([A-Za-z0-9\-]{2,10})\s+",
                        ],
                        "description": "宿舍号",
                    },
                    {
                        "name": "bed_no",
                        "type": "TEXT",
                        "required": False,
                        "patterns": [
                            r"(?:床位|床号)[:：]?\s*([A-Za-z0-9]{1,4})",
                            r"(?:^|\n)\s*\d{0,3}\s*[0-9A-Za-z一二三四五六七八九十]{1,6}(?:栋|号楼)\s+[A-Za-z0-9\-]{2,10}\s+([A-Za-z0-9]{1,4})\s+",
                        ],
                        "description": "床位",
                    },
                    {
                        "name": "name",
                        "type": "PERSON",
                        "required": False,
                        "patterns": [
                            r"姓名[:：]?\s*([A-Za-z\u4e00-\u9fa5]{2,8})",
                            r"(?:^|\n)\s*\d{0,3}\s*[0-9A-Za-z一二三四五六七八九十]{1,6}(?:栋|号楼)\s+[A-Za-z0-9\-]{2,10}\s+[A-Za-z0-9]{1,4}\s+([\u4e00-\u9fa5]{2,4})\s+",
                        ],
                        "description": "姓名",
                    },
                    {
                        "name": "student_id",
                        "type": "STUDENT_ID",
                        "required": False,
                        "patterns": [
                            r"学号[:：]?\s*([0-9]{6,12})",
                            r"(?:^|\n)\s*\d{0,3}\s*[0-9A-Za-z一二三四五六七八九十]{1,6}(?:栋|号楼)\s+[A-Za-z0-9\-]{2,10}\s+[A-Za-z0-9]{1,4}\s+[\u4e00-\u9fa5]{2,4}\s+([0-9]{6,12})",
                        ],
                        "description": "学号",
                    },
                ]

                extract_details = []
                extracted_fields = 0
                total_confidence = 0.0
                for f in template_fields:
                    val = None
                    conf = 0.0
                    pos = None
                    if dorm_sample:
                        val = dorm_sample.get(f["name"])
                        if val:
                            conf = 0.88
                            pos = {"context": "dorm-sample"}
                    if val is None:
                        val, conf, pos = self._extract_by_patterns(
                            text, f.get("patterns", []), f["type"]
                        )
                    if val is not None:
                        extracted_fields += 1
                        total_confidence += conf
                    extract_details.append(
                        {
                            "field_name": f["name"],
                            "field_value": val,
                            "field_type": f["type"],
                            "confidence": round(conf, 4),
                            "source_position": pos,
                            "source": "dormitory",
                        }
                    )

                avg_confidence = (
                    total_confidence / extracted_fields if extracted_fields else 0.0
                )
                processing_time = round(time.time() - start_time, 2)
                extract_result = {
                    "file_id": None,
                    "template_id": "auto-dormitory",
                    "extract_main": {
                        "total_fields": len(template_fields),
                        "extracted_fields": extracted_fields,
                        "confidence": round(avg_confidence, 4),
                        "status": "success" if extracted_fields > 0 else "failed",
                    },
                    "extract_details": extract_details,
                    "processing_time": processing_time,
                }
                template_config = {
                    "template_id": "auto-dormitory",
                    "fields": template_fields,
                }
                fail_reason = None if extracted_fields else "未找到住宿表关键字段"
                return {
                    "extract_result": extract_result,
                    "template_config": template_config,
                    "fail_reason": fail_reason,
                }

        # 班级成员表：直接按名单三元组生成仅包含班级/姓名/性别(补充学号)的模板与提取结果
        if doc_hint == "班级成员表":
            roster_rows = self._extract_roster_rows(text)
            # 若未检测到表格行且正文中未出现明显“班级成员”提示，则回退到通用字段提取，避免住宿表等被误识别
            has_roster_hint = bool(re.search(r"班级成员表|成员名单|班级名单|班级表|班级花名册", text))
            if not roster_rows and not has_roster_hint:
                pass
            else:
                roster_sample = roster_rows[0] if roster_rows else self._extract_roster_sample(
                    text)
                roster_class = roster_sample.get(
                    "class") if roster_sample else None
                if not roster_class and roster_rows:
                    counter = {}
                    for r in roster_rows:
                        cls = r.get("class")
                        if not cls:
                            continue
                        counter[cls] = counter.get(cls, 0) + 1
                    if counter:
                        roster_class = max(
                            counter.items(), key=lambda kv: kv[1])[0]
                if not roster_class:
                    roster_class = self._infer_roster_class(text)
                if not roster_class:
                    roster_class = self._infer_roster_class_relaxed(text)
                if not roster_class:
                    roster_class = self._infer_roster_class_any(text)
                template_fields = [
                    {"name": "class", "type": "CLASS", "required": False,
                     "patterns": [r"班级[:：]?\s*([A-Za-z0-9一-龥]{2,12})"],
                     "description": "班级"},
                    {"name": "name", "type": "PERSON", "required": False,
                     "patterns": [r"姓名[:：]?\s*([A-Za-z\u4e00-\u9fa5]{2,6})"],
                     "description": "姓名"},
                    {"name": "gender", "type": "TEXT", "required": False,
                     "patterns": [r"性别[:：]?\s*(男|女)"],
                     "description": "性别"},
                    {"name": "student_id", "type": "STUDENT_ID", "required": False,
                     "patterns": [r"学号[:：]?\s*([0-9]{6,12})"],
                     "description": "学号"},
                ]

                extract_details = []
                extracted_fields = 0
                total_confidence = 0.0
                for f in template_fields:
                    val = None
                    conf = 0.0
                    pos = None
                    if roster_sample:
                        val = roster_sample.get(f["name"])
                        if val:
                            conf = 0.85
                            pos = {"context": "roster-sample"}
                    if val is None and f["name"] == "class" and roster_class:
                        val = roster_class
                        conf = 0.9
                        pos = {"context": "roster-majority"}
                    # 如果三元组未命中，再按正则试一次
                    if val is None:
                        val, conf, pos = self._extract_by_patterns(
                            text, f.get("patterns", []), f["type"]
                        )
                    if val is not None:
                        extracted_fields += 1
                        total_confidence += conf
                    extract_details.append({
                        "field_name": f["name"],
                        "field_value": val,
                        "field_type": f["type"],
                        "confidence": round(conf, 4),
                        "source_position": pos,
                        "source": "roster",
                    })

                avg_confidence = (
                    total_confidence / extracted_fields if extracted_fields else 0.0
                )
                processing_time = round(time.time() - start_time, 2)
                extract_result = {
                    "file_id": None,
                    "template_id": "auto-roster",
                    "extract_main": {
                        "total_fields": len(template_fields),
                        "extracted_fields": extracted_fields,
                        "confidence": round(avg_confidence, 4),
                        "status": "success" if extracted_fields > 0 else "failed",
                    },
                    "extract_details": extract_details,
                    "processing_time": processing_time,
                }
                template_config = {"template_id": "auto-roster",
                                   "fields": template_fields}
                fail_reason = None if extracted_fields else "未找到班级/姓名/性别三元组"
                return {
                    "extract_result": extract_result,
                    "template_config": template_config,
                    "fail_reason": fail_reason,
                }
        entities = self.extract_entities(text)
        # 预定义字段候选与抽取策略
        candidates = {
            "name": {"type": "PERSON", "keywords": ["姓名", "学生", "申请人", "联系人"]},
            "id_number": {"type": "ID_NUMBER", "keywords": ["身份证", "证件号码", "身份证号", "证号"]},
            "student_id": {"type": "STUDENT_ID", "keywords": ["学号", "学生编号", "学籍号"]},
            "address": {"type": "ADDRESS", "keywords": ["家庭住址", "住址", "地址"]},
            "issuer": {"type": "ORG", "keywords": ["发证单位", "发证机关", "盖章单位"]},
            "admission_school": {"type": "ORG", "keywords": ["录取学校", "录取院校", "录取单位"]},
            "date": {"type": "DATE", "keywords": ["日期", "时间", "填表日期"]},
            "reason": {"type": "TEXT", "keywords": ["原因", "说明", "事由", "申请事项"]},
            "certificate_id": {"type": "CERT_ID", "keywords": ["证件编号", "编号", "证号"]},
            "class": {"type": "CLASS", "keywords": ["班级", "班级名称"]},
            "contact": {"type": "PERSON", "keywords": ["联系人", "家长"]},
            "phone": {"type": "PHONE", "keywords": ["电话", "联系方式", "手机号"]},
        }

        extract_details = []
        template_fields = []

        for field_key, meta in candidates.items():
            value, confidence, position = self._extract_by_keywords_and_entities(
                text, entities, meta["keywords"], meta["type"]
            )
            extract_details.append(
                {
                    "field_name": field_key,
                    "field_value": value,
                    "field_type": meta["type"],
                    "confidence": round(confidence, 4),
                    "source_position": position,
                    "source": "auto",
                }
            )
            template_fields.append(
                {
                    "name": field_key,
                    "type": meta["type"],
                    "required": False,
                    "patterns": self._build_patterns_from_keywords(meta["keywords"]),
                    "description": f"自动推断字段: {field_key}",
                }
            )

        # 证明/学籍类专用字段补强
        proof_overrides = self.extract_proof_fields(text)
        if proof_overrides:
            existing = {d["field_name"]: d for d in extract_details}
            for item in proof_overrides:
                name = item["field_name"]
                if name in existing:
                    if item["field_value"]:
                        existing[name]["field_value"] = item["field_value"]
                        existing[name]["confidence"] = item.get(
                            "confidence", existing[name]["confidence"]
                        )
                else:
                    extract_details.append(item)
                    template_fields.append(
                        {
                            "name": name,
                            "type": item.get("field_type", "TEXT"),
                            "required": False,
                            "patterns": [],
                            "description": f"自动推断字段: {name}",
                        }
                    )

        extracted_fields = sum(
            1 for d in extract_details if d.get("field_value")
        )
        total_confidence = sum(
            d.get("confidence", 0) for d in extract_details if d.get("field_value")
        )
        avg_confidence = total_confidence / extracted_fields if extracted_fields else 0.0
        processing_time = round(time.time() - start_time, 2)

        extract_result = {
            "file_id": None,
            "template_id": "auto-generated-from-doc",
            "extract_main": {
                "total_fields": len(template_fields),
                "extracted_fields": extracted_fields,
                "confidence": round(avg_confidence, 4),
                "status": "success" if extracted_fields > 0 else "failed",
            },
            "extract_details": extract_details,
            "processing_time": processing_time,
        }

        fail_reason = (
            None if extracted_fields > 0 else "未识别到可提取的字段，请检查文本内容或清晰度"
        )
        template_config = {
            "template_id": "auto-generated-from-doc",
            "fields": template_fields,
        }
        return {
            "extract_result": extract_result,
            "template_config": template_config,
            "fail_reason": fail_reason,
        }

    def _extract_by_keywords_and_entities(
        self, text: str, entities: List[Dict], keywords: List[str], field_type: str
    ) -> tuple:
        """
        先按关键词所在行提取冒号后的内容，再根据实体类型兜底。
        """
        # 关键词行抽取
        for kw in keywords:
            pattern = rf"{re.escape(kw)}[:：]?\s*([^\n\r]+)"
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(1).strip()
                if value:
                    start = match.start(1)
                    end = match.end(1)
                    context = text[max(0, start - 15):min(len(text), end + 15)]
                    return value, 0.9, {"start": start, "end": end, "context": context}

        # 实体兜底
        for ent in entities:
            if ent["type"] == field_type or (field_type == "ORG" and ent["type"] == "ORG"):
                return ent["text"], 0.8, {
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                    "context": ent.get("context"),
                }

        # 特殊模式
        if field_type == "ID_NUMBER":
            m = re.search(r"\b(\d{17}[\dXx])\b", text)
            if m:
                return m.group(1), 0.9, {
                    "start": m.start(1),
                    "end": m.end(1),
                    "context": text[max(0, m.start(1) - 10): m.end(1) + 10],
                }

        if field_type == "CERT_ID":
            m = re.search(
                r"(?:证件编号|编号|证号)[:：]?\s*([A-Za-z0-9\-]{4,30})", text
            )
            if m:
                return m.group(1).strip(), 0.85, {
                    "start": m.start(1),
                    "end": m.end(1),
                    "context": text[max(0, m.start(1) - 10): m.end(1) + 10],
                }

        if field_type == "CLASS":
            m = re.search(
                r"(?:班级|班)[:：]?\s*([A-Za-z0-9一-龥\-]{2,20})", text
            )
            if m:
                return m.group(1).strip(), 0.8, {
                    "start": m.start(1),
                    "end": m.end(1),
                    "context": text[max(0, m.start(1) - 10): m.end(1) + 10],
                }

        return None, 0.0, None

    @staticmethod
    def _build_patterns_from_keywords(keywords: List[str]) -> List[str]:
        patterns: List[str] = []
        for kw in keywords:
            patterns.append(rf"{re.escape(kw)}[:：]?\s*([^\n\r]+)")
        return patterns

    @staticmethod
    def _extract_roster_sample(text: str) -> Optional[Dict[str, str]]:
        """从文本中提取首条班级/姓名/性别三元组作为兜底样例"""
        cls_token = r"(?:班级)?"
        cls_body = r"[A-Za-z0-9一-龥]{2,12}"
        roster_patterns = [
            ("cls-name-gender",
             rf"({cls_body}{cls_token})\s+([一-龥]{{1,4}})\s+(男|女)"),
            ("name-gender-cls",
             rf"([一-龥]{{1,4}})\s+(男|女)\s+({cls_body}{cls_token})"),
        ]
        for label, p in roster_patterns:
            for m in re.finditer(p, text):
                if label == "cls-name-gender":
                    return {
                        "class": m.group(1),
                        "name": m.group(2),
                        "gender": m.group(3),
                    }
                return {
                    "class": m.group(3),
                    "name": m.group(1),
                    "gender": m.group(2),
                }
        return None

    @staticmethod
    def _infer_roster_class(text: str) -> Optional[str]:
        """统计出现频次最高的班级标记作为兜底班级"""
        cls_token = r"(?:班级)?"
        cls_body = r"[A-Za-z0-9一-龥]{2,12}"
        patterns = [
            rf"({cls_body}{cls_token})\s+[一-龥]{{1,4}}\s+(?:男|女)",
            rf"[一-龥]{{1,4}}\s+(?:男|女)\s+({cls_body}{cls_token})",
        ]
        counter = {}
        for p in patterns:
            for m in re.finditer(p, text):
                cls = m.group(1)
                if not cls:
                    continue
                counter[cls] = counter.get(cls, 0) + 1
        if not counter:
            return None
        # 返回频次最高的班级
        return max(counter.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _infer_roster_class_relaxed(text: str) -> Optional[str]:
        """更宽松：只要“班级token + 性别”或“班级token + 姓名”即计数"""
        cls_body = r"[A-Za-z0-9一-龥]{2,12}"
        patterns = [
            rf"({cls_body})\s+(?:男|女)",
            rf"({cls_body})\s+[一-龥]{{1,4}}\s+(?:男|女)",
        ]
        counter = {}
        for p in patterns:
            for m in re.finditer(p, text):
                cls = m.group(1)
                if not cls:
                    continue
                counter[cls] = counter.get(cls, 0) + 1
        if not counter:
            return None
        return max(counter.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _infer_roster_class_any(text: str) -> Optional[str]:
        """兜底：提取重复出现的班级样式 token（含数字/字母/汉字），返回最高频"""
        tokens = re.findall(r"[A-Za-z0-9一-龥]{2,12}", text)
        counter = {}
        for tok in tokens:
            # 优先含数字或常见班级前缀（软升/软工等）
            if not re.search(r"\d", tok) and not re.search(r"软|计|信|工|班", tok):
                continue
            counter[tok] = counter.get(tok, 0) + 1
        if not counter:
            return None
        return max(counter.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _extract_roster_rows(text: str) -> List[Dict[str, Any]]:
        """提取全量班级成员行，兼容不同字段顺序，输出 number/class/name/gender 列表"""
        rows: List[Dict[str, Any]] = []

        def is_valid_class(tok: str) -> bool:
            if not tok:
                return False
            tok = tok.strip()
            if len(tok) < 3 or len(tok) > 12:
                return False
            if not re.search(r"\d", tok):
                return False
            if not re.search(r"[A-Za-z\u4e00-\u9fa5]", tok):
                return False
            if tok.isdigit():
                return False
            return True

        def is_valid_name(tok: str) -> bool:
            return bool(re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", tok or ""))

        patterns = [
            ("cls-name-gender", re.compile(
                r"(?:^|\s)(?:(\d{1,3})\s+)?([A-Za-z0-9\u4e00-\u9fa5]{2,12})(?:班级)?\s+([\u4e00-\u9fa5]{2,4})\s+(男|女)")),
            ("cls-gender-name", re.compile(
                r"(?:^|\s)(?:(\d{1,3})\s+)?([A-Za-z0-9\u4e00-\u9fa5]{2,12})(?:班级)?\s+(男|女)\s+([\u4e00-\u9fa5]{2,4})")),
            ("gender-cls-name", re.compile(
                r"(?:^|\s)(?:(\d{1,3})\s+)?(男|女)\s+([A-Za-z0-9\u4e00-\u9fa5]{2,12})(?:班级)?\s+([\u4e00-\u9fa5]{2,4})")),
            ("name-cls-gender", re.compile(
                r"(?:^|\s)(?:(\d{1,3})\s+)?([\u4e00-\u9fa5]{2,4})\s+([A-Za-z0-9\u4e00-\u9fa5]{2,12})(?:班级)?\s+(男|女)")),
        ]
        seen_spans = set()
        seen_pair = set()

        def add_row(num_raw: Optional[str], cls: str, name: str, gender: str) -> None:
            if not is_valid_class(cls) or not is_valid_name(name):
                return
            number = None
            if num_raw and str(num_raw).isdigit():
                number = int(num_raw)
            key = (cls, name, gender)
            if key in seen_pair:
                return
            seen_pair.add(key)
            rows.append(
                {
                    "number": number if number is not None else len(rows) + 1,
                    "class": cls,
                    "name": name,
                    "gender": gender,
                }
            )

        for label, pattern in patterns:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                if label == "cls-name-gender":
                    num, cls, name, gender = m.group(
                        1), m.group(2), m.group(3), m.group(4)
                elif label == "cls-gender-name":
                    num, cls, gender, name = m.group(
                        1), m.group(2), m.group(3), m.group(4)
                elif label == "gender-cls-name":
                    num, gender, cls, name = m.group(
                        1), m.group(2), m.group(3), m.group(4)
                else:  # name-cls-gender
                    num, name, cls, gender = m.group(
                        1), m.group(2), m.group(3), m.group(4)
                add_row(num, cls, name, gender)
        # 按出现顺序已有排序；若存在显式序号则按序号排序
        rows.sort(key=lambda r: r.get("number") or 1_000_000)
        return rows

    @staticmethod
    def _extract_dorm_row_sample(text: str) -> Dict[str, Optional[str]]:
        """提取住宿表中的首个样例行（楼栋/宿舍号/床位/姓名/学号）。"""
        patterns = [
            re.compile(
                r"(?:^|\n)\s*(?:\d{1,3}\s+)?"
                r"(?P<building>[0-9A-Za-z一二三四五六七八九十]{1,6}(?:栋|号楼))\s+"
                r"(?P<dorm_no>[A-Za-z0-9\-]{2,10})\s+"
                r"(?P<bed_no>[A-Za-z0-9]{1,4})\s+"
                r"(?P<name>[\u4e00-\u9fa5]{2,4})\s+"
                r"(?P<student_id>\d{6,12})"
            ),
            re.compile(
                r"(?:^|\n)\s*(?:\d{1,3}\s+)?"
                r"(?P<building>[0-9A-Za-z一二三四五六七八九十]{1,6}(?:栋|号楼))\s+"
                r"(?P<dorm_no>[A-Za-z0-9\-]{2,10})\s+"
                r"(?P<name>[\u4e00-\u9fa5]{2,4})\s+"
                r"(?P<student_id>\d{6,12})"
            ),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            groups = match.groupdict()
            return {
                "building": groups.get("building"),
                "dorm_no": groups.get("dorm_no"),
                "bed_no": groups.get("bed_no"),
                "name": groups.get("name"),
                "student_id": groups.get("student_id"),
            }
        return {}

    # ---------------------
    # OCR 诊断与动态模板工具
    # ---------------------
    def _classify_doc_type_free(self, text: str) -> Dict:
        """
        基于关键词的粗分类，输出 document_type + 概率分布。
        尽量与 LLM 归一化那边的 document_type 命名保持一致。
        """
        candidates = {
            # 高校学业类
            "成绩单": ["成绩单", "成绩表", "课程", "学分", "GPA", "绩点", "总评", "统计时间"],
            "在校证明": ["在校生", "在读", "学生证明", "在校证明", "学籍证明"],
            "学籍信息卡": ["学籍信息卡", "学籍卡", "学籍信息", "学籍状态", "注册学籍", "学籍在线验证报告", "在线验证报告", "预计毕业日期", "在线验证码", "学习形式", "学历类别"],
            "班级成员表": ["班级成员表", "班级成员", "成员名单", "班级名单", "班级表", "班级花名册", "序号", "班级", "姓名", "性别"],
            "住宿表": ["住宿表", "宿舍表", "宿舍名单", "住宿名单", "楼栋", "宿舍号", "寝室", "床位", "宿舍"],
            "考试证书": ["合格证书", "证书编号", "校验码", "查询网址", "全国计算机等级考试", "Certificate Number"],
            "准考证": ["准考证", "准考证号", "报到时间", "考试时间", "考场号", "座位号", "英语四级", "英语六级", "CET"],
            "毕业证书/学历证书": ["毕业证书", "学历证书", "普通高等学校", "经审核准予毕业"],
            "录取凭证": ["录取通知书", "录取通知", "录取学校", "新生", "录取专业"],
            "课程表": ["课程表", "上课时间", "周一", "周二", "节次", "教室"],

            # 贷款 / 奖助学金 / 资助类
            "生源地助学贷款合同": [
                "生源地助学贷款",
                "贷款合同编号",
                "助学贷款在线服务系统",
                "国家开发银行",
            ],
            "贷款合同": ["贷款合同", "借款合同", "借款人", "还款方式", "贷款金额"],
            "奖学金证明": ["奖学金", "助学金", "励志奖学金", "国家奖学金", "发放", "获奖"],
            "贫困证明": ["贫困", "经济困难", "低保", "家庭经济", "资助", "困难证明"],
            "收入证明": ["收入证明", "工资收入", "薪资", "月收入", "年收入"],

            # 请假/休学申请
            "请假条": ["请假条", "请假申请", "请假", "事假", "病假", "假期", "销假"],
            "休学申请": ["休学申请", "休学", "复学", "休学原因"],

            # 通知/会议类
            "通知": ["通知", "公告", "告知", "决定", "安排"],
            "会议纪要": ["会议纪要", "会议记录", "会议时间", "参会人员", "议题"],

            # 宽泛的证明类兜底 & 合同兜底
            "证明类其他": ["兹证明", "特此证明", "证明", "盖章"],
            "合同": ["合同", "协议", "甲方", "乙方", "签订"],
        }

        scores = {}
        for doc_type, kws in candidates.items():
            hit = sum(1 for kw in kws if kw in text)
            scores[doc_type] = hit

        card_signals = sum(1 for kw in ("学籍在线验证报告", "学籍状态", "预计毕业日期", "在线验证码", "学习形式", "学历类别") if kw in text)
        if card_signals >= 2:
            scores["学籍信息卡"] = scores.get("学籍信息卡", 0) + card_signals
            scores["班级成员表"] = max(0, scores.get("班级成员表", 0) - 1)

        # 同时出现“准考证号”与“证书编号/校验码”时，优先判为考试证书
        cert_signals = sum(1 for kw in ("合格证书", "证书编号", "校验码", "查询网址") if kw in text)
        if cert_signals >= 2:
            scores["考试证书"] = scores.get("考试证书", 0) + cert_signals
            scores["准考证"] = max(0, scores.get("准考证", 0) - 1)

        best_type = max(scores, key=scores.get) if scores else "未知"
        best_score = scores.get(best_type, 0)
        total = sum(scores.values()) or 1
        confidence = best_score / total if total else 0.0

        if best_score == 0:
            best_type = "未知"
            confidence = 0.0

        probs = (
            {k: round(v / total, 4) for k, v in scores.items()}
            if total
            else {}
        )
        return {
            "document_type": best_type,
            "confidence": round(confidence, 4),
            "probabilities": probs,
        }

    def _assess_ocr_quality(self, text: str) -> Dict:
        if not text or len(text.strip()) == 0:
            return {"level": "critical", "problems": ["文本缺失"]}
        noise_chars = sum(
            1 for ch in text if not ch.isprintable() or ord(ch) < 32
        )
        noise_ratio = noise_chars / max(len(text), 1)
        messy_tokens = sum(1 for ch in text if ch in {"�", "?", "*"})
        messy_ratio = messy_tokens / max(len(text), 1)
        lines = text.splitlines()
        short_lines = sum(1 for ln in lines if len(ln.strip()) <= 2)
        digit_lines = sum(
            1
            for ln in lines
            if sum(c.isdigit() for c in ln) > len(ln) * 0.5
        )
        problems = []
        if messy_ratio > 0.02 or noise_ratio > 0.02:
            problems.append("噪声字符过多")
        if digit_lines > len(lines) * 0.5:
            problems.append("表格未被解析")
        if short_lines > len(lines) * 0.4:
            problems.append("文本缺失")
        if "�" in text or "?" in text:
            problems.append("乱码")
        score = 1.0 - min(
            1.0,
            messy_ratio * 5
            + noise_ratio * 3
            + 0.1 * short_lines / max(len(lines), 1),
        )
        if score >= 0.75:
            level = "good"
        elif score >= 0.5:
            level = "medium"
        elif score >= 0.3:
            level = "bad"
        else:
            level = "critical"
        return {"level": level, "problems": list(set(problems))}

    def _expected_fields_for_type(self, doc_type: str) -> List[Dict]:
        """
        不同文档类型下，预期应该能提取到哪些字段（用于模板 & 缺失诊断）。
        这里字段名尽量跟 LLM 归一化 schema 能映射得上。
        """
        mapping = {
            "成绩单": [
                ("semester", "TEXT", "学年/学期"),
                ("course_name", "TEXT", "课程名称"),
                ("credit", "NUMBER", "学分"),
                ("score", "NUMBER", "成绩"),
                ("gpa", "NUMBER", "平均绩点"),
                ("stat_date", "DATE", "统计时间"),
            ],
            "贫困证明": [
                ("applicant_name", "PERSON", "申请人姓名"),
                ("id_number", "ID_NUMBER", "身份证号"),
                ("address", "ADDRESS", "家庭住址"),
                ("issuer", "ORG", "发证单位"),
                ("issue_date", "DATE", "出具日期"),
                ("reason", "TEXT", "困难原因"),
            ],
            "班级成员表": [
                ("class", "CLASS", "班级"),
                ("name", "PERSON", "姓名"),
                ("gender", "TEXT", "性别"),
            ],
            "奖学金证明": [
                ("applicant_name", "PERSON", "获奖人姓名"),
                ("id_number", "ID_NUMBER", "身份证号"),
                ("school", "ORG", "学校"),
                ("college", "ORG", "学院"),
                ("major", "TEXT", "专业"),
                ("scholarship_name", "TEXT", "奖学金名称"),
                ("scholarship_amount", "NUMBER", "奖学金金额"),
                ("issue_date", "DATE", "出具日期"),
                ("issuer", "ORG", "出具单位"),
            ],
            "在校证明": [
                ("student_name", "PERSON", "学生姓名"),
                ("student_id", "STUDENT_ID", "学号"),
                ("school", "ORG", "学校"),
                ("college", "ORG", "学院"),
                ("major", "TEXT", "专业"),
                ("class", "CLASS", "班级"),
                ("enroll_date", "DATE", "入学日期"),
                ("expected_grad_date", "DATE", "预计毕业日期"),
                ("status", "TEXT", "学籍状态"),
                ("issue_date", "DATE", "出具日期"),
            ],
            "学籍信息卡": [
                ("student_name", "PERSON", "姓名"),
                ("student_id", "STUDENT_ID", "学号"),
                ("class", "CLASS", "班级"),
                ("major", "TEXT", "专业"),
                ("school", "ORG", "学校"),
                ("college", "ORG", "学院"),
                ("status", "TEXT", "学籍状态"),
            ],
            "毕业证书/学历证书": [
                ("name", "PERSON", "姓名"),
                ("gender", "TEXT", "性别"),
                ("birth_date", "DATE", "出生日期"),
                ("school_name", "ORG", "学校名称"),
                ("college", "ORG", "学院/系所"),
                ("major", "TEXT", "专业"),
                ("degree_level", "TEXT", "学历层次"),
                ("education_type", "TEXT", "学历类别"),
                ("study_mode", "TEXT", "学习形式/学制"),
                ("enroll_date", "DATE", "入学日期"),
                ("expected_grad_date", "DATE", "毕业/结业日期"),
                ("certificate_id", "CERT_ID", "证书编号"),
                ("issue_date", "DATE", "证书签发日期"),
            ],
            "生源地助学贷款合同": [
                ("name", "PERSON", "借款人姓名"),
                ("id_number", "ID_NUMBER", "身份证号"),
                ("address", "ADDRESS", "家庭住址"),
                ("school_name", "ORG", "学校"),
                ("college", "ORG", "学院"),
                ("major", "TEXT", "专业"),
                ("loan_amount", "NUMBER", "贷款金额"),
                ("loan_years", "NUMBER", "贷款年限"),
                ("bank_name", "ORG", "贷款银行"),
                ("account_number", "NUMBER", "贷款/还款账号"),
                ("contract_id", "CERT_ID", "合同编号"),
                ("issue_date", "DATE", "签订日期"),
            ],
            "贷款合同": [
                ("name", "PERSON", "借款人姓名"),
                ("id_number", "ID_NUMBER", "身份证号"),
                ("loan_amount", "NUMBER", "贷款金额"),
                ("loan_years", "NUMBER", "贷款年限"),
                ("bank_name", "ORG", "贷款银行"),
                ("account_number", "NUMBER", "账号"),
                ("contract_id", "CERT_ID", "合同编号"),
                ("issue_date", "DATE", "签订日期"),
            ],
            "收入证明": [
                ("name", "PERSON", "姓名"),
                ("id_number", "ID_NUMBER", "身份证号"),
                ("employer", "ORG", "单位"),
                ("income", "NUMBER", "月/年收入"),
                ("issue_date", "DATE", "开具日期"),
            ],
            "请假条": [
                ("name", "PERSON", "请假人姓名"),
                ("leave_type", "TEXT", "假期类型"),
                ("start_date", "DATE", "开始日期"),
                ("end_date", "DATE", "结束日期"),
                ("days", "NUMBER", "天数"),
                ("reason", "TEXT", "请假原因"),
                ("approver", "PERSON", "批准人"),
            ],
            "休学申请": [
                ("name", "PERSON", "申请人"),
                ("id_number", "ID_NUMBER", "身份证"),
                ("student_id", "STUDENT_ID", "学号"),
                ("reason", "TEXT", "休学原因"),
                ("start_date", "DATE", "休学开始"),
                ("end_date", "DATE", "休学结束"),
            ],
            "通知": [
                ("title", "TEXT", "通知标题"),
                ("issuer", "ORG", "发布单位"),
                ("date", "DATE", "发布日期"),
                ("content", "TEXT", "通知内容"),
            ],
            "录取凭证": [
                ("candidate_name", "PERSON", "姓名"),
                ("exam_id", "TEXT", "准考证号"),
                ("admit_school", "ORG", "录取学校"),
                ("major", "TEXT", "专业"),
                ("admit_date", "DATE", "录取日期"),
            ],
            "课程表": [
                ("student_name", "PERSON", "姓名"),
                ("class", "CLASS", "班级"),
                ("week", "TEXT", "周次"),
                ("course_slot", "TEXT", "课程安排"),
            ],
            "会议纪要": [
                ("meeting_title", "TEXT", "会议主题"),
                ("meeting_date", "DATE", "会议日期"),
                ("attendees", "TEXT", "参会人员"),
                ("summary", "TEXT", "会议摘要"),
            ],
            "合同": [
                ("party_a", "ORG", "甲方"),
                ("party_b", "ORG", "乙方"),
                ("sign_date", "DATE", "签订日期"),
                ("amount", "NUMBER", "金额"),
            ],
            "住宿表": [
                ("building", "TEXT", "楼栋"),
                ("dorm_no", "TEXT", "宿舍号"),
                ("bed_no", "TEXT", "床位"),
                ("name", "PERSON", "姓名"),
                ("student_id", "STUDENT_ID", "学号"),
            ],
            "考试证书": [
                ("name", "PERSON", "姓名"),
                ("id_number", "ID_NUMBER", "身份证件号"),
                ("certificate_id", "CERT_ID", "证书编号"),
                ("verify_code", "TEXT", "校验码"),
                ("verify_url", "TEXT", "查询网址"),
            ],
            "准考证": [
                ("ticket_no", "TEXT", "准考证号"),
                ("name", "PERSON", "姓名"),
                ("gender", "TEXT", "性别"),
                ("id_number", "ID_NUMBER", "证件号码"),
                ("school_name", "ORG", "所属学校"),
                ("class", "CLASS", "院系班级"),
                ("student_id", "STUDENT_ID", "学号"),
                ("exam_date", "DATE", "考试日期"),
                ("report_time", "TEXT", "报到时间"),
                ("exam_time", "TEXT", "考试时间"),
                ("exam_site", "TEXT", "考试地点"),
                ("room_no", "TEXT", "考场号"),
                ("seat_no", "TEXT", "座位号"),
            ],
        }

        # 通用兜底字段
        default_fields = [
            ("name", "PERSON", "姓名"),
            ("id_number", "ID_NUMBER", "证件号/身份证号"),
            ("issuer", "ORG", "发证/发布单位"),
            ("date", "DATE", "日期"),
            ("reason", "TEXT", "原因/说明"),
        ]
        items = mapping.get(doc_type, default_fields)
        return [
            {
                "field_name": f[0],
                "field_type": f[1],
                "description": f[2],
            }
            for f in items
        ]

    def _extract_loan_contract_fields(self, text: str) -> Dict[str, Any]:
        """
        专门提取生源地助学贷款合同的字段。
        返回 {field_name: {"value": ..., "confidence": ..., "source": ...}}
        """
        fields = {}

        # 学生姓名: "学生姓名 周宇清" 或 "贵校学生_周宇清_的"
        m_name = re.search(r"学生姓名\s*([^\s\n]{2,10})", text)
        if not m_name:
            m_name = re.search(r"贵校学生[_\s]*([^\s_]{2,10})[_\s]*的", text)
        if m_name:
            fields["name"] = {"value": m_name.group(1), "confidence": 0.95, "source": m_name.group(0)}

        # 身份证号码: "身份证号码" 后面跟18位数字
        m_id = re.search(r"身份证号码?\s*(\d{17}[\dXx])", text)
        if m_id:
            fields["id_number"] = {"value": m_id.group(1), "confidence": 0.95, "source": m_id.group(0)}

        # 户籍地址: "户籍地址 xxx"
        m_addr = re.search(r"户籍地址\s*([^\n]{5,50}?)(?=\s*就学信息|\s*贷款信息|\n|$)", text)
        if m_addr:
            fields["address"] = {"value": m_addr.group(1).strip(), "confidence": 0.9, "source": m_addr.group(0)}

        # 就读高校: "就读高校" 后面或前面的学校名
        m_school = re.search(r"(?:就读高校|账户名称)\s*([^\n]{2,30}?(?:大学|学院|学校))", text)
        if not m_school:
            m_school = re.search(r"([^\n]{2,30}?(?:大学|学院|学校))\s*[:：]?\s*我行已成功受理", text)
        if m_school:
            fields["school_name"] = {"value": m_school.group(1).strip(), "confidence": 0.9, "source": m_school.group(0)}

        # 院系名称: "院系名称 xxx"
        m_college = re.search(r"院系名称\s*([^\n]{2,30}?(?:学院|系|部))", text)
        if m_college:
            fields["college"] = {"value": m_college.group(1).strip(), "confidence": 0.9, "source": m_college.group(0)}

        # 专业名称: "专业名称 xxx" 或 "(专升本)xxx"
        m_major = re.search(r"专业名称\s*([^\n]{2,30})", text)
        if not m_major:
            m_major = re.search(r"[（(]专升本[）)]\s*([^\n\s]{2,20})", text)
        if m_major:
            major_val = m_major.group(1).strip()
            # 清理可能的干扰文字
            major_val = re.sub(r"贷款信息.*", "", major_val).strip()
            if major_val:
                fields["major"] = {"value": major_val, "confidence": 0.9, "source": m_major.group(0)}

        # 入学年份: "入学年份 2024"
        m_enroll = re.search(r"入学年份\s*(\d{4})", text)
        if m_enroll:
            fields["enroll_year"] = {"value": m_enroll.group(1), "confidence": 0.9, "source": m_enroll.group(0)}

        # 学制: "学制 2年"
        m_duration = re.search(r"学制\s*(\d+)\s*年?", text)
        if m_duration:
            fields["study_years"] = {"value": m_duration.group(1), "confidence": 0.9, "source": m_duration.group(0)}

        # 贷款合同编号: 一般是纯数字
        m_contract = re.search(r"贷款合同编号\s*(\d{10,20})", text)
        if m_contract:
            fields["contract_id"] = {"value": m_contract.group(1), "confidence": 0.9, "source": m_contract.group(0)}

        # 用款期限: "用款期限 87（月）"
        m_term = re.search(r"用款期限\s*(\d+)\s*[（\(]?月[）\)]?", text)
        if m_term:
            fields["loan_term_months"] = {"value": m_term.group(1), "confidence": 0.9, "source": m_term.group(0)}

        # 本次用款金额: "本次用款金额 10000.00（元）"
        m_amount = re.search(r"(?:本次用款金额|贷款金额)\s*(\d+(?:\.\d+)?)\s*[（\(]?元[）\)]?", text)
        if m_amount:
            fields["loan_amount"] = {"value": float(m_amount.group(1)), "confidence": 0.95, "source": m_amount.group(0)}

        # 账户名称（高校）: "账户名称 马鞍山学院"
        m_acc_name = re.search(r"账户名称\s*([^\n]{2,30}?(?:大学|学院|学校))", text)
        if m_acc_name:
            fields["account_name"] = {"value": m_acc_name.group(1).strip(), "confidence": 0.85, "source": m_acc_name.group(0)}

        # 账号: "账号" 后面跟数字
        m_acc_no = re.search(r"账号\s*(\d{6,25})", text)
        if m_acc_no:
            fields["account_number"] = {"value": m_acc_no.group(1), "confidence": 0.9, "source": m_acc_no.group(0)}

        # 开户行: "开户行 xxx银行xxx支行"
        m_bank = re.search(r"开户行\s*([^\n]{5,50}?(?:银行|支行|分行))", text)
        if m_bank:
            fields["bank_name"] = {"value": m_bank.group(1).strip(), "confidence": 0.9, "source": m_bank.group(0)}

        # 回执验证码: "回执验证码为：874653"
        m_verify = re.search(r"回执验证码[为是]?[:：]?\s*(\d{4,10})", text)
        if m_verify:
            fields["verify_code"] = {"value": m_verify.group(1), "confidence": 0.95, "source": m_verify.group(0)}

        # 签发日期: 最后的日期格式
        m_date = re.search(r"(\d{4}[-年]\d{1,2}[-月]\d{1,2}日?)(?:\s*$|\s*\n)", text)
        if m_date:
            norm_date = normalize_date(m_date.group(1))
            if norm_date:
                fields["issue_date"] = {"value": norm_date, "confidence": 0.85, "source": m_date.group(0)}

        # 发放银行全称: "安徽砀山农村商业银行股份有限公司官庄支行"
        m_issuer_bank = re.search(r"([\u4e00-\u9fa5]+银行[\u4e00-\u9fa5]*(?:支行|分行))", text)
        if m_issuer_bank:
            fields["issuer_bank"] = {"value": m_issuer_bank.group(1), "confidence": 0.85, "source": m_issuer_bank.group(0)}

        return fields

    def _extract_loose_fields(self, text: str, expected_fields: List[Dict]) -> List[Dict]:
        results = []
        # 班级成员表提要：提前抓取第一条三元组（班级/姓名/性别）作为兜底来源
        roster_sample = self._extract_roster_sample(text)

        # 助学贷款合同专项提取：如果文本包含助学贷款关键词，优先使用专项方法
        loan_contract_fields = {}
        if any(kw in text for kw in ("生源地助学贷款", "助学贷款", "贷款合同编号", "用款期限")):
            loan_contract_fields = self._extract_loan_contract_fields(text)
            logger.debug(f"助学贷款专项提取结果: {loan_contract_fields}")

        # 常见模式
        patterns = {
            "ID_NUMBER": r"\b(\d{17}[\dXx])\b",
            "STUDENT_ID": r"(?:学号|学籍号)[:：]?\s*([A-Za-z0-9]{6,20})",
            "DATE": r"(\d{4}[年\-\.\/]\d{1,2}[月\-\.\/]?\d{0,2}日?)",
            "PHONE": r"(?<!\d)(1[3-9]\d{9})(?!\d)",
            "NUMBER": r"(\d+(?:\.\d+)?)",
        }
        for field in expected_fields:
            name = field["field_name"]
            ftype = field["field_type"]
            desc = field.get("description", "")
            value = None
            confidence = 0.0
            reason = ""
            src = ""

            # 优先使用助学贷款专项提取的结果
            if name in loan_contract_fields:
                loan_field = loan_contract_fields[name]
                value = loan_field["value"]
                confidence = loan_field["confidence"]
                src = loan_field["source"]
                reason = "助学贷款专项提取"

            # 班级成员表兜底：用首条三元组赋值
            if value is None and roster_sample:
                if name in ("class", "班级"):
                    value = roster_sample.get("class")
                    confidence = 0.82
                    src = "roster-sample"
                    reason = "名单行三元组"
                elif name in ("name", "姓名", "student_name"):
                    value = roster_sample.get("name")
                    confidence = 0.82
                    src = "roster-sample"
                    reason = "名单行三元组"
                elif name in ("gender", "性别"):
                    value = roster_sample.get("gender")
                    confidence = 0.82
                    src = "roster-sample"
                    reason = "名单行三元组"
            # 人名增强
            if ftype == "PERSON" and value is None:
                m_person = re.search(
                    r"(?:姓名|学生|申请人|联系人)[:：]?\s*([A-Za-z\u4e00-\u9fa5]{2,10})",
                    text,
                )
                if m_person and not any(
                    ch.isdigit() for ch in m_person.group(1)
                ):
                    value = m_person.group(1).strip()
                    confidence = 0.85
                    src = m_person.group(0)
                    reason = "姓名关键词匹配"
            # 日期增强
            if ftype == "DATE" and value is None:
                m_date = re.search(patterns["DATE"], text)
                if m_date:
                    norm = normalize_date(m_date.group(1))
                    if norm:
                        value = norm
                        confidence = 0.82
                        src = m_date.group(0)
                        reason = "日期标准化"
            # 先根据关键词行
            m_kw = re.search(
                rf"{re.escape(desc or name)}[:：]?\s*([^\n\r]+)", text
            )
            if m_kw and not value:
                value = m_kw.group(1).strip()
                confidence = 0.8
                src = m_kw.group(0)
                reason = "关键词行匹配"
            # 再根据通用模式
            if value is None and ftype in patterns:
                m = re.search(patterns[ftype], text)
                if m:
                    value = m.group(1).strip()
                    confidence = 0.75
                    src = m.group(0)
                    reason = "模式匹配"
            # 身份类宽松兜底
            if value is None and ftype in ("ID_NUMBER", "STUDENT_ID", "PHONE"):
                m = re.search(r"(\d[\dXx\s]{10,20})", text)
                if m and len(re.sub(r"\s", "", m.group(1))) >= 11:
                    value = re.sub(r"\s", "", m.group(1))
                    confidence = 0.6
                    src = m.group(0)
                    reason = "宽松数字提取"

            results.append(
                {
                    "field_name": name,
                    "field_value": value,
                    "confidence": round(confidence, 4),
                    "source_text": src,
                    "reasoning": reason or "未匹配到明确来源",
                }
            )
        return results

    def extract_entities(self, text: str, use_cache: bool = True) -> List[Dict]:
        """
        基于正则的实体抽取（姓名/证件/日期/机构/课程/联系方式/申请/财务等），
        加入简单缓存：同一段 text 多次调用时避免重复扫描。
        """
        if (
            use_cache
            and self._entity_cache_text == text
            and self._entity_cache_result
        ):
            logger.debug(
                "Named entity extraction cache hit",
                total=len(self._entity_cache_result),
            )
            return self._entity_cache_result

        logger.info("Named entity extraction requested")
        entities: List[Dict] = []

        def add_entity(etype, match, group_id=1):
            try:
                g_start = match.start(group_id)
                g_end = match.end(group_id)
                g_text = match.group(group_id)
            except IndexError:
                g_start = match.start(0)
                g_end = match.end(0)
                g_text = match.group(0)
            entities.append(
                {
                    "type": etype,
                    "text": g_text,
                    "start": g_start,
                    "end": g_end,
                    "context": text[
                        max(0, g_start - 10): min(len(text), g_end + 10)
                    ],
                }
            )

        # PERSON：支持“姓名 张三”“学生 张三”以及无冒号空格分隔
        person_patterns = [
            r"(?:姓名|学生|申请人|联系人)[:：]?\s*([^\s\n]{2,10})",
            r"姓名\s+([^\s\n]{2,10})",
        ]
        for p in person_patterns:
            for m in re.finditer(p, text):
                if not any(ch.isdigit() for ch in m.group(1)):
                    add_entity("PERSON", m)

        # 班级成员表三元组：班级+姓名+性别 或 姓名+性别+班级
        cls_token = r"(?:班级)?"
        cls_body = r"[A-Za-z0-9一-龥]{2,12}"
        roster_patterns = [
            ("cls-name-gender",
             rf"({cls_body}{cls_token})\s*([一-龥]{{2,4}})\s+(男|女)"),
            ("name-gender-cls",
             rf"([一-龥]{{2,4}})\s+(男|女)\s*({cls_body}{cls_token})"),
        ]
        for label, p in roster_patterns:
            for m in re.finditer(p, text):
                cls = None
                name = None
                gender = None
                gender_start = None
                gender_end = None
                if label == "cls-name-gender":
                    cls = m.group(1)
                    name = m.group(2)
                    gender = m.group(3)
                    gender_start = m.start(3)
                    gender_end = m.end(3)
                else:
                    name = m.group(1)
                    gender = m.group(2)
                    gender_start = m.start(2)
                    gender_end = m.end(2)
                    cls = m.group(3)

                if cls:
                    entities.append(
                        {
                            "type": "CLASS",
                            "text": cls,
                            "start": m.start(1 if label == "cls-name-gender" else 3),
                            "end": m.end(1 if label == "cls-name-gender" else 3),
                            "context": m.group(0),
                        }
                    )
                if name and not any(ch.isdigit() for ch in name):
                    entities.append(
                        {
                            "type": "PERSON",
                            "text": name,
                            "start": m.start(2 if label == "cls-name-gender" else 1),
                            "end": m.end(2 if label == "cls-name-gender" else 1),
                            "context": m.group(0),
                        }
                    )
                if gender:
                    entities.append(
                        {
                            "type": "GENDER",
                            "text": gender,
                            "start": gender_start,
                            "end": gender_end,
                            "context": m.group(0),
                        }
                    )

        # 身份证（关键词优先）
        id_kw_patterns = [
            r"(?:身份证号码|身份证号|身份证)[:：]?\s*(\d{17}[\dXx])",
        ]
        for p in id_kw_patterns:
            for m in re.finditer(p, text):
                add_entity("ID_NUMBER", m)

        # 微信号/QQ号（请假单常见）
        for m in re.finditer(r"(?:微信号|微信)[:：]?\s*([A-Za-z][A-Za-z0-9_-]{4,20})", text):
            add_entity("WECHAT", m)
        for m in re.finditer(r"(?:QQ号|QQ)[:：]?\s*(\d{5,12})", text):
            add_entity("QQ", m)

        # 邮箱（已有 EMAIL，但请假单可能有前缀 E-mail:_）
        for m in re.finditer(r"(?:E-?mail)[:：]?\s*[_ ]*([A-Za-z0-9_.+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9\-.]+)", text, flags=re.IGNORECASE):
            add_entity("EMAIL", m)

        # 手机号（允许前缀下划线）
        for m in re.finditer(r"(?:本人手机号|手机号|手机)[:：]?\s*[_ ]*(1[3-9]\d{9})", text):
            add_entity("PHONE", m)

        # 班级（更宽松，含“级/班/专业班”等）
        for m in re.finditer(r"(?:班级|所在班级|行政班)[:：]?\s*([A-Za-z0-9一-龥]{2,20})", text):
            add_entity("CLASS", m)

        # 学院/院系（key-value 形式）
        for m in re.finditer(r"(?:学院|院系|系所|院系名称)[:：]?\s*([^\n\r]{2,40})", text):
            val = m.group(1).strip()
            if val and not self._is_label_like(val):
                add_entity("COLLEGE_NAME", m)

        # 专业（key-value 形式，避免抓到"贷款信息..."）
        for m in re.finditer(r"(?:专业|专业名称)[:：]?\s*([^\n\r]{2,40})", text):
            val = m.group(1).strip()
            if val and ("贷款" not in val) and not self._is_label_like(val):
                add_entity("MAJOR_NAME", m)

        # 合同编号/贷款合同编号
        for m in re.finditer(r"(?:贷款合同编号|合同编号)[:：]?\s*([A-Za-z0-9\-]{6,40})", text):
            add_entity("LOAN_CONTRACT_NO", m)

        # 回执验证码/在线验证码（4~10位纯数字）
        for m in re.finditer(r"(?:回执验证码|在线验证码|验证码)[为是]?[:：]?\s*(\d{4,10})", text):
            add_entity("VERIFY_CODE", m)

        # 银行账号（关键词优先）
        for m in re.finditer(r"(?:银行账号|账号|卡号)[:：]?\s*(\d{6,30})", text):
            add_entity("BANK_ACCOUNT_NO", m)

        # 开户行/贷款银行
        for m in re.finditer(r"(?:开户行|开户银行|贷款银行|银行)[:：]?\s*([^\n\r]{4,60})", text):
            val = m.group(1).strip()
            if val and ("学院" not in val) and not self._is_label_like(val):
                add_entity("LOAN_BANK_NAME", m)

        # 学号（关键词优先，允许中间有空格/下划线/非数字噪声）
        for m in re.finditer(r"(?:学号|学籍号|学生编号)\s*[:：]?\s*[_\s]*([0-9]{6,12})", text):
            add_entity("STUDENT_ID", m)

        # 兜底：窗口扫描（学号 后 0~10 字符内出现 6~12 位数字）
        for m in re.finditer(r"学号.{0,10}?([0-9]{6,12})", text):
            add_entity("STUDENT_ID", m)

        # 其他通用模式
        for m in re.finditer(r"\b(\d{17}[\dXx])\b", text):
            add_entity("ID_NUMBER", m)
        for m in re.finditer(r"\b(\d{8,15})\b(?=.*学号)", text):
            add_entity("STUDENT_ID", m)
        for m in re.finditer(r"(\d{4}[-年/\.]\d{1,2}[-月/\.]?\d{1,2}[日]?)", text):
            add_entity("DATE", m)
        for m in re.finditer(r"(?<!\d)(1[3-9]\d{9})(?!\d)", text):
            add_entity("PHONE", m)
        for m in re.finditer(r"([A-Za-z0-9_.+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9\-.]+)", text):
            add_entity("EMAIL", m)
        for m in re.finditer(r"([A-Za-z0-9一-龥]{2,10})", text):
            add_entity("CLASS", m)
        for m in re.finditer(r"([一-龥]{1,4})", text):
            add_entity("NATION", m)
        for m in re.finditer(r"(男|女)", text):
            add_entity("GENDER", m)
        for m in re.finditer(r"(在籍|在校|毕业|结业|肄业|休学|退学)", text):
            add_entity("STATUS_ACADEMIC", m)
        for m in re.finditer(r"(?:分院|学院|院系|系所)[:：]?\s*([^\s\n]{2,30})", text):
            add_entity("COLLEGE_NAME", m)
        for m in re.finditer(r"(?:专业)[:：]?\s*([^\s\n]{2,30})", text):
            add_entity("MAJOR_NAME", m)
        for m in re.finditer(r"(?:层次|学历类别|学习形式)[:：]?\s*([^\s\n]{2,20})", text):
            add_entity("PROGRAM_LEVEL", m)
        for m in re.finditer(r"(?:入学日期|入学时间|入学)[:：]?\s*([0-9]{4}[-年/\.][0-9]{1,2}[-月/\.]?[0-9]{0,2})", text):
            add_entity("ENROLL_DATE", m)
        for m in re.finditer(r"(?:预计毕业日期|毕业日期|毕业时间)[:：]?\s*([0-9]{4}[-年/\.][0-9]{1,2}[-月/\.]?[0-9]{0,2})", text):
            add_entity("EXPECTED_GRAD_DATE", m)
        for m in re.finditer(r"(?:贷款金额|贷款数额|借款金额)[:：]?\s*([0-9]+(?:\.[0-9]+)?)", text):
            add_entity("LOAN_AMOUNT", m)
        for m in re.finditer(r"(?:用款期限|贷款期限|贷款年限|还款期限)[:：]?\s*([0-9]{1,3})", text):
            add_entity("LOAN_TERM_MONTH", m)
        for m in re.finditer(r"(?:贷款年度|贷款年度)[:：]?\s*([0-9]{4})", text):
            add_entity("LOAN_YEAR", m)
        for m in re.finditer(r"(?:贷款银行|开户行|银行)[:：]?\s*([^\s\n]{2,40})", text):
            add_entity("LOAN_BANK_NAME", m)
        for m in re.finditer(r"(?:合同编号|贷款合同编号)[:：]?\s*([A-Za-z0-9\\-]{4,40})", text):
            add_entity("LOAN_CONTRACT_NO", m)
        for m in re.finditer(r"(国家奖学金|励志奖学金|奖学金|助学金|资助|困难补助|减免)", text):
            add_entity("SCHOLARSHIP_NAME", m, 1)
        for m in re.finditer(r"(?:奖学金金额|资助金额|金额)[:：]?\s*([0-9]+(?:\.[0-9]+)?)", text):
            add_entity("SCHOLARSHIP_AMOUNT", m)
        for m in re.finditer(r"(?:申请类别|申请类型|请假类别|请假类型)[:：]?\s*([^\s\n]{2,10})", text):
            add_entity("APPLICATION_TYPE", m)
        for m in re.finditer(r"(?:请假|事由|原因|申请理由)[:：]?\s*([^\n]{2,60})", text):
            add_entity("APPLICATION_REASON", m)
        for m in re.finditer(r"(?:开始日期|起始日期|请假开始|休学开始)[:：]?\s*([0-9]{4}[-年/\.][0-9]{1,2}[-月/\.]?[0-9]{0,2})", text):
            add_entity("LEAVE_START_DATE", m)
        for m in re.finditer(r"(?:结束日期|终止日期|请假结束|休学结束)[:：]?\s*([0-9]{4}[-年/\.][0-9]{1,2}[-月/\.]?[0-9]{0,2})", text):
            add_entity("LEAVE_END_DATE", m)
        for m in re.finditer(r"(?:证书编号|编号)[:：]?\s*([A-Za-z0-9\\-]{4,40})", text):
            add_entity("CERT_ID", m)
        for m in re.finditer(r"(?:签发单位|发证单位|发放单位|主管单位)[:：]?\s*([^\s\\n]{2,40})", text):
            add_entity("DOC_ISSUER_ORG", m)
        for m in re.finditer(r"(?:地址|住址|联系地址)[:：]?\s*([^\\n]{6,80})", text):
            add_entity("ADDRESS_FULL", m)
        for m in re.finditer(r"(?:账号|银行账号|卡号)[:：]?\s*([0-9]{6,30})", text):
            add_entity("BANK_ACCOUNT_NO", m)
        for m in re.finditer(r"(?:回执验证码|在线验证码|验证码)[为是]?[:：]?\s*(\d{4,10})", text):
            add_entity("VERIFY_CODE", m)

        # ===== 请假/销假专项 =====
        # 起止日期（包含“自...至...”等范围）
        try:
            s, e = extract_date_range(text)
            if s:
                entities.append({"type": "LEAVE_START_DATE", "text": s, "start": 0, "end": 0, "context": "date_range"})
            if e:
                entities.append({"type": "LEAVE_END_DATE", "text": e, "start": 0, "end": 0, "context": "date_range"})
        except Exception:
            pass

        # 兼容“自 2026年01月02日至 2026年01月05日”这类带“日”后缀的范围格式
        if not any(e.get("type") == "LEAVE_START_DATE" for e in entities):
            m = re.search(
                r"自\s*([0-9]{4}[^\d]{0,2}[0-9]{1,2}[^\d]{0,2}[0-9]{1,2}日?)\s*(?:至|到|~|\-|—)\s*([0-9]{4}[^\d]{0,2}[0-9]{1,2}[^\d]{0,2}[0-9]{1,2}日?)",
                text,
            )
            if m:
                s = normalize_date(m.group(1)) or m.group(1)
                e = normalize_date(m.group(2)) or m.group(2)
                entities.append({"type": "LEAVE_START_DATE", "text": s, "start": m.start(1), "end": m.end(1), "context": m.group(0)})
                entities.append({"type": "LEAVE_END_DATE", "text": e, "start": m.start(2), "end": m.end(2), "context": m.group(0)})

        for m in re.finditer(r"(?:外出地点|外出地点、时间|外出地点|地点)[:：]?\s*([^\n]{2,80})", text):
            add_entity("LEAVE_LOCATION", m)

        for m in re.finditer(r"(?:拟返校时间|返校时间|拟返校)[:：]?\s*([^\n]{4,40})", text):
            add_entity("RETURN_DATE", m)

        for m in re.finditer(r"(?:请假(?:离校)?原因|请假事由|事由|原因)[:：]?\s*([^\n]{2,120})", text):
            add_entity("LEAVE_REASON", m)

        for m in re.finditer(r"(?:请假天数|请假\s*天数|天数)[:：]?\s*([0-9]{1,3}(?:\.[0-9])?)", text):
            add_entity("LEAVE_DAYS", m)

        for m in re.finditer(r"(?:指导教师|指导老师)[:：]?\s*([\u4e00-\u9fa5]{2,6})", text):
            add_entity("TEACHER", m)

        for m in re.finditer(r"(?:辅导员(?:老师)?|辅导员)[:：]?\s*([\u4e00-\u9fa5]{2,6})", text):
            add_entity("COUNSELOR", m)

        for m in re.finditer(r"(?:家长姓名及手机|家长姓名(?:及手机)?|家长姓名)[:：]?\s*([\u4e00-\u9fa5]{2,6})", text):
            add_entity("PARENT_NAME", m)
        for m in re.finditer(r"(?:家长姓名及手机|家长手机|家长电话|家长联系方式)[:：]?\s*(?:[\u4e00-\u9fa5]{2,6}\s*)?[_ ]*(1[3-9]\d{9})", text):
            add_entity("PARENT_PHONE", m)

        # 在身份证/学号/手机号等 add_entity 后做一次归一化清洗（避免后续把 _184.. 这类当值）
        for e in entities:
            if e.get("type") == "PHONE":
                nv = normalize_phone(e.get("text"))
                if nv:
                    e["text"] = nv
            if e.get("type") == "PARENT_PHONE":
                nv = normalize_phone(e.get("text"))
                if nv:
                    e["text"] = nv
            if e.get("type") == "STUDENT_ID":
                nv = normalize_student_id(e.get("text"))
                if nv:
                    e["text"] = nv
            if e.get("type") == "ID_NUMBER":
                nv = normalize_id_number(e.get("text"))
                if nv:
                    e["text"] = nv

        # 去重
        unique = []
        seen = set()
        for e in entities:
            key = (e["type"], e["text"], e["start"], e["end"])
            if key not in seen:
                seen.add(key)
                unique.append(e)

        logger.debug(
            "Entity extraction counts",
            total=len(unique),
            persons=len([e for e in unique if e["type"] == "PERSON"]),
            orgs=len([e for e in unique if e["type"] == "ORG"]),
            dates=len([e for e in unique if e["type"] == "DATE"]),
        )

        if use_cache:
            self._entity_cache_text = text
            self._entity_cache_result = unique

        return unique

    def map_entities_to_slots(self, entities: List[Dict], text: str) -> List[Dict]:
        """把实体映射为字段候选，供上层融合。"""

        def pick_by_type(t: str) -> Optional[str]:
            for e in entities or []:
                if e.get("type") == t:
                    v = self._clean_kv_value(e.get("text"))
                    if v and not self._is_bad_field_value(v):
                        return v
            return None

        mapped: List[Dict] = []

        # 基础字段
        name = pick_by_type("PERSON")
        if name:
            mapped.append({"field_name": "name", "field_type": "PERSON", "field_value": name, "confidence": 0.85, "status": "ok"})

        sid = pick_by_type("STUDENT_ID")
        if sid:
            mapped.append({"field_name": "student_id", "field_type": "STUDENT_ID", "field_value": sid, "confidence": 0.85, "status": "ok"})

        idno = pick_by_type("ID_NUMBER")
        if idno:
            mapped.append({"field_name": "id_number", "field_type": "ID_NUMBER", "field_value": idno, "confidence": 0.9, "status": "ok"})

        phone = pick_by_type("PHONE")
        if phone:
            mapped.append({"field_name": "phone", "field_type": "PHONE", "field_value": phone, "confidence": 0.85, "status": "ok"})

        # 请假/销假字段
        reason = pick_by_type("LEAVE_REASON")
        if reason:
            mapped.append({"field_name": "reason", "field_type": "TEXT", "field_value": reason, "confidence": 0.8, "status": "ok"})

        s_date = pick_by_type("LEAVE_START_DATE")
        if s_date:
            mapped.append({"field_name": "start_date", "field_type": "DATE", "field_value": normalize_date(s_date) or s_date, "confidence": 0.75, "status": "ok"})

        e_date = pick_by_type("LEAVE_END_DATE")
        if e_date:
            mapped.append({"field_name": "end_date", "field_type": "DATE", "field_value": normalize_date(e_date) or e_date, "confidence": 0.75, "status": "ok"})

        days = pick_by_type("LEAVE_DAYS")
        if days:
            mapped.append({"field_name": "days", "field_type": "NUMBER", "field_value": days, "confidence": 0.7, "status": "ok"})

        loc = pick_by_type("LEAVE_LOCATION")
        if loc:
            mapped.append({"field_name": "address", "field_type": "ADDRESS", "field_value": loc, "confidence": 0.7, "status": "ok"})

        return_date = pick_by_type("RETURN_DATE")
        if return_date:
            mapped.append({"field_name": "expected_grad_date", "field_type": "DATE", "field_value": normalize_date(return_date) or return_date, "confidence": 0.6, "status": "ok"})

        parent_phone = pick_by_type("PARENT_PHONE")
        if parent_phone:
            mapped.append({"field_name": "contact", "field_type": "PHONE", "field_value": parent_phone, "confidence": 0.6, "status": "ok"})

        teacher = pick_by_type("TEACHER")
        if teacher:
            mapped.append({"field_name": "issuer", "field_type": "PERSON", "field_value": teacher, "confidence": 0.55, "status": "ok"})

        counselor = pick_by_type("COUNSELOR")
        if counselor:
            mapped.append({"field_name": "admission_school", "field_type": "PERSON", "field_value": counselor, "confidence": 0.55, "status": "ok"})

        # 贷款/奖助金额
        money = pick_by_type("MONEY_AMOUNT")
        if money:
            amt = parse_money_amount(money)
            mapped.append({"field_name": "loan_amount", "field_type": "NUMBER", "field_value": amt if amt is not None else money, "confidence": 0.7, "status": "ok"})

        term_m = pick_by_type("LOAN_TERM_MONTH")
        if term_m:
            years = months_to_years(term_m)
            mapped.append({"field_name": "loan_years", "field_type": "NUMBER", "field_value": years if years is not None else term_m, "confidence": 0.65, "status": "ok"})

        return mapped

    # ---------------------
    # Key-Value 对齐（高校文档通用）
    # ---------------------
    def align_key_values(self, text: str, lines: Optional[List[str]] = None) -> List[Dict]:
        """
        识别 key/value 并对齐，返回[{key, value, confidence}].
        """
        if not lines:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        kv_keywords = [
            # 基本信息
            "姓名",
            "学生姓名",
            "性别",
            "出生日期",
            "民族",
            "身份证号码",
            "身份证号",
            "户籍地址",
            # 学校信息
            "学校名称",
            "学校",
            "学院",
            "分院",
            "系所",
            "院系名称",
            "专业",
            "专业名称",
            "层次",
            "学制",
            "学历类别",
            "学习形式",
            "学籍状态",
            "入学日期",
            "入学年份",
            "预计毕业日期",
            "在线验证码",
            "验证网址",
            "就读高校",
            # 助学贷款相关
            "贷款合同编号",
            "用款期限",
            "本次用款金额",
            "账户名称",
            "账号",
            "开户行",
            "回执验证码",
        ]
        key_union = "|".join([re.escape(k) for k in kv_keywords])
        # 使用非贪婪匹配 + 字符类限制，防止 ReDoS 攻击
        # 限制 value 部分最多匹配 100 个非换行字符
        inline_pattern = (
            rf"({key_union})\s*[:：]?\s*([^\n\r]{{1,100}}?)(?=\s*(?:{key_union})\s*[:：]?|$)"
        )
        pairs = []
        for idx, ln in enumerate(lines):
            # 单行多对 key/value（靠关键字串联）
            for m_inline in re.finditer(inline_pattern, ln):
                k = m_inline.group(1).strip("：: ")
                v = self._clean_value(m_inline.group(2))
                if k and v and not self._is_possible_key(v):
                    pairs.append({"key": k, "value": v, "confidence": 0.85})
            # 内联 key:value
            m = re.match(r"(.{1,20})[:：]\s*(.+)", ln)
            if m and self._is_possible_key(m.group(1)):
                k = m.group(1).strip()
                v = self._clean_value(m.group(2))
                if v:
                    pairs.append(
                        {"key": k, "value": v, "confidence": 0.9}
                    )
                    continue
            # 同一行以空格分隔的 key value（如“姓名 周宇清”）
            if self._is_possible_key(ln) and (" " in ln or "\t" in ln):
                parts = re.split(r"[:：\s]+", ln, maxsplit=1)
                if len(parts) == 2:
                    k = parts[0].strip("：: ")
                    v = self._clean_value(parts[1])
                    if k and v and not self._is_possible_key(v):
                        pairs.append(
                            {"key": k, "value": v, "confidence": 0.82}
                        )
                        continue
            # key 在当前行，值在下一行
            if self._is_possible_key(ln) and idx + 1 < len(lines):
                nxt = self._clean_value(lines[idx + 1])
                if nxt and not self._is_possible_key(nxt):
                    pairs.append(
                        {
                            "key": ln.strip("：:"),
                            "value": nxt,
                            "confidence": 0.8,
                        }
                    )
        # 去重
        unique = []
        seen = set()
        for p in pairs:
            key = (p["key"], p["value"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    @staticmethod
    def _is_possible_key(text: str) -> bool:
        text = text.strip("：: \t")
        if not text or len(text) > 20:
            return False
        keywords = [
            "姓名",
            "性别",
            "出生",
            "民族",
            "学校",
            "学院",
            "分院",
            "系所",
            "专业",
            "层次",
            "学制",
            "学历",
            "学习形式",
            "学籍状态",
            "入学",
            "毕业",
            "验证码",
            "验证网址",
            "课程",
            "成绩",
            "学年",
            "学期",
            "学分",
        ]
        return any(k in text for k in keywords)

    @staticmethod
    def _clean_value(val: str) -> str:
        v = val.strip()
        v = re.sub(r"[\"'<>]", "", v)
        return v if v else ""

    def _parse_table_html_as_grid(self, html: str) -> List[Dict]:
        """
        将 PP-Structure 返回的 html 转为行列结构[{row: int, cells: [...]}, ...]
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            rows = []
            for i, tr in enumerate(soup.find_all("tr")):
                cells = []
                for td in tr.find_all(["td", "th"]):
                    txt = td.get_text(separator=" ", strip=True)
                    txt = self._fix_broken_numbers(txt)
                    cells.append(txt)
                if cells:
                    rows.append({"row": i, "cells": cells})
            return rows
        except Exception:
            # fallback:简单按行分割
            lines = [
                ln.strip() for ln in re.split(r"\n+", html) if ln.strip()
            ]
            return [
                {"row": i, "cells": [self._fix_broken_numbers(ln)]}
                for i, ln in enumerate(lines)
            ]

    def _extract_transcript_from_tables(
        self, tables: List[List[Dict]]
    ) -> List[Dict]:
        """
        从解析后的表格 grid 提取成绩表行：学期/课程/学分/成绩/等级
        """
        results = []
        for tbl in tables:
            # tbl 为 [{row:int, cells:[...]}]
            for row in tbl:
                cells = row.get("cells") or []
                if len(cells) < 3:
                    continue
                rec = {
                    "semester": None,
                    "course_name": None,
                    "credit": None,
                    "score": None,
                    "grade": None,
                    "raw": cells,
                }
                # 简单映射：首列学期/课程，后续查找数字/等级
                rec["semester"] = (
                    cells[0] if re.search(r"\d{4}", cells[0]) else None
                )
                rec["course_name"] = cells[1] if len(cells) > 1 else None
                # 查找学分/成绩/等级
                for c in cells:
                    if (
                        rec["credit"] is None
                        and re.fullmatch(r"\d+(\.\d+)?", c)
                    ):
                        rec["credit"] = c
                        continue
                    if (
                        rec["score"] is None
                        and re.fullmatch(r"\d{1,3}(\.\d+)?", c)
                    ):
                        rec["score"] = c
                        continue
                    if (
                        rec["grade"] is None
                        and re.fullmatch(r"[A-F][+-]?", c, re.IGNORECASE)
                    ):
                        rec["grade"] = c
                if rec["semester"] is None and rec["course_name"] is None:
                    rec["course_name"] = cells[0]
                # 只在除 raw 之外至少有一个字段不为空时才收集
                if any(k != "raw" and v is not None for k, v in rec.items()):
                    results.append(rec)
        return results

    def parse_table_plain(self, table_lines: List[str]) -> List[List[Dict]]:
        """
        从无表格线的对齐行解析为 grid，每行 cells 数组。
        """
        if not table_lines:
            return []
        rows = []
        for ln in table_lines:
            # 跳过噪声行
            if len(ln) < 4:
                continue
            # 按空格/制表/竖线分割
            parts = re.split(r"[|\t]", ln)
            if len(parts) == 1:
                parts = re.split(r"\s{2,}", ln)
            cells = [p.strip() for p in parts if p.strip()]
            if len(cells) < 2:
                continue
            rows.append({"row": len(rows), "cells": cells})
        return [rows]

    # ---------------------
    # 文档分块与模式推断
    # ---------------------
    def segment_document(self, text: str) -> Dict:
        """
        粗略分块：header（前 3 行）、table（检测线性对齐行）、meta（含日期/编号/盖章词）、body（其余）
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        header = lines[:3]
        body_lines: List[str] = []
        table_lines: List[str] = []
        meta_lines: List[str] = []
        for ln in lines:
            if re.search(r"日期|时间|编号|盖章|证明", ln):
                meta_lines.append(ln)
            digit_ratio = sum(ch.isdigit() for ch in ln) / max(len(ln), 1)
            separators = (
                ln.count(" ")
                + ln.count("\t")
                + ln.count("|")
                + ln.count(",")
            )
            if separators >= 3 or digit_ratio > 0.35:
                table_lines.append(ln)
            else:
                body_lines.append(ln)
        return {
            "header": header,
            "meta": meta_lines,
            "table": table_lines,
            "body": body_lines,
        }

    def infer_doc_pattern(self, text: str, segments: Dict) -> Dict:
        """
        根据信号词与分块特征推断 Doc Pattern。
        """
        doc_type = self._classify_doc_type_free(text).get(
            "document_type", "未知"
        )
        has_table = bool(segments.get("table"))
        pattern = {
            "doc_type": doc_type,
            "has_table": has_table,
            "signals": {
                "header_keywords": [
                    ln
                    for ln in segments.get("header", [])
                    if any(k in ln for k in ["通知", "证明", "成绩", "请假"])
                ],
                "meta_keywords": [
                    ln
                    for ln in segments.get("meta", [])
                    if any(k in ln for k in ["编号", "日期", "盖章"])
                ],
            },
        }
        return pattern

    def extract_fields(
        self, text: str, template_config: Dict, file_id: Optional[int] = None
    ) -> Dict:
        """
        根据模板配置（包含字段、正则、校验规则）提取结构化字段。
        """
        start_time = time.time()
        fields = template_config.get("fields", [])
        extract_details = []
        extracted_fields = 0
        total_confidence = 0.0

        logger.info(
            "NLP extract_fields start",
            file_id=file_id,
            template_id=template_config.get("template_id"),
            total_fields=len(fields),
        )

        for field_config in fields:
            logger.debug(
                "NLP extracting field",
                field_name=field_config.get("name"),
                field_type=field_config.get("type"),
                patterns_count=len(field_config.get("patterns", [])),
            )
            field_result = self._extract_single_field(text, field_config)
            if field_result["field_value"] is not None:
                extracted_fields += 1
                total_confidence += field_result["confidence"]
            extract_details.append(field_result)

        avg_confidence = (
            total_confidence / extracted_fields if extracted_fields else 0.0
        )
        processing_time = time.time() - start_time

        result = {
            "file_id": file_id,
            "template_id": template_config.get("template_id"),
            "extract_main": {
                "total_fields": len(fields),
                "extracted_fields": extracted_fields,
                "confidence": round(avg_confidence, 4),
                "status": "success" if extracted_fields > 0 else "failed",
            },
            "extract_details": extract_details,
            "processing_time": round(processing_time, 2),
        }

        logger.info(
            "NLP extract_fields done",
            extracted_fields=extracted_fields,
            total_fields=len(fields),
            confidence=result["extract_main"]["confidence"],
            processing_time=result["processing_time"],
        )
        return result

    def _extract_single_field(self, text: str, field_config: Dict) -> Dict:
        """
        单字段抽取流程：先走正则 -> 必填则尝试 NER -> 校验 -> 打分。
        """
        field_name = field_config["name"]
        field_type = field_config["type"]
        patterns = field_config.get("patterns", [])
        required = field_config.get("required", False)
        validation = field_config.get("validation", {})

        field_value, confidence, position = self._extract_by_patterns(
            text, patterns, field_type
        )

        if field_value is None and required and field_type in [
            "PERSON",
            "ORG",
            "DATE",
        ]:
            logger.debug(
                "Field not found by regex, fallback to NER",
                field_name=field_name,
            )
            field_value, confidence, position = self._extract_by_ner(
                text, field_type
            )

        validation_result = self._validate_field(
            field_value, validation, field_type
        )
        if not validation_result["is_valid"]:
            confidence *= 0.5

        logger.debug(
            "NLP field result",
            field_name=field_name,
            value_preview=str(field_value)[:50]
            if field_value is not None
            else None,
            confidence=confidence,
            required=required,
            validation_pass=validation_result["is_valid"],
        )
        return {
            "field_name": field_name,
            "field_value": field_value,
            "field_type": field_type,
            "confidence": round(confidence, 4),
            "source_position": position,
            "validation_result": validation_result,
        }

    def _extract_by_patterns(
        self, text: str, patterns: List[str], field_type: str
    ) -> tuple:
        """
        使用正则按顺序匹配，命中即返回。
        """
        for pattern in patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(
                        1) if match.groups() else match.group(0)
                    value = value.strip()
                    start = match.start()
                    end = match.end()
                    context = text[
                        max(0, start - 20): min(len(text), end + 20)
                    ]
                    position = {
                        "start": start,
                        "end": end,
                        "context": context,
                    }
                    logger.debug(
                        "Regex matched",
                        pattern=pattern,
                        start=start,
                        end=end,
                        value_preview=value[:50],
                    )
                    return value, 0.9, position
            except re.error as exc:
                logger.warning(f"Invalid regex pattern '{pattern}': {exc}")
                continue
        return None, 0.0, None

    def _extract_by_ner(self, text: str, entity_type: str) -> tuple:
        """
        简单 NER 回退：复用实体抽取结果。
        """
        entities = self.extract_entities(text)
        matches = [e for e in entities if e["type"] == entity_type]
        if not matches:
            return None, 0.0, None
        entity = matches[0]
        position = {
            "start": entity.get("start"),
            "end": entity.get("end"),
            "context": entity.get("context"),
        }
        return entity["text"], 0.85, position

    def _validate_field(
        self, value: Any, validation_rules: Dict, field_type: str
    ) -> Dict:
        """
        执行字段级验证规则，失败会降低置信度。
        """
        if value is None:
            return {"is_valid": False, "message": "Value not found"}
        try:
            if "pattern" in validation_rules:
                if not re.match(validation_rules["pattern"], str(value)):
                    return {
                        "is_valid": False,
                        "message": f"Value does not match pattern: {validation_rules['pattern']}",
                    }

            if "min_length" in validation_rules and len(str(value)) < validation_rules["min_length"]:
                return {
                    "is_valid": False,
                    "message": f"Value too short (min: {validation_rules['min_length']})",
                }

            if "max_length" in validation_rules and len(str(value)) > validation_rules["max_length"]:
                return {
                    "is_valid": False,
                    "message": f"Value too long (max: {validation_rules['max_length']})",
                }

            if field_type == "NUMBER":
                try:
                    num_value = float(value)
                    if "min" in validation_rules and num_value < validation_rules["min"]:
                        return {
                            "is_valid": False,
                            "message": f"Value below minimum ({validation_rules['min']})",
                        }
                    if "max" in validation_rules and num_value > validation_rules["max"]:
                        return {
                            "is_valid": False,
                            "message": f"Value above maximum ({validation_rules['max']})",
                        }
                except ValueError:
                    return {
                        "is_valid": False,
                        "message": "Invalid numeric value",
                    }

            if "enum" in validation_rules and value not in validation_rules["enum"]:
                return {
                    "is_valid": False,
                    "message": f"Value not in allowed list: {validation_rules['enum']}",
                }

            return {"is_valid": True, "message": "验证通过"}
        except Exception as exc:
            logger.error(f"Validation error: {exc}")
            return {"is_valid": False, "message": f"Validation error: {exc}"}

    # 额外的“标签型值”判定：包含冒号且像表头/提示语
    @classmethod
    def _is_bad_field_value(cls, val: Optional[str]) -> bool:
        if not val:
            return True
        s = str(val).strip()
        if not s:
            return True
        if cls._is_label_like(s):
            return True
        # 典型提示语/表头
        if re.fullmatch(r".*(姓名|手机|手机号|联系方式|微信|QQ|邮箱|E-?mail)[:：]?$", s):
            return True
        # 值里仍包含“字段名:”说明没取到真实值
        if re.search(r"(姓名|学号|班级|学院|专业|电话|手机号|地址)[:：]", s) and len(s) <= 20:
            return True
        return False

    @staticmethod
    def _clean_kv_value(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        s = s.replace("_", "").strip()
        # 去掉前缀如“班级：”
        s = re.sub(r"^(?:姓名|学号|班级|专业班级|学院|院系|专业|电话|手机号|地址|家庭地址)[:：]\s*", "", s)
        # 只剩下冒号说明没值
        if s in (":", "："):
            return None
        return s or None

    def classify_document(
        self, text: str, candidates: Optional[List[str]] = None
    ) -> Dict:
        """关键词 + 实体加权的文档类型分类（增强请假单/销假单，避免误判班级成员表）。"""
        logger.info("Document classification requested")
        entities = self.extract_entities(text)
        entity_types = {e.get("type") for e in entities}

        has_leave_keywords = any(
            k in text
            for k in [
                "请假",
                "销假",
                "请、销假",
                "请假离校原因",
                "拟返校时间",
                "辅导员老师",
                "指导老师意见",
                "本人承诺",
            ]
        )
        has_roster_table = (
            all(k in text for k in ["序号", "班级", "姓名", "性别"]) and not has_leave_keywords
        )

        # 维持原有结构：doc_type -> keyword list
        keywords_map = {
            # 高校学业类
            "成绩单": ["成绩单", "成绩表", "课程", "学分", "GPA", "绩点", "总评", "统计时间"],
            "在校证明": ["在校生", "在读", "学生证明", "在校证明", "学籍证明"],
            "学籍信息卡": ["学籍信息卡", "学籍卡", "学籍信息", "学籍状态", "注册学籍", "学籍在线验证报告", "在线验证报告", "预计毕业日期", "在线验证码", "学习形式", "学历类别"],
            "班级成员表": ["班级成员表", "班级成员", "成员名单", "班级名单", "班级表", "班级花名册", "序号", "班级", "姓名", "性别"],
            "住宿表": ["住宿表", "宿舍表", "宿舍名单", "住宿名单", "楼栋", "宿舍号", "寝室", "床位", "宿舍"],
            "考试证书": ["合格证书", "证书编号", "校验码", "查询网址", "全国计算机等级考试", "Certificate Number"],
            "准考证": ["准考证", "准考证号", "报到时间", "考试时间", "考场号", "座位号", "英语四级", "英语六级", "CET"],
            "毕业证书/学历证书": ["毕业证书", "学历证书", "普通高等学校", "经审核准予毕业"],
            "录取凭证": ["录取通知书", "录取通知", "录取学校", "新生", "录取专业"],
            "课程表": ["课程表", "上课时间", "周一", "周二", "节次", "教室"],
            # 贷款 / 奖助学金 / 资助类
            "生源地助学贷款合同": ["生源地助学贷款", "贷款合同编号", "助学贷款在线服务系统", "国家开发银行"],
            "贷款合同": ["贷款合同", "借款合同", "借款人", "还款方式", "贷款金额"],
            "奖学金证明": ["奖学金", "助学金", "励志奖学金", "国家奖学金", "发放", "获奖"],
            "贫困证明": ["贫困", "经济困难", "低保", "家庭经济", "资助", "困难证明"],
            "收入证明": ["收入证明", "工资收入", "薪资", "月收入", "年收入"],
            # 请假/休学申请
            "请假条": ["请假条", "请假申请", "请假", "事假", "病假", "假期", "销假"],
            "休学申请": ["休学申请", "休学", "复学", "休学原因"],
            # 通知/会议类
            "通知": ["通知", "公告", "告知", "决定", "安排"],
            "会议纪要": ["会议纪要", "会议记录", "会议时间", "参会人员", "议题"],
            # 宽泛的证明类兜底 & 合同兜底
            "证明类其他": ["兹证明", "特此证明", "证明", "盖章"],
            "合同": ["合同", "协议", "甲方", "乙方", "签订"],
        }
        weight_map = {
            "请假条": 2.2,
            "班级成员表": 1.0,
            "住宿表": 1.2,
            "考试证书": 1.7,
            "准考证": 1.6,
            "学籍信息卡": 1.5,
        }
        has_exam_keywords = any(
            k in text for k in ("准考证", "准考证号", "报到时间", "考试时间", "考场号", "座位号", "英语四级", "英语六级", "CET")
        )
        has_cert_keywords = any(
            k in text for k in ("合格证书", "证书编号", "校验码", "查询网址", "全国计算机等级考试")
        )
        has_student_card_keywords = any(
            k in text for k in ("学籍在线验证报告", "学籍状态", "预计毕业日期", "在线验证码", "学习形式", "学历类别")
        )

        scores: Dict[str, float] = {}
        for doc_type, kw_list in keywords_map.items():
            keyword_score = sum(1 for kw in kw_list if kw in text)

            # 一点点实体加成
            entity_bonus = 0
            if doc_type in ("成绩单", "课程表"):
                entity_bonus += 2 if "COURSE" in entity_types else 0
                entity_bonus += 2 if "SCORE" in entity_types else 0
            if doc_type in ("在校证明", "学籍信息卡", "请假条"):
                entity_bonus += 2 if "PERSON" in entity_types else 0
                entity_bonus += 2 if "STUDENT_ID" in entity_types else 0
                entity_bonus += 1 if "PHONE" in entity_types else 0

            total_score = (keyword_score + entity_bonus) * float(weight_map.get(doc_type, 1.0))

            if has_leave_keywords and doc_type == "班级成员表":
                total_score *= 0.1
            if has_exam_keywords and doc_type == "班级成员表":
                total_score *= 0.15
            if has_student_card_keywords and doc_type == "班级成员表":
                total_score *= 0.2
            if has_roster_table and doc_type == "班级成员表":
                total_score *= 1.8
            if has_exam_keywords and doc_type == "准考证":
                total_score *= 1.8
            if has_cert_keywords and doc_type == "考试证书":
                total_score *= 1.8
            if has_cert_keywords and doc_type == "准考证":
                total_score *= 0.3
            if has_student_card_keywords and doc_type == "学籍信息卡":
                total_score *= 1.8

            scores[doc_type] = total_score

        best_type = "未知"
        confidence = 0.0
        if scores and max(scores.values()) > 0:
            best_type = max(scores, key=scores.get)
            total = sum(scores.values()) or 1.0
            confidence = float(scores[best_type]) / total

        total = sum(scores.values()) or 1.0
        probabilities = {k: round(v / total, 4) for k, v in scores.items()}

        logger.info("Document classification done", doc_type=best_type, confidence=confidence)
        return {
            "document_type": best_type,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }

    def extract_proof_fields(self, text: str) -> List[Dict]:
        """针对证明/学籍/在校类文本的轻量字段抽取（兼容上层调用）。"""
        results: List[Dict] = []

        def add(field_name: str, field_type: str, value: Optional[str], confidence: float) -> None:
            results.append(
                {
                    "field_name": field_name,
                    "field_type": field_type,
                    "field_value": value if value else None,
                    "confidence": confidence,
                }
            )

        name = self._match_first([r"姓名[:：]?\s*([^\s\n]{2,10})", r"学生姓名[:：]?\s*([^\s\n]{2,10})"], text)
        student_id = self._match_first([r"学号[:：]?\s*[_\s]*([0-9]{6,12})", r"学籍号[:：]?\s*([0-9]{6,12})"], text)
        gender = self._match_first([r"性别[:：]?\s*([男女])"], text)
        birth = self._match_date([r"出生日期[:：]?\s*([0-9]{4}[年\-/.][0-9]{1,2}[月\-/.][0-9]{1,2}日?)"], text)
        school = self._match_first([r"学校名称[:：]?\s*([^\n]{2,40})", r"([^\s]{2,30}(?:大学|学院|学校))"], text)
        college = self._match_first([r"学院[:：]?\s*([^\n]{2,40})", r"院系名称[:：]?\s*([^\n]{2,40})"], text)
        major = self._match_first([r"专业(?:名称)?[:：]?\s*([^\n]{2,40})"], text)
        status = self._match_first([r"学籍状态[:：]?\s*([^\s\n]{2,20})"], text)
        enroll = self._match_date([r"入学(?:日期|时间)?[:：]?\s*([0-9]{4}[年\-/.][0-9]{1,2}[月\-/.]?[0-9]{0,2}日?)"], text)
        grad = self._match_date([r"(?:预计)?毕业(?:日期|时间)?[:：]?\s*([0-9]{4}[年\-/.][0-9]{1,2}[月\-/.]?[0-9]{0,2}日?)"], text)
        vcode = self._match_first([r"(?:在线验证码|回执验证码|验证码)[:：]?\s*(\d{4,10})"], text)
        vurl = self._match_first([r"https?://[^\s]+"], text)

        add("name", "PERSON", self._clean_kv_value(name), 0.95)
        add("student_id", "STUDENT_ID", self._clean_kv_value(student_id), 0.9)
        add("gender", "TEXT", self._clean_kv_value(gender), 0.85)
        add("birth_date", "DATE", birth, 0.8)
        add("school_name", "ORG", self._clean_kv_value(school), 0.85)
        add("college", "ORG", self._clean_kv_value(college), 0.8)
        add("major", "TEXT", self._clean_kv_value(major), 0.8)
        add("status", "TEXT", self._clean_kv_value(status), 0.75)
        add("enroll_date", "DATE", enroll, 0.75)
        add("expected_grad_date", "DATE", grad, 0.75)
        add("verify_code", "TEXT", self._clean_kv_value(vcode), 0.7)
        add("verify_url", "TEXT", self._clean_kv_value(vurl), 0.6)

        return results

    @staticmethod
    def _match_first(patterns: List[str], text: str) -> Optional[str]:
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                if m.groups():
                    return m.group(1).strip()
                return m.group(0).strip()
        return None

    @staticmethod
    def _match_date(patterns: List[str], text: str) -> Optional[str]:
        for p in patterns:
            m = re.search(p, text)
            if m:
                val = m.group(1).strip() if m.groups() else m.group(0).strip()
                return normalize_date(val) or val
        return None

    def extract_relations(self, text: str, entities: Optional[List[Dict]] = None) -> List[Dict]:
        """关系抽取：规则版先保留接口兼容上层调用。

        目前关系层不作为核心输出，默认返回空列表即可避免 500。
        后续如需可在此加入：
        - 请假：PERSON-LEAVE_REASON/DATE_RANGE 的关联
        - 贷款：PERSON-LOAN_CONTRACT_NO/LOAN_AMOUNT 的关联
        """
        _ = text
        _ = entities
        return []

