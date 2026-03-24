"""Compatibility patch helpers for NLPService.

Kept in a tiny module to avoid editing the huge nlp_service.py when the editor mapper
is unstable. Imported and attached to NLPService at module import time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _reconstruct_text(self, ocr_raw: Optional[Dict], fallback_text: str) -> Dict[str, Any]:
    """Reconstruct plain text/lines/tables from OCR raw output.

    This is a *best-effort* helper used by analyze_ocr_text / upstream pipeline.
    It must never raise.
    """
    text = fallback_text or ""
    lines: List[str] = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parsed_tables: List[Any] = []

    if not ocr_raw:
        return {"text": text, "lines": lines, "parsed_tables": parsed_tables}

    try:
        if isinstance(ocr_raw, dict):
            raw_lines = ocr_raw.get("lines")
            if isinstance(raw_lines, list) and raw_lines:
                lines = [str(x).strip() for x in raw_lines if str(x).strip()]
                text = "\n".join(lines)

            # PP-Structure style table html
            for key in ("result", "structure", "data"):
                arr = ocr_raw.get(key)
                if not isinstance(arr, list):
                    continue
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    html = item.get("html")
                    if isinstance(html, str) and "<table" in html.lower():
                        try:
                            parsed_tables.append(self._parse_table_html_as_grid(html))
                        except Exception:
                            continue
    except Exception:
        pass

    return {"text": text, "lines": lines, "parsed_tables": parsed_tables}


def _fix_broken_numbers(self, s: str) -> str:
    """Fix OCR-broken numeric tokens (best-effort)."""
    if not s:
        return s
    # 常见：数字之间被空格断开
    return "".join(str(s).split())


def _template_from_expected(self, doc_type: str, expected_fields: List[Dict]) -> Dict[str, Any]:
    """Build a minimal template config from expected_fields."""
    fields = []
    for f in expected_fields or []:
        fields.append(
            {
                "name": f.get("field_name"),
                "type": f.get("field_type", "TEXT"),
                "required": False,
                "patterns": [],
                "description": f.get("description", ""),
            }
        )
    return {"template_id": f"auto-{doc_type or 'doc'}", "fields": fields}


def _build_fail_analysis(
    self,
    extracted_fields: List[Dict],
    expected_fields: List[Dict],
    quality: Dict,
    doc_type_info: Dict,
) -> Dict[str, Any]:
    """Simple fail analysis used by analyze_ocr_text."""
    expected_names = {f.get("field_name") for f in expected_fields or [] if f.get("field_name")}
    got_names = {f.get("field_name") for f in extracted_fields or [] if f.get("field_value")}
    missing = sorted(list(expected_names - got_names))
    return {
        "doc_type": doc_type_info.get("document_type"),
        "ocr_quality": quality.get("level"),
        "missing_fields": missing,
        "problems": quality.get("problems") or [],
    }


def _suggest_repair(self, quality: Dict, doc_type_info: Dict, text: str) -> List[str]:
    """Return human-readable suggestions (safe default)."""
    suggestions: List[str] = []
    if not text or len(text.strip()) < 10:
        suggestions.append("文本过短，建议重新拍照/提高分辨率")
    if quality and quality.get("level") in ("bad", "critical"):
        suggestions.append("OCR 质量较差：建议开启图像增强/去噪/倾斜矫正")
    if doc_type_info and doc_type_info.get("document_type") == "未知":
        suggestions.append("未能稳定判别文档类型：建议保证标题区域清晰")
    return suggestions
