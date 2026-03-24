"""NLP utility helpers.

This module exists to keep `nlp_service.py` from growing too large.
Only put small, dependency-free helpers here.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """Normalize common Chinese/ISO-ish date formats.

    Supports:
    - yyyy-mm-dd
    - yyyy/m/d
    - yyyy年m月d日

    Returns yyyy-mm-dd, or None when not recognized.
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    if not s:
        return None

    m = re.search(r"(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*[月\-/.]\s*(\d{1,2})", s)
    if m:
        y = m.group(1)
        mo = int(m.group(2))
        d = int(m.group(3))
        return f"{y}-{mo:02d}-{d:02d}"

    # Sometimes only year-month is present; keep as yyyy-mm (optional)
    m = re.search(r"(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*[月\-/.]?", s)
    if m and re.search(r"\d{4}[-/.]\d{1,2}$", s):
        y = m.group(1)
        mo = int(m.group(2))
        return f"{y}-{mo:02d}"

    return None


def normalize_phone(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    digits = re.sub(r"\D", "", str(s))
    return digits if len(digits) == 11 and digits.startswith("1") else None


def normalize_id_number(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    raw = str(s).strip()
    # keep X/x
    m = re.search(r"\b(\d{17}[\dXx])\b", raw)
    if m:
        return m.group(1)
    digits = re.sub(r"\D", "", raw)
    return digits if len(digits) == 18 else None


def normalize_student_id(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    digits = re.sub(r"\D", "", str(s))
    return digits if 6 <= len(digits) <= 12 else None


def parse_money_amount(text: Optional[str]) -> Optional[float]:
    """Parse Chinese money amount in OCR text.

    Examples:
    - 10000
    - 10000.00
    - 10,000.00
    - 10000（元） / 10000元
    """
    if not text:
        return None
    s = str(text)
    s = s.replace(",", "")
    # 兼容：10000元 / 10000（元） / 10000人民币
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\(\s*元\s*\)|元|人民币)?", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def months_to_years(months: Optional[str]) -> Optional[float]:
    if months is None:
        return None
    try:
        m = float(str(months).strip())
        return round(m / 12.0, 2)
    except Exception:
        return None


def extract_date_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract a (start_date, end_date) pair from common patterns.

    Supports:
    - 自 yyyy-mm-dd 至 yyyy-mm-dd
    - yyyy年m月d日- yyyy年m月d日
    - yyyy/mm/dd ~ yyyy/mm/dd
    """
    if not text:
        return (None, None)

    patterns = [
        r"自\s*([0-9]{4}[^\d]{0,2}[0-9]{1,2}[^\d]{0,2}[0-9]{1,2})\s*(?:至|到|~|\-|—)\s*([0-9]{4}[^\d]{0,2}[0-9]{1,2}[^\d]{0,2}[0-9]{1,2})",
        r"([0-9]{4}[^\d]{0,2}[0-9]{1,2}[^\d]{0,2}[0-9]{1,2})\s*(?:至|到|~|\-|—)\s*([0-9]{4}[^\d]{0,2}[0-9]{1,2}[^\d]{0,2}[0-9]{1,2})",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            s = normalize_date(m.group(1))
            e = normalize_date(m.group(2))
            return (s, e)

    return (None, None)
