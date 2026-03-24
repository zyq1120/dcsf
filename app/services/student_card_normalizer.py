"""
Normalization utilities for Chinese student enrollment cards ("学籍卡/学籍信息卡").
This module post-processes noisy OCR + NER output into a clean, flat structure.
Only standard library is used so it can be embedded anywhere in the stack.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------
# Shared patterns and dictionaries
# ---------------------
ID_PATTERN = re.compile(r"\b\d{17}[\dXx]\b")
DATE_PATTERN = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
SCHOOLING_PATTERN = re.compile(r"(?<!\d)\d{1,2}(?:\.\d+)?年\b")
STUDENT_ID_PATTERN = re.compile(r"\b\d{8,12}\b")
VERIFY_CODE_PATTERN = re.compile(r"\b\d{8,12}\b")
CLASS_PATTERN = re.compile(r"[A-Za-z\u4e00-\u9fa5]{1,10}\d{1,4}")

EDUCATION_TYPE_WHITELIST: Sequence[str] = [
    "普通高等教育",
    "成人高等教育",
    "网络教育",
    "开放教育",
    "高等职业教育",
]

NATION_WHITELIST: Sequence[str] = [
    "汉族",
    "回族",
    "藏族",
    "维吾尔族",
    "蒙古族",
    "壮族",
    "苗族",
    "彝族",
    "土家族",
    "满族",
    "朝鲜族",
    "白族",
    "侗族",
    "瑶族",
    "傣族",
    "黎族",
    "哈尼族",
    "哈萨克族",
    "畲族",
    "拉祜族",
    "水族",
    "东乡族",
    "纳西族",
    "景颇族",
    "柯尔克孜族",
    "土族",
    "达斡尔族",
    "仫佬族",
    "羌族",
    "布朗族",
    "撒拉族",
    "毛南族",
    "仡佬族",
    "锡伯族",
    "阿昌族",
    "普米族",
    "塔吉克族",
    "怒族",
    "乌孜别克族",
    "俄罗斯族",
    "鄂温克族",
    "德昂族",
    "保安族",
    "裕固族",
    "京族",
    "塔塔尔族",
    "独龙族",
    "鄂伦春族",
    "赫哲族",
    "门巴族",
    "珞巴族",
    "基诺族",
]

STUDY_MODE_CANDIDATES: Sequence[str] = [
    "普通全日制",
    "全日制",
    "非全日制",
    "业余",
    "函授",
    "开放教育",
    "网络教育",
]

LABELS: Sequence[str] = [
    "姓名",
    "学生姓名",
    "名字",
    "性别",
    "民族",
    "学号",
    "入学日期",
    "入学时间",
    "入学年月",
    "证件号码",
    "身份证号码",
    "学历类别",
    "班级",
    "班级名称",
    "班级名",
    "学制",
    "分院",
    "系所",
    "院系",
    "院系名称",
    "学院",
    "学校",
    "学校名称",
    "专业",
    "专业名称",
    "学习形式",
    "学习方式",
    "培养方式",
    "方向",
]

VERIFY_CODE_LABELS: Sequence[str] = [
    "验证码", "校验码", "验证号码", "核验码", "查询码", "验证代码"]

INVALID_LABEL_VALUES = set(LABELS) | set(VERIFY_CODE_LABELS) | {
    "学制",
    "分院",
    "系所",
    "证件号码",
    "班级",
    "学历类别",
}


def normalize_student_card(raw: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Normalize OCR + NER output of a student enrollment card (学籍卡/学籍信息卡).
    Returns a flat dict with cleaned fields; values are None when not confidently found.
    """
    data = raw.get("data") or {}
    raw_text = str(data.get("text") or raw.get("text") or "")
    entities = data.get("entities") or []
    doc_type = _normalize_doc_type(data)
    label_segments = _find_label_segments(
        raw_text, list(LABELS) + list(VERIFY_CODE_LABELS))

    id_number = extract_id_number(raw_text, entities)
    student_id = extract_student_id(
        raw_text, label_segments, entities, id_number)
    verify_code = extract_verify_code(
        raw_text, id_number, student_id, label_segments)
    enroll_date = extract_enroll_date(raw_text, entities, label_segments)
    schooling_length = extract_schooling_length(raw_text, label_segments)
    class_name = extract_class_name(raw_text, label_segments)
    education_type = extract_education_type(raw_text, label_segments)
    nation = extract_nation(raw_text, label_segments)
    gender = extract_gender(raw_text, label_segments)
    study_mode = extract_study_mode(raw_text, label_segments)
    major = extract_major(raw_text, label_segments, study_mode)
    college, department, school_name = extract_orgs(
        raw_text, entities, label_segments)
    name = extract_name(raw_text, entities, label_segments)

    # If verify_code equals id_number, drop verify_code per rule
    if verify_code and id_number and verify_code == id_number:
        verify_code = None

    if not school_name:
        school_name = college

    return {
        "doc_type": doc_type,
        "name": name,
        "gender": gender,
        "nation": nation,
        "id_number": id_number,
        "student_id": student_id,
        "major": major,
        "class_name": class_name,
        "college": college,
        "department": department,
        "school_name": school_name,
        "education_type": education_type,
        "study_mode": study_mode,
        "enroll_date": enroll_date,
        "schooling_length": schooling_length,
        "verify_code": verify_code,
        "raw_text": raw_text,
    }


# ---------------------
# Helper functions
# ---------------------
def _normalize_doc_type(data: Dict[str, Any]) -> Optional[str]:
    candidates = data.get("document_type_candidates") or {}
    for key in ("学籍信息卡", "学籍卡"):
        if key in candidates:
            return "学籍卡"
    return data.get("document_type")


def _is_invalid_value(value: Optional[str]) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    return not stripped or stripped in INVALID_LABEL_VALUES


def _normalize_date_tuple(groups: Tuple[str, str, str]) -> str:
    year, month, day = groups
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _find_label_segments(text: str, labels: Iterable[str]) -> Dict[str, List[str]]:
    """
    Locate label positions and slice the text between labels as value segments.
    Returns mapping of label -> list of segments following that label.
    """
    positions: List[Tuple[int, int, str]] = []
    for label in labels:
        for match in re.finditer(re.escape(label), text):
            positions.append((match.start(), match.end(), label))
    positions.sort(key=lambda x: x[0])

    segments: Dict[str, List[str]] = {}
    for idx, (start, end, label) in enumerate(positions):
        next_start = positions[idx + 1][0] if idx + \
            1 < len(positions) else len(text)
        segment = text[end:next_start].strip()
        if segment:
            segments.setdefault(label, []).append(segment)
    return segments


def extract_id_number(text: str, entities: Sequence[Dict[str, Any]]) -> Optional[str]:
    """
    Prefer ID_NUMBER entities; fallback to VERIFY_CODE entities that look like ID; then regex on full text.
    """
    for ent in entities:
        if ent.get("type") in ("ID_NUMBER", "VERIFY_CODE"):
            candidate = (ent.get("text") or "").strip()
            if ID_PATTERN.fullmatch(candidate):
                return candidate
            embedded = ID_PATTERN.search(candidate)
            if embedded:
                return embedded.group(0)
    match = ID_PATTERN.search(text)
    return match.group(0) if match else None


def extract_verify_code(
    text: str,
    id_number: Optional[str],
    student_id: Optional[str],
    label_segments: Dict[str, List[str]],
) -> Optional[str]:
    """
    Extract an 8–12 digit verify code that is not the ID number or student ID.
    """
    for label in VERIFY_CODE_LABELS:
        for segment in label_segments.get(label, []):
            match = VERIFY_CODE_PATTERN.search(segment)
            if match:
                candidate = match.group(0)
                if candidate not in {id_number, student_id}:
                    return candidate

    # Conservative fallback: search entire text but skip known numbers
    for match in VERIFY_CODE_PATTERN.finditer(text):
        candidate = match.group(0)
        if candidate in {id_number, student_id}:
            continue
        return candidate
    return None


def extract_enroll_date(
    text: str, entities: Sequence[Dict[str, Any]], label_segments: Dict[str, List[str]]
) -> Optional[str]:
    """
    Prefer date near 入学日期; fallback to date entities; then any date in text.
    """
    for label in ("入学日期", "入学时间", "入学年月"):
        for segment in label_segments.get(label, []):
            match = DATE_PATTERN.search(segment)
            if match:
                # type: ignore[arg-type]
                return _normalize_date_tuple(match.groups())

    # 部分模板会把一坨信息挤在“系所”后面，这里也试一下
    for segment in label_segments.get("系所", []):
        match = DATE_PATTERN.search(segment)
        if match:
            # type: ignore[arg-type]
            return _normalize_date_tuple(match.groups())

    for ent in entities:
        if ent.get("type") == "DATE":
            match = DATE_PATTERN.search(ent.get("text") or "")
            if match:
                # type: ignore[arg-type]
                return _normalize_date_tuple(match.groups())

    match = DATE_PATTERN.search(text)
    if match:
        return _normalize_date_tuple(match.groups())  # type: ignore[arg-type]
    return None


def extract_schooling_length(text: str, label_segments: Dict[str, List[str]]) -> Optional[str]:
    """
    Find 学制 value like '2年'.
    """
    for segment in label_segments.get("学制", []):
        match = SCHOOLING_PATTERN.search(segment)
        if match:
            return match.group(0)

    # 你这类版式经常把“2年 普通高等教育 软升243”都塞到“系所”后面
    for segment in label_segments.get("系所", []):
        match = SCHOOLING_PATTERN.search(segment)
        if match:
            return match.group(0)

    label_window = re.search(r"学制.{0,60}", text)
    if label_window:
        match = SCHOOLING_PATTERN.search(label_window.group(0))
        if match:
            return match.group(0)

    match = SCHOOLING_PATTERN.search(text)
    return match.group(0) if match else None


def extract_student_id(
    text: str,
    label_segments: Dict[str, List[str]],
    entities: Sequence[Dict[str, Any]],
    id_number: Optional[str],
) -> Optional[str]:
    """
    Extract 学号 as 8–12 digit number.
    """
    # 1) 实体中优先 STUDENT_ID / VERIFY_CODE 但排除身份证
    for ent in entities:
        if ent.get("type") in ("STUDENT_ID", "VERIFY_CODE"):
            candidate = (ent.get("text") or "").strip()
            if STUDENT_ID_PATTERN.fullmatch(candidate) and candidate != id_number:
                return candidate

    # 2) 标签“学号”后面的片段
    for segment in label_segments.get("学号", []):
        match = STUDENT_ID_PATTERN.search(segment)
        if match:
            return match.group(0)

    # 3) “系所”后面那一坨里找 8~12 位数字
    for segment in label_segments.get("系所", []):
        match = STUDENT_ID_PATTERN.search(segment)
        if match:
            candidate = match.group(0)
            if candidate != id_number:
                return candidate

    # 4) 文本中 “学号: 242040390” 这种形式
    match = re.search(r"学号[:：]?\s*([\d]{8,12})", text)
    if match:
        candidate = match.group(1)
        if candidate != id_number:
            return candidate

    # 5) 兜底：全局 8–12 位数字，排除身份证
    for candidate in re.findall(r"\b\d{8,12}\b", text):
        if candidate != id_number:
            return candidate

    return None


def _find_class_in_segment(segment: str) -> Optional[str]:
    for match in CLASS_PATTERN.finditer(segment):
        candidate = match.group(0)
        if any(token in candidate for token in ("年", "月", "日")):
            continue
        if not _is_invalid_value(candidate):
            return candidate
    return None


def extract_class_name(text: str, label_segments: Dict[str, List[str]]) -> Optional[str]:
    """
    Extract 班级名称; avoids using labels themselves.
    """
    for label in ("班级", "班级名称", "班级名", "系所"):
        for segment in label_segments.get(label, []):
            candidate = _find_class_in_segment(segment)
            if candidate:
                return candidate

    match = re.search(r"班级[:：]?\s*([A-Za-z\u4e00-\u9fa5]{1,10}\d{1,4})", text)
    if match:
        candidate = match.group(1)
        if not _is_invalid_value(candidate):
            return candidate

    return _find_class_in_segment(text)


def extract_education_type(text: str, label_segments: Dict[str, List[str]]) -> Optional[str]:
    """
    Extract from whitelist; ignores labels like '证件号码'.
    """
    for label in ("学历类别",):
        for segment in label_segments.get(label, []):
            for candidate in EDUCATION_TYPE_WHITELIST:
                if candidate in segment:
                    return candidate

    # 很多模板是把 “普通高等教育 软升243” 放在“系所”后面
    for segment in label_segments.get("系所", []):
        for candidate in EDUCATION_TYPE_WHITELIST:
            if candidate in segment:
                return candidate

    for candidate in EDUCATION_TYPE_WHITELIST:
        if candidate in text:
            return candidate
    return None


def extract_nation(text: str, label_segments: Dict[str, List[str]]) -> Optional[str]:
    """
    Find first matching nation from whitelist.
    """
    for label in ("民族",):
        for segment in label_segments.get(label, []):
            for nation in NATION_WHITELIST:
                if nation in segment:
                    return nation
    for nation in NATION_WHITELIST:
        if nation in text:
            return nation
    return None


def extract_gender(text: str, label_segments: Dict[str, List[str]]) -> Optional[str]:
    """
    Extract gender as 男/女.
    """
    for label in ("性别",):
        for segment in label_segments.get(label, []):
            match = re.search(r"(男|女)", segment)
            if match:
                return match.group(1)
    match = re.search(r"性别[:：]?\s*(男|女)", text)
    return match.group(1) if match else None


def extract_study_mode(text: str, label_segments: Dict[str, List[str]]) -> Optional[str]:
    """
    Extract 学习形式/方式 from a shortlist.
    """
    for label in ("学习形式", "学习方式", "培养方式"):
        for segment in label_segments.get(label, []):
            for mode in STUDY_MODE_CANDIDATES:
                if mode in segment:
                    return mode
    for mode in STUDY_MODE_CANDIDATES:
        if mode in text:
            return mode
    return None


def extract_major(
    text: str,
    label_segments: Dict[str, List[str]],
    study_mode: Optional[str],
) -> Optional[str]:
    """
    Extract major/profession; prefers:
    1) 第一行中 '丨' 左侧的最后一个词（过滤姓名/性别等标签）
    2) 专业相关 label 切片
    3) 在学习形式（普通全日制/非全日制）前面的词
    """

    # 0) 优先：利用首行和分隔符 "丨"
    # 场景：
    #   "姓名 周宇清 性别 男 软件工程丨普通全日制 ..."
    #   "软件工程丨普通全日制 证件号码 入学日期 ..."
    header = text.splitlines()[0] if text else ""
    if "丨" in header:
        left = header.split("丨", 1)[0]
        # 在左半边里抽最后一个“看起来像专业名”的 token
        tokens = re.findall(r"[\u4e00-\u9fa5A-Za-z·]{2,30}", left)
        for token in reversed(tokens):
            token = token.strip()
            if not token:
                continue
            if "学院" in token or "大学" in token:
                continue
            if _is_invalid_value(token):
                continue
            # 例如：["姓名", "周宇清", "性别", "男", "软件工程"] -> 选最后一个 "软件工程"
            return token

    # 1) 其次：基于 label 的切片（“专业”、“专业名称”后面的值）
    for label in ("专业", "专业名称"):
        for segment in label_segments.get(label, []):
            candidate = _pick_major_token(segment)
            if candidate:
                return candidate

    # 2) 再次：利用已识别的学习形式，在其前面找一个专业名
    if study_mode:
        # 在 study_mode 前面的一个 token 看成专业
        # 比如 "软件工程 普通全日制" / "软件工程丨普通全日制"
        pattern = rf"([\u4e00-\u9fa5A-Za-z·]{{2,30}})[|丨｜:/\-－—–】\]\s]*{re.escape(study_mode)}"
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip().strip("|丨｜:/-－—–")
            if candidate and not _is_invalid_value(candidate) and "学院" not in candidate and "大学" not in candidate:
                return candidate

    # 3) 兜底：在 “全日制/非全日制” 前拿一个词当专业
    match = re.search(
        r"([\u4e00-\u9fa5A-Za-z·]{2,30})[】\]）》」】]?\s*(?:全日制|非全日制)",
        text,
    )
    if match:
        candidate = match.group(1).strip()
        if candidate and not _is_invalid_value(candidate) and "学院" not in candidate and "大学" not in candidate:
            return candidate

    return None


def _pick_major_token(segment: str) -> Optional[str]:
    tokens = re.findall(r"[\u4e00-\u9fa5A-Za-z·]{2,30}", segment)
    for token in tokens:
        if "学院" in token or "大学" in token:
            continue
        if not _is_invalid_value(token):
            return token
    return None


def extract_orgs(
    text: str, entities: Sequence[Dict[str, Any]], label_segments: Dict[str, List[str]]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Derive college/department/school_name from ORG-like strings.
    """
    candidates: List[str] = []
    for ent in entities:
        if ent.get("type") == "ORG":
            value = (ent.get("text") or "").strip()
            if value and not _is_invalid_value(value):
                candidates.append(value)

    candidates.extend(_find_org_like_strings(text))
    for label in ("学院", "学校", "学校名称", "院系"):
        for segment in label_segments.get(label, []):
            candidates.extend(_find_org_like_strings(segment))

    candidates = _deduplicate(candidates)

    school_name = _first_contains(candidates, "大学")
    college = _first_contains(candidates, "学院")
    department = None

    # 尝试在“系所/分院/院系”后面的片段里找更细的院系名
    for label in ("系所", "分院", "院系"):
        for segment in label_segments.get(label, []):
            for org in _find_org_like_strings(segment):
                if not _is_invalid_value(org):
                    department = org
                    break
            if department:
                break
        if department:
            break

    if department is None:
        window_match = re.search(
            r"(系所|分院)[:：]?\s*(.{0,120})", text, flags=re.DOTALL)
        if window_match:
            window = window_match.group(2)
            org = re.search(r"[\u4e00-\u9fa5A-Za-z·]{2,40}学院", window)
            if org:
                dept_candidate = org.group(0).strip()
                if not _is_invalid_value(dept_candidate):
                    department = dept_candidate

    return college, department, school_name


def _find_org_like_strings(text: str) -> List[str]:
    return [
        match.group(0)
        for match in re.finditer(r"[\u4e00-\u9fa5A-Za-z·]{2,40}(大学|学院)", text)
        if not _is_invalid_value(match.group(0))
    ]


def _first_contains(candidates: Sequence[str], keyword: str) -> Optional[str]:
    for cand in candidates:
        if keyword in cand:
            return cand
    return None


def _deduplicate(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def extract_name(
    text: str, entities: Sequence[Dict[str, Any]], label_segments: Dict[str, List[str]]
) -> Optional[str]:
    """
    Extract 姓名, preferring PERSON entities then label segments.
    """
    for ent in entities:
        if ent.get("type") == "PERSON":
            candidate = (ent.get("text") or "").strip()
            if not _is_invalid_value(candidate):
                return candidate

    for label in ("姓名", "学生姓名", "名字"):
        for segment in label_segments.get(label, []):
            match = re.search(r"([\u4e00-\u9fa5·]{2,8})", segment)
            if match:
                candidate = match.group(1)
                if not _is_invalid_value(candidate):
                    return candidate

    match = re.search(r"姓名[:：]?\s*([\u4e00-\u9fa5·]{2,8})", text)
    if match:
        candidate = match.group(1)
        if not _is_invalid_value(candidate):
            return candidate
    return None


# ---------------------
# Demo
# ---------------------
if __name__ == "__main__":
    from pprint import pprint

    sample_raw = {
        "code": 200,
        "data": {
            "document_type": "学籍卡",
            "document_type_candidates": {
                "学籍信息卡": 1,
                "学籍卡": 0.95,
                "其他": 0.001,
                "在校证明": 0.01,
            },
            "entities": [
                {"text": "周宇清", "type": "PERSON"},
                {"text": "男", "type": "PERSON"},
                {"text": "341321200211201034", "type": "ID_NUMBER"},
                {"text": "242040390", "type": "VERIFY_CODE"},
                {"text": "2024年09月07日", "type": "DATE"},
                {"text": "大数据与人工智能学院", "type": "ORG"},
                {"text": "大数据与人工智能学院", "type": "ORG"},
                {"text": "汉族", "type": "NATION"},
            ],
            "fields": {},
            "text": (
                "姓名 周宇清 性别 男 软件工程丨普通全日制 证件号码 入学日期 学历类别 学制 班级 分院 民族 学号 系所 "
                "2024年09月07日 2年 普通高等教育 软升243\n 大数据与人工智能学院 大数据与人工智能学院 汉族 学号 242040390"
            ),
        },
    }

    normalized = normalize_student_card(sample_raw)
    pprint(normalized)
