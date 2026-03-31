import os
import pytest

# Ensure LLM disabled for tests
os.environ.setdefault("ENABLE_LLM_FALLBACK", "False")

from app.services.nlp_service import NLPService  # noqa: E402


@pytest.fixture()
def nlp():
    return NLPService()


def _slot_map(details):
    return {d.get("field_name"): d.get("field_value") for d in details or []}


def test_leave_form_entities_and_slots(nlp: NLPService):
    text = (
        "马鞍山学院2026届毕业生请、销假单\n"
        "姓名：周宇清 学号：242040390 专业班级：软升243 指导教师：甘丽\n"
        "请假离校原因: 1、论文调研、实习；\n"
        "外出地点、时间： （调研、实习等）单位名称、地址：_江苏省南京市\n"
        "家长姓名及手机：周和平 18689221796\n"
        "外出时间：自 2026年01月02日至 2026年01月05日；拟返校时间: 2026年01月06日\n"
        "本人手机号：_18455716598\n"
    )
    ents = nlp.extract_entities(text)
    assert any(e["type"] == "PERSON" and "周宇清" in e["text"] for e in ents)
    assert any(e["type"] == "STUDENT_ID" and e["text"] == "242040390" for e in ents)
    assert any(e["type"] == "PARENT_PHONE" and e["text"] == "18689221796" for e in ents)
    assert any(e["type"] == "LEAVE_LOCATION" for e in ents)
    assert any(e["type"] == "LEAVE_START_DATE" for e in ents)
    assert any(e["type"] == "LEAVE_END_DATE" for e in ents)

    slots = _slot_map(nlp.map_entities_to_slots(ents, text))
    assert slots.get("name") == "周宇清"
    assert slots.get("student_id") == "242040390"
    assert slots.get("phone") == "18455716598"
    assert slots.get("contact") == "18689221796"
    assert slots.get("start_date") == "2026-01-02"
    assert slots.get("end_date") == "2026-01-05"


def test_loan_contract_amount_and_term(nlp: NLPService):
    text = (
        "贷款信息 贷款合同编号 4363011120242001 用款期限 87（月） 本次用款金额 10000.00（元）\n"
        "开户行 中国民生银行马鞍山江东大道支行\n"
        "回执验证码为：874653。\n"
    )
    ents = nlp.extract_entities(text)
    assert any(e["type"] == "LOAN_CONTRACT_NO" for e in ents)
    assert any(e["type"] == "LOAN_TERM_MONTH" for e in ents)


def test_classification_not_roster_when_leave_keywords(nlp: NLPService):
    text = "请、销假单 姓名：周宇清 班级：软升243 请假离校原因：论文调研"
    cls = nlp.classify_document(text)
    assert cls.get("document_type") in ("请假条", "休学申请", "通知", "证明类其他", "在校证明", "学籍信息卡")
    assert cls.get("document_type") != "班级成员表"


def test_dormitory_table_auto_template(nlp: NLPService):
    text = (
        "住宿表\n"
        "序号 楼栋 宿舍号 床位 姓名 学号\n"
        "1 7栋 708 1 张三 242040390\n"
    )
    auto = nlp.infer_fields_auto(text)
    tpl = auto.get("template_config") or {}
    details = _slot_map((auto.get("extract_result") or {}).get("extract_details"))

    assert tpl.get("template_id") == "auto-dormitory"
    assert details.get("building") == "7栋"
    assert details.get("dorm_no") == "708"
    assert details.get("bed_no") == "1"
    assert details.get("name") == "张三"
    assert details.get("student_id") == "242040390"


def test_class_table_alias_hits_roster(nlp: NLPService):
    text = "班级表\n序号 班级 姓名 性别 学号\n1 软升243 周宇清 女 242040390"
    cls = nlp.classify_document(text)
    assert cls.get("document_type") == "班级成员表"


def test_admission_ticket_not_roster_and_template(nlp: NLPService):
    text = (
        "2025年下半年全国大学英语四级考试\n"
        "准考证\n"
        "准考证号：340801252117128\n"
        "姓名：周宇清\n"
        "性别：男\n"
        "证件号码：341321200211201034\n"
        "所属学校：马鞍山学院\n"
        "院系班级：大数据与人工智能学院 软升243\n"
        "学号：242040390\n"
        "考试日期 2025-12-13\n"
        "报到时间 08:40\n"
        "考试时间 09:00-11:20\n"
        "考试地点 马鞍山学院 G414A\n"
        "考场号 171\n"
        "座位号 28\n"
    )
    cls = nlp.classify_document(text)
    assert cls.get("document_type") == "准考证"

    auto = nlp.infer_fields_auto(text)
    tpl = auto.get("template_config") or {}
    details = _slot_map((auto.get("extract_result") or {}).get("extract_details"))
    assert tpl.get("template_id") == "auto-admission-ticket"
    assert details.get("ticket_no") == "340801252117128"
    assert details.get("name") == "周宇清"
    assert details.get("student_id") == "242040390"
    assert details.get("room_no") == "171"
    assert details.get("seat_no") == "28"


def test_exam_certificate_not_admission_ticket(nlp: NLPService):
    text = (
        "全国计算机等级考试\n"
        "二级合格证书\n"
        "姓名：杨思雨\n"
        "身份证件号：341282200212154935\n"
        "准考证号：2468340019010107\n"
        "证书编号：24683400186130\n"
        "校验码：H5T0CS6HYAE5DOCV\n"
        "查询网址：www.neea.edu.cn\n"
    )
    cls = nlp.classify_document(text)
    assert cls.get("document_type") == "考试证书"

    auto = nlp.infer_fields_auto(text)
    tpl = auto.get("template_config") or {}
    details = _slot_map((auto.get("extract_result") or {}).get("extract_details"))
    assert tpl.get("template_id") == "auto-exam-certificate"
    assert details.get("name") == "杨思雨"
    assert details.get("id_number") == "341282200212154935"
    assert details.get("certificate_id") == "24683400186130"
    assert details.get("verify_code") == "H5T0CS6HYAE5DOCV"
    assert details.get("verify_url") == "www.neea.edu.cn"


def test_student_card_auto_template_and_fields(nlp: NLPService):
    text = (
        "教育部学籍在线验证报告\n"
        "姓名\n周宇清\n"
        "性别\n男\n"
        "出生日期\n2002年11月20日\n"
        "学校名称\n马鞍山学院\n"
        "层次\n本科\n"
        "专业\n软件工程\n"
        "学历类别\n普通高等教育\n"
        "学习形式\n普通全日制\n"
        "分院\n大数据与人工智能学院\n"
        "入学日期\n2024年09月07日\n"
        "学籍状态\n在籍 (注册学籍)\n"
        "预计毕业日期\n2026年07月01日\n"
        "在线验证码AQT52ZQTS2RJWC0Q\n"
        "在线查验网址：https://www.chsi.com.cn/xlcx/bgcx.jsp\n"
    )
    auto = nlp.infer_fields_auto(text)
    tpl = auto.get("template_config") or {}
    details = _slot_map((auto.get("extract_result") or {}).get("extract_details"))

    assert tpl.get("template_id") == "auto-student-card"
    assert details.get("name") == "周宇清"
    assert details.get("gender") == "男"
    assert details.get("school_name") == "马鞍山学院"
    assert details.get("major") == "软件工程"
    assert details.get("status") and "在籍" in details.get("status")
    assert details.get("verify_url") == "https://www.chsi.com.cn/xlcx/bgcx.jsp"
    # 关键防回归：不应把“姓名/层次”等标签词错误写入 class/college
    assert details.get("college") != "层次"


