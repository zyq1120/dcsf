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

