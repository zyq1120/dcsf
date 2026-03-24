from app.services.nlp_service import NLPService


LEAVE_TEXT = """马鞍山学院2026届毕业生请、销假单 姓名：周宇清 学号：242040390 专业班级：软升243 指导教师：甘丽
论文题目：基于OCR+NLP的高校教务材料智能处理系统的设计与实现 请假离校原因: 1、论文调研、实习；
外出地点、时间： 1、（调研、实习等）单位名称、地址：_江苏省南京市
家长姓名及手机：周和平 本人手机号：_18455716598 微信号：ThisMyWeChats Zyq1120@yahoo.com E-mail:_ QQ号：18689221796
拟返校时间: 日 与指导老师、辅导员老师联系方式："""


def test_leave_doc_should_not_be_roster():
    svc = NLPService()
    cls = svc.classify_document(LEAVE_TEXT)
    # 允许输出“请假条/请假单”，但不应是“班级成员表”
    assert cls["document_type"] != "班级成员表"


def test_leave_entities_phone_email_studentid():
    svc = NLPService()
    entities = svc.extract_entities(LEAVE_TEXT, use_cache=False)
    types = {e["type"] for e in entities}
    assert "PHONE" in types
    assert "EMAIL" in types
    assert "STUDENT_ID" in types

    # phone 应该能抽到 18455716598
    phones = [e["text"] for e in entities if e["type"] == "PHONE"]
    assert any("18455716598" in p for p in phones)

    # 学号应能抽到 242040390
    sids = [e["text"] for e in entities if e["type"] == "STUDENT_ID"]
    assert any("242040390" in s for s in sids)

