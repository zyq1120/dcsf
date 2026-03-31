# app/services/document_type_service.py
from typing import Dict, Optional, List, Any
import re
from loguru import logger


class DocumentTypeService:
    """
    统一的文档类型识别服务：
    输入：全文 text、ocr_analysis、nlp_cls、llm_cls
    输出：doc_type（具体类型）、primary_type（一级分类）、candidates（概率字典）
    """

    # ===== 这里放我们刚才讨论的那一大坨映射 =====
    # 高校文档 -> 一级类型
    PRIMARY_TYPE_MAP: Dict[str, str] = {
        # ===== 成绩 / 课程 =====
        "成绩单": "transcript",
        "学业成绩表": "transcript",
        "成绩单/学业成绩表": "transcript",
        "课程表": "transcript",
        "课表": "transcript",
        "课表/选课单": "transcript",

        # ===== 学籍 / 在校信息 =====
        "学籍卡": "student_record",
        "学籍信息卡": "student_record",
        "班级成员表": "student_record",
        "住宿表": "student_record",
        "学籍证明": "student_record",
        "在校证明": "enrollment_proof",
        "学籍证明/在校证明": "enrollment_proof",

        # ===== 证明 / 证书类 =====
        "贫困证明": "certificate",
        "收入证明": "certificate",
        "奖学金证明": "certificate",
        "证明类其他": "certificate",
        "实习证明": "certificate",
        "实习鉴定表": "certificate",
        "实习协议书": "certificate",
        "毕业实习鉴定": "certificate",

        # ===== 申请 / 审批类 =====
        "请假条": "leave_application",
        "休学申请": "leave_application",
        "复学申请": "leave_application",
        "缓考申请表": "application_exam",
        "重修申请表": "application_exam",
        "转专业申请表": "application_change_major",
        "转学申请表": "application_transfer",
        "奖学金申请表": "application_scholarship",
        "奖学金审批表": "application_scholarship",
        "助学金申请表": "application_grant",
        "贫困生认定申请表": "application_grant",

        # ===== 录取 / 学历 / 学位 =====
        "录取通知": "admission",
        "录取凭证": "admission",
        "准考证": "admission",
        "考试证书": "certificate",
        "毕业证书/学历证书": "certificate",
        "学历证书": "certificate",
        "毕业证书": "certificate",
        "学位证书": "certificate",

        # ===== 合同 / 贷款 =====
        "合同": "contract",
        "贷款合同": "loan_contract",
        "生源地助学贷款合同": "loan_contract",

        # ===== 通知 / 会议 / 纪律 =====
        "通知": "notice",
        "通知/公告": "notice",
        "会议纪要": "meeting_minutes",
        "处分决定书": "discipline",
        "解除处分决定书": "discipline",

        # ===== 证件类 =====
        "学生证": "card",
        "校园卡": "card",

        # ===== 兜底 =====
        "其他": "other",
        "generic_document": "other",
    }

    # 关键字规则（我之前给你的 KEYWORD_RULES 可以完整粘过来）
    KEYWORD_RULES: List[Dict] = [
        # ---- 成绩单 / 学业成绩 ----
        {
            "doc_type": "成绩单",
            "keywords_any": ["成绩单", "学业成绩表", "成绩列表", "课程成绩", "学分绩点"],
            "keywords_all": [],
            "exclude": ["贷款合同", "借款人"],
            "min_hits": 1,
        },
        # 课程表 / 课表
        {
            "doc_type": "课程表",
            "keywords_any": ["课程表", "课表", "上课时间", "节次", "周一", "周二", "周三", "周四", "周五"],
            "keywords_all": [],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 学籍信息 / 在校证明 ----
        {
            "doc_type": "学籍卡",
            "keywords_any": ["学籍卡", "学籍信息卡"],
            "keywords_all": [],
            "exclude": ["贷款合同", "借款人"],
            "min_hits": 1,
        },
        {
            "doc_type": "学籍信息卡",
            "keywords_any": ["学籍状态", "预计毕业日期", "学信网", "在线验证码", "注册学籍"],
            "keywords_all": ["学籍"],
            "exclude": ["贷款合同", "借款人"],
            "min_hits": 2,
        },
        {
            "doc_type": "班级成员表",
            "keywords_any": ["班级成员表", "班级成员名单", "班级成员", "成员名单", "班级名单", "班级表", "班级花名册"],
            "keywords_all": ["班级"],
            "exclude": ["贷款合同", "借款人", "准考证", "准考证号", "考场号", "座位号"],
            "min_hits": 1,
        },
        {
            "doc_type": "班级成员表",
            "keywords_any": ["序号", "班级", "姓名"],
            "keywords_all": ["班级", "姓名"],
            "exclude": ["贷款合同", "借款人", "准考证", "准考证号", "考场号", "座位号"],
            "min_hits": 2,
        },
        {
            "doc_type": "准考证",
            "keywords_any": ["准考证", "准考证号", "报到时间", "考试时间", "考场号", "座位号", "英语四级", "英语六级", "CET"],
            "keywords_all": [],
            "exclude": [],
            "min_hits": 2,
        },
        {
            "doc_type": "考试证书",
            "keywords_any": ["合格证书", "证书编号", "校验码", "查询网址", "全国计算机等级考试", "Certificate Number"],
            "keywords_all": [],
            "exclude": [],
            "min_hits": 2,
        },
        {
            "doc_type": "住宿表",
            "keywords_any": ["住宿表", "宿舍表", "宿舍名单", "住宿名单", "楼栋", "宿舍号", "寝室", "床位"],
            "keywords_all": [],
            "exclude": ["贷款合同", "借款人"],
            "min_hits": 1,
        },
        {
            "doc_type": "在校证明",
            "keywords_any": ["在校证明", "在读证明", "兹证明", "特此证明"],
            "keywords_all": ["学生", "学号"],
            "exclude": ["贷款合同"],
            "min_hits": 1,
        },
        {
            "doc_type": "学籍证明",
            "keywords_any": ["学籍证明"],
            "keywords_all": ["学籍", "学生"],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 贫困 / 收入 / 奖学金证明 ----
        {
            "doc_type": "贫困证明",
            "keywords_any": ["家庭经济困难", "贫困证明", "困难学生"],
            "keywords_all": ["兹证明"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "收入证明",
            "keywords_any": ["收入证明", "工资收入", "月收入"],
            "keywords_all": ["证明"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "奖学金证明",
            "keywords_any": ["奖学金", "奖励", "表彰"],
            "keywords_all": ["特此证明"],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 实习相关 ----
        {
            "doc_type": "实习证明",
            "keywords_any": ["实习证明", "实习单位", "实习期间", "实习岗位"],
            "keywords_all": ["实习"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "实习鉴定表",
            "keywords_any": ["实习鉴定", "实习表现", "考核意见", "指导教师评语"],
            "keywords_all": ["实习"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "实习协议书",
            "keywords_any": ["实习协议", "三方协议", "实践教学安排"],
            "keywords_all": ["甲方", "乙方"],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 请假 / 休学 / 复学 ----
        {
            "doc_type": "请假条",
            "keywords_any": ["请假条", "请假申请", "请假单"],
            "keywords_all": ["请假", "申请人"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "休学申请",
            "keywords_any": ["休学申请", "休学申请表", "休学期间"],
            "keywords_all": ["休学"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "复学申请",
            "keywords_any": ["复学申请", "复学申请表"],
            "keywords_all": ["复学"],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 缓考 / 重修 ----
        {
            "doc_type": "缓考申请表",
            "keywords_any": ["缓考申请", "缓考申请表", "缓考课程"],
            "keywords_all": ["缓考"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "重修申请表",
            "keywords_any": ["重修申请", "重修申请表", "重修课程"],
            "keywords_all": ["重修"],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 转专业 / 转学 ----
        {
            "doc_type": "转专业申请表",
            "keywords_any": ["转专业申请", "转专业申请表"],
            "keywords_all": ["转专业"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "转学申请表",
            "keywords_any": ["转学申请", "转学申请表"],
            "keywords_all": ["转学"],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 奖助学金申请 ----
        {
            "doc_type": "奖学金申请表",
            "keywords_any": ["奖学金申请", "奖学金申请表"],
            "keywords_all": ["奖学金"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "奖学金审批表",
            "keywords_any": ["奖学金评审表", "奖学金审批表", "评奖评优"],
            "keywords_all": ["奖学金"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "助学金申请表",
            "keywords_any": ["助学金申请", "助学金申请表"],
            "keywords_all": ["助学金"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "贫困生认定申请表",
            "keywords_any": ["家庭经济困难学生认定", "贫困生认定申请表"],
            "keywords_all": ["认定", "家庭经济困难"],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 录取 / 学历 / 学位 ----
        {
            "doc_type": "录取通知",
            "keywords_any": ["录取通知书", "兹录取你", "报到注意事项"],
            "keywords_all": ["录取"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "录取凭证",
            "keywords_any": ["录取凭证", "新生报到凭证"],
            "keywords_all": ["录取"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "学位证书",
            "keywords_any": ["学位证书"],
            "keywords_all": ["学位"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "毕业证书/学历证书",
            "keywords_any": ["毕业证书", "学历证书"],
            "keywords_all": ["毕业"],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 通知 / 会议 / 处分 ----
        {
            "doc_type": "通知",
            "keywords_any": ["通知", "关于", "决定"],
            "keywords_all": [],
            "exclude": ["会议纪要"],
            "min_hits": 1,
        },
        {
            "doc_type": "会议纪要",
            "keywords_any": ["会议纪要", "会议记录"],
            "keywords_all": [],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "处分决定书",
            "keywords_any": ["处分决定书", "警告处分", "严重警告", "记过", "留校察看", "开除学籍"],
            "keywords_all": ["处分"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "解除处分决定书",
            "keywords_any": ["解除处分", "撤销处分"],
            "keywords_all": [],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 学生证 / 校园卡 ----
        {
            "doc_type": "学生证",
            "keywords_any": ["学生证"],
            "keywords_all": ["学校", "学生"],
            "exclude": [],
            "min_hits": 1,
        },
        {
            "doc_type": "校园卡",
            "keywords_any": ["校园卡", "一卡通"],
            "keywords_all": [],
            "exclude": [],
            "min_hits": 1,
        },

        # ---- 生源地助学贷款合同 / 贷款合同（保留你原来的规则，但收紧） ----
        {
            "doc_type": "生源地助学贷款合同",
            "keywords_any": ["生源地助学贷款", "生源地信用助学贷款"],
            "keywords_all": ["贷款合同编号", "借款人"],
            "exclude": ["学籍卡", "学籍信息卡"],
            "min_hits": 2,
        },
        {
            "doc_type": "贷款合同",
            "keywords_any": ["贷款合同", "借款人", "贷款金额", "还款期限"],
            "keywords_all": ["贷款"],
            "exclude": ["学籍卡", "学籍信息卡"],
            "min_hits": 2,
        },
    ]

    @staticmethod
    def _priority_of_type(doc_type: str) -> int:
        """
        分数接近时，用类型优先级做最后 tie-break。
        高优先级：贷款合同 / 学籍卡 / 成绩单 / 在校证明
        """
        high = {
            "生源地助学贷款合同",
            "贷款合同",
            "成绩单",
            "学籍卡",
            "学籍信息卡",
            "班级成员表",
            "住宿表",
            "准考证",
            "考试证书",
            "在校证明",
            "学籍证明",
        }
        mid = {
            "课程表", "课表",
            "贫困证明", "收入证明", "奖学金证明",
            "实习证明", "实习鉴定表",
            "请假条", "休学申请", "复学申请",
            "录取通知", "录取凭证",
        }
        if doc_type in high:
            return 3
        if doc_type in mid:
            return 2
        return 1

    @staticmethod
    def _infer_from_keywords_fallback(text: str) -> Optional[str]:
        """
        LLM + NLP 都给不出靠谱结果时，最后兜底的关键字猜测。
        """
        if any(k in text for k in ("成绩单", "学业成绩表", "成绩列表")):
            return "成绩单"
        if any(k in text for k in ("合格证书", "证书编号", "校验码", "查询网址", "全国计算机等级考试")):
            return "考试证书"
        if any(k in text for k in ("准考证", "准考证号", "考场号", "座位号", "英语四级", "英语六级", "CET")):
            return "准考证"
        if any(k in text for k in ("课程表", "课表")):
            return "课程表"
        if any(k in text for k in ("住宿表", "宿舍表", "宿舍名单", "住宿名单")):
            return "住宿表"
        if ("班级" in text and "姓名" in text) or ("班级" in text and "序号" in text):
            return "班级成员表"
        if "学籍卡" in text or "学籍信息卡" in text:
            return "学籍卡"
        if "在校证明" in text or "在读证明" in text:
            return "在校证明"
        if "生源地助学贷款" in text or "贷款合同编号" in text:
            return "生源地助学贷款合同"
        return None

    def detect(
        self,
        text: str,
        ocr_analysis: Optional[Dict],
        nlp_cls: Optional[Dict],
        llm_cls: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        统一入口：
        返回 {
          "doc_type": str,          # 最终类型
          "primary_type": str,      # 一级类型，如 transcript / student_record / loan_contract
          "candidates": Dict[str,float]  # 合并后的候选概率
        }
        """
        text = text or ""
        lower = text.lower()
        candidates: Dict[str, float] = {}

        # 1) 先合并 LLM 分类结果（权重大）
        if llm_cls:
            dt = llm_cls.get("document_type")
            prob = float(llm_cls.get("confidence", 0) or 0.0)
            if dt:
                candidates[dt] = max(candidates.get(dt, 0.0), prob)
            for k, v in (llm_cls.get("probabilities") or {}).items():
                candidates[k] = max(candidates.get(k, 0.0), float(v))

        # 2) 再合并 NLP 分类结果
        if nlp_cls:
            dt = nlp_cls.get("document_type")
            prob = float(nlp_cls.get("confidence", 0) or 0.0) * 0.9
            if dt:
                candidates[dt] = max(candidates.get(dt, 0.0), prob)
            for k, v in (nlp_cls.get("probabilities") or {}).items():
                candidates[k] = max(candidates.get(k, 0.0), float(v) * 0.9)

        # 3) 再合并 OCR 文本分析
        if ocr_analysis and ocr_analysis.get("document_type"):
            dt_block = ocr_analysis["document_type"]
            dt = dt_block.get("document_type")
            prob = float(dt_block.get("confidence", 0) or 0.0) * 0.8
            if dt:
                candidates[dt] = max(candidates.get(dt, 0.0), prob)
            for k, v in (dt_block.get("probabilities") or {}).items():
                candidates[k] = max(candidates.get(k, 0.0), float(v) * 0.8)

        # 4) 关键字规则打补丁（给匹配到的类型加一点权重）
        for rule in self.KEYWORD_RULES:
            dt = rule["doc_type"]
            kw_any = rule.get("keywords_any") or []
            kw_all = rule.get("keywords_all") or []
            exclude = rule.get("exclude") or []
            min_hits = rule.get("min_hits", 1)

            if any(e in text for e in exclude):
                continue

            hits_any = sum(1 for k in kw_any if k and k in text)
            if hits_any < min_hits:
                continue
            if kw_all and not all(k in text for k in kw_all):
                continue

            bonus = 0.15 + 0.05 * min(hits_any, 5)
            candidates[dt] = max(candidates.get(dt, 0.0), bonus)

        # 4.5) 班级成员表结构化特征强兜底：大量“班级+姓名+性别”三元组时直接抬高权重
        roster_triplet_pattern = re.compile(
            # 班级/姓名/性别
            r"(?:[A-Za-z0-9一-龥]{2,12}(?:班级)?\s*[一-龥]{2,4}\s*(?:男|女))|"
            # 姓名/性别/班级
            r"(?:[一-龥]{2,4}\s*(?:男|女)\s*[A-Za-z0-9一-龥]{2,12}(?:班级)?)"
        )
        roster_hits = len(list(roster_triplet_pattern.finditer(text)))
        exam_context = any(k in text for k in ("准考证", "准考证号", "考场号", "座位号", "英语四级", "英语六级", "CET"))
        roster_header = ("序号" in text and "班级" in text and "姓名" in text) or (
            "班级" in text and "性别" in text)
        if roster_hits >= 5 and roster_header and not exam_context:
            candidates["班级成员表"] = max(candidates.get("班级成员表", 0.0), 0.99)
        elif roster_hits >= 3 and roster_header and not exam_context:
            candidates["班级成员表"] = max(candidates.get("班级成员表", 0.0), 0.9)

        # 5) 没有任何候选时，最后兜底
        if not candidates:
            fallback = self._infer_from_keywords_fallback(text)
            if fallback:
                candidates[fallback] = 0.6

        if not candidates:
            final_type = "其他"
        else:
            # 取概率最高 + 类型优先级的 doc_type
            best_dt = None
            best_score = -1.0
            for dt, prob in candidates.items():
                score = prob * 10 + self._priority_of_type(dt)
                if score > best_score:
                    best_score = score
                    best_dt = dt
            final_type = best_dt or "其他"

        primary_type = self.PRIMARY_TYPE_MAP.get(final_type, "other")

        logger.info(
            "DocumentTypeService.detect",
            doc_type=final_type,
            primary_type=primary_type,
            candidates=candidates,
        )

        return {
            "doc_type": final_type,
            "primary_type": primary_type,
            "candidates": candidates,
        }
