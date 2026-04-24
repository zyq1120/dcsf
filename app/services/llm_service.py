"""
LLM Service (Nvidia only): 提供 OCR 增强、字段兜底抽取、多模态读图、文档分类。
仅使用 Nvidia OpenAI 兼容接口。
"""
import ast
import json
import re
import time
from typing import Dict, List, Optional


import requests
from loguru import logger

from app.config import settings
from app.utils.errors import ServiceError

# ─── 常量 ────────────────────────────────────────────────────────────────────

_MIME_MAP: Dict[str, str] = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}

_DOC_TYPE_MAP: Dict[str, str] = {
    "student id": "学生证",
    "student id card": "学生证",
    "student certificate": "考试证书",
    "certificate": "考试证书",
    "admission ticket": "准考证",
    "exam ticket": "准考证",
}

_CET_KEYWORDS = ("CET", "英语六级", "英语四级", "成绩报告单", "总分")

_BASIC_INFO_FIELDS = frozenset({
    "name", "id_number", "student_id", "phone", "gender",
    "school_name", "college", "major", "class",
})

_BASIC_INFO_PLACEHOLDERS = frozenset({
    "姓名", "个人信息", "手机号码", "电话",
    "[姓名]", "[ID号]", "[学号]", "[电话]", "[性别]",
    "[学校名称]", "[学院]", "[专业]", "[班级]",
})

_NON_RETRYABLE_KEYWORDS = (
    "配额不足",
    "Quota",
    "RateLimit",
    "api_key 未配置",
    "免费额度已用完",
    "404",
    "Not Found",
    "404 Client Error",
)

# ─── Vision 少样本提示词（抽为模块级常量，避免每次调用重建）────────────────

# _VISION_PROMPT = llm_prompts.LLM_VISION_COMPACT_PROMPT
_VISION_PROMPT = r"""
先记住你只能输出 json
你是一个【通用文档结构化归一化助手】，负责从各种中文文档中提取结构化信息并归一化。
文档类型可以包含但不限于：个人证明、合同、通知、证明材料、票据、以及高校相关文档
（成绩单、在校证明、学籍信息卡、贷款合同、奖学金证明、请假条等）。

无论输入是截图、扫描件还是 OCR/规则处理后的 JSON+文本，你的核心任务只有一个：
——理解文档内容，并填充统一的结构化 JSON。

====================【输出格式（硬约束）】====================
最终你只输出一个 JSON 对象，结构必须严格为：

{
  "academic_info": {
    "degree_level": null | string,
    "education_type": null | string,
    "enroll_date": null | string,
    "expected_grad_date": null | string,
    "status": null | string,
    "study_mode": null | string
  },
  "basic_info": {
    "address": null | string,
    "birth_date": null | string,
    "class": null | string,
    "college": null | string,
    "contact": null | string,
    "gender": null | string,
    "id_number": null | string,
    "major": null | string,
    "name": null | string,
    "phone": null | string,
    "school_name": null | string,
    "student_id": null | string
  },
  "certificate_info": {
    "certificate_id": null | string,
    "issue_date": null | string,
    "issuer": null | string,
    "verify_code": null | string,
    "verify_url": null | string
  },
  "confidence_overall": number,
  "courses": array,
  "document_type": string,
  "financial_info": {
    "account_number": null | string,
    "bank_name": null | string,
    "loan_amount": null | number,
    "loan_years": null | number | string,
    "scholarship_amount": null | number,
    "scholarship_name": null | string
  },
  "leave_info": {
    "approve_status": null | string,
    "days": null | number | string,
    "end_date": null | string,
    "issuer": null | string,
    "reason": null | string,
    "start_date": null | string
  },
  "summary": string,
  "tables": array,
  "text": string
}

严格要求：
- 所有字段都必须保留，不允许新增或删除字段。
- 不知道的字段填 null，不要乱编。
- 禁止输出 code、message、debug、fields、document_type_candidates、meta、tables_raw 等任何附加字段。
- 必须返回合法 JSON（键名加双引号，字符串加双引号，不能有注释）。

====================【整体处理流程】====================

【第 1 步：判断文档类型 document_type（通用规则）】
根据标题、抬头、关键短语、正文内容进行判断。
常见类型示例（不限于此）：
- “成绩单”“成绩证明”“成绩表” → document_type = "成绩单"
- “在校证明”“学籍证明”“学籍信息卡”“学生在读证明” → "在校证明" 或 "学籍信息卡"
- “毕业证书”“学历证书” → "毕业证书/学历证书"
- “生源地助学贷款合同”“助学贷款合同”“贷款合同” → 细化为真实类型名称
- “奖学金证明”“助学金发放证明”“贫困证明” → "奖学金证明" 或 "贫困证明/资助证明"
- “请假条”“休学申请”“病假条”等 → "请假条" 或 "休学申请"
- “通知”“公告”“关于……的通知” → "通知"
- 其他合同、协议、证明 → 尽量用简短中文，如“劳动合同”“实习协议”“收入证明”等
- 实在无法确定 → document_type = "其他"

【第 2 步：公共字段抽取（适用于所有类型）】
尽量从文档中抽取并填写以下字段：

- basic_info.name：姓名（2~4 字中文人名，不能是“姓名”“学生姓名”“借款人”等标签本身）
- basic_info.id_number：居民身份证号（18 位，或文中明确标注的有效身份证）
- basic_info.student_id：学号/学籍号/工号（根据上下文判断）
- basic_info.school_name：学校/单位全称，如“马鞍山学院”“某某中学”“某某公司”
- basic_info.college：院系/学院/部门，如“大数据与人工智能学院”“人力资源部”
- basic_info.major：专业/岗位信息，去掉“专业名称/名称”等标签
- academic_info.enroll_date：入学/入职/开始日期（能判断时填）
- academic_info.expected_grad_date：预计毕业/结束日期（能判断时填）
- academic_info.degree_level：学历层次，如“专科”“本科”“硕士”“博士”
- academic_info.education_type：学历类别，如“普通高等教育”“成人教育”“专升本”等
- certificate_info.issue_date：证书或证明的出具日期
- certificate_info.issuer：出具单位（学校名称、学院、部门、公司等）
- basic_info.address：家庭住址、联系地址等完整地址
- basic_info.phone：手机号/联系电话
- basic_info.contact：其他联系信息（如 QQ、邮箱、家长电话等）

要求：
- 文中没有出现就保持 null，不要臆造。
- 如果某字段只是“字段名/标签”（如“姓名：”“身份证号：”没有后面的值），则填 null。

【第 3 步：按 document_type 应用“类型特定规则”】

====== 3.1 成绩单 / 成绩证明 / 成绩表 ======
- document_type 统一使用 "成绩单"。
- courses：从表格或正文中抽取课程列表，每个元素可包含：
  - 课程名称、学分、成绩、绩点、学年学期等字段（名称可根据已有数据灵活，但整个 courses 必须是数组）。
- summary：用 1~3 句话概括：
  - 这是某人的成绩单；
  - 涉及学校/学院/专业；
  - 涉及哪些学年/学期，总学分/平均绩点等关键信息（有则写，没有就省略）。

====== 3.2 贷款合同（通用规则） ======
适用于“贷款合同”“借款合同”等，不限于高校场景。

- document_type：尽量细化，例如“贷款合同”“个人借款合同”“房贷合同”等。
- financial_info：
  - account_number：银行卡号或贷款账号
  - bank_name：银行/放款机构名称
  - loan_amount：贷款金额（数值形式，单位默认为人民币元，如“10000.00”）
  - loan_years：贷款年限（可为数字或包含“X 年”的字符串）
- basic_info：
  - name：借款人姓名
  - id_number：借款人身份证号
  - address：借款人地址（如有）
- certificate_info：
  - certificate_id：合同编号/协议编号
  - issue_date：签署日期
  - issuer：贷款机构/银行/学校/公司等

summary 示例（通用）：
- “个人贷款合同：借款人张三，向中国某银行申请贷款金额 100000 元，期限 3 年。”
- 尽量包含：借款人姓名、贷款机构、金额、期限等要素。

====== 3.2.1 生源地助学贷款合同【专项规则】======
当文档明显属于“生源地助学贷款”场景时，在 3.2 的基础上应用以下规则：

1）姓名修正：
- 若 basic_info.name 是“姓名/开户行/账号”等标签，视为无效，需要从正文重新提取。
- 如果 text 中包含：
  - “贵校学生XXX的生源地信用助学贷款申请”
  - 或 “贵校学生_XXX的生源地信用助学贷款申请”
  - 则 XXX 为姓名（去掉下划线），写入 basic_info.name。
- 若 basic_info.contact 是噪声、与联系方式无关，可置为 null。

2）地址填充：
- basic_info.address 为空时，
- 若 text 中出现类似 “安徽省_宿州市_砀山县_官庄镇黄集村周庄” 的地址：
  - 去掉下划线，填入完整地址。

3）专业填充：
- 当 basic_info.major 为空且 text 中含：
  - “专业名称 学制 …（专升本）软件工程”
- 则 basic_info.major 设为“软件工程（专升本）”或“软件工程”（不保留“名称”标签）。

4）学校/学院拆分：
- 同时出现 “马鞍山学院” 和 “大数据与人工智能学院”：
  - school_name = “马鞍山学院”
  - college = “大数据与人工智能学院”

5）文档类型细化：
- text 中包含 “贷款合同编号” 且有 “生源地助学贷款” 等关键词：
  - document_type = “生源地助学贷款合同”。

6）摘要特化：
- 若姓名/学校/学院/专业/金额/期限信息齐备，summary 示例：
  - “生源地助学贷款合同：借款人周宇清，马鞍山学院大数据与人工智能学院软件工程（专升本）专业，贷款金额 10000 元，期限 2 年。”
- 姓名缺失时，可用“借款人姓名缺失”代替姓名位置，但不能写占位符“姓名”本身。

【样例约束（生源地助学贷款场景）】
- text 中包含：
  - “贵校学生_周宇清的生源地信用助学贷款申请”
  - “安徽省_宿州市_砀山县_官庄镇黄集村周庄”
  - “专业名称 学制 …（专升本）软件工程”
  - “贷款合同编号 生源地助学贷款在线服务系统...”
- 最终必须满足：
  - basic_info.name = “周宇清”
  - basic_info.address = “安徽省宿州市砀山县官庄镇黄集村周庄”
  - basic_info.major 包含 “软件工程”
  - document_type = “生源地助学贷款合同”
  - summary 中使用真实姓名“周宇清”，不能出现“借款人 姓名”等占位符。

====== 3.3 奖学金证明 / 贫困证明 / 资助类文档 ======
- document_type：如“奖学金证明”“贫困证明”“资助证明”等。
- financial_info：
  - scholarship_name：奖项名称/资助项目名称
  - scholarship_amount：金额（如“2000”表示 2000 元）
- certificate_info：
  - certificate_id：文号/编号
  - issue_date：出具日期
  - issuer：出具单位（学院、学校、资助中心等）
- summary：说明谁、获得/享受了什么资助或奖学金、金额多少。

====== 3.4 在校证明 / 学籍信息卡 ======
- document_type：使用“在校证明”或“学籍信息卡”。
- 重点填写：
  - basic_info：姓名、性别、出生日期、学号、学校名称、学院、专业、班级等。
  - academic_info：入学日期、预计毕业日期、在读状态（如“在读”“休学中”）、学制/学习形式。
  - certificate_info：出具单位（学院/学校）、出具日期。
- summary：说明该学生目前在何校、何专业在读，入学时间、预计毕业时间等。

====== 3.5 请假条 / 休学申请 ======
- document_type：如“请假条”“休学申请”。
- leave_info：
  - reason：请假/休学原因
  - start_date：起始日期
  - end_date：结束日期
  - days：天数（可以是数字或“X天”的字符串）
  - approve_status：审批意见，如“已批准”“未批准”“待审批”
  - issuer：批准人/审批部门（如有）
- summary：简要说明谁因何事在什么时间段请假/休学。

====== 3.6 通知 / 公告 / 其他通用通知类 ======
- document_type：如“通知”“公告”。
- summary：用 1~3 句概括：
  - 通知的主题（关于什么）
  - 针对的对象（全体学生、某学院、某部门等）
  - 关键时间/地点/要求（有则写）

====== 3.7 毕业证书 / 学历证书【专项规则】======
当 document_type = "毕业证书/学历证书" 时，应用以下专项规则：

1）顶部区域优先抽取 basic_info：
- 姓名（注意不能是“姓名”标签本身）
- 性别
- 出生日期
- 学校名称
- 专业名称

2）正文 text 提取关键信息：
- 入学日期 → academic_info.enroll_date
- (结)业日期 → academic_info.expected_grad_date
- 证书签发日期 → certificate_info.issue_date（如有明确）
- “学制 X年”：
  - academic_info.study_mode：可设为“普通全日制”（如上下文明确）
  - financial_info.loan_years：填入 X 或 “X 年”
- “学历类别 普通高等教育”等：
  - academic_info.education_type
- “专科/本科/硕士/博士”等：
  - academic_info.degree_level
- “证书编号”：
  - certificate_info.certificate_id

3）姓名标签处理：
- 如果 basic_info.name 解析到的是“姓名”等纯标签，必须纠正或置 null，不可当作姓名。

4）summary 模板参考：
- “毕业证书：姓名张三，学校马鞍山学院，专业软件工程，学历类别普通高等教育，入学日期 2020-09-01，结业日期 2023-06-30。”

====== 3.8 其他合同 / 协议 / 证明 ======
- 尽量用 document_type 概括文档类型，如“劳动合同”“实习协议”“收入证明”等。
- basic_info、certificate_info、financial_info 尽量填满能识别的字段。
- summary 概括主体（谁和谁）、文档性质（合同/证明/协议）、关键金额/期限/时间点。

====================【第 4 步：summary 和 text】====================
- summary：
  - 使用 1~3 句自然语言。
  - 尽量包含：文档类型、主体（姓名/单位）、学校/学院/部门、金额/学制/年限等关键信息。
  - 不要复述全部原文，只提炼重点。

- text：
  - 若有 OCR 原文，可复用或适度清洗后写入。
  - 若没有完整原文，也要把你能辨认的关键信息按原文顺序串成一段文字。
  - 除非完全无法辨认，否则 text 不应为空字符串。

====================【第 5 步：confidence_overall】====================
- 取值范围 0~1。
- 高置信度：0.9~1.0（字段基本都能从文中明确找到）
- 中等置信度：0.6~0.9（大部分信息可靠，少量字段有推测）
- 低置信度：0~0.5（大量依赖推测、文本模糊或缺失严重）
- 即便置信度很低，也必须输出完整 schema（所有字段都存在，只是值多为 null）。

请严格遵守以上规则，最终只输出一个 JSON 对象，无任何多余说明。
    "你是高校文档结构化抽取器。请直接阅读图片并返回严格 JSON。\n"
    "必须包含以下字段：document_type, text, summary, confidence_overall, "
    "basic_info{name,id_number,student_id,phone,gender,school_name,college,major,class}, "
    "academic_info{}, certificate_info{}, financial_info{}, leave_info{}, courses[], tables[]。\n"
    "规则：无法识别填 null；text 至少 20 字；只输出 JSON，不要解释。"
    "禁止输出 Markdown 代码块、禁止输出说明文字。\n"
"""

_VISION_JSON_EXAMPLE: Dict = {
    "document_type": "准考证",
    "text": "图片识别出的原文文本",
    "summary": "文档摘要",
    "confidence_overall": 0.85,
    "basic_info": {
        "name": None, "id_number": None, "student_id": None,
        "phone": None, "gender": None, "school_name": None,
        "college": None, "major": None, "class": None,
    },
    "academic_info": {},
    "certificate_info": {},
    "financial_info": {},
    "leave_info": {},
    "courses": [],
    "tables": [],
}


# ─── 主类 ─────────────────────────────────────────────────────────────────────

class LLMService:
    """
    Nvidia OpenAI-compatible LLM 服务封装。
    提供：OCR 增强、字段抽取、文档分类、多模态图像解析。
    """

    def __init__(self) -> None:
        self.enabled: bool = settings.ENABLE_LLM_FALLBACK
        self.provider: str = "nvidia"

        self.api_key: str = (
            getattr(settings, "LLM_API_KEY_NVIDIA", "") or settings.LLM_API_KEY
        )
        self.model_text: str = (
            getattr(settings, "LLM_MODEL_NVIDIA", None)
            or settings.LLM_MODEL
            or "minimaxai/minimax-m2.5"
        )
        self.model_vision: str = (
            getattr(settings, "LLM_MODEL_NVIDIA_VISION", None)
            or getattr(settings, "LLM_MODEL_NVIDIA", None)
            or self.model_text
        )
        self.model_vision_fallback: Optional[str] = getattr(
            settings, "LLM_MODEL_NVIDIA_VISION_FALLBACK", None
        )
        # 向后兼容别名
        self.model: str = self.model_text

        self.timeout: int = settings.LLM_TIMEOUT
        self.vision_timeout: int = getattr(settings, "LLM_VISION_TIMEOUT", self.timeout)
        self.retry: int = settings.LLM_RETRY
        self.trust_env: bool = getattr(settings, "LLM_TRUST_ENV", False)
        self.raw_log_limit: int = int(getattr(settings, "LLM_RAW_LOG_LIMIT", 3000) or 3000)

        self._session = requests.Session()
        self._session.trust_env = self.trust_env
        if not self.trust_env:
            self._session.proxies = {}

        logger.info(
            "LLMService initialized",
            enabled=self.enabled,
            provider=self.provider,
            model_text=self.model_text,
            model_vision=self.model_vision,
            model_vision_fallback=self.model_vision_fallback,
            timeout=self.timeout,
            vision_timeout=self.vision_timeout,
            retry=self.retry,
            trust_env=self.trust_env,
            raw_log_limit=self.raw_log_limit,
        )
        logger.debug("LLM connectivity check skipped at startup to avoid blocking")

    # ──────────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────────

    def enhance_ocr(
        self,
        ocr_text: str,
        confidence: float,
        override: Optional[Dict] = None,
    ) -> Dict:
        """对 OCR 输出做 LLM 纠错；失败时透明降级到原文。"""
        base = {"text": ocr_text, "confidence": confidence, "source": "ocr"}
        if not self.enabled:
            return {**base, "enhanced": False}

        prompt = (
            "你是 OCR 纠错器，请在保持原文语义的基础上纠正常见 OCR 错误，输出纯文本。\n"
            "要求：不编造内容；保持换行；只返回纠正后的文本。\n\n"
            f"原文：\n{ocr_text}"
        )
        try:
            fixed = self._call_llm(prompt, override=override)
            return {**base, "enhanced": True, "text": fixed.strip(), "source": "llm"}
        except Exception as exc:
            logger.warning(f"OCR enhance failed: {exc}")
            return {**base, "enhanced": False}

    def extract_with_llm(
        self,
        text: str,
        template_config: Dict,
        override: Optional[Dict] = None,
        ocr_context: Optional[Dict] = None,
        file_content_b64: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """字段兜底抽取：默认纯文本；若携带图片则走多模态并融合 OCR 上下文。"""
        if not self.enabled:
            return None

        fields = template_config.get("fields") or []
        field_desc = "\n".join(
            f"- {f.get('name')} ({f.get('type', 'TEXT')}): {f.get('description', '')}"
            for f in fields
        )
        prompt = (
            "你是高校文档结构化解析引擎。请从文本中提取下列字段，输出严格 JSON："
            '{"extract_details":[{"field_name":"","field_value":"","field_type":"","confidence":0.0}]}。\n'
            "规则：\n"
            "1) 缺失用 null；2) 不要返回 NaN/undefined；3) confidence 0~1；4) 只输出 JSON。\n"
            f"字段列表：\n{field_desc}\n文本：\n{text}"
        )

        # 文件场景：将 OCR 证据 JSON + 原图一起交给多模态模型做兜底。
        if file_content_b64:
            ocr_context = ocr_context or {}
            ocr_context_json = json.dumps(ocr_context, ensure_ascii=False)[:5000]
            vision_prompt = (
                _VISION_PROMPT
                + "\n\n【当前任务】这是 LLM 兜底场景。请优先依据 OCR 与规则抽取证据，不要编造。"
                + "\n输出要求：仅输出 JSON；未知填 null；当图片与 OCR 冲突时，以图片可见事实为准，并在 text 中保留可确认信息。"
                + f"\n\n【模板字段】\n{field_desc or '(未提供模板字段)'}"
                + f"\n\n【OCR_兜底上下文_JSON】\n{ocr_context_json}"
                + f"\n\n【OCR_清洗文本】\n{text[:5000]}"
            )
            response: Optional[str] = None
            try:
                response = self._call_vision(vision_prompt, file_content_b64, file_name, override=override)
                self._log_raw_response("LLM fallback vision raw response", response)
                parsed = self._parse_json_dict_response(response)
                if not parsed:
                    parsed = self._repair_json_with_llm(response, override=override)
                if not parsed:
                    return None
                return self._normalize_llm_image_result(parsed)
            except Exception as exc:
                logger.warning(f"LLM fallback with image failed, downgrade to text-only: {exc}")

        try:
            raw = self._call_llm(prompt, override=override)
            return self._parse_json_dict_response(raw)
        except Exception as exc:
            logger.error(f"LLM extract_with_llm parse failed: {exc}")
            return None

    def classify_with_llm(
        self,
        text: str,
        override: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """文档类型分类，输出 {"document_type":…,"confidence":…,"probabilities":{}}。"""
        if not self.enabled:
            return None

        prompt = (
            "根据以下高校文档文本判断文档类型，输出 JSON："
            '{"document_type":"","confidence":0.0,"probabilities":{}}。'
            "类型示例：成绩单、学籍卡、在校证明、毕业证书/学历证书、学位证书、合同、申请、通知、课表、其他。\n"
            f"文本：\n{text}"
        )
        fallback = {
            "document_type": "generic_document",
            "confidence": 0.0,
            "probabilities": {},
            "source": "llm",
        }
        try:
            raw = self._call_llm(prompt, override=override)
            parsed = self._parse_json_dict_response(raw)
            if parsed:
                parsed["source"] = "llm"
                return parsed
        except Exception as exc:
            logger.warning(f"LLM classify parse failed: {exc}")
        return fallback

    def extract_with_llm_image(
        self,
        file_content_b64: str,
        file_name: Optional[str],
        template_config: Dict,
        override: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """多模态图像直读，返回结构化抽取结果。"""
        if not self.enabled:
            logger.warning("LLM image extract skipped: LLM disabled")
            return None

        # 可选字段补充提示（最多取前5个，避免 prompt 过长）
        opt_fields = template_config.get("fields", [])[:5]
        prompt = _VISION_PROMPT
        if opt_fields:
            names = ", ".join(f.get("name", "") for f in opt_fields)
            prompt += f"\n可选字段：{names}"

        logger.info(
            "LLM image extract triggered",
            provider=self.provider,
            model=self.model_vision,
            has_image=bool(file_content_b64),
        )

        response: Optional[str] = None
        try:
            response = self._call_vision(prompt, file_content_b64, file_name, override=override)
            self._log_raw_response("LLM vision raw response", response)

            parsed = self._parse_json_dict_response(response)
            if not parsed:
                parsed = self._repair_json_with_llm(response, override=override)
            if not parsed:
                logger.error(
                    "LLM vision response: invalid JSON payload",
                    response_snippet=(response[: self.raw_log_limit] if response else "(empty)"),
                )
                logger.warning(
                    "LLM vision expected valid JSON example",
                    example=json.dumps(_VISION_JSON_EXAMPLE, ensure_ascii=False),
                )
                return None

            parsed = self._normalize_llm_image_result(parsed)
            logger.info(
                "LLM vision response received",
                doc_type=parsed.get("document_type"),
                text_length=len(str(parsed.get("text") or "").strip()),
                text_preview=str(parsed.get("text") or "")[:150],
                summary=str(parsed.get("summary") or "")[:150],
                confidence_overall=parsed.get("confidence_overall"),
                has_basic_info=bool(any(v for v in (parsed.get("basic_info") or {}).values() if v)),
                has_academic_info=bool(any(v for v in (parsed.get("academic_info") or {}).values() if v)),
            )
            return parsed

        except json.JSONDecodeError as exc:
            logger.error(
                "LLM image JSON decode failed",
                error=str(exc),
                response_snippet=(response[: self.raw_log_limit] if response else "(empty)"),
            )
            return None
        except ServiceError:
            raise
        except Exception as exc:
            logger.error(
                "LLM image parse failed",
                error_type=type(exc).__name__,
                error_detail=str(exc),
                response_snippet=(response[: self.raw_log_limit] if response else "(empty)"),
            )
            return None

    def _validate_connectivity(self) -> None:
        if not self.enabled:
            logger.debug("LLM connectivity check skipped (disabled)")
            return
        if not self.api_key:
            logger.warning("LLM connectivity check skipped: api_key 未配置")
            return
        try:
            resp = self._call_text("返回 ok", timeout=min(self.timeout, 8) if self.timeout else 8)
            logger.info("LLM connectivity ok", provider=self.provider, sample=str(resp)[:20])
        except Exception as exc:
            logger.warning(f"LLM connectivity check failed: {exc}")

    def probe_connectivity(self) -> Dict[str, object]:
        """主动探测 LLM 连通性，供健康检查/管理界面使用。"""
        result: Dict[str, object] = {
            "provider": self.provider,
            "enabled": bool(self.enabled),
            "model": self.model_text,
            "vision_model": self.model_vision,
            "endpoint": self._resolve_base_url(),
            "connected": False,
        }

        if not self.enabled:
            result["reason"] = "LLM 已禁用"
            return result
        if not self.api_key:
            result["reason"] = "LLM API key 未配置"
            return result

        timeout = min(self.timeout, 8) if self.timeout else 8
        try:
            sample = self._call_text("返回 ok", timeout=timeout)
            result.update(
                {
                    "connected": True,
                    "sample": str(sample)[:80],
                }
            )
            return result
        except Exception as exc:
            result["reason"] = str(exc)
            return result

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：LLM 调用层
    # ──────────────────────────────────────────────────────────────────────────

    def _call_llm(
        self,
        prompt: str,
        override: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> str:
        if not self.enabled:
            raise ServiceError("LLM 未启用")

        api_key = (override or {}).get("api_key") or self.api_key
        model = (override or {}).get("model") or self.model_text
        if not api_key:
            raise ServiceError("LLM API key 未配置", detail="请配置 LLM API 密钥")

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retry + 2):
            try:
                logger.debug("LLM call attempt", attempt=attempt, model=model)
                return self._call_text(prompt, model=model, api_key=api_key, timeout=timeout)
            except Exception as exc:
                last_exc = exc
                if isinstance(exc, ServiceError):
                    detail = getattr(exc, "detail", "") or ""
                    if any(kw in detail for kw in _NON_RETRYABLE_KEYWORDS):
                        logger.warning(f"LLM 不可重试错误，停止重试: {exc}")
                        break

                backoff = min(3 * attempt, 8)
                logger.warning(f"LLM 调用失败，第 {attempt} 次重试（{backoff}s后）: {exc}")
                time.sleep(backoff)

        if isinstance(last_exc, ServiceError):
            raise last_exc
        raise ServiceError("LLM 调用失败", detail=str(last_exc))

    def _resolve_base_url(self, base: Optional[str] = None) -> str:
        """解析 Nvidia OpenAI 兼容 endpoint。"""
        return (
            base
            or getattr(settings, "LLM_ENDPOINT_NVIDIA", None)
            or getattr(settings, "LLM_ENDPOINT", None)
            or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")

    def _call_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        model = model or self.model_text
        api_key = api_key or self.api_key
        if not api_key:
            raise ServiceError("LLM API key 未配置", detail="请配置 LLM API 密钥")

        url = f"{self._resolve_base_url()}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        resp = self._post(url, headers=self._auth_headers(api_key), json=payload, timeout=timeout or self.timeout)
        logger.debug("LLM API response", status=resp.status_code, request_id=resp.headers.get("X-Request-Id"))
        return resp.json()["choices"][0]["message"]["content"]

    def _call_vision(
        self,
        prompt: str,
        file_content_b64: str,
        file_name: Optional[str],
        override: Optional[Dict] = None,
    ) -> str:
        override = override or {}
        model = override.get("vision_model") or override.get("model") or self.model_vision
        api_key = override.get("api_key") or getattr(settings, "LLM_API_KEY_NVIDIA", "") or self.api_key
        base = self._resolve_base_url(override.get("endpoint"))
        if not api_key:
            raise ServiceError("LLM API key 未配置", detail="请配置 LLM API 密钥")

        # 限制 vision timeout 区间，防止过长等待耽误 fallback
        vision_timeout = float(self.vision_timeout or self.timeout or 45)
        vision_timeout = min(max(vision_timeout, 10.0), 45.0)

        data_url = self._build_data_url(file_content_b64, file_name)

        models_to_try = [model]
        if self.model_vision_fallback and self.model_vision_fallback not in models_to_try:
            models_to_try.append(self.model_vision_fallback)

        logger.info(
            "LLM vision call preparing",
            model=model, fallback_model=self.model_vision_fallback,
            endpoint=base, image_bytes=len(file_content_b64) if file_content_b64 else 0,
        )

        url = f"{base}/chat/completions"
        headers = self._auth_headers(api_key)
        last_exc: Optional[Exception] = None

        for idx, model_name in enumerate(models_to_try, start=1):
            payload = {
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "response_format": {"type": "json_object"},
            }
            t0 = time.time()
            try:
                logger.debug("LLM vision request sending", url=url, model=model_name,
                             timeout=vision_timeout, attempt=idx)
                resp = self._post(url, headers=headers, json=payload, timeout=vision_timeout)
                elapsed = round(time.time() - t0, 2)
                logger.info("LLM vision request succeeded", model=model_name,
                            status=resp.status_code, elapsed=elapsed,
                            request_id=resp.headers.get("X-Request-Id"), attempt=idx)
                return resp.json()["choices"][0]["message"]["content"]

            except Exception as exc:
                last_exc = exc
                elapsed = round(time.time() - t0, 2)
                is_timeout = "timed out" in str(exc).lower()
                logger.error("LLM vision request failed", model=model_name, elapsed=elapsed,
                             attempt=idx, error_type=type(exc).__name__, error_detail=str(exc)[:300])
                if is_timeout and idx < len(models_to_try):
                    logger.warning("LLM vision timeout, trying fallback vision model",
                                   from_model=model_name, to_model=models_to_try[idx])
                    continue
                raise

        if last_exc:
            raise last_exc
        raise ServiceError("LLM 调用失败", detail="多模态请求失败")

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：HTTP 层
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _auth_headers(api_key: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _post(self, url: str, **kwargs) -> requests.Response:
        """统一 POST，识别配额/限流错误并转为 ServiceError。"""
        kwargs.setdefault("timeout", self.timeout)
        try:
            resp = self._session.post(url, **kwargs)
            try:
                resp.raise_for_status()
                return resp
            except requests.exceptions.HTTPError as exc:
                self._raise_service_error_from_response(resp, exc)

        except requests.exceptions.ProxyError as exc:
            logger.error(f"LLM 代理失败: {exc}")
            raise ServiceError("LLM 调用失败",
                               detail="代理连接失败，请检查服务器网络/代理设置，或将 LLM_TRUST_ENV=True。")
        except requests.exceptions.SSLError as exc:
            logger.error(f"LLM SSL 失败: {exc}")
            raise ServiceError("LLM 调用失败", detail="SSL 握手失败，请检查证书或网络环境。")
        except requests.exceptions.RequestException as exc:
            detail = self._extract_error_detail(exc)
            logger.error(f"LLM 请求失败: {detail}")
            raise ServiceError("LLM 调用失败", detail=detail)

    @staticmethod
    def _raise_service_error_from_response(resp: requests.Response, http_exc: Exception) -> None:
        """将 HTTP 错误体解析为具体 ServiceError，不可达时退回原始文本。"""
        detail = str(http_exc)
        try:
            data = resp.json()
            err = data.get("error") or {}
            code = err.get("code") or err.get("type")
            msg = err.get("message") or detail

            if code == "AllocationQuota.FreeTierOnly":
                logger.error(f"LLM 配额错误: {code} - {msg}")
                raise ServiceError("LLM 配额不足",
                                   detail="免费额度已用完，请检查 Nvidia API 配额或账单设置后再试。")

            if code in {"QuotaExceeded", "RateLimitExceeded"}:
                logger.error(f"LLM 限流/配额错误: {code} - {msg}")
                raise ServiceError("LLM 调用受限", detail=f"LLM 配额或限流受限：{msg}")

            detail = f"{code}: {msg}"
        except ServiceError:
            raise
        except Exception:
            detail = f"{http_exc}; body={resp.text[:500]}"

        logger.error(f"LLM 请求失败: {detail}")
        raise ServiceError("LLM 调用失败", detail=detail)

    @staticmethod
    def _extract_error_detail(exc: Exception) -> str:
        resp = getattr(exc, "response", None)
        if resp is None:
            return str(exc)
        try:
            data = resp.json()
            err = data.get("error") or {}
            code = err.get("code") or err.get("type")
            msg = err.get("message") or str(exc)
            return f"{code}: {msg}"
        except Exception:
            return f"{exc}; body={resp.text[:500]}"

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：JSON 解析 / 修复
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_json_dict_response(self, raw: Optional[str]) -> Optional[Dict]:
        """宽松解析模型输出（支持代码围栏、尾随逗号、Python literal）。"""
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list):
            raw = "\n".join(str(x.get("text") if isinstance(x, dict) else x) for x in raw)

        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        candidates = [text]
        candidates.extend(self._extract_braced_chunks(text))
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 < end:
            candidates.append(text[start:end])

        for candidate in candidates:
            result = self._try_parse_candidate(candidate)
            if result is not None:
                return result
        return None

    @staticmethod
    def _try_parse_candidate(candidate: str) -> Optional[Dict]:
        for text in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        return None

    @staticmethod
    def _extract_braced_chunks(text: str) -> List[str]:
        """提取所有顶级 {...} 块。"""
        chunks: List[str] = []
        stack = 0
        start = -1
        in_str = esc = False

        for i, ch in enumerate(text):
            if in_str:
                esc = not esc and ch == "\\"
                if not esc and ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if stack == 0:
                    start = i
                stack += 1
            elif ch == "}" and stack > 0:
                stack -= 1
                if stack == 0 and start >= 0:
                    chunks.append(text[start:i + 1])
                    start = -1
        return chunks

    def _repair_json_with_llm(
        self,
        raw_response: str,
        override: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """使用文本模型将残缺的多模态输出修复为严格 JSON。"""
        if not raw_response:
            return None

        repair_prompt = (
            "将下面内容修正为严格 JSON 对象。"
            "只返回 JSON 对象本身，不要解释，不要代码块。\n"
            "要求包含键：document_type,text,summary,confidence_overall,basic_info,academic_info,"
            "certificate_info,financial_info,leave_info,courses,tables。\n"
            f"原始内容：\n{raw_response[:6000]}"
        )
        try:
            api_key = (override or {}).get("api_key") or self.api_key
            model = (override or {}).get("model") or self.model_text
            fixed = self._call_text(repair_prompt, model=model, api_key=api_key,
                                    timeout=min(self.timeout, 12))
            parsed = self._parse_json_dict_response(fixed)
            if parsed:
                logger.info("LLM vision JSON repaired by text model")
            return parsed
        except Exception as exc:
            logger.warning(f"LLM vision JSON repair failed: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # 内部：结果归一化
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_llm_image_result(self, parsed: Dict) -> Dict:
        """将多模态输出归一化到稳定 schema，并修正常见字段混淆。"""
        out = dict(parsed)

        out["confidence_overall"] = self._normalize_confidence_value(out.get("confidence_overall"))
        out["document_type"] = self._normalize_doc_type(out.get("document_type"), out.get("text"))
        out.setdefault("text", "")
        out.setdefault("summary", "")
        out.setdefault("courses", [])
        out.setdefault("tables", [])

        for sec in ("academic_info", "certificate_info", "financial_info", "leave_info"):
            if not isinstance(out.get(sec), dict):
                out[sec] = {}

        out["basic_info"] = self._normalize_basic_info(out.get("basic_info"))

        # CET 特定修正
        text_s = str(out.get("text") or "")
        if any(k in text_s for k in _CET_KEYWORDS):
            out["document_type"] = "考试证书"
            basic = out["basic_info"]
            for k in ("class", "major"):
                v = basic.get(k)
                if isinstance(v, (int, float)) or (
                    isinstance(v, str) and re.fullmatch(r"\d{2,4}", v.strip())
                ):
                    basic[k] = None

        return out

    @staticmethod
    def _normalize_basic_info(raw: object) -> Dict:
        basic: Dict = raw if isinstance(raw, dict) else {}
        basic = {k: basic.get(k) for k in _BASIC_INFO_FIELDS}

        # 清除占位字符串
        for k, v in list(basic.items()):
            if isinstance(v, str) and (not v.strip() or v.strip() in _BASIC_INFO_PLACEHOLDERS):
                basic[k] = None

        # 身份证号与学号互换修正
        id_number = str(basic.get("id_number") or "").strip()
        student_id = str(basic.get("student_id") or "").strip()
        is_id = bool(re.fullmatch(r"\d{17}[\dXx]", id_number))
        is_sid = bool(re.fullmatch(r"\d{17}[\dXx]", student_id))
        if not is_id and is_sid:
            basic["id_number"], basic["student_id"] = student_id, id_number or None

        # 格式校验
        if basic.get("id_number") and not re.fullmatch(r"\d{17}[\dXx]", str(basic["id_number"]).strip()):
            basic["id_number"] = None
        if basic.get("phone") and not re.fullmatch(r"1\d{10}", str(basic["phone"]).strip()):
            basic["phone"] = None
        if basic.get("name"):
            nm = str(basic["name"]).strip()
            if len(nm) < 2 or len(nm) > 12 or nm in {"学生", "个人信息", "姓名"}:
                basic["name"] = None

        return basic

    @staticmethod
    def _normalize_confidence_value(value: object, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            num = float(value)
        else:
            s = str(value).strip().lower()
            if not s:
                return default
            label_map = {
                "high": 0.85, "high confidence": 0.85, "strong": 0.85,
                "medium": 0.6, "moderate": 0.6,
                "low": 0.3, "weak": 0.3,
            }
            if s in label_map:
                return label_map[s]
            if s.endswith("%"):
                try:
                    num = float(s[:-1].strip()) / 100.0
                except Exception:
                    return default
            else:
                try:
                    num = float(s)
                except Exception:
                    return default

        if 1.0 < num <= 100.0:
            num /= 100.0
        return max(0.0, min(1.0, num))

    @staticmethod
    def _normalize_doc_type(doc_type: object, text: object) -> str:
        t = str(doc_type or "").strip()
        text_s = str(text or "")

        mapped = _DOC_TYPE_MAP.get(t.lower())
        if mapped:
            t = mapped

        if any(k in text_s for k in _CET_KEYWORDS):
            if t in {"", "学生证", "Student ID", "Student ID Card", "student id card"}:
                t = "考试证书"

        return t or "其他"

    @staticmethod
    def _build_data_url(file_content_b64: str, file_name: Optional[str]) -> str:
        """将 base64 内容构建为 data URL（支持前端传入的原始 data URL）。"""
        if file_content_b64.strip().startswith("data:"):
            return file_content_b64

        mime = "image/png"
        if file_name and "." in file_name:
            ext = file_name.lower().rsplit(".", 1)[-1]
            mime = _MIME_MAP.get(ext, mime)

        raw_b64 = file_content_b64.split(",")[-1]
        return f"data:{mime};base64,{raw_b64}"

    def _log_raw_response(self, title: str, response: Optional[str]) -> None:
        if not response:
            logger.info(f"{title} | len=0 | snippet=(empty)")
            return
        snippet = response[: self.raw_log_limit].replace("\n", "\\n")
        logger.info(f"{title} | len={len(response)} | snippet={snippet}")