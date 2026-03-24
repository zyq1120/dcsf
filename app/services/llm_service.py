"""
LLM Service (NVIDIA only): 提供 OCR 增强、字段兜底抽取、多模态读图、文档分类。
已移除 OpenAI/Gemini/Qwen，所有调用走 NVIDIA OpenAI-compatible 接口。
"""
import json
import time
from typing import Dict, Optional, List
import os
import requests
from loguru import logger

from app.config import settings
from app.utils.errors import ServiceError
from app.services.llm_prompts import LLM_NORMALIZE_PROMPT as DEFAULT_LLM_NORMALIZE_PROMPT



class LLMService:
    def __init__(self):
        self.enabled = False
        self.provider = "nvidia"
        self.api_key = None
        self.model_text = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        self.model_vision = "nvidia/neva-22b"
        self.model = self.model_text
        self.timeout = 60
        self.retry = 1
        self.trust_env = False
        # 固定只使用 NVIDIA
        self.enabled = settings.ENABLE_LLM_FALLBACK
        self.provider = "nvidia"

        # 再从 settings 覆盖
        self.enabled = getattr(settings, "ENABLE_LLM_FALLBACK", False)
        self.api_key = (
                getattr(settings, "LLM_API_KEY_NVIDIA", None)
                or getattr(settings, "LLM_API_KEY", None)
                or getattr(settings, "LLM_API_KEY_QWEN", None)
        )
        self.model_text = (
                getattr(settings, "LLM_MODEL_NVIDIA_TEXT", None)
                or getattr(settings, "LLM_MODEL_TEXT", None)
                or "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        )

        self.model_vision = (
            getattr(settings, "LLM_MODEL_NVIDIA_VISION", None)
            or getattr(settings, "LLM_MODEL_VISION", None)
            or "nvidia/nemotron-nano-12b-v2-vl"
        )

        self.model = self.model_text
        self.timeout = getattr(settings, "LLM_TIMEOUT", 60)
        self.retry = getattr(settings, "LLM_RETRY", 1)
        self.trust_env = getattr(settings, "LLM_TRUST_ENV", False)

        import requests
        self._session = requests.Session()
        self._session.trust_env = self.trust_env
        if not self.trust_env:
            self._session.proxies = {}

        logger.info(
            "LLMService initialized (NVIDIA only)",
            enabled=self.enabled,
            model_text=self.model_text,
            model_vision=self.model_vision,
            timeout=self.timeout,
            retry=self.retry,
            trust_env=self.trust_env,
        )

        self._validate_connectivity()

    def enhance_ocr(
        self,
        ocr_text: str,
        confidence: float,
        override: Optional[Dict] = None,
    ) -> Dict:
        if not self.enabled:
            return {
                "enhanced": False,
                "text": ocr_text,
                "confidence": confidence,
                "source": "ocr",
            }

        prompt = (
            "你是 OCR 纠错器，请在保持原文语义的基础上纠正常见 OCR 错误，输出纯文本。\n"
            "要求：不编造内容；保持换行；只返回纠正后的文本。"
        )

        try:
            fixed = self._call_llm(
                prompt + "\n\n原文：\n" + ocr_text,
                override=override,
            )
            return {
                "enhanced": True,
                "text": fixed.strip(),
                "confidence": confidence,
                "source": "llm",
            }
        except Exception as exc:
            logger.warning(f"OCR enhance failed: {exc}")
            return {
                "enhanced": False,
                "text": ocr_text,
                "confidence": confidence,
                "source": "ocr",
            }

    def extract_with_llm(
        self,
        text: str,
        template_config: Dict,
        override: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        基于文本的字段兜底抽取，按 template_config.fields 输出 {"extract_details":[...]}。
        """
        if not self.enabled:
            return None

        fields = template_config.get("fields") or []
        desc = []
        for f in fields:
            name = f.get("name")
            ftype = f.get("type", "TEXT")
            d = f.get("description", "")
            desc.append(f"- {name} ({ftype}): {d}")

        prompt = (
            "你是高校文档结构化解析引擎。请从文本中提取下列字段，输出严格 JSON："
            '{"extract_details":[{"field_name":"","field_value":"","field_type":"","confidence":0.0}]}。\n'
            "规则：\n"
            "1) 缺失用 null；2) 不要返回 NaN/undefined；3) confidence 0~1；4) 只输出 JSON。\n"
            "字段列表：\n" + "\n".join(desc) + "\n文本：\n" + text
        )

        try:
            raw = self._call_llm(prompt, override=override)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            parsed = json.loads(raw[start:end])
            return parsed
        except Exception as exc:
            logger.error(f"LLM extract_with_llm parse failed: {exc}")
            return None

    def classify_with_llm(
        self,
        text: str,
        override: Optional[Dict] = None,
    ) -> Optional[Dict]:
        if not self.enabled:
            return None

        prompt = (
            "根据以下高校文档文本判断文档类型，输出 JSON："
            '{"document_type":"","confidence":0.0,"probabilities":{}}。'
            "类型示例：成绩单、学籍卡、在校证明、毕业证书/学历证书、学位证书、合同、申请、通知、课表、其他。\n文本：\n"
            + text
        )

        try:
            raw = self._call_llm(prompt, override=override)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            parsed = json.loads(raw[start:end])
            parsed["source"] = "llm"
            return parsed
        except Exception as exc:
            logger.warning(f"LLM classify parse failed: {exc}")
            return {
                "document_type": "generic_document",
                "confidence": 0.0,
                "probabilities": {},
                "source": "llm",
            }

    def extract_with_llm_image(
        self,
        file_content_b64: str,
        file_name: Optional[str],
        template_config: Dict,
        override: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        多模态直读（NVIDIA Vision 模型），返回 extract_details/courses 可选。
        """
        if not self.enabled:
            return None

        fields_desc: List[str] = []
        for field in template_config.get("fields", []):
            desc = field.get("description", "请提取该字段")
            fields_desc.append(
                f"- {field['name']} ({field.get('type', 'TEXT')}): {desc}"
            )

        prompt = (
            "你是高校文档结构化解析引擎，直接阅读图片内容（含表格/手写）并输出统一 JSON。\n"
            "必须严格可解析，且遵循以下归一化指导（保持固定 schema，未知为 null）：\n"
            + DEFAULT_LLM_NORMALIZE_PROMPT
            + "\n规则补充：不要用表头/列标题（如“姓名/性别/班级/班主任”等词）作为字段值；"
            + "无法识别时请返回 null；若字段为姓名/班级/联系方式等，请填写真实识别内容，"
            + "严禁返回“姓名”“性别”等表头词。\n"
            + "字段列表（可选，若能识别则填充）：\n"
            + ("\n".join(fields_desc) if fields_desc else "可选字段")
        )

        logger.info(
            "LLM image extract triggered",
            provider="nvidia",
            model=self.model_vision,
            has_image=bool(file_content_b64),
        )

        try:
            response = self._call_nvidia_vision(
                prompt,
                file_content_b64,
                file_name,
                override=override,
            )
            start = response.find("{")
            end = response.rfind("}") + 1
            parsed = json.loads(response[start:end])
            return parsed
        except Exception as exc:
            logger.error(f"LLM image JSON parse failed: {exc}")
            return None

    def _call_llm(
        self,
        prompt: str,
        override: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> str:
        if not self.enabled:
            raise ServiceError("LLM 未启用")

        override = override or {}
        api_key = override.get("api_key") or self.api_key
        model = override.get("model") or self.model_text

        if not api_key:
            raise ServiceError("LLM API key 未配置", detail="请配置 NVIDIA API Key")

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.retry + 2):
            try:
                logger.debug(
                    "LLM call attempt",
                    attempt=attempt,
                    provider="nvidia",
                    model=model,
                    timeout=timeout or self.timeout,
                )
                return self._call_nvidia(
                    prompt=prompt,
                    model=model,
                    api_key=api_key,
                    timeout=timeout,
                    override=override,
                )
            except Exception as exc:
                last_exc = exc

                if isinstance(exc, ServiceError):
                    detail = getattr(exc, "detail", "") or ""
                    if any(
                        kw in detail
                        for kw in (
                            "Quota",
                            "RateLimit",
                            "api_key 未配置",
                            "Unauthorized",
                            "Permission",
                        )
                    ):
                        logger.warning(f"LLM 调用出现不可重试错误，停止重试: {exc}")
                        break

                backoff = min(3 * attempt, 8)
                logger.warning(
                    f"LLM 调用失败，第 {attempt} 次重试即将开始: {exc}，等待 {backoff}s"
                )
                time.sleep(backoff)

        if isinstance(last_exc, ServiceError):
            raise last_exc
        raise ServiceError("LLM 调用失败", detail=str(last_exc))

    def _call_nvidia(
        self,
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        override: Optional[Dict] = None,
    ) -> str:
        override = override or {}
        model = model or self.model_text
        api_key = api_key or self.api_key

        if not api_key:
            raise ServiceError("LLM API key 未配置", detail="请配置 NVIDIA API Key")

        base = (
            override.get("endpoint")
            or getattr(settings, "LLM_ENDPOINT_NVIDIA", None)
            or getattr(settings, "LLM_ENDPOINT", None)
            or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")

        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0,
        }

        resp = self._post(url, headers=headers, json=payload, timeout=timeout or self.timeout)
        logger.debug(
            "NVIDIA API response",
            status=resp.status_code,
            request_id=resp.headers.get("X-Request-Id"),
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_nvidia_vision(
        self,
        prompt: str,
        file_content_b64: str,
        file_name: Optional[str],
        override: Optional[Dict] = None,
    ) -> str:
        override = override or {}

        model = override.get("model") or self.model_vision
        api_key = override.get("api_key") or self.api_key
        base = (
            override.get("endpoint")
            or getattr(settings, "LLM_ENDPOINT_NVIDIA", None)
            or getattr(settings, "LLM_ENDPOINT", None)
            or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")

        if not api_key:
            raise ServiceError("LLM API key 未配置", detail="请配置 NVIDIA API Key")

        if file_content_b64.strip().startswith("data:"):
            data_url = file_content_b64
        else:
            mime = "image/png"
            if file_name and "." in file_name:
                ext = file_name.lower().split(".")[-1]
                mime = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "webp": "image/webp",
                    "bmp": "image/bmp",
                    "tif": "image/tiff",
                    "tiff": "image/tiff",
                }.get(ext, mime)
            data_url = f"data:{mime};base64,{file_content_b64.split(',')[-1]}"

        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0,
        }

        resp = self._post(url, headers=headers, json=payload, timeout=self.timeout)
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _post(self, url: str, **kwargs) -> requests.Response:
        """
        统一 POST 封装：
        - 解析 NVIDIA 接口错误
        - 转换为 ServiceError
        """
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        try:
            resp = self._session.post(url, **kwargs)
            try:
                resp.raise_for_status()
                return resp
            except requests.exceptions.HTTPError as exc:
                detail = str(exc)
                try:
                    data = resp.json()
                    err = data.get("error") or {}
                    code = err.get("code") or err.get("type") or resp.status_code
                    msg = err.get("message") or detail
                    detail = f"{code}: {msg}"
                except Exception:
                    detail = f"{exc}; body={resp.text[:500]}"

                logger.error(f"LLM 请求失败: {detail}")
                raise ServiceError("LLM 调用失败", detail=detail)

        except requests.exceptions.ProxyError as exc:
            logger.error(f"LLM 请求代理失败: {exc}")
            raise ServiceError(
                "LLM 调用失败",
                detail="代理连接失败，请检查服务器网络/代理设置，或调整 LLM_TRUST_ENV。",
            )
        except requests.exceptions.SSLError as exc:
            logger.error(f"LLM SSL 失败: {exc}")
            raise ServiceError(
                "LLM 调用失败",
                detail="SSL 握手失败，请检查证书或网络环境。",
            )
        except requests.exceptions.RequestException as exc:
            detail = str(exc)
            try:
                resp = getattr(exc, "response", None)
                if resp is not None:
                    try:
                        data = resp.json()
                        err = data.get("error") or {}
                        code = err.get("code") or err.get("type") or resp.status_code
                        msg = err.get("message") or detail
                        detail = f"{code}: {msg}"
                    except Exception:
                        detail = f"{exc}; body={resp.text[:500]}"
            except Exception:
                pass

            logger.error(f"LLM 请求失败: {detail}")
            raise ServiceError("LLM 调用失败", detail=detail)

    def _validate_connectivity(self):
        if not self.enabled:
            logger.debug("LLM connectivity check skipped (disabled)")
            return

        if not self.api_key:
            logger.warning("LLM connectivity check skipped: api_key 未配置")
            return

        try:
            resp = self._call_nvidia(
                "返回 ok",
                timeout=min(self.timeout, 8) if self.timeout else 8,
            )
            logger.info(
                "LLM connectivity ok",
                provider="nvidia",
                sample=str(resp)[:20],
            )
        except Exception as exc:
            logger.warning(f"LLM connectivity check failed: {exc}")