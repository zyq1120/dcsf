"""
Utility helpers for standardized API responses.
"""
from typing import Any, Dict


def ok(data: Any = None, message: str = "success") -> Dict[str, Any]:
    return {"code": 200, "message": message, "data": data}


def fail(code: int, message: str, detail: Any = None) -> Dict[str, Any]:
    payload = {"code": code, "message": message}
    if detail is not None:
        payload["detail"] = detail
    return payload

