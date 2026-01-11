from app.services.factory import (
    get_ocr_service,
    get_nlp_service,
    get_llm_service,
    get_final_ai_service,
)

# --- NLPService compatibility patch (safe no-op if already present) ---
try:
    from app.services.nlp_service import NLPService
    from app.services import nlp_service_compat_patch as _nlp_patch

    for _name in (
        "_reconstruct_text",
        "_build_fail_analysis",
        "_template_from_expected",
        "_suggest_repair",
        "_fix_broken_numbers",
    ):
        if not hasattr(NLPService, _name) and hasattr(_nlp_patch, _name):
            setattr(NLPService, _name, getattr(_nlp_patch, _name))
except Exception:
    # Don't fail package import due to optional patch
    pass

__all__ = [
    "get_ocr_service",
    "get_nlp_service",
    "get_llm_service",
    "get_final_ai_service",
]
