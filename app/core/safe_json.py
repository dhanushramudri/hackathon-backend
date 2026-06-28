import math
from typing import Any

from fastapi.responses import JSONResponse

def _sanitize_nan(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return super().render(_sanitize_nan(content))
