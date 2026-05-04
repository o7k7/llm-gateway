import re
from functools import lru_cache

from magika import Magika


@lru_cache(maxsize=1)
class CodeDetectionService:
    def __init__(self):
        self.magika = Magika()

    _QUICK_CODE_REGEX = re.compile(
        r"^(print\(.*\)|console\.log\(.*\)|System\.out\.println"
        r"\(.*\)|SELECT .* FROM .*|def .*\(.*\):|class .*[:\(])",
        re.IGNORECASE,
    )

    def is_code(self, text: str) -> bool:
        clean_text = text.strip()
        if not clean_text:
            return False

        if len(clean_text) < 50 and self._QUICK_CODE_REGEX.match(clean_text):
            return True

        prediction = self.magika.identify_bytes(clean_text.encode("utf-8"))
        return prediction.output.group in {"code", "web"}
