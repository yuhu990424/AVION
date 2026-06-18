from __future__ import annotations

import re
from dataclasses import dataclass, field


DEFAULT_POSITIVE_TOKENS = [
    "overhead",
    "aerial view",
    "satellite imagery",
    "nadir",
    "orthorectified",
    "multispectral",
    "SAR",
]

DEFAULT_NEGATIVE_TOKENS = [
    "street",
    "indoor",
    "selfie",
    "portrait",
    "close-up",
    "ground level",
]


@dataclass(frozen=True)
class RSFlagConfig:
    positive_tokens: list[str] = field(default_factory=lambda: list(DEFAULT_POSITIVE_TOKENS))
    negative_tokens: list[str] = field(default_factory=lambda: list(DEFAULT_NEGATIVE_TOKENS))
    min_words: int = 6
    max_words: int = 20


@dataclass(frozen=True)
class RSFlagResult:
    rs_flag: int
    positive_terms_detected: list[str]
    negative_terms_detected: list[str]
    word_count: int
    reasons: list[str]


def _contains_term(text: str, term: str) -> bool:
    # Word-boundary matching for both single tokens and short phrases.
    escaped = re.escape(term.lower())
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def count_words(caption: str) -> int:
    return len([part for part in re.split(r"\s+", caption.strip()) if part])


def evaluate_rs_flag(caption: str, config: RSFlagConfig | None = None) -> RSFlagResult:
    config = config or RSFlagConfig()
    positives = [term for term in config.positive_tokens if _contains_term(caption, term)]
    negatives = [term for term in config.negative_tokens if _contains_term(caption, term)]
    words = count_words(caption)

    reasons: list[str] = []
    if not positives:
        reasons.append("missing_positive_token")
    if negatives:
        reasons.append("contains_negative_token")
    if words < config.min_words:
        reasons.append("too_short")
    if words > config.max_words:
        reasons.append("too_long")

    return RSFlagResult(
        rs_flag=0 if reasons else 1,
        positive_terms_detected=positives,
        negative_terms_detected=negatives,
        word_count=words,
        reasons=reasons,
    )

