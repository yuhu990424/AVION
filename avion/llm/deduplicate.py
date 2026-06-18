from __future__ import annotations

import re

from avion.llm.schemas import RemoteSensingCandidate


def normalize_caption_for_dedup(caption: str) -> str:
    caption = caption.lower().strip()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption)
    return caption


def deduplicate_candidates(candidates: list[RemoteSensingCandidate]) -> list[RemoteSensingCandidate]:
    seen: set[str] = set()
    unique: list[RemoteSensingCandidate] = []
    for candidate in candidates:
        key = normalize_caption_for_dedup(candidate.caption)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique

