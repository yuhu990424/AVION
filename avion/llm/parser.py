from __future__ import annotations

import json

from avion.llm.schemas import CandidateResponse


def parse_candidate_response(text: str, expected_count: int | None = None) -> CandidateResponse:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Gemini JSON response must be an object.")
    return CandidateResponse.from_dict(payload, expected_count=expected_count)

