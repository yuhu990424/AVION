from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_VIEWPOINTS = {"overhead", "aerial", "nadir", "satellite"}


@dataclass(frozen=True)
class RemoteSensingCandidate:
    caption: str
    viewpoint: str
    visual_cues: list[str] = field(default_factory=list)
    spatial_cues: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RemoteSensingCandidate":
        caption = str(row.get("caption", "")).strip()
        viewpoint = str(row.get("viewpoint", "")).strip().lower()
        if not caption:
            raise ValueError("Candidate caption is empty.")
        if viewpoint not in VALID_VIEWPOINTS:
            raise ValueError(f"Invalid viewpoint: {viewpoint}")
        return cls(
            caption=caption,
            viewpoint=viewpoint,
            visual_cues=[str(item) for item in row.get("visual_cues", [])],
            spatial_cues=[str(item) for item in row.get("spatial_cues", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "viewpoint": self.viewpoint,
            "visual_cues": self.visual_cues,
            "spatial_cues": self.spatial_cues,
        }


@dataclass(frozen=True)
class CandidateResponse:
    candidates: list[RemoteSensingCandidate]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], expected_count: int | None = None) -> "CandidateResponse":
        rows = payload.get("candidates")
        if not isinstance(rows, list):
            raise ValueError("Gemini response must contain a candidates list.")
        candidates = [RemoteSensingCandidate.from_dict(row) for row in rows]
        if expected_count is not None and len(candidates) != expected_count:
            raise ValueError(f"Expected {expected_count} candidates, got {len(candidates)}.")
        return cls(candidates=candidates)

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": [candidate.to_dict() for candidate in self.candidates]}

