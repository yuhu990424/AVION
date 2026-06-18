from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avion.utils.io import read_json


@dataclass(frozen=True)
class ClassInfo:
    class_id: int
    raw_name: str
    canonical_name: str
    prompt_name: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ClassInfo":
        return cls(
            class_id=int(row["class_id"]),
            raw_name=str(row.get("raw_name", row.get("class_name", ""))),
            canonical_name=str(row["canonical_name"]),
            prompt_name=str(row.get("prompt_name", row["canonical_name"])),
        )


@dataclass(frozen=True)
class ClassificationSample:
    sample_id: str
    image_path: str
    class_id: int
    class_name: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ClassificationSample":
        return cls(
            sample_id=str(row["sample_id"]),
            image_path=str(row["image_path"]),
            class_id=int(row["class_id"]),
            class_name=str(row["class_name"]),
        )


@dataclass(frozen=True)
class ClassificationMetadata:
    dataset: str
    version: str
    classes: list[ClassInfo]
    samples: list[ClassificationSample]

    @classmethod
    def from_json(cls, path: str | Path) -> "ClassificationMetadata":
        payload = read_json(path)
        return cls(
            dataset=str(payload["dataset"]),
            version=str(payload.get("version", "unknown")),
            classes=[ClassInfo.from_dict(row) for row in payload["classes"]],
            samples=[ClassificationSample.from_dict(row) for row in payload["samples"]],
        )

    @property
    def class_names(self) -> list[str]:
        return [info.canonical_name for info in sorted(self.classes, key=lambda item: item.class_id)]

    def samples_by_class(self) -> dict[str, list[ClassificationSample]]:
        grouped: dict[str, list[ClassificationSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.class_name, []).append(sample)
        for rows in grouped.values():
            rows.sort(key=lambda sample: sample.sample_id)
        return grouped

