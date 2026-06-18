from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    task: str
    expected_classes: int | None = None
    expected_images: int | None = None


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "aid": DatasetSpec("aid", "AID", "classification", 30, 10000),
    "resisc45": DatasetSpec("resisc45", "RESISC45", "classification", 45, 31500),
    "eurosat": DatasetSpec("eurosat", "EuroSAT", "classification", 10, 27000),
    "whu_rs19": DatasetSpec("whu_rs19", "WHU-RS19", "classification", 19, 1005),
    "patternnet": DatasetSpec("patternnet", "PatternNet", "classification", 38, 30400),
    "ucmerced": DatasetSpec("ucmerced", "UCMerced", "classification", 21, 2100),
    "rsitmd": DatasetSpec("rsitmd", "RSITMD", "retrieval"),
    "rsicd": DatasetSpec("rsicd", "RSICD", "retrieval"),
}


def metadata_path(data_root: str | Path, dataset_key: str) -> Path:
    return Path(data_root) / "processed" / dataset_key / "metadata.json"

