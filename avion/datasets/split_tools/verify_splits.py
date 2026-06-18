from __future__ import annotations

from typing import Any


def verify_fewshot_split(split: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    train_ids = {row["sample_id"] for row in split.get("train", [])}
    test_ids = {row["sample_id"] for row in split.get("test", [])}
    overlap = train_ids & test_ids
    if overlap:
        errors.append(f"train/test overlap: {len(overlap)} samples")
    shots = int(split["num_shots"])
    counts: dict[str, int] = {}
    for row in split.get("train", []):
        counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
    bad = {name: count for name, count in counts.items() if count != shots}
    if bad:
        errors.append(f"classes with wrong shot count: {bad}")
    return errors


def verify_base_novel_split(split: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    base = set(split.get("base_classes", []))
    novel = set(split.get("novel_classes", []))
    if base & novel:
        errors.append("base/novel class overlap")
    train_classes = {row["class_name"] for row in split.get("train_base", [])}
    if not train_classes <= base:
        errors.append("train_base contains non-base classes")
    if any(row["class_name"] in novel for row in split.get("train_base", [])):
        errors.append("novel class appears in training")
    return errors

