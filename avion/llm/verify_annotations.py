from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from avion.llm.deduplicate import normalize_caption_for_dedup
from avion.llm.rs_flag import evaluate_rs_flag
from avion.llm.schemas import RemoteSensingCandidate
from avion.utils.io import read_json, read_jsonl, write_json


def verify_annotation_rows(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    expected_kp: int = 30,
) -> dict[str, Any]:
    expected_classes = [row["canonical_name"] for row in metadata.get("classes", [])]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    negative_terms: Counter[str] = Counter()
    word_counts: list[int] = []
    rs_flag_counts: Counter[int] = Counter()
    duplicate_counts: dict[str, int] = {}

    for idx, row in enumerate(rows):
        class_name = str(row.get("class_name", ""))
        by_class[class_name].append(row)
        try:
            RemoteSensingCandidate.from_dict(row)
        except Exception as exc:
            errors.append(f"row {idx} schema error: {exc}")
        flag = evaluate_rs_flag(str(row.get("caption", "")))
        row_flag = row.get("rs_flag")
        if row_flag is not None and int(row_flag) != flag.rs_flag:
            errors.append(f"row {idx} rs_flag mismatch: stored={row_flag} computed={flag.rs_flag}")
        rs_flag_counts[flag.rs_flag] += 1
        word_counts.append(flag.word_count)
        negative_terms.update(flag.negative_terms_detected)

    for class_name in expected_classes:
        count = len(by_class.get(class_name, []))
        if count != expected_kp:
            errors.append(f"class {class_name} expected {expected_kp} candidates, found {count}")
        normalized = [normalize_caption_for_dedup(str(row.get("caption", ""))) for row in by_class.get(class_name, [])]
        duplicates = len(normalized) - len(set(normalized))
        duplicate_counts[class_name] = duplicates

    unexpected_classes = sorted(set(by_class) - set(expected_classes))
    if unexpected_classes:
        errors.append(f"unexpected classes: {unexpected_classes}")

    total = len(rows)
    return {
        "dataset": metadata.get("dataset"),
        "expected_kp": expected_kp,
        "num_expected_classes": len(expected_classes),
        "num_rows": total,
        "classes": {
            class_name: {
                "count": len(by_class.get(class_name, [])),
                "duplicates": duplicate_counts.get(class_name, 0),
            }
            for class_name in expected_classes
        },
        "rs_flag_pass_rate": float(rs_flag_counts[1] / total) if total else 0.0,
        "rs_flag_counts": {str(key): value for key, value in sorted(rs_flag_counts.items())},
        "negative_terms": dict(negative_terms),
        "word_count_min": min(word_counts) if word_counts else None,
        "word_count_max": max(word_counts) if word_counts else None,
        "word_count_mean": float(sum(word_counts) / len(word_counts)) if word_counts else None,
        "num_duplicate_captions": int(sum(duplicate_counts.values())),
        "errors": errors,
        "ok": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-kp", type=int, default=30)
    args = parser.parse_args()

    report = verify_annotation_rows(
        read_json(args.metadata),
        list(read_jsonl(args.annotations)),
        expected_kp=args.expected_kp,
    )
    write_json(report, args.out)


if __name__ == "__main__":
    main()

