from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Any

from avion.utils.io import ensure_dir, read_json


def _parse_seed(text: str) -> int | None:
    match = re.fullmatch(r"seed(\d+)", text)
    return int(match.group(1)) if match else None


def _parse_shots(text: str) -> int | None:
    match = re.fullmatch(r"shots_(\d+)", text)
    return int(match.group(1)) if match else None


def infer_run_fields(run_dir: str | Path) -> dict[str, Any]:
    parts = Path(run_dir).parts
    fields: dict[str, Any] = {}
    for part in reversed(parts):
        seed = _parse_seed(part)
        if seed is not None:
            fields["seed"] = seed
            break
    for part in parts:
        shots = _parse_shots(part)
        if shots is not None:
            fields["shots"] = shots
            break

    if "base2new" in parts:
        idx = parts.index("base2new")
        fields["protocol"] = "base_to_novel"
        if idx + 1 < len(parts):
            fields["split"] = parts[idx + 1]
        if idx + 2 < len(parts):
            fields["dataset"] = parts[idx + 2]
        if idx + 4 < len(parts):
            fields["model"] = parts[idx + 4]
        if idx + 5 < len(parts):
            fields["tag"] = parts[idx + 5]
        return fields

    if "retrieval" in parts:
        idx = parts.index("retrieval")
        fields["protocol"] = "retrieval"
        if idx + 1 < len(parts):
            fields["dataset"] = parts[idx + 1]
        if idx + 2 < len(parts):
            fields["model"] = parts[idx + 2]
        if idx + 3 < len(parts):
            fields["tag"] = parts[idx + 3]
        return fields

    for idx, part in enumerate(parts):
        if _parse_shots(part) is not None:
            fields["protocol"] = "few_shot"
            if idx - 1 >= 0:
                fields["dataset"] = parts[idx - 1]
            if idx + 1 < len(parts):
                fields["model"] = parts[idx + 1]
            if idx + 2 < len(parts):
                fields["tag"] = parts[idx + 2]
            return fields
    return fields


def collect_metric_rows(root: str | Path, protocol: str | None = None) -> list[dict[str, Any]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("metrics.json")):
        metrics = read_json(path)
        fields = infer_run_fields(path.parent)
        row = {"run_dir": str(path.parent), **fields, **metrics}
        if protocol is None or row.get("protocol") == protocol:
            rows.append(row)
    return rows


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["protocol", "dataset", "shots", "model", "tag", "seed", "split", "run_dir"]
    keys = sorted({key for row in rows for key in row})
    return [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    keys = _fieldnames(rows)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    keys = _fieldnames(rows)
    with target.open("w", encoding="utf-8") as handle:
        if not rows:
            handle.write("No metrics found.\n")
            return
        handle.write("| " + " | ".join(keys) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(keys)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |\n")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = ["protocol", "dataset", "shots", "model", "tag", "split"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_keys)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    ignored_numeric = {"seed", "epochs"}
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        summary = {field: value for field, value in zip(group_keys, key) if value is not None}
        seeds = sorted(row["seed"] for row in group_rows if isinstance(row.get("seed"), int))
        if seeds:
            summary["seeds"] = ",".join(str(seed) for seed in seeds)
        metric_keys = sorted(
            {
                metric
                for row in group_rows
                for metric, value in row.items()
                if metric not in group_keys and metric not in ignored_numeric and _is_number(value)
            }
        )
        for metric in metric_keys:
            values = [float(row[metric]) for row in group_rows if _is_number(row.get(metric))]
            if not values:
                continue
            summary[f"{metric}_mean"] = sum(values) / len(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary["num_runs"] = len(group_rows)
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--md")
    parser.add_argument("--summary-csv")
    parser.add_argument("--summary-md")
    parser.add_argument("--protocol", choices=["few_shot", "base_to_novel", "retrieval"])
    args = parser.parse_args()
    rows = collect_metric_rows(args.root, protocol=args.protocol)
    write_csv(rows, args.csv)
    if args.md:
        write_markdown(rows, args.md)
    summaries = summarize_metric_rows(rows)
    if args.summary_csv:
        write_csv(summaries, args.summary_csv)
    if args.summary_md:
        write_markdown(summaries, args.summary_md)
    print(f"Collected {len(rows)} metrics files.")
    if args.summary_csv or args.summary_md:
        print(f"Summarized {len(summaries)} metric groups.")


if __name__ == "__main__":
    main()
