from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def collect_metrics(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append({"path": str(path), **metrics})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = collect_metrics(Path(args.root))
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        print("No metrics.json files found.")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    print("\t".join(keys))
    for row in rows:
        print("\t".join(str(row.get(key, "")) for key in keys))


if __name__ == "__main__":
    main()

