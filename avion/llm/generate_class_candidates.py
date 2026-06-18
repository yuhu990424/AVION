from __future__ import annotations

import argparse
from pathlib import Path

from avion.llm.gemini_client import GeminiCandidateGenerator
from avion.llm.rs_flag import evaluate_rs_flag
from avion.utils.hashing import stable_json_sha256
from avion.utils.io import read_json, write_jsonl


def build_rows(dataset: str, class_names: list[str], kp: int, model_name: str) -> list[dict[str, object]]:
    generator = GeminiCandidateGenerator(model_name=model_name)
    rows: list[dict[str, object]] = []
    for class_name in class_names:
        response = generator.generate(class_name, n=kp)
        raw_hash = stable_json_sha256(response.to_dict())
        for index, candidate in enumerate(response.candidates):
            flag = evaluate_rs_flag(candidate.caption)
            rows.append(
                {
                    "dataset": dataset,
                    "class_name": class_name,
                    "candidate_index": index,
                    "caption": candidate.caption,
                    "viewpoint": candidate.viewpoint,
                    "visual_cues": candidate.visual_cues,
                    "spatial_cues": candidate.spatial_cues,
                    "llm_model": model_name,
                    "prompt_version": "rs_scene_v1",
                    "kp": kp,
                    "raw_response_sha256": raw_hash,
                    "rs_flag": flag.rs_flag,
                    "positive_terms_detected": flag.positive_terms_detected,
                    "negative_terms_detected": flag.negative_terms_detected,
                    "word_count": flag.word_count,
                    "teacher_similarity": None,
                    "mad_z": None,
                    "kept": None,
                    "aggregation_weight": None,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kp", type=int, default=30)
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    metadata = read_json(args.metadata)
    class_names = [row["canonical_name"] for row in metadata["classes"]]
    rows = build_rows(metadata["dataset"], class_names, kp=args.kp, model_name=args.model)
    write_jsonl(rows, Path(args.out))


if __name__ == "__main__":
    main()

