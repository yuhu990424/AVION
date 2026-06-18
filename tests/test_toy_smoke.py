from pathlib import Path

from avion.smoke.build_toy_artifacts import build_toy_artifacts


def test_build_toy_artifacts(tmp_path):
    paths = build_toy_artifacts(tmp_path)
    assert Path(paths["metadata"]).exists()
    assert Path(paths["split"]).exists()
    assert Path(paths["annotations"]).exists()
    assert (Path(paths["prototype_dir"]) / "text_prototypes_selective.pt").exists()
    assert (Path(paths["prototype_dir"]) / "candidate_scores.jsonl").exists()

