from avion.datasets.base import ClassificationMetadata, ClassificationSample, ClassInfo
from avion.datasets.split_tools.build_base_novel_splits import build_base_novel_split
from avion.datasets.split_tools.build_fewshot_splits import build_fewshot_split
from avion.datasets.split_tools.verify_splits import verify_base_novel_split, verify_fewshot_split


def _metadata() -> ClassificationMetadata:
    classes = [
        ClassInfo(i, f"Class {i}", f"class {i}", f"class {i}")
        for i in range(4)
    ]
    samples = []
    for class_id in range(4):
        for idx in range(5):
            samples.append(
                ClassificationSample(
                    sample_id=f"toy/class{class_id}/{idx}",
                    image_path=f"/tmp/class{class_id}_{idx}.jpg",
                    class_id=class_id,
                    class_name=f"class {class_id}",
                )
            )
    return ClassificationMetadata("toy", "v1", classes, samples)


def test_fewshot_split_counts():
    split = build_fewshot_split(_metadata(), shots=2, seed=1)
    assert len(split["train"]) == 8
    assert verify_fewshot_split(split) == []


def test_base_novel_split_no_leakage():
    split = build_base_novel_split(_metadata(), shots=2, seed=1)
    assert set(split["base_classes"]).isdisjoint(set(split["novel_classes"]))
    assert verify_base_novel_split(split) == []

