from avion.datasets.verify_metadata import verify_classification_metadata


def test_verify_classification_metadata_counts_for_unknown_dataset():
    metadata = {
        "dataset": "toy",
        "classes": [{"class_id": 0, "raw_name": "A", "canonical_name": "a", "prompt_name": "a"}],
        "samples": [],
    }
    report = verify_classification_metadata(metadata)
    assert report["num_classes"] == 1
    assert report["num_samples"] == 0
    assert report["ok"]

