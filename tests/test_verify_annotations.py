from avion.llm.verify_annotations import verify_annotation_rows


def test_verify_annotations_reports_ok():
    metadata = {
        "dataset": "toy",
        "classes": [{"canonical_name": "airport"}],
    }
    rows = [
        {
            "dataset": "toy",
            "class_name": "airport",
            "candidate_index": 0,
            "caption": "An aerial view of an airport with runways and taxiways.",
            "viewpoint": "aerial",
            "visual_cues": [],
            "spatial_cues": [],
            "rs_flag": 1,
        }
    ]
    report = verify_annotation_rows(metadata, rows, expected_kp=1)
    assert report["ok"]
    assert report["rs_flag_pass_rate"] == 1.0

