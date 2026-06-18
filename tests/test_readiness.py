from avion.utils.readiness import build_readiness_checks, build_readiness_report


def test_readiness_report_detects_missing_and_present(tmp_path):
    ckpt_root = tmp_path / "checkpoints"
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    student = ckpt_root / "georsclip" / "RS5M_ViT-B-32.pt"
    student.parent.mkdir(parents=True)
    student.write_text("placeholder", encoding="utf-8")

    checks = build_readiness_checks(
        data_root=data_root,
        ckpt_root=ckpt_root,
        cache_root=cache_root,
        protocols=["retrieval"],
        retrieval_datasets=["rsitmd"],
    )
    report = build_readiness_report(checks)
    assert report["present"] == 1
    assert report["missing"] == 3
    assert report["ready"] is False
    assert report["missing_truncated"] is False


def test_readiness_report_can_truncate_missing(tmp_path):
    checks = build_readiness_checks(
        data_root=tmp_path / "data",
        ckpt_root=tmp_path / "checkpoints",
        cache_root=tmp_path / "cache",
        protocols=["retrieval"],
        retrieval_datasets=["rsitmd"],
    )
    report = build_readiness_report(checks, max_missing=2)
    assert report["missing"] == 4
    assert len(report["missing_required"]) == 2
    assert report["missing_truncated"] is True
