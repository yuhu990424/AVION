from avion.models.verify_checkpoints import verify_georsclip_checkpoints


def test_verify_georsclip_missing_files(tmp_path):
    report = verify_georsclip_checkpoints(tmp_path)
    assert report["RS5M_ViT-B-32.pt"]["exists"] is False
    assert report["RS5M_ViT-H-14.pt"]["exists"] is False

