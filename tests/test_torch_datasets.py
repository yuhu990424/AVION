from pathlib import Path

from PIL import Image

from avion.datasets.torch_datasets import ClassificationImageDataset, RetrievalPairDataset


def test_classification_image_dataset(tmp_path):
    image_path = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image_path)
    rows = [
        {
            "sample_id": "toy/a/x.jpg",
            "image_path": str(image_path),
            "class_id": 0,
            "class_name": "a",
        }
    ]
    dataset = ClassificationImageDataset(rows)
    item = dataset[0]
    assert item.sample_id == "toy/a/x.jpg"
    assert item.label == 0


def test_retrieval_pair_dataset(tmp_path):
    image_path = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image_path)
    metadata = {
        "dataset": "toy",
        "images": [{"image_id": "img1", "image_path": str(image_path), "split": "train"}],
        "captions": [{"caption_id": "cap1", "image_id": "img1", "caption": "a caption"}],
        "pairs": [{"image_id": "img1", "caption_id": "cap1", "split": "train"}],
    }
    dataset = RetrievalPairDataset(metadata)
    item = dataset[0]
    assert item["image_id"] == "img1"
    assert item["caption"] == "a caption"

