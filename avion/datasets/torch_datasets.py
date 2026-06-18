from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from avion.utils.io import read_json


@dataclass(frozen=True)
class DatasetItem:
    image: Any
    label: int
    sample_id: str
    class_name: str
    image_path: str


class ClassificationImageDataset:
    def __init__(self, rows: list[dict[str, Any]], transform: Callable[[Image.Image], Any] | None = None) -> None:
        self.rows = list(rows)
        self.transform = transform

    @classmethod
    def from_metadata(cls, metadata_path: str, transform: Callable[[Image.Image], Any] | None = None) -> "ClassificationImageDataset":
        payload = read_json(metadata_path)
        return cls(payload["samples"], transform=transform)

    @classmethod
    def from_split(
        cls,
        split_path: str,
        split_key: str,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> "ClassificationImageDataset":
        payload = read_json(split_path)
        if split_key not in payload:
            raise KeyError(f"Split key {split_key} not found in {split_path}")
        return cls(payload[split_key], transform=transform)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> DatasetItem:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            tensor_or_image = self.transform(image) if self.transform else image.copy()
        return DatasetItem(
            image=tensor_or_image,
            label=int(row["class_id"]),
            sample_id=str(row["sample_id"]),
            class_name=str(row["class_name"]),
            image_path=str(row["image_path"]),
        )


class RetrievalPairDataset:
    def __init__(
        self,
        metadata: dict[str, Any],
        split: str = "train",
        image_transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.metadata = metadata
        self.split = split
        self.image_transform = image_transform
        self.images = {row["image_id"]: row for row in metadata["images"]}
        self.captions = {row["caption_id"]: row for row in metadata["captions"]}
        self.pairs = [row for row in metadata["pairs"] if row.get("split", "train") == split]

    @classmethod
    def from_metadata(
        cls,
        metadata_path: str,
        split: str = "train",
        image_transform: Callable[[Image.Image], Any] | None = None,
    ) -> "RetrievalPairDataset":
        return cls(read_json(metadata_path), split=split, image_transform=image_transform)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        image_row = self.images[pair["image_id"]]
        caption_row = self.captions[pair["caption_id"]]
        with Image.open(image_row["image_path"]) as image:
            image = image.convert("RGB")
            tensor_or_image = self.image_transform(image) if self.image_transform else image.copy()
        return {
            "image": tensor_or_image,
            "caption": caption_row["caption"],
            "image_id": pair["image_id"],
            "caption_id": pair["caption_id"],
            "image_path": image_row["image_path"],
        }

