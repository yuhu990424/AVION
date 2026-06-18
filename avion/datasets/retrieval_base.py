from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avion.utils.io import read_json


@dataclass(frozen=True)
class RetrievalImage:
    image_id: str
    image_path: str
    split: str


@dataclass(frozen=True)
class RetrievalCaption:
    caption_id: str
    image_id: str
    caption: str


@dataclass(frozen=True)
class RetrievalPair:
    image_id: str
    caption_id: str
    split: str


@dataclass(frozen=True)
class RetrievalMetadata:
    dataset: str
    images: list[RetrievalImage]
    captions: list[RetrievalCaption]
    pairs: list[RetrievalPair]

    @classmethod
    def from_json(cls, path: str | Path) -> "RetrievalMetadata":
        payload = read_json(path)
        return cls(
            dataset=str(payload["dataset"]),
            images=[RetrievalImage(**row) for row in payload["images"]],
            captions=[RetrievalCaption(**row) for row in payload["captions"]],
            pairs=[RetrievalPair(**row) for row in payload["pairs"]],
        )


def build_retrieval_metadata_from_rs_json(
    dataset: str,
    annotation_json: str | Path,
    image_root: str | Path,
) -> dict[str, Any]:
    """Build metadata from RSICD/RSITMD-style JSON.

    Supports the common shape:
    {"images": [{"filename": ..., "split": ..., "sentences": [{"raw": ...}]}]}.
    """
    payload = read_json(annotation_json)
    rows = payload.get("images", payload if isinstance(payload, list) else None)
    if not isinstance(rows, list):
        raise ValueError("Retrieval annotation JSON must contain an images list.")

    image_root = Path(image_root)
    images: list[dict[str, str]] = []
    captions: list[dict[str, str]] = []
    pairs: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        filename = str(row.get("filename") or row.get("file_name") or row.get("image") or row.get("imgid") or "")
        if not filename:
            raise ValueError(f"Missing filename for retrieval row {index}")
        split = str(row.get("split", "train")).lower()
        image_id = str(row.get("image_id") or Path(filename).stem)
        image_path = image_root / filename
        images.append({"image_id": image_id, "image_path": str(image_path.resolve()), "split": split})

        sentence_rows = row.get("sentences") or row.get("captions") or []
        if isinstance(sentence_rows, list):
            for sent_idx, sentence in enumerate(sentence_rows):
                if isinstance(sentence, dict):
                    caption = str(sentence.get("raw") or sentence.get("caption") or sentence.get("text") or "").strip()
                else:
                    caption = str(sentence).strip()
                if not caption:
                    continue
                caption_id = f"{image_id}_{sent_idx}"
                captions.append({"caption_id": caption_id, "image_id": image_id, "caption": caption})
                pairs.append({"image_id": image_id, "caption_id": caption_id, "split": split})

    return {"dataset": dataset, "images": images, "captions": captions, "pairs": pairs}

