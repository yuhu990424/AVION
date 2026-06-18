from __future__ import annotations

import argparse
from pathlib import Path

from avion.datasets.base import ClassificationMetadata
from avion.datasets.torch_datasets import ClassificationImageDataset
from avion.evaluation.metrics import top1_accuracy
from avion.models.georsclip_loader import GeoRSCLIPConfig, load_georsclip
from avion.utils.io import write_json
from avion.utils.tensor import require_torch


def zero_shot_classification(
    metadata_path: str,
    model_config: GeoRSCLIPConfig,
    out_path: str | None = None,
    template: str = "a photo of a {}",
    batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, float]:
    torch = require_torch()
    metadata = ClassificationMetadata.from_json(metadata_path)
    bundle = load_georsclip(model_config, device=device)
    dataset = ClassificationImageDataset.from_metadata(metadata_path, transform=bundle.preprocess_val)
    class_prompts = [template.format(name) for name in metadata.class_names]
    with torch.no_grad():
        text_features = bundle.encode_text(class_prompts)
        logits_all = []
        labels_all = []
        for start in range(0, len(dataset), batch_size):
            items = [dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))]
            images = torch.stack([item.image for item in items]).to(device)
            labels = torch.as_tensor([item.label for item in items], device=device)
            image_features = bundle.encode_image(images)
            logits = image_features @ text_features.t()
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())
        logits_np = torch.cat(logits_all, dim=0).numpy()
        labels_np = torch.cat(labels_all, dim=0).numpy()
    metrics = {"accuracy": top1_accuracy(logits_np, labels_np), "num_samples": float(len(dataset))}
    if out_path:
        write_json(metrics, out_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out")
    parser.add_argument("--backbone", default="ViT-B/32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--checkpoint")
    parser.add_argument("--template", default="a photo of a {}")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = GeoRSCLIPConfig(args.backbone, args.pretrained, args.checkpoint)
    metrics = zero_shot_classification(
        args.metadata,
        config,
        out_path=args.out,
        template=args.template,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(metrics)


if __name__ == "__main__":
    main()

