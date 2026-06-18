from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from avion.datasets.torch_datasets import ClassificationImageDataset, DatasetItem
from avion.evaluation.classification import evaluate_classification
from avion.evaluation.metrics import harmonic_mean
from avion.models.georsclip_loader import GeoRSCLIPConfig, load_georsclip
from avion.models.prompted_clip import PromptConfig, PromptedCLIP
from avion.trainers.avion_classification import ClassificationLossConfig, compute_avion_classification_loss
from avion.trainers.checkpoint import save_checkpoint
from avion.trainers.optimizer import build_adamw
from avion.utils.io import ensure_dir, read_json, write_json
from avion.utils.manifest import build_run_manifest, write_run_manifest
from avion.utils.seed import set_seed
from avion.utils.tensor import require_torch


@dataclass(frozen=True)
class ClassificationRunConfig:
    protocol: str
    split_path: str
    output_dir: str
    student_checkpoint: str | None = None
    student_backbone: str = "ViT-B/32"
    student_pretrained: str = "openai"
    teacher_image_cache: str | None = None
    teacher_text_prototypes: str | None = None
    epochs: int = 100
    batch_size: int = 4
    lr: float = 5e-4
    seed: int = 1
    device: str = "cpu"
    precision: str = "fp32"
    visual_prompt_tokens: int = 8
    text_prompt_tokens: int = 4
    prompt_init_std: float = 0.02
    lambda_img: float = 0.5
    lambda_text: float = 0.5
    lambda_logit: float = 1.0
    logit_warmup_ratio: float = 0.30
    temperature: float = 2.0


def _collate(items: list[DatasetItem]) -> dict[str, Any]:
    torch = require_torch()
    return {
        "images": torch.stack([item.image for item in items]),
        "labels": torch.as_tensor([item.label for item in items], dtype=torch.long),
        "sample_ids": [item.sample_id for item in items],
        "class_names": [item.class_name for item in items],
    }


def _class_map_from_rows(rows: list[dict[str, Any]]) -> tuple[list[str], dict[int, int], dict[str, int]]:
    by_id = {int(row["class_id"]): row["class_name"] for row in rows}
    sorted_ids = sorted(by_id)
    class_names = [by_id[class_id] for class_id in sorted_ids]
    global_to_local = {class_id: local for local, class_id in enumerate(sorted_ids)}
    name_to_local = {name: idx for idx, name in enumerate(class_names)}
    return class_names, global_to_local, name_to_local


def _load_feature_cache(path: str | None, name: str) -> dict[str, Any]:
    if not path:
        raise ValueError(f"{name} is required for AVION training.")
    torch = require_torch()
    payload = torch.load(path, map_location="cpu")
    if "features" in payload:
        return payload["features"]
    if "prototypes" in payload:
        return payload["prototypes"]
    raise ValueError(f"Unsupported cache format for {name}: {path}")


def _require_same_dim(left: Any, right: Any, left_name: str, right_name: str) -> None:
    if int(left.shape[-1]) != int(right.shape[-1]):
        raise ValueError(
            f"Embedding dimension mismatch: {left_name} has dim {left.shape[-1]}, "
            f"{right_name} has dim {right.shape[-1]}. AVION embedding/text alignment "
            "requires teacher and student embeddings in the same projection space. "
            "Verify the GeoRSCLIP checkpoints or add an explicit projection strategy."
        )


def _load_split_rows(split: dict[str, Any], protocol: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
    if protocol == "few_shot":
        return split["train"], split["test"], None
    if protocol == "base_to_novel":
        return split["train_base"], split["test_base"], split["test_novel"]
    raise ValueError(f"Unsupported classification protocol: {protocol}")


def _evaluate(
    model: PromptedCLIP,
    dataset: ClassificationImageDataset,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    torch = require_torch()
    model.eval()
    if not dataset.rows:
        return {"accuracy": 0.0}
    class_names, global_to_local, _ = _class_map_from_rows(dataset.rows)
    logits_all = []
    labels_all = []
    with torch.no_grad():
        text_features = model.encode_text(class_names)
        for start in range(0, len(dataset), batch_size):
            items = [dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))]
            batch = _collate(items)
            images = batch["images"].to(device)
            labels = torch.as_tensor([global_to_local[int(label)] for label in batch["labels"].tolist()])
            image_features = model.encode_image(images)
            logits = model.logits(image_features, text_features)
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels)
    if not logits_all:
        return {"accuracy": 0.0}
    return evaluate_classification(torch.cat(logits_all).numpy(), torch.cat(labels_all).numpy())


def run_classification(config: ClassificationRunConfig) -> dict[str, float]:
    torch = require_torch()
    set_seed(config.seed)
    split = read_json(config.split_path)
    train_rows, test_rows, test_novel_rows = _load_split_rows(split, config.protocol)
    class_names, global_to_local, name_to_local = _class_map_from_rows(train_rows)

    student_bundle = load_georsclip(
        GeoRSCLIPConfig(config.student_backbone, config.student_pretrained, config.student_checkpoint),
        device=config.device,
    )
    model = PromptedCLIP(
        student_bundle.model,
        student_bundle.tokenizer,
        PromptConfig(
            vision_tokens_per_layer=config.visual_prompt_tokens,
            text_tokens_per_layer=config.text_prompt_tokens,
            init_std=config.prompt_init_std,
        ),
    ).to(config.device)

    train_dataset = ClassificationImageDataset(train_rows, transform=student_bundle.preprocess_train)
    test_dataset = ClassificationImageDataset(test_rows, transform=student_bundle.preprocess_val)
    test_novel_dataset = ClassificationImageDataset(test_novel_rows, transform=student_bundle.preprocess_val) if test_novel_rows is not None else None

    teacher_image_features = _load_feature_cache(config.teacher_image_cache, "teacher_image_cache")
    teacher_text_prototypes = _load_feature_cache(config.teacher_text_prototypes, "teacher_text_prototypes")
    teacher_text_tensor = torch.stack([teacher_text_prototypes[name] for name in class_names]).to(config.device)

    optimizer = build_adamw(model.trainable_parameters(), lr=config.lr)
    loss_config = ClassificationLossConfig(
        lambda_img=config.lambda_img,
        lambda_text=config.lambda_text,
        lambda_logit=config.lambda_logit,
        logit_warmup_ratio=config.logit_warmup_ratio,
        temperature=config.temperature,
    )
    total_steps = max(math.ceil(len(train_dataset) / config.batch_size) * config.epochs, 1)
    global_step = 0
    train_history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(len(train_dataset), generator=generator).tolist()
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            items = [train_dataset[idx] for idx in indices]
            batch = _collate(items)
            images = batch["images"].to(config.device)
            labels = torch.as_tensor([global_to_local[int(label)] for label in batch["labels"].tolist()], dtype=torch.long, device=config.device)
            teacher_images = torch.stack([teacher_image_features[sample_id] for sample_id in batch["sample_ids"]]).to(config.device)

            student_image = model.encode_image(images)
            student_text = model.encode_text(class_names)
            _require_same_dim(student_image, teacher_images, "student_image", "teacher_image")
            _require_same_dim(student_text, teacher_text_tensor, "student_text", "teacher_text_prototypes")
            student_logits = model.logits(student_image, student_text)
            teacher_logits = model.logit_scale.exp().detach() * teacher_images @ teacher_text_tensor.t()

            loss_dict = compute_avion_classification_loss(
                student_logits=student_logits,
                labels=labels,
                student_image_features=student_image,
                teacher_image_features=teacher_images,
                student_text_features=student_text,
                teacher_text_prototypes=teacher_text_tensor,
                teacher_logits=teacher_logits,
                global_step=global_step,
                total_steps=total_steps,
                config=loss_config,
                active_class_indices=list(range(len(class_names))) if config.protocol == "base_to_novel" else None,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            optimizer.step()
            epoch_losses.append(float(loss_dict["loss"].detach().cpu()))
            global_step += 1
        train_history.append({"epoch": epoch, "loss": float(sum(epoch_losses) / max(len(epoch_losses), 1))})

    metrics: dict[str, float] = {}
    if config.protocol == "few_shot":
        metrics.update(_evaluate(model, test_dataset, config.batch_size, config.device))
    else:
        base_metrics = _evaluate(model, test_dataset, config.batch_size, config.device)
        novel_metrics = _evaluate(model, test_novel_dataset, config.batch_size, config.device) if test_novel_dataset else {"accuracy": 0.0}
        metrics.update(
            {
                "Base": base_metrics["accuracy"],
                "Novel": novel_metrics["accuracy"],
                "HM": harmonic_mean(base_metrics["accuracy"], novel_metrics["accuracy"]),
            }
        )
    metrics["final_train_loss"] = train_history[-1]["loss"] if train_history else 0.0
    metrics["epochs"] = float(config.epochs)

    out_dir = ensure_dir(config.output_dir)
    write_json(asdict(config), out_dir / "config.json")
    try:
        import yaml

        with (out_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(config), handle, sort_keys=True)
    except Exception:
        write_json(asdict(config), out_dir / "config.yaml")
    write_json(metrics, out_dir / "metrics.json")
    write_json({"history": train_history}, out_dir / "log.json")
    save_checkpoint(
        {
            "prompts": model.prompts.state_dict(),
            "logit_scale": model.logit_scale.detach().cpu(),
            "metrics": metrics,
            "config": asdict(config),
        },
        out_dir / f"model.pth.tar-{config.epochs}",
    )
    manifest = build_run_manifest(
        config=asdict(config),
        artifacts={
            "split": config.split_path,
            "teacher_image_cache": config.teacher_image_cache or "",
            "teacher_text_prototypes": config.teacher_text_prototypes or "",
            "student_checkpoint": config.student_checkpoint or "",
        },
        seed=config.seed,
        extra={"protocol": config.protocol},
    )
    write_run_manifest(manifest, out_dir / "run_manifest.json")
    return metrics
