from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from avion.datasets.torch_datasets import RetrievalPairDataset
from avion.evaluation.retrieval import build_relevance_matrix, evaluate_retrieval
from avion.models.georsclip_loader import GeoRSCLIPConfig, load_georsclip
from avion.models.prompted_clip import PromptConfig, PromptedCLIP
from avion.trainers.avion_retrieval import RetrievalLossConfig, compute_avion_retrieval_loss
from avion.trainers.checkpoint import save_checkpoint
from avion.trainers.classification_runner import _require_same_dim
from avion.trainers.optimizer import build_adamw
from avion.utils.io import ensure_dir, read_json, write_json
from avion.utils.manifest import build_run_manifest, write_run_manifest
from avion.utils.seed import set_seed
from avion.utils.tensor import require_torch


@dataclass(frozen=True)
class RetrievalRunConfig:
    metadata_path: str
    teacher_retrieval_cache: str
    output_dir: str
    student_checkpoint: str | None = None
    student_backbone: str = "ViT-B/32"
    student_pretrained: str = "openai"
    train_split: str = "train"
    eval_split: str = "test"
    epochs: int = 50
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


def _load_retrieval_cache(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = require_torch()
    payload = torch.load(path, map_location="cpu")
    if "image_features" not in payload or "caption_features" not in payload:
        raise ValueError(f"Unsupported retrieval cache format: {path}")
    return payload["image_features"], payload["caption_features"]


def _collate_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    torch = require_torch()
    return {
        "images": torch.stack([item["image"] for item in items]),
        "captions": [str(item["caption"]) for item in items],
        "image_ids": [str(item["image_id"]) for item in items],
        "caption_ids": [str(item["caption_id"]) for item in items],
    }


def _rows_for_eval(metadata: dict[str, Any], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_rows = [row for row in metadata["images"] if row.get("split", "train") == split]
    caption_ids = {pair["caption_id"] for pair in metadata["pairs"] if pair.get("split", "train") == split}
    caption_rows = [row for row in metadata["captions"] if row["caption_id"] in caption_ids]
    return image_rows, caption_rows


def _encode_eval_images(model: PromptedCLIP, rows: list[dict[str, Any]], transform: Any, batch_size: int, device: str) -> Any:
    torch = require_torch()
    features = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            tensors = []
            for row in batch_rows:
                with Image.open(row["image_path"]) as image:
                    tensors.append(transform(image.convert("RGB")))
            features.append(model.encode_image(torch.stack(tensors).to(device)).detach().cpu())
    return torch.cat(features, dim=0) if features else torch.empty((0, 0))


def _encode_eval_captions(model: PromptedCLIP, rows: list[dict[str, Any]], batch_size: int) -> Any:
    torch = require_torch()
    features = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            features.append(model.encode_text([str(row["caption"]) for row in batch_rows]).detach().cpu())
    return torch.cat(features, dim=0) if features else torch.empty((0, 0))


def _evaluate_retrieval(
    model: PromptedCLIP,
    metadata: dict[str, Any],
    split: str,
    image_transform: Any,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    image_rows, caption_rows = _rows_for_eval(metadata, split)
    if not image_rows or not caption_rows:
        return {"I2T_R1": 0.0, "I2T_R5": 0.0, "I2T_R10": 0.0, "T2I_R1": 0.0, "T2I_R5": 0.0, "T2I_R10": 0.0, "mR": 0.0}
    image_features = _encode_eval_images(model, image_rows, image_transform, batch_size, device)
    caption_features = _encode_eval_captions(model, caption_rows, batch_size)
    _require_same_dim(image_features, caption_features, "retrieval_image_features", "retrieval_caption_features")
    similarity = (model.logit_scale.exp().detach().cpu() * image_features @ caption_features.t()).numpy()
    relevance = build_relevance_matrix(
        [str(row["image_id"]) for row in image_rows],
        [str(row["image_id"]) for row in caption_rows],
    )
    return evaluate_retrieval(np.asarray(similarity), relevance)


def run_retrieval(config: RetrievalRunConfig) -> dict[str, float]:
    torch = require_torch()
    set_seed(config.seed)
    metadata = read_json(config.metadata_path)
    train_dataset = RetrievalPairDataset(metadata, split=config.train_split)
    if len(train_dataset) == 0:
        raise ValueError(f"No retrieval pairs found for split '{config.train_split}' in {config.metadata_path}")

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
    train_dataset.image_transform = student_bundle.preprocess_train

    teacher_image_features, teacher_caption_features = _load_retrieval_cache(config.teacher_retrieval_cache)
    optimizer = build_adamw(model.trainable_parameters(), lr=config.lr)
    loss_config = RetrievalLossConfig(
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
            batch = _collate_pairs(items)
            images = batch["images"].to(config.device)
            captions = batch["captions"]
            teacher_images = torch.stack([teacher_image_features[image_id] for image_id in batch["image_ids"]]).to(config.device)
            teacher_text = torch.stack([teacher_caption_features[caption_id] for caption_id in batch["caption_ids"]]).to(config.device)

            student_images = model.encode_image(images)
            student_text = model.encode_text(captions)
            _require_same_dim(student_images, teacher_images, "student_image", "teacher_image")
            _require_same_dim(student_text, teacher_text, "student_text", "teacher_caption")
            student_similarity = model.logits(student_images, student_text)
            teacher_similarity = model.logit_scale.exp().detach() * teacher_images @ teacher_text.t()
            loss_dict = compute_avion_retrieval_loss(
                student_similarity=student_similarity,
                teacher_similarity=teacher_similarity,
                student_image_features=student_images,
                teacher_image_features=teacher_images,
                student_text_features=student_text,
                teacher_text_features=teacher_text,
                global_step=global_step,
                total_steps=total_steps,
                config=loss_config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            optimizer.step()
            epoch_losses.append(float(loss_dict["loss"].detach().cpu()))
            global_step += 1
        train_history.append({"epoch": epoch, "loss": float(sum(epoch_losses) / max(len(epoch_losses), 1))})

    metrics = _evaluate_retrieval(
        model,
        metadata,
        split=config.eval_split,
        image_transform=student_bundle.preprocess_val,
        batch_size=config.batch_size,
        device=config.device,
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
            "metadata": config.metadata_path,
            "teacher_retrieval_cache": config.teacher_retrieval_cache,
            "student_checkpoint": config.student_checkpoint or "",
        },
        seed=config.seed,
        extra={
            "protocol": "retrieval",
            "assumption": "Batch image-caption diagonal pairs are used as positives; this follows the reproduction plan because the paper does not fully specify retrieval training.",
        },
    )
    write_run_manifest(manifest, out_dir / "run_manifest.json")
    return metrics
