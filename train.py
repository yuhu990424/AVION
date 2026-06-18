from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from avion.config.merge import get_nested, load_and_merge
from avion.trainers.classification_runner import ClassificationRunConfig, run_classification
from avion.trainers.retrieval_runner import RetrievalRunConfig, run_retrieval


def _pick(cli_value, config_value, default):
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _protocol_from_config(config: dict[str, Any]) -> str | None:
    protocol = get_nested(config, "DATASET.PROTOCOL")
    if protocol is None:
        protocol = get_nested(config, "TRAIN.PROTOCOL")
    return str(protocol) if protocol is not None else None


def _prompt_tag(config: dict[str, Any]) -> str:
    vp = int(get_nested(config, "MODEL.PROMPT.VISUAL_TOKENS_PER_LAYER", 8))
    tp = int(get_nested(config, "MODEL.PROMPT.TEXT_TOKENS_PER_LAYER", 4))
    kp = int(get_nested(config, "PROTOTYPE.KP", 30))
    beta = get_nested(config, "PROTOTYPE.BETA", 10)
    gamma = get_nested(config, "PROTOTYPE.GAMMA", 2)
    zeta = get_nested(config, "PROTOTYPE.MAD_THRESHOLD", 3)
    return f"vp{vp}_tp{tp}_kp{kp}_beta{beta:g}_gamma{gamma:g}_zeta{zeta:g}"


def _default_artifact_paths(config: dict[str, Any], protocol: str, seed: int) -> dict[str, str | None]:
    dataset = get_nested(config, "DATASET.KEY")
    if not dataset:
        return {
            "split": None,
            "metadata": None,
            "teacher_image_cache": None,
            "teacher_text_prototypes": None,
            "teacher_retrieval_cache": None,
            "output_dir": None,
        }
    dataset = str(dataset)
    shots = int(get_nested(config, "DATASET.NUM_SHOTS", 16))
    data_root = Path(str(get_nested(config, "DATA_ROOT", "/data/avion-repro/data")))
    cache_root = Path(str(get_nested(config, "CACHE_ROOT", "/data/avion-repro/cache")))
    output_root = Path(str(get_nested(config, "OUTPUT_ROOT", "/data/avion-repro/output")))
    model_name = str(get_nested(config, "MODEL.NAME", "AVION_GeoRSCLIP"))
    tag = _prompt_tag(config)

    if protocol == "few_shot":
        return {
            "split": str(data_root / "splits" / dataset / f"seed{seed}" / f"fewshot_{shots}.json"),
            "metadata": None,
            "teacher_image_cache": str(cache_root / "teacher_images" / dataset / "fewshot" / f"seed{seed}" / f"shots_{shots}" / "vith14_image_features.pt"),
            "teacher_text_prototypes": str(cache_root / "prototypes" / dataset / "fewshot" / f"seed{seed}" / f"shots_{shots}" / "text_prototypes_selective.pt"),
            "teacher_retrieval_cache": None,
            "output_dir": str(output_root / dataset / f"shots_{shots}" / model_name / tag / f"seed{seed}"),
        }
    if protocol == "base_to_novel":
        return {
            "split": str(data_root / "splits" / dataset / f"seed{seed}" / "base_to_novel.json"),
            "metadata": None,
            "teacher_image_cache": str(cache_root / "teacher_images" / dataset / "base2new" / f"seed{seed}" / "vith14_base_image_features.pt"),
            "teacher_text_prototypes": str(cache_root / "prototypes" / dataset / "base2new" / f"seed{seed}" / "text_prototypes_selective.pt"),
            "teacher_retrieval_cache": None,
            "output_dir": str(output_root / "base2new" / "train_base" / dataset / f"shots_{shots}" / model_name / tag / f"seed{seed}"),
        }
    if protocol == "retrieval":
        return {
            "split": None,
            "metadata": str(data_root / "processed" / dataset / "metadata.json"),
            "teacher_image_cache": None,
            "teacher_text_prototypes": None,
            "teacher_retrieval_cache": str(cache_root / "teacher_retrieval" / dataset / "vith14_retrieval_features.pt"),
            "output_dir": str(output_root / "retrieval" / dataset / model_name / tag / f"seed{seed}"),
        }
    return {
        "split": None,
        "metadata": None,
        "teacher_image_cache": None,
        "teacher_text_prototypes": None,
        "teacher_retrieval_cache": None,
        "output_dir": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AVION training entrypoint.")
    parser.add_argument("--config", nargs="+", action="append", default=[], help="YAML config files to merge.")
    parser.add_argument("--protocol", choices=["few_shot", "base_to_novel", "retrieval"])
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without launching training.")
    parser.add_argument("--split", help="Few-shot or base-to-novel split JSON.")
    parser.add_argument("--metadata", help="Retrieval metadata JSON.")
    parser.add_argument("--output-dir", help="Run output directory.")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--teacher-image-cache")
    parser.add_argument("--teacher-text-prototypes")
    parser.add_argument("--teacher-retrieval-cache")
    parser.add_argument("--train-split")
    parser.add_argument("--eval-split")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--visual-prompt-tokens", type=int)
    parser.add_argument("--text-prompt-tokens", type=int)
    args = parser.parse_args()
    config_paths = [path for group in args.config for path in group]
    merged_config = load_and_merge(config_paths) if config_paths else {}
    if args.shots is not None:
        merged_config.setdefault("DATASET", {})["NUM_SHOTS"] = args.shots
    protocol = args.protocol or _protocol_from_config(merged_config)
    if protocol is None:
        raise SystemExit("Missing protocol. Pass --protocol or set DATASET.PROTOCOL in a config.")
    seed = int(_pick(args.seed, get_nested(merged_config, "SEED"), 1))
    default_paths = _default_artifact_paths(merged_config, protocol, seed)
    split_path = _pick(args.split, default_paths.get("split"), None)
    metadata_path = _pick(args.metadata, default_paths.get("metadata"), None)
    output_dir = _pick(args.output_dir, default_paths.get("output_dir"), None)
    teacher_image_cache = _pick(args.teacher_image_cache, default_paths.get("teacher_image_cache"), None)
    teacher_text_prototypes = _pick(args.teacher_text_prototypes, default_paths.get("teacher_text_prototypes"), None)
    teacher_retrieval_cache = _pick(args.teacher_retrieval_cache, default_paths.get("teacher_retrieval_cache"), None)

    if args.dry_run:
        print(
            {
                "protocol": protocol,
                "configs": config_paths,
                "split": split_path,
                "metadata": metadata_path,
                "teacher_image_cache": teacher_image_cache,
                "teacher_text_prototypes": teacher_text_prototypes,
                "teacher_retrieval_cache": teacher_retrieval_cache,
                "output_dir": output_dir,
                "merged_config_keys": sorted(merged_config.keys()),
                "status": "dry_run_ok",
            }
        )
        return
    if protocol == "retrieval":
        required = {
            "--metadata": metadata_path,
            "--output-dir": output_dir,
            "--teacher-retrieval-cache": teacher_retrieval_cache,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(f"Missing required arguments for retrieval training: {', '.join(missing)}")
        prompt_init = str(get_nested(merged_config, "MODEL.PROMPT.INIT", "normal_0.02"))
        init_std = 0.02
        if prompt_init.startswith("normal_"):
            try:
                init_std = float(prompt_init.split("_", 1)[1])
            except ValueError:
                init_std = 0.02
        retrieval_config = RetrievalRunConfig(
            metadata_path=str(metadata_path),
            teacher_retrieval_cache=str(teacher_retrieval_cache),
            output_dir=str(output_dir),
            student_checkpoint=_pick(args.student_checkpoint, get_nested(merged_config, "MODEL.STUDENT.CKPT"), None),
            student_backbone=get_nested(merged_config, "MODEL.STUDENT.BACKBONE", "ViT-B/32"),
            student_pretrained=get_nested(merged_config, "MODEL.STUDENT.PRETRAINED", "openai"),
            train_split=str(_pick(args.train_split, get_nested(merged_config, "RETRIEVAL.TRAIN_SPLIT"), "train")),
            eval_split=str(_pick(args.eval_split, get_nested(merged_config, "RETRIEVAL.EVAL_SPLIT"), "test")),
            epochs=int(_pick(args.epochs, get_nested(merged_config, "TRAIN.EPOCHS.RETRIEVAL"), 50)),
            batch_size=int(_pick(args.batch_size, get_nested(merged_config, "TRAIN.BATCH_SIZE"), 4)),
            lr=float(_pick(args.lr, get_nested(merged_config, "TRAIN.LR"), 5e-4)),
            seed=seed,
            device=str(_pick(args.device, None, "cpu")),
            visual_prompt_tokens=int(_pick(args.visual_prompt_tokens, get_nested(merged_config, "MODEL.PROMPT.VISUAL_TOKENS_PER_LAYER"), 8)),
            text_prompt_tokens=int(_pick(args.text_prompt_tokens, get_nested(merged_config, "MODEL.PROMPT.TEXT_TOKENS_PER_LAYER"), 4)),
            lambda_img=float(get_nested(merged_config, "DISTILL.LAMBDA_IMG", 0.5)),
            lambda_text=float(get_nested(merged_config, "DISTILL.LAMBDA_TEXT", 0.5)),
            lambda_logit=float(get_nested(merged_config, "DISTILL.LAMBDA_LOGIT", 1.0)),
            logit_warmup_ratio=float(get_nested(merged_config, "DISTILL.LOGIT_WARMUP_RATIO", 0.30)),
            temperature=float(get_nested(merged_config, "DISTILL.TEMPERATURE", 2.0)),
            prompt_init_std=init_std,
        )
        print(run_retrieval(retrieval_config))
        return
    required = {
        "--split": split_path,
        "--output-dir": output_dir,
        "--teacher-image-cache": teacher_image_cache,
        "--teacher-text-prototypes": teacher_text_prototypes,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required arguments for training: {', '.join(missing)}")
    default_epochs = 100 if protocol == "few_shot" else 50
    config_epoch_key = "TRAIN.EPOCHS.FEWSHOT" if protocol == "few_shot" else "TRAIN.EPOCHS.BASE_TO_NOVEL"
    epochs = _pick(args.epochs, get_nested(merged_config, config_epoch_key), default_epochs)
    prompt_init = str(get_nested(merged_config, "MODEL.PROMPT.INIT", "normal_0.02"))
    init_std = 0.02
    if prompt_init.startswith("normal_"):
        try:
            init_std = float(prompt_init.split("_", 1)[1])
        except ValueError:
            init_std = 0.02
    run_config = ClassificationRunConfig(
        protocol=protocol,
        split_path=str(split_path),
        output_dir=str(output_dir),
        student_checkpoint=_pick(args.student_checkpoint, get_nested(merged_config, "MODEL.STUDENT.CKPT"), None),
        student_backbone=get_nested(merged_config, "MODEL.STUDENT.BACKBONE", "ViT-B/32"),
        student_pretrained=get_nested(merged_config, "MODEL.STUDENT.PRETRAINED", "openai"),
        teacher_image_cache=str(teacher_image_cache),
        teacher_text_prototypes=str(teacher_text_prototypes),
        epochs=int(epochs),
        batch_size=int(_pick(args.batch_size, get_nested(merged_config, "TRAIN.BATCH_SIZE"), 4)),
        lr=float(_pick(args.lr, get_nested(merged_config, "TRAIN.LR"), 5e-4)),
        seed=seed,
        device=str(_pick(args.device, None, "cpu")),
        visual_prompt_tokens=int(_pick(args.visual_prompt_tokens, get_nested(merged_config, "MODEL.PROMPT.VISUAL_TOKENS_PER_LAYER"), 8)),
        text_prompt_tokens=int(_pick(args.text_prompt_tokens, get_nested(merged_config, "MODEL.PROMPT.TEXT_TOKENS_PER_LAYER"), 4)),
        lambda_img=float(get_nested(merged_config, "DISTILL.LAMBDA_IMG", 0.5)),
        lambda_text=float(get_nested(merged_config, "DISTILL.LAMBDA_TEXT", 0.5)),
        lambda_logit=float(get_nested(merged_config, "DISTILL.LAMBDA_LOGIT", 1.0)),
        logit_warmup_ratio=float(get_nested(merged_config, "DISTILL.LOGIT_WARMUP_RATIO", 0.30)),
        temperature=float(get_nested(merged_config, "DISTILL.TEMPERATURE", 2.0)),
        prompt_init_std=init_std,
    )
    metrics = run_classification(run_config)
    print(metrics)


if __name__ == "__main__":
    main()
