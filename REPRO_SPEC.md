# Reproduction Specification

## Scope

This reproduction is limited to the AVION main method:

- Few-shot classification with shots `1, 2, 4, 8, 16`.
- Base-to-novel generalization with deterministic random 50/50 class splits.
- Retrieval on RSITMD and RSICD.

The following are out of scope for this pass:

- B0-B7 incremental ablations.
- P0-P6 prototype ablations.
- LLM generator ablations.
- Backbone ablations.
- Cross-dataset transfer.
- ImageNet generalization.

## Paths

The source repository lives at:

```text
/home/yu.hu/AVION
```

Large artifacts should live outside the git repository:

```text
AVION_DATA=/data/avion-repro/data
AVION_CKPT=/data/avion-repro/checkpoints
AVION_CACHE=/data/avion-repro/cache
AVION_OUTPUT=/data/avion-repro/output
```

## Main Configuration

```text
teacher: GeoRSCLIP ViT-H/14
student: GeoRSCLIP ViT-B/32
LLM: gemini-2.5-flash
captions_per_class: 30
visual_prompt_tokens_per_layer: 8
text_prompt_tokens_per_layer: 4
optimizer: AdamW
learning_rate: 5e-4
batch_size: 4
fewshot_epochs: 100
base_to_novel_epochs: 50
retrieval_epochs: 50
lambda_img: 0.5
lambda_text: 0.5
lambda_logit: 1.0
logit_warmup_ratio: 0.30
distillation_temperature: 2.0
prototype_beta: 10
prototype_gamma: 2
mad_threshold: 3.0
```

## Retrieval Assumption

The paper specifies the retrieval datasets, fixed-gallery protocol, and metrics,
but does not fully spell out the retrieval training objective. This
implementation uses a transparent engineering assumption consistent with AVION:

```text
symmetric CLIP-style contrastive task loss
+ image/text/logit teacher distillation
+ frozen-backbone prompt tuning
```

This assumption must be reported with retrieval results.

