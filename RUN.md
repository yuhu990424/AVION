# Run Plan

The execution gates are:

1. Verify environment.
2. Verify datasets and metadata.
3. Verify GeoRSCLIP checkpoints.
4. Run zero-shot GeoRSCLIP sanity checks.
5. Generate Gemini annotations and verify JSON schema.
6. Build RS-Flag reports and teacher prototype diagnostics.
7. Run one small 16-shot classification smoke run.
8. Run the full few-shot grid.
9. Run base-to-novel.
10. Run retrieval with fixed galleries.
11. Collect tables and manifests.

## Current Entrypoints

All training scripts default to dry-run mode. Set `AVION_RUN_TRAINING=1` only
after datasets, GeoRSCLIP checkpoints, Gemini annotations, and teacher caches
are present.

Before launching a real run, inspect missing external artifacts:

```bash
python verify_all.py --stage readiness
bash scripts/avion/check_readiness.sh all
```

```bash
bash scripts/avion/few_shot.sh /data/avion-repro/data aid 16 GeoRSCLIP 1 cpu
bash scripts/avion/base2new.sh /data/avion-repro/data aid GeoRSCLIP 1 cpu
bash scripts/avion/retrieval.sh /data/avion-repro/data rsitmd GeoRSCLIP 1 cpu
bash scripts/avion/run_all_main.sh /data/avion-repro/data cpu
```

`run_all_main.sh` runs one smoke triplet by default. Use
`AVION_FULL_GRID=1` to expand the full few-shot, base-to-novel, and retrieval
grid; combine it with `AVION_RUN_TRAINING=1` only when the external artifacts
are ready.

The merged YAML entrypoint can also be called directly:

```bash
python train.py \
  --config configs/paths/default.yaml \
  --config configs/models/georsclip_vitb32_student_vith14_teacher.yaml \
  --config configs/trainers/AVION/few_shot/aid.yaml \
  --shots 16 \
  --seed 1 \
  --dry-run
```

Few-shot, base-to-novel, and retrieval trainers are connected. Real runs still
depend on external artifacts that are intentionally not stored in git:

- dataset metadata and splits under `AVION_DATA`
- GeoRSCLIP checkpoints under `AVION_CKPT`
- Gemini candidate annotations and teacher caches under `AVION_CACHE`
- experiment outputs under `AVION_OUTPUT`

After runs finish, collect protocol-specific raw metrics and seed summaries:

```bash
bash scripts/avion/eval_fewshot.sh /data/avion-repro/output
bash scripts/avion/eval_base2new.sh /data/avion-repro/output
bash scripts/avion/eval_retrieval.sh /data/avion-repro/output
bash scripts/avion/collect_tables.sh /data/avion-repro/output
```
