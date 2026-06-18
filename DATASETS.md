# Datasets

Datasets are not stored in git.

Expected external roots:

```text
/data/avion-repro/data/raw
/data/avion-repro/data/processed
/data/avion-repro/data/splits
/data/avion-repro/data/annotations
```

Classification datasets:

- AID: 30 classes, 10,000 images.
- RESISC-45: 45 classes, 31,500 images.
- EuroSAT RGB: 10 classes, 27,000 images.
- WHU-RS19: 19 classes, 1,005 images.
- PatternNet: 38 classes, 30,400 images.
- UCMerced: 21 classes, 2,100 images.

Retrieval datasets:

- RSITMD.
- RSICD.

All raw datasets are converted into canonical metadata JSON before training.
Trainers should never infer labels by scanning folders at runtime.

