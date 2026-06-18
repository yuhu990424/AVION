#!/usr/bin/env bash
set -euo pipefail

bash scripts/data/download_classification.sh
bash scripts/data/download_retrieval.sh

