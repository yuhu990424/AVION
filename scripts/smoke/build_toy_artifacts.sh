#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/tmp/avion_toy}
python -m avion.smoke.build_toy_artifacts --root "${ROOT}"

