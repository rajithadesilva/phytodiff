#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/workspace/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"

