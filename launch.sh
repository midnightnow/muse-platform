#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -r requirements.txt
exec uvicorn agent.main:app --host 0.0.0.0 --port "${PORT:-9000}"