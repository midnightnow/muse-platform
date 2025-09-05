#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting agent-001…"
: "${TARGETS:=http://localhost:8000/health}"
: "${PORT:=9000}"

python - <<'PY'
import sys; sys.exit(0)
PY

pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

export APP_NAME=agent-001
export TARGETS
echo "🌐 Listening on :$PORT"
uvicorn agent.main:app --host 0.0.0.0 --port "$PORT" &
PID=$!

for i in {1..40}; do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null; then
    echo "✅ Agent healthy"; break; fi
  sleep 0.25
done

echo "🔎 Sample /check:"
curl -fsS "http://localhost:$PORT/check" | sed 's/","/\n  "/g' | head -n 40 || true

trap 'kill $PID' EXIT
wait $PID