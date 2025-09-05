#!/usr/bin/env bash
set -euo pipefail
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:9000/health}"

# Pure Python parsing - no jq dependency required
status="$(python3 - <<'PY' "$HEALTH_URL"
import json,sys,urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as r:
        data = json.load(r)
        print(data.get("status", "fail"))
except Exception as e:
    print("fail")
    print(f"Error: {e}", file=sys.stderr)
PY
)"

if [[ "$status" == "ok" || "$status" == "degraded" ]]; then
    echo "✅ Smoke OK ($status) — $HEALTH_URL"
    exit 0
else
    echo "❌ Smoke FAIL ($status) — $HEALTH_URL"  
    exit 1
fi