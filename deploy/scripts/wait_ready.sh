#!/usr/bin/env bash
# Wait until the API reports every service loaded (up to 30 minutes; the first start downloads models).
#   bash deploy/scripts/wait_ready.sh
PORT="${SJ_PORT:-8000}"
for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:${PORT}/api/v1/ready" >/dev/null; then
    echo "ready"
    exit 0
  fi
  sleep 5
done
echo "not ready after 30 minutes" >&2
exit 1
