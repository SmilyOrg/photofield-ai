#!/usr/bin/env bash
set -eo pipefail

DOCKER_REPO="${DOCKER_REPO:-ghcr.io/smilyorg/photofield-ai}"
VERSION="${VERSION:-$(git describe --tags --match "v*" --always --dirty)}"
SERVER_LOG=$(mktemp)
DOCKER_LOG=$(mktemp)
trap "rm -f '$SERVER_LOG' '$DOCKER_LOG'" EXIT

# --- SERVER ---
kill $(lsof -ti:8081) 2>/dev/null || true
uv run python main.py >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true; rm -f '$SERVER_LOG' '$DOCKER_LOG'" EXIT

printf "server:  "
for i in $(seq 1 30); do
  if curl -sf -o /dev/null http://localhost:8081/health 2>/dev/null; then
    echo "ready (${i}s)"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "FAILED (timeout)"
    cat "$SERVER_LOG"
    exit 1
  fi
  sleep 1
done

# --- PROBE ---
echo "probe:"
httpyac send --all --bail --output none --output-failed short README.md examples.http | grep -v '^----'

# --- BENCHMARK ---
echo "bench:"
uv run python benchmark.py 2>/dev/null | awk '
  /^[A-Z].*:/ { split($0, a, " "); label=tolower(a[1]) }
  /Throughput/ { printf "  %s=%.1f req/s\n", label, $2 }
'

# --- DOCKER ---
printf "docker:  "
if docker build --progress=plain --build-arg VERSION="$VERSION" \
    -t "$DOCKER_REPO:$VERSION" \
    -t "$DOCKER_REPO:latest" . >"$DOCKER_LOG" 2>&1; then
  SIZE=$(docker image ls "$DOCKER_REPO:latest" --format '{{.Size}}')
  echo "ok  $SIZE"
else
  echo "FAILED"
  cat "$DOCKER_LOG"
  exit 1
fi
