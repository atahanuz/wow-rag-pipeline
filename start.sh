#!/usr/bin/env bash
# Boots the FastAPI backend (:8000) and the Vite frontend (:5173).
# Open http://localhost:5173 once both have started.

set -e
cd "$(dirname "$0")"

cleanup() {
  echo
  echo "shutting down..."
  kill 0
}
trap cleanup EXIT INT TERM

echo "starting backend on http://127.0.0.1:8000 ..."
uvicorn server:app --port 8000 --host 127.0.0.1 --reload &

echo "starting frontend on http://localhost:5173 ..."
(cd ui && npm run dev) &

wait
