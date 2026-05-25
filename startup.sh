#!/bin/bash
set -e

python3 -m gunicorn \
  --bind "0.0.0.0:${PORT:-8000}" \
  --timeout 600 \
  --worker-class uvicorn.workers.UvicornWorker \
  app:app
