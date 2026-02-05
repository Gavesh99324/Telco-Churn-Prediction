#!/bin/bash
# Container entrypoint template
set -e

echo "Starting container..."
exec "$@"
