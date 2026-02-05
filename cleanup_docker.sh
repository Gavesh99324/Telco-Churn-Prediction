#!/bin/bash
# Docker cleanup utility
echo "Cleaning up Docker resources..."
docker-compose down -v
docker system prune -f
echo "Docker cleanup completed!"
