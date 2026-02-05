# Build & deployment commands
.PHONY: help install test clean docker-build

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean artifacts"
	@echo "  make docker-build - Build Docker images"

install:
	pip install -r requirements.txt

test:
	pytest tests/

clean:
	rm -rf artifacts/*
	find . -type d -name __pycache__ -exec rm -rf {} +

docker-build:
	docker-compose build
