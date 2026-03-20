# Ledger Logic Agent – common development tasks
# Usage: make <target>   |  make simulate TARGET=https://...

.DEFAULT_GOAL := help

.PHONY: help dev test shadow simulate deploy smoke analyze logs-fetch clean

PYTHON  := .venv/bin/python
TARGET  ?= http://localhost:8080

help:                ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev:                 ## Run agent locally with hot-reload on port 8080
	uvicorn main:app --reload --port 8080

test:                ## Run integration tests against sandbox API
	$(PYTHON) tests/test_runner.py

shadow:              ## Run shadow checker (read-only verification of existing data)
	$(PYTHON) tests/shadow_check.py

simulate:            ## Run full competition simulator against TARGET server
	$(PYTHON) tests/simulate_evaluator.py --target $(TARGET)

deploy:              ## Build Docker image and deploy to Cloud Run
	./deploy.sh

smoke:               ## Smoke-test the deployed Cloud Run service
	./deploy.sh smoke

analyze:             ## Fetch last 30 min of episodes and print insights
	$(PYTHON) scripts/analyze.py --fetch --freshness 30m

logs-fetch:          ## Fetch all stored episodes from Cloud Logging
	$(PYTHON) scripts/analyze.py --fetch

clean:               ## Remove Python bytecode and cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
