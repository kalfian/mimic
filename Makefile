# Mimic — developer entry points. Run `make help`.
SHELL := /bin/bash
.DEFAULT_GOAL := help

API_DIR := api
WEB_DIR := web
DATA_DIR := data

.PHONY: help setup check-tools dev-api dev-web test test-api test-web lint contract contract-check synth eval calibrate clean-jobs

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

check-tools: ## Verify uv, pnpm, ffmpeg, ffprobe are on PATH
	@for t in uv pnpm ffmpeg ffprobe; do \
	  command -v $$t >/dev/null 2>&1 || { echo "missing required tool: $$t"; exit 1; }; \
	done
	@command -v claude >/dev/null 2>&1 || echo "note: 'claude' CLI not found; AI labeling will be unavailable (fallback labels are used)"
	@echo "tools ok"

setup: check-tools ## Install backend (uv, Python 3.13) and frontend (pnpm) dependencies
	cd $(API_DIR) && uv sync
	cd $(WEB_DIR) && pnpm install
	mkdir -p $(DATA_DIR)/jobs
	cd $(API_DIR) && uv run python -c "import cv2, numpy; print('opencv', cv2.__version__, '| numpy', numpy.__version__)"

dev-api: ## Run FastAPI on :8000 with reload
	cd $(API_DIR) && uv run uvicorn app.main:app --reload --port 8000

dev-web: ## Run Next.js on :3000
	cd $(WEB_DIR) && pnpm dev

test: test-api test-web ## Run all tests

test-api: ## Backend tests (excludes synth + claude_live markers)
	cd $(API_DIR) && uv run pytest

test-web: ## Frontend typecheck + lint + unit tests
	cd $(WEB_DIR) && pnpm typecheck && pnpm lint && pnpm run --if-present test

lint: ## Ruff + ESLint
	cd $(API_DIR) && uv run ruff check . && uv run ruff format --check .
	cd $(WEB_DIR) && pnpm lint

contract: ## Regenerate docs/contract/motion-spec.schema.json + web/lib/types.ts from Pydantic
	cd $(API_DIR) && uv run python -m app.contract

contract-check: ## Fail if generated contract files are stale
	cd $(API_DIR) && uv run python -m app.contract --check

synth: ## Render synthetic scenario videos + ground truth into data/synth (Track S)
	cd $(API_DIR) && uv run python scripts/make_synth.py
	cd $(API_DIR) && uv run python scripts/synth_contact_sheet.py

eval: ## Run the pipeline on data/synth, write <name>.ir.json, check PLAN §11.4 thresholds
	cd $(API_DIR) && uv run python scripts/eval_synth.py

calibrate: ## Monte Carlo check of confidence bands vs §11.4 targets (high band >= 90 % in target)
	cd $(API_DIR) && uv run python scripts/calibrate_confidence.py

clean-jobs: ## Delete all job data (data/jobs + job store DB). Manual retention, PLAN Q3
	rm -rf $(DATA_DIR)/jobs $(DATA_DIR)/mimic.db $(DATA_DIR)/mimic.db-wal $(DATA_DIR)/mimic.db-shm
	mkdir -p $(DATA_DIR)/jobs
	@echo "job data removed"
