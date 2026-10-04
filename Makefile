# Mimic — developer entry points. Run `make help`.
SHELL := /bin/bash
.DEFAULT_GOAL := help

API_DIR := api
WEB_DIR := web
# Same data dir as the API/CLI when MIMIC_DATA_DIR is exported (relative = repo root). The
# repo-root .env is not read here, so export it for reset-data if you set it only in .env.
DATA_DIR := $(or $(MIMIC_DATA_DIR),data)

.PHONY: help setup check-tools dev-api dev-web test test-api test-web lint contract contract-check synth eval calibrate create-admin reset-password list-users clean-jobs reset-data

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

calibrate: ## Monte Carlo check of confidence bands vs §11.4 / continuous §8.4 targets (high band >= 90 %)
	cd $(API_DIR) && uv run python scripts/calibrate_confidence.py
	cd $(API_DIR) && uv run python scripts/calibrate_continuous.py

# Round trip (PLAN-continuous §14): replay a replica page in headless Chrome, re-analyze, compare
# with the source IR. HTML/IR/OUT only count when given on the command line.
RT_HTML := $(if $(filter command line,$(origin HTML)),$(HTML))
RT_IR := $(if $(filter command line,$(origin IR)),$(IR))
RT_OUT := $(if $(filter command line,$(origin OUT)),$(OUT))
.PHONY: roundtrip
roundtrip: ## Replica check HTML=index.html IR=source.json [OUT=dir]; no args = harness self-check on synthetic truth
	@if [ -n "$(RT_HTML)" ] || [ -n "$(RT_IR)" ]; then \
	  if [ -z "$(RT_HTML)" ] || [ -z "$(RT_IR)" ]; then echo "usage: make roundtrip HTML=index.html IR=source.json [OUT=dir]"; exit 2; fi; \
	  cd $(API_DIR) && uv run python scripts/roundtrip.py "$(abspath $(RT_HTML))" "$(abspath $(RT_IR))" $(if $(RT_OUT),--out "$(abspath $(RT_OUT))"); \
	else \
	  cd $(API_DIR) && uv run python scripts/roundtrip_validate.py; \
	fi

# Accounts (docs/PLAN-auth.md §6). Passwords are prompted for (or read from stdin with
# PASSWORD_STDIN=1), never passed as arguments. USER= only counts when given on the command
# line, so the shell's own $USER is never used by accident.
CLI := cd $(API_DIR) && uv run python -m app.cli
CLI_USER := $(if $(filter command line,$(origin USER)),$(USER))
CLI_PW_STDIN := $(if $(PASSWORD_STDIN),--password-stdin)

create-admin: ## Create an admin account (first one claims existing jobs). [USER=<name>] [PASSWORD_STDIN=1]
	$(CLI) create-admin $(if $(CLI_USER),--username "$(CLI_USER)") $(CLI_PW_STDIN)

reset-password: ## Break-glass password reset (enables the account, signs it out). USER=<name> [PASSWORD_STDIN=1]
	@if [ -z "$(CLI_USER)" ]; then echo "usage: make reset-password USER=<username>"; exit 2; fi
	$(CLI) reset-password "$(CLI_USER)" $(CLI_PW_STDIN)

list-users: ## List accounts (username, role, active, must-change, jobs)
	$(CLI) list-users

clean-jobs: ## Delete all jobs (rows + data/jobs/*), keep accounts. Manual retention, PLAN Q3
	$(CLI) clean-jobs --yes

reset-data: ## Wipe everything: job store DB incl. accounts + data/jobs (run create-admin again)
	rm -rf $(DATA_DIR)/jobs $(DATA_DIR)/mimic.db $(DATA_DIR)/mimic.db-wal $(DATA_DIR)/mimic.db-shm
	mkdir -p $(DATA_DIR)/jobs
	@echo "all data removed (jobs + accounts); run 'make create-admin' before signing in"
