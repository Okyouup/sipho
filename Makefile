VENV        := .venv
PYTHON      := $(VENV)/bin/python3
PIP         := $(VENV)/bin/pip
UVICORN     := $(VENV)/bin/uvicorn
SRC_DIR     := src
API_HOST    := 0.0.0.0
API_PORT    := 8000
BUNDLE_OUT  := bundles/bundle.txt
SOURCE_DIRS := src api.py obs_sync.py Ui.html
MEMORY_FILE := src/aegis_memory.json
KNOWLEDGE_FILE := src/aegis_knowledge.json

.PHONY: help install install-dev run serve smoke bundle clean \
        reset-memory check-key deploy logs status restart stop

# ── Default target ─────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  🧠  Aegis-1 — Command Reference"
	@echo ""
	@echo "  ── Setup ──────────────────────────────────────────"
	@echo "  make install        Install all dependencies"
	@echo "  make install-dev    Install + dev/test tools"
	@echo "  make check-key      Verify DEEPSEEK_API_KEY is set"
	@echo ""
	@echo "  ── Run ────────────────────────────────────────────"
	@echo "  make run            Interactive CLI chat"
	@echo "  make run-verbose    Interactive CLI with full telemetry"
	@echo "  make smoke          Run automated smoke tests"
	@echo "  make serve          Start web API on port $(API_PORT)"
	@echo "  make serve-reload   Start web API with auto-reload (dev)"
	@echo ""
	@echo "  ── Memory ─────────────────────────────────────────"
	@echo "  make reset-memory   Delete local memory files (fresh start)"
	@echo "  make show-memory    Print memory file sizes"
	@echo ""
	@echo "  ── Build ──────────────────────────────────────────"
	@echo "  make bundle         Package all source → bundles/bundle.txt"
	@echo "  make clean          Remove caches, logs, .pyc files"
	@echo ""
	@echo "  ── Deploy ─────────────────────────────────────────"
	@echo "  make deploy         Run deploy.sh on this machine"
	@echo "  make logs           Tail systemd service logs"
	@echo "  make status         Check systemd service status"
	@echo "  make restart        Restart systemd service"
	@echo "  make stop           Stop systemd service"
	@echo ""


# ── Setup ─────────────────────────────────────────────────────────────────────
install: $(VENV)/bin/activate
	@echo "→ Installing dependencies..."
	$(PIP) install --upgrade pip -q
	$(PIP) install -q \
		openai \
		fastapi \
		"uvicorn[standard]" \
		numpy \
		sentence-transformers \
		requests \
		nengo \
		torch \
		esdk-obs-python \
		pydantic
	@echo "✓ Dependencies installed"

install-dev: install
	@echo "→ Installing dev tools..."
	$(PIP) install -q pytest pytest-asyncio httpx black ruff
	@echo "✓ Dev tools installed"

$(VENV)/bin/activate:
	@echo "→ Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "✓ Virtual environment created at $(VENV)/"


# ── Key check ─────────────────────────────────────────────────────────────────
check-key:
	@if [ -z "$$DEEPSEEK_API_KEY" ]; then \
		echo "❌  DEEPSEEK_API_KEY is not set."; \
		echo "    Run: export DEEPSEEK_API_KEY=sk-..."; \
		exit 1; \
	else \
		echo "✓ DEEPSEEK_API_KEY is set (starts with: $$(echo $$DEEPSEEK_API_KEY | cut -c1-8)...)"; \
	fi


# ── Run ───────────────────────────────────────────────────────────────────────
run: check-key
	@echo "→ Starting Aegis-1 CLI..."
	$(PYTHON) src/run_aegis.py

run-verbose: check-key
	@echo "→ Starting Aegis-1 CLI (verbose)..."
	$(PYTHON) src/run_aegis.py --verbose

smoke: check-key
	@echo "→ Running smoke tests..."
	$(PYTHON) src/run_aegis.py --smoke

serve: check-key
	@echo "→ Starting Aegis-1 web API on http://$(API_HOST):$(API_PORT)"
	@echo "   UI:   http://localhost:$(API_PORT)"
	@echo "   Docs: http://localhost:$(API_PORT)/docs"
	$(UVICORN) api:app \
		--host $(API_HOST) \
		--port $(API_PORT) \
		--workers 1

serve-reload: check-key
	@echo "→ Starting Aegis-1 web API (dev mode, auto-reload)..."
	$(UVICORN) api:app \
		--host $(API_HOST) \
		--port $(API_PORT) \
		--reload \
		--reload-dir src \
		--reload-dir .


# ── Memory ────────────────────────────────────────────────────────────────────
reset-memory:
	@echo "→ Resetting local memory files..."
	@read -p "  Are you sure? This deletes all saved memories. [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -f $(MEMORY_FILE) $(KNOWLEDGE_FILE); \
		echo "✓ Memory files deleted — Aegis will start fresh"; \
	else \
		echo "  Aborted."; \
	fi

show-memory:
	@echo "→ Memory file status:"
	@if [ -f $(MEMORY_FILE) ]; then \
		echo "  aegis_memory.json    $$(du -h $(MEMORY_FILE) | cut -f1)  ($$(python3 -c "import json; d=json.load(open('$(MEMORY_FILE)')); print(len(d.get('synapses',{})), 'synapses')" 2>/dev/null || echo 'unreadable'))"; \
	else \
		echo "  aegis_memory.json    — not found"; \
	fi
	@if [ -f $(KNOWLEDGE_FILE) ]; then \
		echo "  aegis_knowledge.json $$(du -h $(KNOWLEDGE_FILE) | cut -f1)"; \
	else \
		echo "  aegis_knowledge.json — not found"; \
	fi


# ── Bundle ────────────────────────────────────────────────────────────────────
bundle:
	@echo "→ Bundling source code..."
	@mkdir -p bundles
	@output="$(BUNDLE_OUT)"; \
	echo "========================================" > "$$output"; \
	echo " FULL SOURCE CODE BUNDLE"               >> "$$output"; \
	echo "Generated on $$(date)"                  >> "$$output"; \
	echo "========================================">> "$$output"; \
	for dir in $(SOURCE_DIRS); do \
		find $$dir -type f \
			! -name "*.pyc" \
			! -name "*.json" \
			! -path "*/__pycache__/*" \
			! -path "*/.venv/*" \
		| sort | while read -r file; do \
			echo "  Adding: $$file"; \
			echo ""                                          >> "$$output"; \
			echo "========================================"  >> "$$output"; \
			echo "FILE: $$file"                             >> "$$output"; \
			echo "========================================"  >> "$$output"; \
			echo ""                                          >> "$$output"; \
			cat "$$file"                                     >> "$$output" 2>/dev/null; \
		done; \
	done
	@echo "----------------------------------------"
	@echo "✓ Bundle: $(BUNDLE_OUT)  ($$(du -h $(BUNDLE_OUT) | cut -f1))"


# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	@echo "→ Cleaning caches and build artifacts..."
	find . -type d -name "__pycache__" ! -path "./$(VENV)/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" ! -path "./$(VENV)/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" ! -path "./$(VENV)/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" ! -path "./$(VENV)/*" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf logs/
	@echo "✓ Clean"

clean-all: clean
	@echo "→ Removing virtual environment..."
	rm -rf $(VENV)
	@echo "✓ Full clean — run 'make install' to start fresh"


# ── Deploy (Systemd) ──────────────────────────────────────────────────────────
deploy: check-key
	@echo "→ Running deploy script..."
	chmod +x src/deploy.sh
	bash src/deploy.sh

logs:
	journalctl --user -u aegis -f

status:
	systemctl --user status aegis

restart:
	systemctl --user restart aegis
	@echo "✓ Aegis service restarted"

stop:
	systemctl --user stop aegis
	@echo "✓ Aegis service stopped"