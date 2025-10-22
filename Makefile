# ============================================================
# DisasterGuard • Makefile v5 (dev-friendly, DB helpers, lint)
# ============================================================

PY ?= python3
VENV := .venv
APP_DIR := streamlit_app
ENTRY := $(APP_DIR)/app.py
REQ := requirements.txt          # ← unified source of truth
DB := $(APP_DIR)/data/disasterguard.db
SCHEMA := $(APP_DIR)/schema.sql
BACKUPS := backups
TS := $(shell date +%Y%m%d_%H%M)

ifneq ("$(wildcard .env)","")
  include .env
  export
endif

BLUE=\033[34m
GREEN=\033[32m
YELLOW=\033[33m
RED=\033[31m
RESET=\033[0m

.DEFAULT_GOAL := help

.PHONY: help setup upgrade run run-prod train \
        db-tables db-backup db-shell db-sql db-vacuum reset-db \
        doctor clean fmt lint freeze

help:
	@echo ""
	@echo " DisasterGuard • Common tasks"
	@echo "  make setup     # create venv and install deps"
	@echo "  make upgrade   # upgrade pip & all requirements"
	@echo "  make run       # launch Streamlit app (dev)"
	@echo "  make run-prod  # launch Streamlit (watchdog on)"
	@echo "  make train     # retrain model (CLI)"
	@echo "  make db-tables # list SQLite tables"
	@echo "  make db-backup # backup DB to ./backups/"
	@echo "  make db-shell  # open sqlite shell"
	@echo "  make db-sql SQL='...'; # run one SQL statement"
	@echo "  make db-vacuum # VACUUM & ANALYZE database"
	@echo "  make reset-db  # DROP & RECREATE schema (DANGEROUS)"
	@echo "  make fmt       # format (black)"
	@echo "  make lint      # lint (ruff)"
	@echo "  make freeze    # export exact versions to requirements.txt"
	@echo "  make doctor    # quick environment check"
	@echo "  make clean     # remove __pycache__/ and *.pyc"
	@echo ""

setup:
	@echo " • Creating venv and installing requirements..."
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip setuptools wheel
	. $(VENV)/bin/activate && pip install -r $(REQ)
	@echo " ✓ Setup done. Next:  make run"

upgrade:
	@echo " • Upgrading pip & requirements..."
	. $(VENV)/bin/activate && pip install -U pip setuptools wheel
	. $(VENV)/bin/activate && pip install -U -r $(REQ)
	@echo " ✓ Dependencies up-to-date."

run:
	@echo " • Starting Streamlit (dev)..."
	. $(VENV)/bin/activate && PYTHONPATH=$(APP_DIR) streamlit run $(ENTRY)

run-prod:
	@echo " • Starting Streamlit (prod-ish: watchdog recommended)..."
	. $(VENV)/bin/activate && PYTHONPATH=$(APP_DIR) streamlit run $(ENTRY)
	@echo " Tip: If reloads are slow: pip install watchdog"

train:
	@echo " • Training model via CLI..."
	. $(VENV)/bin/activate && $(PY) $(APP_DIR)/train_model.py
	@echo " ✓ Training complete. Models in $(APP_DIR)/models/"

db-tables:
	@echo " • Listing tables in $(DB)..."
	@sqlite3 $(DB) "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"

db-backup:
	@echo " • Backing up DB to ./$(BACKUPS)/ ..."
	mkdir -p $(BACKUPS)
	cp $(DB) $(BACKUPS)/disasterguard_$(TS).db
	@echo " ✓ Backup -> $(BACKUPS)/disasterguard_$(TS).db"

db-shell:
	@echo " • Opening sqlite3 shell (CTRL+D to exit)..."
	sqlite3 $(DB)

db-sql:
	@test -n "$(SQL)" || (echo "Usage: make db-sql SQL='SELECT ...;'" && exit 2)
	@echo " • Running SQL: $(SQL)"
	@sqlite3 -echo -cmd ".headers on" -cmd ".mode column" $(DB) "$(SQL)"

db-vacuum:
	@echo " • VACUUM & ANALYZE $(DB)..."
	@sqlite3 $(DB) "VACUUM; ANALYZE;"
	@echo " ✓ Done."

reset-db:
	@echo " ! WARNING: This will DELETE $(DB) and recreate an empty schema."
	@echo "   Press Ctrl+C to abort, or waiting 3s to continue..."
	@sleep 3
	rm -f $(DB)
	@echo " • Recreating schema..."
	. $(VENV)/bin/activate && $(PY) -c "import sys; sys.path.append('$(APP_DIR)'); from db import init_db; init_db()"
	@echo " ✓ DB reset complete."

fmt:
	@echo " • Formatting with black..."
	. $(VENV)/bin/activate && pip install -q black && black $(APP_DIR)

lint:
	@echo " • Linting with ruff..."
	. $(VENV)/bin/activate && pip install -q ruff && ruff check $(APP_DIR)

freeze:
	@echo " • Freezing installed versions to $(REQ)..."
	. $(VENV)/bin/activate && pip freeze > $(REQ) && tail -n +1 $(REQ) | head -n 10
	@echo " ✓ Wrote versions to $(REQ)."

doctor:
	@echo " • Doctor: environment & files check"
	@if [ -d "$(VENV)" ]; then echo "  - venv: found"; else echo "  - venv: missing (run make setup)"; fi
	@if [ -f "$(REQ)" ]; then echo "  - requirements.txt: ok"; else echo "  - requirements.txt: missing"; fi
	@if [ -f "$(DB)" ]; then echo "  - SQLite DB: $(DB) exists"; else echo "  - SQLite DB: not found (created on first run)"; fi
	@if [ -f "$(APP_DIR)/models/flood_model.pkl" ]; then echo "  - Model: found"; else echo "  - Model: not found (run make train)"; fi
	@if [ -f "$(APP_DIR)/models/metadata.json" ]; then echo "  - Metadata: found"; else echo "  - Metadata: not found (after train)"; fi
	@. $(VENV)/bin/activate && python -c "import streamlit, pandas, sqlite3; print('  - Python libs: OK')"

clean:
	@echo " • Cleaning __pycache__ and *.pyc..."
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo " ✓ Clean complete."