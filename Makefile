# Makefile (run from repository root)
PY=python
VENV=.venv
APP_DIR=streamlit_app
REQ=$(APP_DIR)/requirements.txt
DB=$(APP_DIR)/data/disasterguard.db
BACKUPS=backups

.PHONY: setup run train db-tables db-backup clean help

help:
	@echo "make setup      # create venv and install deps"
	@echo "make run        # launch Streamlit app"
	@echo "make train      # retrain model (CLI)"
	@echo "make db-tables  # list DB tables"
	@echo "make db-backup  # backup SQLite DB to ./backups/"
	@echo "make clean      # remove __pycache__ dirs"

setup:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r $(REQ)

run:
	. $(VENV)/bin/activate && streamlit run $(APP_DIR)/app.py

train:
	. $(VENV)/bin/activate && $(PY) $(APP_DIR)/train_model.py

db-tables:
	sqlite3 $(DB) "SELECT name FROM sqlite_master WHERE type='table';"

db-backup:
	mkdir -p $(BACKUPS)
	cp $(DB) $(BACKUPS)/disasterguard_$$(date +%Y%m%d_%H%M).db

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +