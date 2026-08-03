# services/ingest — own venv, contract port 7101.

PYTHON ?= python3.12
VENV := .venv
PORT ?= 7101

.PHONY: venv test lint run

venv: ## Create the venv; installs heco-common editable
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test: ## Offline tests (synthetic video, in-process HTTP)
	$(VENV)/bin/pytest -q

lint: ## ruff against the root config
	$(VENV)/bin/ruff check .

run: ## Serve on the contract port
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port $(PORT)
