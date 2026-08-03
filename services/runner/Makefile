# services/runner — own venv, contract port 7100.

PYTHON ?= python3.12
VENV := .venv
PORT ?= 7100

.PHONY: venv test lint run clean

venv: ## Create the venv; installs heco-common editable
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test: ## Offline tests (the loop against fake stage services, MockTransport)
	$(VENV)/bin/pytest -q

lint: ## ruff against the root config
	$(VENV)/bin/ruff check .

run: ## Serve on the contract port
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

clean: ## Remove venv and caches
	rm -rf $(VENV) .pytest_cache .ruff_cache
