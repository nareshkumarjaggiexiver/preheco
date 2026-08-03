# services/match — own venv, contract port 7106.

PYTHON ?= python3.12
VENV := .venv
PORT ?= 7106

.PHONY: venv test lint run clean

venv: ## Create the venv and install pinned deps
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test: ## Offline tests (synthetic clustered-gaussian embeddings)
	$(VENV)/bin/pytest -q

lint: ## ruff against the root config
	$(VENV)/bin/ruff check .

run: ## Serve on the contract port
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

clean: ## Remove venv and local data
	rm -rf $(VENV) data .pytest_cache .ruff_cache
