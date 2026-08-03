# common/ — shared library. Same three targets as every service directory.

PYTHON ?= python3.12
VENV := .venv

.PHONY: venv test lint

venv: ## Create the venv and install the library editable + dev tools
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test: ## Offline unit tests
	$(VENV)/bin/pytest -q

lint: ## ruff against the root config
	$(VENV)/bin/ruff check .
