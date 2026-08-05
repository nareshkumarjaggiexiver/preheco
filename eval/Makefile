# eval/ — the accuracy harness. Own venv, same three targets as every other
# directory in this repo (project hard rule: one venv per service).
#
# The harness itself needs almost nothing: heco-common (for the planner's
# bearer transport and the run logger) plus pytest and ruff. Everything it
# measures runs in the services' own venvs, over HTTP.

PYTHON ?= python3.12
VENV := .venv

# The harness runs from the REPO ROOT (so `python -m eval.run` resolves the
# package), which is why these paths are root-relative.
MANIFEST ?= eval/manifest.example.json
LABEL ?= baseline

.PHONY: venv test lint eval compare clean

venv: ## Create the venv (heco-common editable + dev tools)
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test: ## Offline unit tests — pure metric maths, filters and verdicts
	$(VENV)/bin/pytest -q

lint: ## ruff against the root config
	$(VENV)/bin/ruff check .

eval: ## Run the harness: make eval MANIFEST=path/to/manifest.json LABEL=name
	cd .. && eval/$(VENV)/bin/python -m eval.run --manifest $(MANIFEST) --label $(LABEL)

compare: ## A/B two result files: make compare BEFORE=a.json AFTER=b.json
	cd .. && eval/$(VENV)/bin/python -m eval.compare $(BEFORE) $(AFTER)

clean: ## Remove venv and caches
	rm -rf $(VENV) .pytest_cache .ruff_cache
