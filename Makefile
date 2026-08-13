# heco-pipeline — root orchestration Makefile.
#
# Each service (and common/) owns its OWN virtualenv, created by `make venv`
# inside that directory (project hard rule: one venv per service). This root
# Makefile only iterates; it never installs anything itself.
#
# Directories are discovered, not listed, so a new service under services/*
# with a Makefile joins venv-all / test-all / lint automatically.

PYTHON ?= python3.12

# Accuracy harness defaults (override on the command line). The example
# manifest ships with placeholder paths — point MANIFEST at a real one.
MANIFEST ?= eval/manifest.example.json
LABEL ?= baseline

SERVICE_DIRS := $(patsubst %/Makefile,%,$(wildcard services/*/Makefile))
# The accuracy harness (eval/) is not a service, but it owns a venv and a test
# suite like one, so it joins venv-all / test-all / lint on the same terms.
EVAL_DIR := $(patsubst %/Makefile,%,$(wildcard eval/Makefile))
ALL_DIRS := common counting $(SERVICE_DIRS) $(EVAL_DIR)

# `eval` is also a DIRECTORY, so without .PHONY make would call the target
# up to date and do nothing.
.PHONY: venv-all test-all lint models-all models-restricted clean-venvs eval eval-compare help

help: ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

venv-all: ## Create/refresh the per-directory venvs (common + every service)
	@for d in $(ALL_DIRS); do \
		echo "==> venv: $$d"; \
		$(MAKE) -C $$d venv PYTHON=$(PYTHON) || exit 1; \
	done

test-all: ## Run each directory's pytest suite in its own venv
	@for d in $(ALL_DIRS); do \
		echo "==> test: $$d"; \
		$(MAKE) -C $$d test || exit 1; \
	done

lint: ## ruff check everywhere (root ruff.toml is the single config)
	@for d in $(ALL_DIRS); do \
		echo "==> lint: $$d"; \
		$(MAKE) -C $$d lint || exit 1; \
	done

eval: ## Score ground-truth clips: make eval MANIFEST=eval/clips.json LABEL=baseline
	@eval/.venv/bin/python -m eval.run --manifest $(MANIFEST) --label $(LABEL)

eval-compare: ## A/B two result files: make eval-compare BEFORE=a.json AFTER=b.json
	@eval/.venv/bin/python -m eval.compare $(BEFORE) $(AFTER)

models-all: ## Download pinned model weights for services that need them
	@for d in $(SERVICE_DIRS); do \
		if grep -q '^models:' $$d/Makefile 2>/dev/null; then \
			echo "==> models: $$d"; \
			$(MAKE) -C $$d models || exit 1; \
		fi; \
	done

verify-models: ## Fail if any pinned weight is missing, wrong, or a Docker placeholder
	@./scripts/verify-models.sh

models-restricted: ## Fetch the OPT-IN restricted tier (doc 15 §3) — never part of a default deploy
	@for d in services/*; do \
		if grep -q '^models:' $$d/Makefile 2>/dev/null; then \
			echo "==> models-restricted: $$d"; \
			$(MAKE) -C $$d models LOCK=../../models-restricted.lock || exit 1; \
		fi; \
	done

up: verify-models ## Bring the stack up, but ONLY once every pinned weight checks out
	@docker compose up -d

# WHY `up` HAS A PREREQUISITE. Weights are never committed and never baked
# into the images, so a bind mount is their only source — and Docker turns a
# missing bind-mount source into an empty DIRECTORY rather than failing. That
# produced a stack reporting "Up (healthy)" on every service while the face
# models were dead, counting zero faces for hours. Checking the models.lock
# pins first turns that silent, hours-long failure into a refusal to start.

clean-venvs: ## Remove every per-directory venv
	@for d in $(ALL_DIRS); do rm -rf $$d/.venv; done
