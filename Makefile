# heco-pipeline — root orchestration Makefile.
#
# Each service (and common/) owns its OWN virtualenv, created by `make venv`
# inside that directory (project hard rule: one venv per service). This root
# Makefile only iterates; it never installs anything itself.
#
# Directories are discovered, not listed, so a new service under services/*
# with a Makefile joins venv-all / test-all / lint automatically.

PYTHON ?= python3.12

SERVICE_DIRS := $(patsubst %/Makefile,%,$(wildcard services/*/Makefile))
ALL_DIRS := common $(SERVICE_DIRS)

.PHONY: venv-all test-all lint models-all clean-venvs help

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

models-all: ## Download pinned model weights for services that need them
	@for d in $(SERVICE_DIRS); do \
		if grep -q '^models:' $$d/Makefile 2>/dev/null; then \
			echo "==> models: $$d"; \
			$(MAKE) -C $$d models || exit 1; \
		fi; \
	done

clean-venvs: ## Remove every per-directory venv
	@for d in $(ALL_DIRS); do rm -rf $$d/.venv; done
