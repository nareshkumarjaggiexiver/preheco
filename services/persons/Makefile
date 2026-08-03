# persons service — venv, model download, lint, test, run.
PY ?= python3.12
SERVICE := persons
LOCK := ../../models.lock
PORT ?= 7102

.PHONY: venv models lint test run

venv:
	$(PY) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# Reads this service's rows from repo-root models.lock: filename, sha256, URL.
# Downloads into models/ (gitignored) and hard-fails on checksum mismatch.
models:
	@mkdir -p models
	@awk -v s=$(SERVICE) '$$1==s {print $$2, $$3, $$4}' $(LOCK) | while read f sha url; do \
		if [ -f "models/$$f" ] && echo "$$sha  models/$$f" | sha256sum --check --status -; then \
			echo "models/$$f already present and verified"; \
		else \
			echo "downloading $$f"; \
			curl -L --fail --silent --show-error -o "models/$$f" "$$url" || exit 1; \
			echo "$$sha  models/$$f" | sha256sum --check - || { rm -f "models/$$f"; exit 1; }; \
		fi; \
	done

lint:
	.venv/bin/python -m ruff check .

test:
	.venv/bin/python -m pytest -q

run:
	.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT)
