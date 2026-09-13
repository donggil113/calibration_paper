# Reproduce the study. Every target is idempotent.
PY      ?= python3
RESULTS ?= results/main_foundation
PRESET  ?= main

.PHONY: help install test lint study figures numbers paper check clean data-plan smoke

help:
	@echo "install     install the package and dev extras"
	@echo "test        run the test suite"
	@echo "smoke       ~8 min wiring check on the simulator"
	@echo "study       run the full study (PRESET=$(PRESET) -> $(RESULTS))"
	@echo "reanalyse   rebuild every table from \$$RESULTS/scored.npz, no retraining"
	@echo "figures     build the manuscript figures from \$$RESULTS"
	@echo "numbers     regenerate the manuscript's numbers and tables from \$$RESULTS"
	@echo "paper       figures + numbers + structural check"
	@echo "data-plan   report which cohorts are present, missing, or need credentials"

install:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src experiments scripts tests || true

smoke:
	$(PY) experiments/run_study.py --preset smoke --out results/smoke

study:
	$(PY) experiments/run_study.py --preset $(PRESET) --out $(RESULTS)

reanalyse:
	$(PY) experiments/run_study.py --from-scored $(RESULTS) --preset $(PRESET) --out $(RESULTS)

figures:
	$(PY) experiments/make_figures.py $(RESULTS)

numbers:
	$(PY) experiments/make_numbers.py $(RESULTS)

paper: figures numbers check

check:
	$(PY) scripts/check_manuscript.py paper/

data-plan:
	$(PY) scripts/build_cohorts.py --plan

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ paper/figures/*.pdf paper/figures/*.png
