.PHONY: install evaluate evaluate-all

install:
	pip install -e ".[dev]"

evaluate:
	@echo "Usage: make evaluate CONFIG=configs/dr_tulu.yaml"
	@test -n "$(CONFIG)" || (echo "ERROR: CONFIG not set" && exit 1)
	python scripts/run_evaluation.py --config $(CONFIG)

evaluate-all:
	bash scripts/run_all.sh
