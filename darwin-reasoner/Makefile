.PHONY: install install-gpu test smoke pilot full analyze clean

install:
	python -m pip install -e .[dev]

install-gpu:
	python -m pip install -e .[all]

test:
	pytest -q

smoke:
	bash scripts/smoke.sh

pilot:
	python -m darwin_reasoner.cli pipeline --config configs/pipeline_pilot.yaml

full:
	python -m darwin_reasoner.cli pipeline --config configs/pipeline_iclr.yaml

analyze:
	python scripts/build_paper_results.py --runs runs --out paper_results

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
