.PHONY: install test eda baselines pages

install:
	python3 -m pip install -e '.[dev]'

test:
	python3 -m pytest

eda:
	marimo edit notebooks/01_dataset_understanding_eda.py

baselines:
	marimo edit notebooks/02_data_preparation_baselines.py

pages:
	bash scripts/publish_pages.sh
