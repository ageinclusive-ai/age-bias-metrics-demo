.PHONY: setup lint test format all

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pre-commit install

lint:
	ruff check .

format:
	black . && ruff check . --fix

test:
	pytest -q

all: format lint test
