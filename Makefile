.PHONY: bootstrap dev api web format check check-api check-web clean-runs

bootstrap:
	uv sync --extra model
	npm ci

dev:
	./scripts/dev.sh

api:
	uv run uvicorn llm_lab.app:app --host 127.0.0.1 --port 8000 --reload

web:
	npm run dev:web

format:
	uv run ruff format services/api
	uv run ruff check --fix services/api

check: check-api check-web

check-api:
	uv run --frozen ruff format --check services/api
	uv run --frozen ruff check services/api
	uv run --frozen pytest

check-web:
	npm run lint:web
	npm run typecheck:web
	npm run test:web
	npm run build:web
