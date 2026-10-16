.PHONY: sync lint type test verify live pg-up pg-down pg-test

sync:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

type:
	uv run mypy src tests

test:
	uv run pytest -v

verify: lint type test

live:
	uv run pytest -m live -v tests/live

# Postgres + pgvector for the vector index and integration tests (where it runs: docs/ROADMAP.md S0)
pg-up:
	docker compose up -d --wait pgvector

pg-down:
	docker compose down

pg-test:
	uv run pytest -m pg -v
