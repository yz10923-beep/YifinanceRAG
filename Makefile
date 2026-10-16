.PHONY: sync lint type test verify live pg-up pg-down pg-test data-check data-test course-index ollama-test

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
	uv run pytest -m "pg and not ollama" -v

# The real dataset (FLOWPAY_DATASET_ROOT, on the VM): hashes, counts, chunk rebuild, leakage
data-check:
	uv run yfr data check

data-test:
	uv run pytest -m "dataset and not pg" -v

# The course's own index (its pgvector dump) into database course_kb, tables and data only
# (its guard triggers need functions the dump doesn't carry). Needs `make pg-up`.
course-index:
	docker compose exec -T pgvector sh -c 'dropdb -U flowpay --if-exists course_kb && createdb -U flowpay course_kb && psql -qU flowpay -d course_kb -c "CREATE EXTENSION vector"'
	docker compose exec -T pgvector pg_restore --no-owner --no-privileges --section=pre-data --section=data -U flowpay -d course_kb < "$$(uv run -q yfr data path postgres/flowpay_knowledge.dump)"
	docker compose exec -T pgvector psql -U flowpay -d course_kb -c "SELECT collection_id, count(*) FROM chunks GROUP BY 1"

# Our embedder against the course's stored vectors (needs pg-up, course-index and Ollama)
ollama-test:
	uv run pytest -m ollama -v
