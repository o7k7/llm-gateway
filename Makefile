.PHONY: up up-redis-stack lint-format run-linter

up:
	docker compose up -d

up-redis-stack:
	docker compose up redis-stack -d

lint-format:
	uv run ruff format app tests

run-linter:
	uv run ruff check app