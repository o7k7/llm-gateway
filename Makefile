.PHONY: up up-redis-stack

up:
	docker compose up -d

up-redis-stack:
	docker compose up redis-stack -d