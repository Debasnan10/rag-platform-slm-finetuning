.PHONY: install run test lint docker-up docker-down

install:
	pip install -r requirements.txt

run:
	uvicorn app.api.main:app --reload

test:
	pytest -v

lint:
	ruff check app tests
	mypy app

docker-up:
	docker compose up --build

docker-down:
	docker compose down
