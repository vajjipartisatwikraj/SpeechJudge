# Linux / server shortcuts (Windows: use scripts/*.ps1).
PY ?= python

.PHONY: run worker test test-real load-test docker-up

run:          ## start the API from .env
	$(PY) -m app

worker:       ## start a Celery worker (threads pool shares the loaded models)
	celery -A app.workers.celery_app worker --pool=threads --concurrency=2 --loglevel=INFO

test:         ## unit + integration tests (no models needed)
	$(PY) -m pytest -q

test-real:    ## adds end-to-end tests (real models / running stack)
	$(PY) -m pytest -q --real

load-test:    ## concurrency step test against a running API (see tests/load/README.md)
	$(PY) tests/load/concurrency_load.py --max 15

docker-up:    ## build and start redis + api + worker
	docker compose up --build -d
