.PHONY: install test smoke docker-build preprocess

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

smoke:
	bash scripts/smoke_test_all.sh

docker-build:
	docker compose -f docker/docker-compose.yml build train

preprocess:
	python scripts/prepare_tomatowur.py --config configs/data/tomatowur_v3.yaml

