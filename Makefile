.PHONY: install test smoke docker-build preprocess visualize-stage1-test

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

visualize-stage1-test:
	docker compose -f docker/docker-compose.yml run --rm train \
		python scripts/visualize_encoder_predictions.py \
		--checkpoint outputs/encoder/best.ckpt \
		--processed-root data/processed/v3_10mm_K256 \
		--split test --count 0 \
		--output outputs/encoder/test_visualizations
