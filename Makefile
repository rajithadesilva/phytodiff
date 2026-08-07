.PHONY: install test smoke docker-build preprocess visualize-stage1-test

STAGE1_CHECKPOINT ?= outputs/encoder/best.ckpt
STAGE1_PROCESSED_ROOT ?= data/processed/v3_gt_K256
STAGE1_VIS_OUTPUT ?= outputs/encoder/test_visualizations

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
		--checkpoint $(STAGE1_CHECKPOINT) \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--split test --count 0 \
		--output $(STAGE1_VIS_OUTPUT)
