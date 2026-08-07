.PHONY: install test smoke docker-build preprocess prepare-stage1-models stage1-ablation visualize-stage1-test

STAGE1_CHECKPOINT ?= outputs/stage1_ablation/kpconvx/best.ckpt
STAGE1_PROCESSED_ROOT ?= data/processed/v3_gt_K256
STAGE1_VIS_OUTPUT ?= outputs/stage1_ablation/kpconvx/test_visualizations
STAGE1_ABLATION_OUTPUT ?= outputs/stage1_ablation
STAGE1_EPOCHS ?= 50
STAGE1_ABLATION_RESUME ?=

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

prepare-stage1-models:
	docker compose -f docker/docker-compose.yml run --rm train \
		python -u scripts/prepare_stage1_models.py

stage1-ablation:
	docker compose -f docker/docker-compose.yml run --rm train \
		python -u scripts/run_stage1_ablation.py \
		--processed-root data/processed/v3_gt_K256 \
		--output $(STAGE1_ABLATION_OUTPUT) \
		--max-epochs $(STAGE1_EPOCHS) $(STAGE1_ABLATION_RESUME)

visualize-stage1-test:
	docker compose -f docker/docker-compose.yml run --rm train \
		python scripts/visualize_encoder_predictions.py \
		--checkpoint $(STAGE1_CHECKPOINT) \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--split test --count 0 \
		--output $(STAGE1_VIS_OUTPUT)
