.PHONY: install test smoke docker-build preprocess prepare-stage1-models stage1-benchmark visualize-stage1-test

STAGE1_CHECKPOINT ?= outputs/stage1_benchmark/combined/kpconvx/best.ckpt
STAGE1_PROCESSED_ROOT ?= data/dataset
STAGE1_VIS_OUTPUT ?= outputs/stage1_benchmark/combined/kpconvx/test_visualizations
STAGE1_BENCHMARK_OUTPUT ?= outputs/stage1_benchmark
STAGE1_EPOCHS ?= 50
STAGE1_BENCHMARK_RESUME ?=

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

stage1-benchmark:
	docker compose -f docker/docker-compose.yml run --rm train \
		python -u scripts/run_stage1_benchmark.py \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--output $(STAGE1_BENCHMARK_OUTPUT) \
		--max-epochs $(STAGE1_EPOCHS) $(STAGE1_BENCHMARK_RESUME)

visualize-stage1-test:
	docker compose -f docker/docker-compose.yml run --rm train \
		python scripts/visualize_encoder_predictions.py \
		--checkpoint $(STAGE1_CHECKPOINT) \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--split test --dataset combined --count 0 \
		--output $(STAGE1_VIS_OUTPUT)
