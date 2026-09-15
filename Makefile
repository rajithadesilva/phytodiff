.PHONY: install test smoke docker-build preprocess generate-top-down visualize-dataset prepare-stage1-models stage1-benchmark visualize-stage1-test

DATASET_ROOT ?= data/dataset
TOP_DOWN_OCCLUSION_RADIUS_M ?= 0.003
TOP_DOWN_DEPTH_TOLERANCE_M ?= 0.001
DATASET_VIS_OUTPUT ?= outputs/dataset_preview
DATASET_VIS_COUNT ?= 0
DATASET_VIS_AZIMUTH_DEG ?= 35
DATASET_VIS_ELEVATION_DEG ?= 15

STAGE1_CHECKPOINT ?= outputs/stage1_benchmark/combined/kpconvx/best.ckpt
STAGE1_PROCESSED_ROOT ?= data/dataset
# Visualization: full or top_down.
STAGE1_PCL_TYPE ?= full
STAGE1_VIS_OUTPUT ?= outputs/stage1_benchmark/combined/kpconvx/test_visualizations$(if $(filter top_down,$(STAGE1_PCL_TYPE)),_top_down,)
STAGE1_BENCHMARK_OUTPUT ?= outputs/stage1_benchmark
STAGE1_EPOCHS ?= 50
STAGE1_BENCHMARK_RESUME ?=
# Training and evaluation: full, top_down, or both.
STAGE1_TRAIN_PCL_TYPE ?= full

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

generate-top-down:
	docker compose -f docker/docker-compose.yml run --rm \
		-e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 preprocess \
		python -u scripts/generate_top_down.py "$(DATASET_ROOT)" \
		--occlusion-radius-m "$(TOP_DOWN_OCCLUSION_RADIUS_M)" \
		--depth-tolerance-m "$(TOP_DOWN_DEPTH_TOLERANCE_M)"

visualize-dataset:
	docker compose -f docker/docker-compose.yml run --rm \
		-e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 preprocess \
		python -u scripts/visualize_dataset.py "$(DATASET_ROOT)" \
		--count "$(DATASET_VIS_COUNT)" --output "$(DATASET_VIS_OUTPUT)" \
		--azimuth-deg "$(DATASET_VIS_AZIMUTH_DEG)" \
		--elevation-deg "$(DATASET_VIS_ELEVATION_DEG)"

prepare-stage1-models:
	docker compose -f docker/docker-compose.yml run --rm train \
		python -u scripts/prepare_stage1_models.py

stage1-benchmark:
	docker compose -f docker/docker-compose.yml run --rm train \
		python -u scripts/run_stage1_benchmark.py \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--output $(STAGE1_BENCHMARK_OUTPUT) \
		--pcl-type "$(STAGE1_TRAIN_PCL_TYPE)" \
		--max-epochs $(STAGE1_EPOCHS) $(STAGE1_BENCHMARK_RESUME)

visualize-stage1-test:
	docker compose -f docker/docker-compose.yml run --rm train \
		python scripts/visualize_encoder_predictions.py \
		--checkpoint $(STAGE1_CHECKPOINT) \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--split test --dataset combined --count 0 \
		--pcl-type "$(STAGE1_PCL_TYPE)" \
		--output $(STAGE1_VIS_OUTPUT)
