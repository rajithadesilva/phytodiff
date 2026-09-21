.PHONY: install test smoke docker-build preprocess generate-top-down generate-side visualize-dataset prepare-stage1-models stage1-ablation-1 stage1-ablation-2 stage1-ablation-3 visualize-stage1-test

DATASET_ROOT ?= data/dataset
TOP_DOWN_OCCLUSION_RADIUS_M ?= 0.003
TOP_DOWN_DEPTH_TOLERANCE_M ?= 0.001
SIDE_OCCLUSION_RADIUS_M ?= 0.003
SIDE_DEPTH_TOLERANCE_M ?= 0.001
DATASET_VIS_OUTPUT ?= outputs/dataset_preview
DATASET_VIS_COUNT ?= 0
DATASET_VIS_AZIMUTH_DEG ?= 35
DATASET_VIS_ELEVATION_DEG ?= 15

STAGE1_CHECKPOINT ?= outputs/stage1_ablation_1/combined/kpconvx/best.ckpt
STAGE1_PROCESSED_ROOT ?= data/dataset
# Point-cloud types: full, top_down, side. Use a space-separated ordered list.
# Example: STAGE1_PCL_TYPES= full top_down side for Stage 1 visualization.
STAGE1_PCL_TYPES ?= full top_down side
STAGE1_VIS_OUTPUT ?= outputs/stage1_ablation_1/combined/kpconvx/test_visualizations$(if $(filter 1,$(words $(STAGE1_PCL_TYPES))),$(if $(filter full,$(STAGE1_PCL_TYPES)),,_$(firstword $(STAGE1_PCL_TYPES))),)
STAGE1_ABLATION1_OUTPUT ?= outputs/stage1_ablation_1
STAGE1_ABLATION1_DATASET ?= combined
# Command-line override: make stage1-ablation-1 STAGE1_ABLATION1_PCL_TYPES="full top_down side"
STAGE1_ABLATION1_PCL_TYPES ?= full top_down side
STAGE1_ABLATION1_RESUME ?=
STAGE1_ABLATION2_OUTPUT ?= outputs/stage1_ablation_2
STAGE1_ABLATION2_ABLATION1_OUTPUT ?= $(STAGE1_ABLATION1_OUTPUT)
STAGE1_ABLATION2_RESUME ?=
STAGE1_ABLATION3_OUTPUT ?= outputs/stage1_ablation_3
STAGE1_ABLATION3_BATCH_SIZE ?= 1
STAGE1_ABLATION3_RESUME ?=
STAGE1_EPOCHS ?= 50

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

generate-side:
	docker compose -f docker/docker-compose.yml run --rm \
		-e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 preprocess \
		python -u scripts/generate_side.py "$(DATASET_ROOT)" \
		--occlusion-radius-m "$(SIDE_OCCLUSION_RADIUS_M)" \
		--depth-tolerance-m "$(SIDE_DEPTH_TOLERANCE_M)"

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

stage1-ablation-1:
	docker compose -f docker/docker-compose.yml run --rm train \
		python -u scripts/run_stage1_ablation_1.py \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--output $(STAGE1_ABLATION1_OUTPUT) \
		--dataset $(STAGE1_ABLATION1_DATASET) \
		--pcl-types $(STAGE1_ABLATION1_PCL_TYPES) \
		--max-epochs $(STAGE1_EPOCHS) $(STAGE1_ABLATION1_RESUME)

stage1-ablation-2:
	docker compose -f docker/docker-compose.yml run --rm train \
		python -u scripts/run_stage1_ablation_2.py \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--ablation-1-output $(STAGE1_ABLATION2_ABLATION1_OUTPUT) \
		--output $(STAGE1_ABLATION2_OUTPUT) \
		--max-epochs $(STAGE1_EPOCHS) $(STAGE1_ABLATION2_RESUME)

stage1-ablation-3:
	docker compose -f docker/docker-compose.yml run --rm train \
		python -u scripts/run_stage1_ablation_3.py \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--output $(STAGE1_ABLATION3_OUTPUT) \
		--max-epochs $(STAGE1_EPOCHS) \
		--batch-size $(STAGE1_ABLATION3_BATCH_SIZE) $(STAGE1_ABLATION3_RESUME)

visualize-stage1-test:
	docker compose -f docker/docker-compose.yml run --rm train \
		python scripts/visualize_encoder_predictions.py \
		--checkpoint $(STAGE1_CHECKPOINT) \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--split test --dataset combined --count 0 \
		--pcl-types $(STAGE1_PCL_TYPES) \
		--output $(STAGE1_VIS_OUTPUT)
