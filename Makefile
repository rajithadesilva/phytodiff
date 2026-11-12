# PhytoDiff workflow commands. Run `make help` for the command catalogue.
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Docker commands
# -----------------------------------------------------------------------------
DOCKER_COMPOSE ?= docker compose -f docker/docker-compose.yml
TRAIN_RUN = $(DOCKER_COMPOSE) run --rm train
PREPROCESS_RUN = $(DOCKER_COMPOSE) run --rm \
	-e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 preprocess

# -----------------------------------------------------------------------------
# Data preparation and visualisation
# -----------------------------------------------------------------------------
DATASET_ROOT ?= data/dataset
TOP_DOWN_OCCLUSION_RADIUS_M ?= 0.003
TOP_DOWN_DEPTH_TOLERANCE_M ?= 0.001
SIDE_OCCLUSION_RADIUS_M ?= 0.003
SIDE_DEPTH_TOLERANCE_M ?= 0.001
DATASET_VIS_OUTPUT ?= outputs/dataset_preview
DATASET_VIS_COUNT ?= 0
DATASET_VIS_AZIMUTH_DEG ?= 35
DATASET_VIS_ELEVATION_DEG ?= 15

# -----------------------------------------------------------------------------
# Stage 1: encoder training and visualisation
# -----------------------------------------------------------------------------
STAGE1_PROCESSED_ROOT ?= data/dataset
STAGE1_TRAIN_OUTPUT ?= outputs/encoder/combined/kpconvx
STAGE1_CHECKPOINT ?= $(STAGE1_TRAIN_OUTPUT)/best.ckpt
STAGE1_TRAIN_RESUME ?=
STAGE1_EPOCHS ?= 50
STAGE1_ABLATION_BATCH_SIZE ?= 1
# Ordered, space-separated views for Stage 1 visualisation.
STAGE1_PCL_TYPES ?= full top_down side
STAGE1_VIS_OUTPUT ?= $(STAGE1_TRAIN_OUTPUT)/test_visualizations$(if $(filter 1,$(words $(STAGE1_PCL_TYPES))),$(if $(filter full,$(STAGE1_PCL_TYPES)),,_$(firstword $(STAGE1_PCL_TYPES))),)

# Stage 1 comparison experiments. Variable names remain compatible with old runs.
STAGE1_ABLATION1_OUTPUT ?= outputs/stage1_ablation_1
STAGE1_ABLATION1_DATASET ?= combined
STAGE1_ABLATION1_PCL_TYPES ?= full top_down side
STAGE1_ABLATION1_RESUME ?=
STAGE1_ABLATION2_OUTPUT ?= outputs/stage1_ablation_2
STAGE1_ABLATION2_ABLATION1_OUTPUT ?= $(STAGE1_ABLATION1_OUTPUT)
STAGE1_ABLATION2_RESUME ?=
STAGE1_ABLATION3_OUTPUT ?= outputs/stage1_ablation_3
STAGE1_ABLATION3_RESUME ?=

# -----------------------------------------------------------------------------
# Stage 2: shared model selection and training settings
# -----------------------------------------------------------------------------
STAGE2_MODEL ?= diffts
STAGE2_DATASET ?= combined
STAGE2_PCL_TYPES ?= full top_down side
STAGE2_PROCESSED_ROOT ?= $(STAGE1_PROCESSED_ROOT)
STAGE2_ENCODER_CHECKPOINT ?= $(STAGE1_CHECKPOINT)
STAGE2_FREEZE_STAGE1 ?= true
STAGE2_STAGE1_LR ?= 0.00001
STAGE2_EPOCHS ?= 200
STAGE2_MAX_POINTS ?= 8192
STAGE2_MAX_NODES ?= 256
STAGE2_BATCH_SIZE ?= 16
STAGE2_LEARNING_RATE ?= 0.0001
STAGE2_FLOW_STEPS ?= 20
STAGE2_RESUME ?=
STAGE2_RUN ?= $(TRAIN_RUN) python -u

# DiffTS paths and variant-specific resume/visualisation settings.
STAGE2_OUTPUT_ROOT ?= outputs/diffts
STAGE2_STAGE1_OUTPUT ?= $(STAGE2_OUTPUT_ROOT)/points_stage1_adapter
STAGE2_STAGE1_RESUME ?=
STAGE2_VIS_VARIANT ?= points_stage1
STAGE2_VIS_VIEW ?= full
STAGE2_VIS_STEPS ?= 12
STAGE2_VIS_FRAMES ?= $(STAGE2_VIS_STEPS)
STAGE2_VIS_SAMPLE_STEPS ?=
STAGE2_VIS_COUNT ?= 0
STAGE2_VIS_CHECKPOINT ?= $(if $(filter points_stage1,$(STAGE2_VIS_VARIANT)),$(STAGE2_STAGE1_OUTPUT),$(STAGE2_OUTPUT_ROOT)/$(STAGE2_VIS_VARIANT))/best.ckpt
STAGE2_VIS_OUTPUT ?= $(if $(filter points_stage1,$(STAGE2_VIS_VARIANT)),$(STAGE2_STAGE1_OUTPUT),$(STAGE2_OUTPUT_ROOT)/$(STAGE2_VIS_VARIANT))/test_diffusion_frames

# AG-GFM configuration and dataset/view-specific output paths.
STAGE2_AGGFM_CONFIG ?= configs/flow_matching/aggfm.yaml
STAGE2_AGGFM_OVERRIDES ?=
empty :=
space := $(empty) $(empty)
comma := ,
STAGE2_VIEW_KEY = $(subst $(space),_,$(strip $(STAGE2_PCL_TYPES)))
STAGE2_VIEW_ARRAY = [$(subst $(space),$(comma),$(strip $(STAGE2_PCL_TYPES)))]
STAGE2_AGGFM_OUTPUT ?= outputs/aggfm/$(STAGE2_DATASET)/$(STAGE2_VIEW_KEY)

# Selected-model evaluation, standalone inference and visualisation.
STAGE2_SELECTED_OUTPUT ?= $(if $(filter aggfm,$(STAGE2_MODEL)),$(STAGE2_AGGFM_OUTPUT),$(if $(filter points_stage1,$(STAGE2_VIS_VARIANT)),$(STAGE2_STAGE1_OUTPUT),$(STAGE2_OUTPUT_ROOT)/$(STAGE2_VIS_VARIANT)))
STAGE2_CHECKPOINT ?= $(STAGE2_SELECTED_OUTPUT)/best.ckpt
STAGE2_SPLIT ?= test
STAGE2_SELECTED_VIEWS ?= $(STAGE2_PCL_TYPES)
STAGE2_INFER_VIEW ?= $(if $(filter top_down,$(STAGE2_SELECTED_VIEWS)),top_down,$(firstword $(STAGE2_SELECTED_VIEWS)))
STAGE2_INPUT ?=
STAGE2_EVAL_OUTPUT ?= $(STAGE2_SELECTED_OUTPUT)/evaluation_$(STAGE2_SPLIT).json
STAGE2_INFER_OUTPUT ?= $(STAGE2_SELECTED_OUTPUT)/inference
STAGE2_SELECTED_VIS_OUTPUT ?= $(STAGE2_SELECTED_OUTPUT)/visualizations
STAGE2_SELECTED_VIS_FRAMES ?= 5
STAGE2_SELECTED_VIS_COUNT ?= 1

.PHONY: help docker-build install-local
.PHONY: data-prepare-tomatowur data-prepare-tomatopgt data-prepare-pheno4d
.PHONY: data-generate-top-down data-generate-side data-visualize
.PHONY: stage1-prepare-models stage1-train stage1-visualize-test
.PHONY: stage1-compare-backbones stage1-compare-views stage1-compare-datasets
.PHONY: stage2-train stage2-train-diffts stage2-train-diffts-points
.PHONY: stage2-train-diffts-stage1 stage2-train-aggfm
.PHONY: stage2-evaluate stage2-infer stage2-visualize stage2-visualize-diffts-test
.PHONY: check-tests check-smoke-diffts check-smoke-aggfm
.PHONY: check-tests-local check-smoke-diffts-local data-prepare-tomatowur-local

help: ## Show available workflows and common options (default).
	@awk 'BEGIN { printf "PhytoDiff workflow commands\nUsage: make <command> [VARIABLE=value ...]\n" } /^## / { printf "\n%s\n", substr($$0, 4) } /^[a-zA-Z0-9_-]+:.*## / { split($$0, entry, ":.*## "); printf "  %-36s %s\n", entry[1], entry[2] } END { printf "\nStage 2 options: STAGE2_MODEL=diffts|aggfm, STAGE2_DATASET, STAGE2_PCL_TYPES\nExamples:\n  make stage1-train\n  make stage2-train STAGE2_MODEL=aggfm STAGE2_DATASET=pheno4d STAGE2_PCL_TYPES=side\n\nExisting command names remain available as aliases; variable names and output paths are unchanged.\nSee README.md and docs/TRAINING.md for workflow details.\n" }' $(MAKEFILE_LIST)

## Environment

docker-build: ## Build the Docker training/development image.
	$(DOCKER_COMPOSE) build train

## Data preparation and inspection (Docker)

data-prepare-tomatowur: ## Prepare TomatoWUR caches and derived views.
	$(PREPROCESS_RUN) python scripts/prepare_tomatowur.py --config configs/data/tomatowur_v3.yaml

data-prepare-tomatopgt: ## Prepare TomatoPGT caches and derived views.
	$(PREPROCESS_RUN) python scripts/prepare_tomatopgt.py --config configs/data/tomatopgt_v1.yaml

data-prepare-pheno4d: ## Prepare Pheno4D tomato caches and derived views.
	$(PREPROCESS_RUN) python scripts/prepare_pheno4d.py --config configs/data/pheno4d_tomato.yaml

data-generate-top-down: ## Generate/refresh top-down views from full caches.
	$(PREPROCESS_RUN) python -u scripts/generate_top_down.py "$(DATASET_ROOT)" \
		--occlusion-radius-m "$(TOP_DOWN_OCCLUSION_RADIUS_M)" \
		--depth-tolerance-m "$(TOP_DOWN_DEPTH_TOLERANCE_M)"

data-generate-side: ## Generate/refresh side views from full caches.
	$(PREPROCESS_RUN) python -u scripts/generate_side.py "$(DATASET_ROOT)" \
		--occlusion-radius-m "$(SIDE_OCCLUSION_RADIUS_M)" \
		--depth-tolerance-m "$(SIDE_DEPTH_TOLERANCE_M)"

data-visualize: ## Preview processed point clouds, views and targets.
	$(PREPROCESS_RUN) python -u scripts/visualize_dataset.py "$(DATASET_ROOT)" \
		--count "$(DATASET_VIS_COUNT)" --output "$(DATASET_VIS_OUTPUT)" \
		--azimuth-deg "$(DATASET_VIS_AZIMUTH_DEG)" \
		--elevation-deg "$(DATASET_VIS_ELEVATION_DEG)"

## Stage 1: point cloud encoding (Docker)

stage1-prepare-models: ## Prepare pinned pretrained assets for the encoders.
	$(TRAIN_RUN) python -u scripts/prepare_stage1_models.py

stage1-train: ## Train KPConvX on the combined dataset with all views.
	$(TRAIN_RUN) python -u -m tomato_recon.train.train_encoder --config-name encoder \
		data.dataset=combined data.pcl_types=[full,top_down,side] \
		output.dir=$(STAGE1_TRAIN_OUTPUT) trainer.max_epochs=$(STAGE1_EPOCHS) \
		trainer.batch_size=$(STAGE1_ABLATION_BATCH_SIZE) $(STAGE1_TRAIN_RESUME)

stage1-visualize-test: ## Visualise the Stage 1 checkpoint on held-out plants.
	$(TRAIN_RUN) python scripts/visualize_encoder_predictions.py \
		--checkpoint $(STAGE1_CHECKPOINT) \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--split test --dataset combined --count 0 \
		--pcl-types $(STAGE1_PCL_TYPES) \
		--output $(STAGE1_VIS_OUTPUT)

stage1-compare-backbones: ## Compare encoder architectures (Ablation 1).
	$(TRAIN_RUN) python -u scripts/run_stage1_ablation_1.py \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--output $(STAGE1_ABLATION1_OUTPUT) \
		--dataset $(STAGE1_ABLATION1_DATASET) \
		--pcl-types $(STAGE1_ABLATION1_PCL_TYPES) \
		--max-epochs $(STAGE1_EPOCHS) \
		--batch-size $(STAGE1_ABLATION_BATCH_SIZE) $(STAGE1_ABLATION1_RESUME)

stage1-compare-views: ## Compare view combinations using the winning backbone (Ablation 2).
	$(TRAIN_RUN) python -u scripts/run_stage1_ablation_2.py \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--ablation-1-output $(STAGE1_ABLATION2_ABLATION1_OUTPUT) \
		--output $(STAGE1_ABLATION2_OUTPUT) \
		--max-epochs $(STAGE1_EPOCHS) \
		--batch-size $(STAGE1_ABLATION_BATCH_SIZE) $(STAGE1_ABLATION2_RESUME)

stage1-compare-datasets: ## Compare training/evaluation datasets (Ablation 3).
	$(TRAIN_RUN) python -u scripts/run_stage1_ablation_3.py \
		--processed-root $(STAGE1_PROCESSED_ROOT) \
		--output $(STAGE1_ABLATION3_OUTPUT) \
		--max-epochs $(STAGE1_EPOCHS) \
		--batch-size $(STAGE1_ABLATION_BATCH_SIZE) $(STAGE1_ABLATION3_RESUME)

## Stage 2: skeleton generation (Docker)

stage2-train: ## Train the selected model; defaults to both DiffTS variants.
ifeq ($(STAGE2_MODEL),aggfm)
	$(MAKE) stage2-train-aggfm
else ifeq ($(STAGE2_MODEL),diffts)
	$(MAKE) stage2-train-diffts
else
	$(error STAGE2_MODEL must be diffts or aggfm)
endif

stage2-train-diffts: ## Train both DiffTS conditioning variants in sequence.
	$(MAKE) stage2-train-diffts-points
	$(MAKE) stage2-train-diffts-stage1

stage2-train-diffts-points: ## Train DiffTS conditioned on points and available RGB.
	$(TRAIN_RUN) python -u -m tomato_recon.train.train_diffusion \
		model.diffusion.variant=points \
		model.diffusion.max_points=$(STAGE2_MAX_POINTS) \
		trainer.max_epochs=$(STAGE2_EPOCHS) \
		output.dir=$(STAGE2_OUTPUT_ROOT)/points

stage2-train-diffts-stage1: ## Train DiffTS conditioned on the Stage 1 encoder.
	$(TRAIN_RUN) python -u -m tomato_recon.train.train_diffusion \
		model.diffusion.variant=points_stage1 \
		model.encoder.checkpoint=$(STAGE1_CHECKPOINT) \
		model.diffusion.max_points=$(STAGE2_MAX_POINTS) \
		trainer.max_epochs=$(STAGE2_EPOCHS) \
		trainer.freeze_stage1=$(STAGE2_FREEZE_STAGE1) \
		trainer.stage1_learning_rate=$(STAGE2_STAGE1_LR) \
		output.dir=$(STAGE2_STAGE1_OUTPUT) $(STAGE2_STAGE1_RESUME)

stage2-train-aggfm: ## Train AG-GFM on the selected dataset and views.
	$(STAGE2_RUN) -m tomato_recon.stage2 train --model aggfm \
		--config $(STAGE2_AGGFM_CONFIG) \
		data.dataset=$(STAGE2_DATASET) 'data.pcl_types=$(STAGE2_VIEW_ARRAY)' \
		data.processed_root=$(STAGE2_PROCESSED_ROOT) data.max_nodes=$(STAGE2_MAX_NODES) \
		model.encoder.checkpoint=$(STAGE2_ENCODER_CHECKPOINT) \
		model.aggfm.max_points=$(STAGE2_MAX_POINTS) \
		trainer.freeze_stage1=$(STAGE2_FREEZE_STAGE1) trainer.stage1_learning_rate=$(STAGE2_STAGE1_LR) \
		trainer.max_epochs=$(STAGE2_EPOCHS) trainer.batch_size=$(STAGE2_BATCH_SIZE) \
		trainer.learning_rate=$(STAGE2_LEARNING_RATE) inference.flow_steps=$(STAGE2_FLOW_STEPS) \
		output.dir=$(STAGE2_AGGFM_OUTPUT) $(if $(STAGE2_RESUME),--resume $(STAGE2_RESUME),) $(STAGE2_AGGFM_OVERRIDES)

stage2-evaluate: ## Evaluate the selected model/checkpoint on val or test.
	$(STAGE2_RUN) -m tomato_recon.stage2 evaluate --model $(STAGE2_MODEL) \
		--checkpoint $(STAGE2_CHECKPOINT) --dataset $(STAGE2_DATASET) \
		--pcl-types $(STAGE2_SELECTED_VIEWS) --processed-root $(STAGE2_PROCESSED_ROOT) \
		--split $(STAGE2_SPLIT) --steps $(STAGE2_FLOW_STEPS) --output $(STAGE2_EVAL_OUTPUT)

stage2-infer: ## Reconstruct one point-only input with the selected model.
	$(STAGE2_RUN) -m tomato_recon.stage2 infer --model $(STAGE2_MODEL) \
		--checkpoint $(STAGE2_CHECKPOINT) --input "$(STAGE2_INPUT)" \
		--dataset $(STAGE2_DATASET) --view $(STAGE2_INFER_VIEW) \
		--steps $(STAGE2_FLOW_STEPS) --output $(STAGE2_INFER_OUTPUT)

stage2-visualize: ## Visualise the selected model's generation and predictions.
	$(STAGE2_RUN) -m tomato_recon.stage2 visualize --model $(STAGE2_MODEL) \
		--checkpoint $(STAGE2_CHECKPOINT) --dataset $(STAGE2_DATASET) \
		--pcl-types $(STAGE2_SELECTED_VIEWS) --processed-root $(STAGE2_PROCESSED_ROOT) \
		--split $(STAGE2_SPLIT) --steps $(STAGE2_FLOW_STEPS) \
		--frames $(STAGE2_SELECTED_VIS_FRAMES) --count $(STAGE2_SELECTED_VIS_COUNT) \
		--output $(STAGE2_SELECTED_VIS_OUTPUT)

stage2-visualize-diffts-test: ## Render the original DiffTS held-out denoising frames.
	$(TRAIN_RUN) python -u scripts/visualize_stage2_predictions.py \
		--checkpoint $(STAGE2_VIS_CHECKPOINT) \
		--view $(STAGE2_VIS_VIEW) --frames $(STAGE2_VIS_FRAMES) \
		--count $(STAGE2_VIS_COUNT) --output $(STAGE2_VIS_OUTPUT) $(if $(STAGE2_VIS_SAMPLE_STEPS),--steps $(STAGE2_VIS_SAMPLE_STEPS),)

## Tests and smoke checks (Docker)

check-tests: ## Run the complete unit/integration suite in Docker.
	$(TRAIN_RUN) python -m pytest

check-smoke-diffts: ## Run the existing data-dependent DiffTS training smoke checks.
	$(TRAIN_RUN) bash scripts/smoke_test_all.sh

check-smoke-aggfm: ## Run the synthetic AG-GFM CUDA forward/backward/inference smoke test.
	$(TRAIN_RUN) python -m pytest tests/integration/test_aggfm.py -k cuda -q

## Optional local environment helpers

install-local: ## Install the development package in the host Python environment.
	python -m pip install -e ".[dev]"

check-tests-local: ## Run tests in an already configured host environment.
	python -m pytest

check-smoke-diffts-local: ## Run the DiffTS smoke script in the host environment.
	bash scripts/smoke_test_all.sh

# The original `preprocess` command ran on the host; retain that behavior.
data-prepare-tomatowur-local:
	python scripts/prepare_tomatowur.py --config configs/data/tomatowur_v3.yaml

# -----------------------------------------------------------------------------
# Backwards-compatible command names. Keep old scripts and saved commands valid.
# Host helpers (install/test/smoke/preprocess) retain their original environment.
# -----------------------------------------------------------------------------
.PHONY: install test smoke preprocess generate-top-down generate-side visualize-dataset
.PHONY: prepare-stage1-models train-stage1 visualize-stage1-test
.PHONY: stage1-ablation-1 stage1-ablation-2 stage1-ablation-3
.PHONY: train-stage2 train-stage2-points train-stage2-points-stage1 train-stage2-aggfm
.PHONY: evaluate-stage2-selected infer-stage2-selected visualize-stage2-selected visualize-stage2-test

install: install-local
test: check-tests-local
smoke: check-smoke-diffts-local
preprocess: data-prepare-tomatowur-local
generate-top-down: data-generate-top-down
generate-side: data-generate-side
visualize-dataset: data-visualize
prepare-stage1-models: stage1-prepare-models
train-stage1: stage1-train
visualize-stage1-test: stage1-visualize-test
stage1-ablation-1: stage1-compare-backbones
stage1-ablation-2: stage1-compare-views
stage1-ablation-3: stage1-compare-datasets
train-stage2: stage2-train
train-stage2-points: stage2-train-diffts-points
train-stage2-points-stage1: stage2-train-diffts-stage1
train-stage2-aggfm: stage2-train-aggfm
evaluate-stage2-selected: stage2-evaluate
infer-stage2-selected: stage2-infer
visualize-stage2-selected: stage2-visualize
visualize-stage2-test: stage2-visualize-diffts-test
