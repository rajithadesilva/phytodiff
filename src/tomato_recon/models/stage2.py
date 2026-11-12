"""Lazy Stage 2 registry. Conditioning variants remain a DiffTS setting."""

from importlib import import_module


BACKENDS = {
    "diffts": {
        "factory": "tomato_recon.models.diffusion.diffts_runtime",
        "train": "tomato_recon.train.train_diffts",
        "evaluate": "tomato_recon.evaluate_diffts",
        "infer": "tomato_recon.infer_diffts",
    },
    "aggfm": {
        "factory": "tomato_recon.models.flow_matching.runtime",
        "train": "tomato_recon.train.train_aggfm",
        "evaluate": "tomato_recon.evaluate_aggfm",
        "infer": "tomato_recon.infer_aggfm",
        "visualize": "tomato_recon.visualize_aggfm",
    },
}


def backend(name, operation):
    if name not in BACKENDS:
        raise ValueError(f"unknown Stage 2 model {name!r}; choose diffts or aggfm")
    return import_module(BACKENDS[name][operation])


def create_stage2_model(name, cfg):
    return backend(name, "factory").create_model(cfg)
