"""Selected Stage 2 CLI, preserving the original DiffTS entry points and contracts."""

import argparse
import runpy
import sys

from tomato_recon.config import repository_root
from tomato_recon.models.stage2 import backend


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "evaluate", "infer", "visualize"))
    parser.add_argument("--model", choices=("diffts", "aggfm"), default="diffts")
    selected, rest = parser.parse_known_args(argv)
    if selected.model == "aggfm" or selected.operation == "train":
        return backend(selected.model, selected.operation).main(rest)
    # Adapt the common CLI to the intentionally restricted legacy backend.
    legacy = argparse.ArgumentParser(add_help=False)
    legacy.add_argument("--dataset", default="combined")
    legacy.add_argument("--pcl-types", nargs="+", default=["top_down"])
    legacy.add_argument("--processed-root")
    legacy.add_argument("--view", default="top_down")
    options, remaining = legacy.parse_known_args(rest)
    if options.dataset != "combined":
        raise ValueError("legacy DiffTS selected commands require dataset=combined")
    if options.processed_root:
        # Legacy tools use the root embedded in the checkpoint; verify rather than ignore it.
        from pathlib import Path
        import torch
        path_parser = argparse.ArgumentParser(add_help=False)
        path_parser.add_argument("--checkpoint", required=True)
        path_args, _ = path_parser.parse_known_args(remaining)
        payload = torch.load(path_args.checkpoint, map_location="cpu", weights_only=False)
        if Path(payload["config"]["data"]["processed_root"]).resolve() != Path(options.processed_root).resolve():
            raise ValueError("DiffTS uses the processed root stored in its checkpoint")
    if selected.operation == "evaluate":
        if options.pcl_types == ["top_down"]:
            pass
        elif options.pcl_types == ["full", "top_down", "side"]:
            remaining.append("--all-views")
        else:
            raise ValueError("legacy DiffTS evaluation supports top_down or all three ordered views")
    elif selected.operation == "infer":
        if options.view != "top_down":
            raise ValueError("legacy DiffTS inference requires a top_down.npz input")
    elif selected.operation == "visualize":
        legacy_vis = argparse.ArgumentParser(add_help=False)
        legacy_vis.add_argument("--split", default="test")
        split_args, remaining = legacy_vis.parse_known_args(remaining)
        if split_args.split != "test":
            raise ValueError("legacy DiffTS visualization supports the test split only")
        module = runpy.run_path(str(repository_root() / "scripts/visualize_stage2_predictions.py"))
        for view in options.pcl_types:
            module["main"](remaining + ["--view", view])
        return
    return backend("diffts", selected.operation).main(remaining)


if __name__ == "__main__":
    main(sys.argv[1:])
