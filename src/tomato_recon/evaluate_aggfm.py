"""AG-GFM evaluation on an explicit validation or test split."""

import argparse
import json
from pathlib import Path

from tomato_recon.data.aggfm import dataset_compatibility
from tomato_recon.data.processed import PointCloudViewDataset
from tomato_recon.evaluation.aggfm import evaluate_dataset
from tomato_recon.models.flow_matching.runtime import load_runtime


def evaluate(checkpoint, output, *, split="test", dataset=None, views=None, processed_root=None, steps=None):
    model, encoder, cfg, payload, _ = load_runtime(checkpoint)
    selection = dataset or str(cfg.data.dataset)
    root = processed_root or cfg.data.processed_root
    compatibility = dataset_compatibility(root, selection)
    # A combined model may be evaluated on an individual trained source.
    from tomato_recon.models.flow_matching.runtime import check_evaluation_compatibility
    check_evaluation_compatibility(compatibility, payload["dataset_compatibility"])
    selected_views = tuple(views or cfg.data.pcl_types)
    samples = PointCloudViewDataset(root, split=split, dataset=selection, pcl_types=selected_views)
    report = evaluate_dataset(model, encoder, cfg, samples, steps=steps)
    report.update(schema_version="aggfm-evaluation-v1", stage2_model="aggfm", split=split,
                  matching_threshold_m=0.02, dataset=selection, pcl_types=list(selected_views))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--dataset", choices=("combined", "tomatowur", "tomatopgt", "pheno4d"))
    parser.add_argument("--pcl-types", nargs="+", choices=("full", "top_down", "side"))
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args(argv)
    report = evaluate(args.checkpoint, args.output, split=args.split, dataset=args.dataset,
                       views=args.pcl_types, processed_root=args.processed_root, steps=args.steps)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
