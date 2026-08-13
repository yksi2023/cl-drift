"""Analyze representational drift across continual-learning checkpoints.

Paper pipeline (always runs):
  1. Reference drift vs first checkpoint
  2. Pairwise model similarity matrices
  3. Sample-PV gap drift
  4. Sample UMAP (Fig. 3)
  5. Performance / accuracy matrix
"""
import argparse
import json
import os
from typing import List

import torch

from src.models import MODEL_DEFAULTS, build_model
from src.checkpoints import list_checkpoints
from src.analysis import (
    build_reps_cache,
    run_reference_drift,
    run_model_similarity,
    run_gap_drift,
    plot_cnn_performance,
    run_sample_umap,
)
from src.eval import plot_performance_from_files
from datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze representational drift (paper pipeline)")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="layer1,layer2,layer3,layer4")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for any stochastic analysis steps")
    return parser.parse_args()


def setup_environment(args):
    config_path = os.path.join(args.ckpt_dir, "experiment_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing {config_path}. Re-run training so dataset/model/num_classes are saved."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dataset_name = cfg["dataset"]
    defaults = MODEL_DEFAULTS[dataset_name]
    model_name = cfg.get("model", defaults["model"])
    num_classes = cfg.get("num_classes", defaults["num_classes"])
    img_size = cfg.get("img_size", defaults["img_size"])

    data_manager = build_dataset(
        dataset_name,
        num_classes=num_classes,
        img_size=img_size,
    )
    model = build_model(model_name, num_classes=num_classes)
    print(f"Dataset: {dataset_name} (num_classes={num_classes}, img_size={img_size})")
    print(f"Model:   {model_name}")

    if args.output_dir is None:
        args.output_dir = os.path.join(args.ckpt_dir, "drift_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layer_names: List[str] = [s.strip() for s in args.layers.split(",") if s.strip()]

    meta_path = os.path.join(args.ckpt_dir, "model_after_task_1.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find reference metadata at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        increment = json.load(f)["training_params"]["increment"]

    probe_loader = data_manager.get_loader(
        mode="test",
        label=range(increment),
        batch_size=args.batch_size,
        shuffle=False,
    )

    ckpts = list_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir}")

    return model, probe_loader, layer_names, device


def main():
    args = parse_args()
    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    model, probe_loader, layer_names, device = setup_environment(args)

    print("=" * 60)
    print("DRIFT ANALYSIS")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"Layers: {layer_names}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    print("\n[0/5] Building representation cache...")
    reps_cache, labels, _neuron_indices = build_reps_cache(
        model=model,
        probe_loader=probe_loader,
        ckpt_dir=args.ckpt_dir,
        layer_names=layer_names,
        device=device,
        max_batches=args.max_batches,
        neuron_ratio=1.0,
        seed=args.seed,
    )
    print(f"  Cache built: {len(reps_cache)} checkpoints × {len(layer_names)} layers, "
          f"N={labels.shape[0]} probe samples")

    print("\n[1/5] Reference drift...")
    run_reference_drift(
        reps_cache=reps_cache,
        layer_names=layer_names,
        output_dir=args.output_dir,
    )

    print("\n[2/5] Model similarity...")
    run_model_similarity(
        reps_cache=reps_cache,
        layer_names=layer_names,
        output_dir=args.output_dir,
    )

    print("\n[3/5] Gap drift (sample-PV)...")
    run_gap_drift(
        reps_cache=reps_cache,
        layer_names=layer_names,
        output_dir=args.output_dir,
    )

    print("\n[4/5] Sample UMAP...")
    run_sample_umap(
        reps_cache=reps_cache,
        labels=labels,
        layer_names=layer_names,
        output_dir=args.output_dir,
        pca_var_threshold=0.90,
        color_by="class",
        show_trajectory=False,
    )

    print("\n[5/5] Performance plots...")
    try:
        plot_performance_from_files(args.ckpt_dir, args.output_dir)
    except FileNotFoundError as e:
        print(f"  Skipping line plots: {e}")
    try:
        plot_cnn_performance(args.ckpt_dir, args.output_dir)
    except FileNotFoundError as e:
        print(f"  Skipping accuracy matrix: {e}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
