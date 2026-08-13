#!/usr/bin/env python
"""Train a CNN continual-learning run and save checkpoints.

Paper entry point is ``run_paper.py``; this script is the training worker it
invokes. CLI flags match ``paper_config.json`` (no research-only toggles).
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from datasets import DATASET_CHOICES, build_dataset
from src.continual import incremental_learning
from src.models import MODEL_CHOICES, MODEL_DEFAULTS, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train incremental CNN and save checkpoints")
    # Data / model
    parser.add_argument("--dataset", type=str, default="imagenet1k", choices=DATASET_CHOICES)
    parser.add_argument("--model", type=str, default=None, choices=MODEL_CHOICES,
                        help="Defaults per dataset (see MODEL_DEFAULTS).")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--increment", type=int, default=5)
    # Optimization
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs)")
    parser.add_argument("--channels_last", action="store_true")
    # Method
    parser.add_argument("--method", type=str, default="normal",
                        choices=["normal", "replay", "anchored_replay", "ewc", "lwf"])
    parser.add_argument("--memory_per_class", type=int, default=None,
                        help="Replay / anchored_replay: exemplars per class (required for those methods).")
    parser.add_argument("--ewc_lambda", type=float, default=5e5)
    parser.add_argument("--ewc_protect", type=str, default="all", choices=["first", "all"])
    parser.add_argument("--lwf_lambda", type=float, default=30.0)
    parser.add_argument("--lwf_temperature", type=float, default=2.0)
    parser.add_argument("--anchor_lambda", type=float, default=0.0,
                        help="Anchored replay: strength of the representation-anchoring penalty.")
    parser.add_argument("--anchor_layers", type=str, default="layer3,layer4")
    parser.add_argument("--anchor_loss", type=str, default="mse", choices=["mse", "cosine"])
    # Run
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="experiments/run")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.method in ("replay", "anchored_replay") and args.memory_per_class is None:
        raise SystemExit(f"--memory_per_class is required for method={args.method}")

    os.makedirs(args.save_dir, exist_ok=True)
    config_path = os.path.join(args.save_dir, "experiment_config.json")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            # PyTorch 2.9+: prefer the new TF32 API (legacy flags warn).
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            pass

    defaults = MODEL_DEFAULTS[args.dataset]
    model_name = args.model or defaults["model"]
    num_classes = args.num_classes if args.num_classes is not None else defaults["num_classes"]
    img_size = args.img_size if args.img_size is not None else defaults["img_size"]
    args.model = model_name
    args.num_classes = num_classes
    args.img_size = img_size

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    print(f"Experiment configuration saved to {config_path}")

    data_manager = build_dataset(args.dataset, num_classes=num_classes, img_size=img_size)
    model = build_model(model_name, num_classes=num_classes)
    print(f"Dataset: {args.dataset} (num_classes={num_classes}, img_size={img_size})")
    print(f"Model:   {model_name}")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # No weight decay on norm layers / biases (needed for GN+WS+zero-gamma).
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.SGD(
        [
            {"params": decay_params, "weight_decay": 5e-4},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, momentum=0.9, nesterov=True,
    )

    if args.scheduler == "plateau":
        scheduler_config = {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 3,
            "min_lr": 1e-4,
        }
    elif args.scheduler == "cosine":
        scheduler_config = {
            "type": "CosineAnnealingLR",
            "T_max": args.epochs,
            "eta_min": 1e-4,
        }
    else:
        scheduler_config = None

    incremental_learning(
        model,
        data_manager,
        epochs=args.epochs,
        device=device,
        num_classes=num_classes,
        increment=args.increment,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_config=scheduler_config,
        batch_size=args.batch_size,
        method=args.method,
        memory_per_class=args.memory_per_class,
        save_dir=args.save_dir,
        early_stopping_patience=args.patience,
        channels_last=args.channels_last,
        ewc_lambda=args.ewc_lambda,
        ewc_protect=args.ewc_protect,
        lwf_lambda=args.lwf_lambda,
        lwf_temperature=args.lwf_temperature,
        anchor_lambda=args.anchor_lambda,
        anchor_layers=args.anchor_layers,
        anchor_loss=args.anchor_loss,
    )


if __name__ == "__main__":
    main()
