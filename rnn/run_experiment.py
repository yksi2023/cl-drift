#!/usr/bin/env python
"""Train an RNN continual-learning run and save checkpoints.

Paper entry point is ``run_paper.py``; this script is the training worker it
invokes. CLI flags match ``paper_config.json``.
"""
import argparse
import json
import os

import torch

from datasets import DEFAULT_TASKS, get_default_config, get_task_generator
from src.continual import sequential_learning
from src.models import CognitiveRNN
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sequential cognitive-RNN and save checkpoints"
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--sigma_rec", type=float, default=0.05)
    parser.add_argument("--activation", type=str, default="softplus", choices=["softplus"])
    parser.add_argument("--w_rec_init", type=str, default="diag", choices=["diag"])
    parser.add_argument("--num_iterations", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--method", type=str, default="normal", choices=["normal", "replay"])
    parser.add_argument("--memory_per_task", type=int, default=300,
                        help="Replay: trials stored per past task.")
    parser.add_argument("--replay_num_tasks", type=int, default=5,
                        help="Replay: past tasks sampled per step.")
    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS,
                        help=f"Task sequence. Default: {DEFAULT_TASKS}")
    parser.add_argument("--early_stop_patience", type=int, default=500)
    parser.add_argument("--early_stop_delta", type=float, default=1e-3)
    parser.add_argument("--train_pool_size", type=int, default=30)
    parser.add_argument("--train_seed", type=int, default=12345,
                        help="Base seed for training-set generation (≠ test seed 42).")
    parser.add_argument("--save_dir", type=str, default="experiments/rnn_drift")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.method == "replay" and args.memory_per_task is None:
        raise SystemExit("--memory_per_task is required for method=replay")

    os.makedirs(args.save_dir, exist_ok=True)
    config_path = os.path.join(args.save_dir, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    print(f"Experiment configuration saved to {config_path}")

    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    config = get_default_config(args.tasks)
    model = CognitiveRNN(
        input_size=config["n_input"],
        hidden_size=args.hidden_size,
        output_size=config["n_output"],
        dt=config["dt"],
        tau=config["dt"] / config["alpha"],
        sigma_rec=args.sigma_rec,
        activation=args.activation,
        w_rec_init=args.w_rec_init,
    )
    model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    tasks = [(name, get_task_generator(name)) for name in args.tasks]
    print(f"Task sequence ({len(tasks)}): {[t[0] for t in tasks]}")

    sequential_learning(
        model=model,
        tasks=tasks,
        config=config,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        method=args.method,
        save_dir=args.save_dir,
        train_pool_size=args.train_pool_size,
        train_seed=args.train_seed,
        early_stop_patience=args.early_stop_patience,
        early_stop_delta=args.early_stop_delta,
        memory_per_task=args.memory_per_task,
        replay_num_tasks=args.replay_num_tasks,
    )


if __name__ == "__main__":
    main()
