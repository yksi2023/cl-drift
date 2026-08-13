#!/usr/bin/env python3
"""Single entry point for reproducing the paper experiments.

Examples:
  python run_paper.py cnn                 # Fig. 2–3 (10 seeds, paper default)
  python run_paper.py cnn_anchor          # Fig. 4
  python run_paper.py rnn                 # Fig. 5 (+ temporal panels)
  python run_paper.py cnn --seeds 0       # single-seed test run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "paper_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def parse_seeds(arg: Optional[str], cfg: dict) -> List[int]:
    if arg:
        return [int(x) for x in arg.split(",") if x.strip() != ""]
    return list(cfg["seeds"])


def run(cmd: Sequence[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd), f"(cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def flat_train_args(train: dict) -> List[str]:
    """Convert a train-config dict into CLI flags for run_experiment.py."""
    args: List[str] = []
    for key, value in train.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


def cnn_method_extra(method: str, train: dict) -> List[str]:
    if method == "normal":
        return ["--method", "normal"]
    if method == "replay":
        return [
            "--method", "replay",
            "--memory_per_class", str(train["memory_per_class"]),
        ]
    if method == "ewc":
        return [
            "--method", "ewc",
            "--ewc_lambda", str(train["ewc_lambda"]),
            "--ewc_protect", str(train["ewc_protect"]),
        ]
    if method == "lwf":
        return [
            "--method", "lwf",
            "--lwf_lambda", str(train["lwf_lambda"]),
            "--lwf_temperature", str(train["lwf_temperature"]),
        ]
    raise ValueError(f"Unknown CNN method: {method}")


def lambda_tag(lam: float) -> str:
    return str(lam).replace(".", "p")


def cnn_done(save_dir: Path) -> bool:
    return (save_dir / "comprehensive_evaluation.json").is_file()


def rnn_done(save_dir: Path) -> bool:
    return (save_dir / "performance_history.json").is_file()


def cnn_analyzed(save_dir: Path) -> bool:
    return (save_dir / "drift_analysis" / "metrics.json").is_file()


def run_cnn(cfg: dict, seeds: List[int], force: bool) -> None:
    section = cfg["cnn"]
    train = section["train"]
    prefix = section["prefix"]
    methods = section["methods"]
    cnn_dir = ROOT / "cnn"
    exp_root = cnn_dir / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)

    common = flat_train_args({
        k: v for k, v in train.items()
        if k not in ("memory_per_class", "ewc_lambda", "ewc_protect", "lwf_lambda", "lwf_temperature")
    })

    # --- train ---
    for seed in seeds:
        for method in methods:
            save_dir = exp_root / f"{prefix}{method}_seed{seed}"
            if cnn_done(save_dir) and not force:
                print(f"[skip train] {save_dir.name}")
                continue
            cmd = [
                sys.executable, "run_experiment.py",
                *common,
                *cnn_method_extra(method, train),
                "--seed", str(seed),
                "--save_dir", str(save_dir),
            ]
            run(cmd, cnn_dir)

    # --- analyze ---
    for seed in seeds:
        for method in methods:
            save_dir = exp_root / f"{prefix}{method}_seed{seed}"
            if not save_dir.is_dir():
                continue
            if cnn_analyzed(save_dir) and not force:
                print(f"[skip analyze] {save_dir.name}")
                continue
            cmd = [
                sys.executable, "analyze_drift.py",
                "--ckpt_dir", str(save_dir),
                "--layers", section["analyze_layers"],
            ]
            run(cmd, cnn_dir)

    # --- aggregate ---
    out_dir = exp_root / "paper_cnn_aggregate_report"
    cmd = [
        sys.executable, "aggregate_seeds.py",
        "--exp_root", str(exp_root),
        "--prefix", prefix,
        "--methods", ",".join(methods),
        "--layers", section["aggregate_layers"],
        "--output_dir", str(out_dir),
    ]
    run(cmd, cnn_dir)
    print(f"\nCNN aggregate panels → {out_dir}")


def run_cnn_anchor(cfg: dict, seeds: List[int], force: bool) -> None:
    section = cfg["cnn_anchor"]
    train = section["train"]
    prefix = section["prefix"]
    cnn_dir = ROOT / "cnn"
    exp_root = cnn_dir / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)

    common = flat_train_args(train)
    mem = str(section["memory_per_class"])
    loss = section["anchor_loss"]
    layers = section["anchor_layers"]

    jobs: List[tuple] = []
    for seed in seeds:
        jobs.append((
            f"{prefix}replay_l0_seed{seed}",
            [
                "--method", "replay",
                "--memory_per_class", mem,
                "--seed", str(seed),
            ],
        ))
        for lam in section["lambdas"]:
            tag = f"anchored_{loss}_l{lambda_tag(lam)}"
            jobs.append((
                f"{prefix}{tag}_seed{seed}",
                [
                    "--method", "anchored_replay",
                    "--memory_per_class", mem,
                    "--anchor_lambda", str(lam),
                    "--anchor_loss", loss,
                    "--anchor_layers", layers,
                    "--seed", str(seed),
                ],
            ))

    for name, extra in jobs:
        save_dir = exp_root / name
        if cnn_done(save_dir) and not force:
            print(f"[skip train] {save_dir.name}")
            continue
        cmd = [
            sys.executable, "run_experiment.py",
            *common,
            *extra,
            "--save_dir", str(save_dir),
        ]
        run(cmd, cnn_dir)

    for name, _ in jobs:
        save_dir = exp_root / name
        if not save_dir.is_dir():
            continue
        if cnn_analyzed(save_dir) and not force:
            print(f"[skip analyze] {save_dir.name}")
            continue
        cmd = [
            sys.executable, "analyze_drift.py",
            "--ckpt_dir", str(save_dir),
            "--layers", section["analyze_layers"],
        ]
        run(cmd, cnn_dir)

    out_dir = exp_root / "paper_anchor_report"
    cmd = [
        sys.executable, "compare_anchor.py",
        "--exp_root", str(exp_root),
        "--glob", f"{prefix}*",
        "--out_dir", str(out_dir),
    ]
    run(cmd, cnn_dir)
    print(f"\nFig. 4 panels → {out_dir}")
    print("  task1_fwd_acc_vs_lambda.pdf")
    print("  fwd_acc_vs_final_drift.pdf")


def run_rnn(cfg: dict, seeds: List[int], force: bool) -> None:
    section = cfg["rnn"]
    train = section["train"]
    prefix = section["prefix"]
    methods = section["methods"]
    rnn_dir = ROOT / "rnn"
    exp_root = rnn_dir / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)

    base = flat_train_args({
        k: v for k, v in train.items() if k != "memory_per_task"
    })

    for seed in seeds:
        for method in methods:
            save_dir = exp_root / f"{prefix}{method}_seed{seed}"
            if rnn_done(save_dir) and not force:
                print(f"[skip train] {save_dir.name}")
                continue
            extra = ["--method", method]
            if method == "replay":
                extra += ["--memory_per_task", str(train["memory_per_task"])]
            cmd = [
                sys.executable, "run_experiment.py",
                *base,
                *extra,
                "--seed", str(seed),
                "--save_dir", str(save_dir),
            ]
            run(cmd, rnn_dir)

    out_dir = exp_root / "paper_rnn_aggregate_report"
    cmd = [
        sys.executable, "aggregate_seeds.py",
        "--exp_root", str(exp_root),
        "--prefix", prefix,
        "--methods", ",".join(methods),
        "--probe", section["probe"],
        "--output_dir", str(out_dir),
    ]
    run(cmd, rnn_dir)
    print(f"\nRNN aggregate panels → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce paper experiments (train → analyze → aggregate).",
    )
    parser.add_argument(
        "experiment",
        choices=["cnn", "cnn_anchor", "rnn"],
        help="cnn: Figs 2–3; cnn_anchor: Fig 4; rnn: Fig 5",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (default: paper setting in paper_config.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if outputs already exist.",
    )
    args = parser.parse_args()
    cfg = load_config()
    seeds = parse_seeds(args.seeds, cfg)

    print(f"Experiment: {args.experiment}")
    print(f"Seeds ({len(seeds)}): {seeds}")

    if args.experiment == "cnn":
        run_cnn(cfg, seeds, args.force)
    elif args.experiment == "cnn_anchor":
        run_cnn_anchor(cfg, seeds, args.force)
    else:
        run_rnn(cfg, seeds, args.force)


if __name__ == "__main__":
    main()
