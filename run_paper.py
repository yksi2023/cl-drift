#!/usr/bin/env python3
"""Single entry point for reproducing the paper experiments.

Examples:
  python run_paper.py cnn                 # Fig. 2–3 and S1
  python run_paper.py cnn_anchor          # Fig. 4
  python run_paper.py rnn                 # Fig. 5 and S2
  python run_paper.py cnn --smoke         # seed 0 only
  python run_paper.py cnn_anchor --smoke  # seed 0, two lambdas
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "paper_config.json"
FIGURES = ROOT / "figures"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def parse_seeds(arg: Optional[str], cfg: dict, smoke: bool) -> List[int]:
    if arg:
        return [int(x) for x in arg.split(",") if x.strip() != ""]
    if smoke:
        return list(cfg.get("smoke_seeds", [0]))
    return list(cfg["seeds"])


def parse_lambdas(section: dict, smoke: bool) -> List[float]:
    if smoke:
        return [float(x) for x in section.get("smoke_lambdas", [0.3])]
    return [float(x) for x in section["lambdas"]]


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


def _copy_panel(src: Path, dest: Path) -> None:
    if not src.is_file():
        print(f"[figures] missing {src}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"[figures] {dest.relative_to(ROOT)}")


def _umap_panel(exp_root: Path, prefix: str, method: str, seed: int) -> Optional[Path]:
    path = (
        exp_root / f"{prefix}{method}_seed{seed}"
        / "drift_analysis" / "sample_umap" / "umap_by_class_paper_layer4.pdf"
    )
    return path if path.is_file() else None


def collect_figure3(cfg: dict, seeds: List[int]) -> None:
    section = cfg["cnn"]
    exp_root = ROOT / "cnn" / "experiments"
    dest = FIGURES / "figure3"
    umap_seeds = section.get("figure3_umap", {})
    for letter, method in (("a", "lwf"), ("b", "replay")):
        seed = int(umap_seeds.get(method, seeds[0] if seeds else 0))
        src = _umap_panel(exp_root, section["prefix"], method, seed)
        if src is None:
            print(f"[figures] missing Fig. 3 UMAP for {method} seed {seed}")
            continue
        _copy_panel(src, dest / f"{letter}_{method}_umap_layer4.pdf")
        legend = src.parent / "legend_by_class.pdf"
        if legend.is_file():
            _copy_panel(legend, dest / "legend_by_class.pdf")


def collect_figure2(cfg: dict) -> None:
    agg = ROOT / "cnn" / "experiments" / "paper_cnn_aggregate_report"
    dest = FIGURES / "figure2"
    rows = [
        ("accuracy_matrix.pdf", "accuracy", "a"),
        ("similarity_matrix_layer3.pdf", "similarity_layer3", "e"),
        ("gap_drift_sample_pv.pdf", "gap_drift", "i"),
    ]
    methods = ["normal", "ewc", "lwf", "replay"]
    for filename, stem, start_letter in rows:
        start = ord(start_letter)
        for i, method in enumerate(methods):
            letter = chr(start + i)
            _copy_panel(agg / method / filename, dest / f"{letter}_{method}_{stem}.pdf")


def collect_figure_s1() -> None:
    replay = ROOT / "cnn" / "experiments" / "paper_cnn_aggregate_report" / "replay"
    dest = FIGURES / "figure_s1"
    layer_dir = replay / "sample_similarity_matrices" / "layer4"
    _copy_panel(layer_dir / "sample_sim_task1_layer4.pdf", dest / "a_sample_sim_task1_layer4.pdf")
    _copy_panel(layer_dir / "sample_sim_task10_layer4.pdf", dest / "b_sample_sim_task10_layer4.pdf")
    _copy_panel(layer_dir / "sample_sim_task20_layer4.pdf", dest / "c_sample_sim_task20_layer4.pdf")
    _copy_panel(replay / "sample_sim_cka_matrix_layer4.pdf", dest / "d_cka_matrix_layer4.pdf")
    _copy_panel(replay / "sample_similarity_cka.pdf", dest / "e_cka_vs_checkpoint.pdf")
    _copy_panel(replay / "sample_similarity_gap_cka.pdf", dest / "f_cka_vs_gap.pdf")


def collect_figure4() -> None:
    src_dir = ROOT / "cnn" / "experiments" / "paper_anchor_report"
    dest = FIGURES / "figure4"
    _copy_panel(src_dir / "task1_fwd_acc_vs_lambda.pdf", dest / "a_acc_vs_lambda.pdf")
    _copy_panel(src_dir / "fwd_acc_vs_final_drift.pdf", dest / "b_fwd_acc_vs_drift.pdf")


def collect_figure5() -> None:
    agg = ROOT / "rnn" / "experiments" / "paper_rnn_aggregate_report"
    dest = FIGURES / "figure5"
    _copy_panel(agg / "normal" / "accuracy_matrix.pdf", dest / "a_normal_accuracy.pdf")
    _copy_panel(agg / "normal" / "pearson_matrix_fdgo.pdf", dest / "b_normal_pearson.pdf")
    _copy_panel(agg / "normal" / "vector_drift_fdgo.pdf", dest / "c_normal_vector_drift.pdf")
    _copy_panel(agg / "replay" / "accuracy_matrix.pdf", dest / "d_replay_accuracy.pdf")
    _copy_panel(agg / "replay" / "pearson_matrix_fdgo.pdf", dest / "e_replay_pearson.pdf")
    _copy_panel(agg / "replay" / "vector_drift_fdgo.pdf", dest / "f_replay_vector_drift.pdf")


def collect_figure_s2() -> None:
    agg = ROOT / "rnn" / "experiments" / "paper_rnn_aggregate_report"
    dest = FIGURES / "figure_s2"
    for letter, method, split in (
        ("a", "normal", "fix1"),
        ("b", "normal", "stim1_go1"),
        ("c", "replay", "fix1"),
        ("d", "replay", "stim1_go1"),
    ):
        src = (
            agg / method / "temporal_similarity"
            / f"cross_checkpoint_pearson_fdgo_{split}.pdf"
        )
        _copy_panel(src, dest / f"{letter}_{method}_{split}.pdf")


def cnn_done(save_dir: Path) -> bool:
    return (save_dir / "comprehensive_evaluation.json").is_file()


def rnn_done(save_dir: Path) -> bool:
    return (save_dir / "performance_history.json").is_file()


def cnn_analyzed(save_dir: Path, *, need_sample_sim: bool = False) -> bool:
    drift = save_dir / "drift_analysis"
    if not (drift / "metrics.json").is_file():
        return False
    if need_sample_sim:
        return (
            drift / "sample_similarity_evolution" / "similarity_evolution_metrics.json"
        ).is_file()
    return True


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
            if cnn_analyzed(save_dir, need_sample_sim=True) and not force:
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
    collect_figure2(cfg)
    collect_figure3(cfg, seeds)
    collect_figure_s1()
    print(f"\nPaper panels → {FIGURES / 'figure2'}, figure3, figure_s1")


def run_cnn_anchor(cfg: dict, seeds: List[int], lambdas: List[float], force: bool) -> None:
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
    print(f"Anchor lambdas ({len(lambdas)}): {lambdas}")

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
        for lam in lambdas:
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
    collect_figure4()
    print(f"\nPaper panels → {FIGURES / 'figure4'}")


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
    collect_figure5()
    collect_figure_s2()
    print(f"\nPaper panels → {FIGURES / 'figure5'}, figure_s2")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce paper experiments (train → analyze → aggregate).",
    )
    parser.add_argument(
        "experiment",
        choices=["cnn", "cnn_anchor", "rnn"],
        help="cnn: Figs 2–3 and S1; cnn_anchor: Fig 4; rnn: Fig 5 and S2",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (default: paper setting in paper_config.json).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short pipeline test: seed 0; cnn_anchor also uses two lambdas (0.3 and 1000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if outputs already exist.",
    )
    args = parser.parse_args()
    cfg = load_config()
    seeds = parse_seeds(args.seeds, cfg, args.smoke)

    print(f"Experiment: {args.experiment}")
    print(f"Seeds ({len(seeds)}): {seeds}")

    if args.experiment == "cnn":
        run_cnn(cfg, seeds, args.force)
    elif args.experiment == "cnn_anchor":
        lambdas = parse_lambdas(cfg["cnn_anchor"], args.smoke)
        run_cnn_anchor(cfg, seeds, lambdas, args.force)
    else:
        run_rnn(cfg, seeds, args.force)


if __name__ == "__main__":
    main()
