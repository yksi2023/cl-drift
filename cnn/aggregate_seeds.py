"""Aggregate multi-seed CNN runs into paper figure panels.

Reads pre-computed ``drift_analysis/`` outputs (no GPU).

Per method:
  1. accuracy_matrix.pdf
  2. similarity_matrix_layer3.pdf   (Fig. 2; residual stage 3 only)
  3. gap_drift_sample_pv.pdf
  4. Fig. S1 (replay): sample-sim heatmaps, CKA matrix, CKA vs checkpoint / gap
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis._plot_utils import (
    SINGLE_FIGSIZE,
    WIDE_FIGSIZE,
    LEGEND_FONT_SIZE,
    LEGEND_TITLE_SIZE,
    apply_paper_axis_style,
    layer_display_name,
    layer_errorbar_kwargs,
    layer_color_map,
    layer_marker_map,
    savefig_compact,
    sparse_ticks,
    sparse_value_ticks,
)
from src.analysis.sample_similarity import PAPER_HEATMAP_LAYER, PAPER_HEATMAP_TASKS
from src.analysis.model_similarity import FIGURE2_SIMILARITY_LAYER


def discover_seed_dirs(exp_root: str, prefix: str, method: str) -> List[str]:
    pattern = os.path.join(exp_root, f"{prefix}{method}_seed*")
    return [d for d in sorted(glob.glob(pattern)) if os.path.isdir(d)]


def load_accuracy_matrix(exp_dir: str) -> Optional[np.ndarray]:
    path = os.path.join(exp_dir, "performance_history.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        perf = json.load(f)
    raw_task_names = sorted(perf.keys(), key=lambda k: int(k.split("_")[1]))
    num_tasks = len(raw_task_names)
    num_stages = max(len(perf[n]) for n in raw_task_names)
    matrix = np.full((num_tasks, num_stages), np.nan)
    for i, name in enumerate(raw_task_names):
        for j, entry in enumerate(perf[name]):
            if entry is not None:
                acc = entry.get("accuracy")
                if acc is not None:
                    matrix[i, j] = acc
    return matrix


def load_similarity_matrix(exp_dir: str, layer: str) -> Optional[np.ndarray]:
    safe_layer = layer.replace(".", "_").replace("/", "_")
    npy_path = os.path.join(
        exp_dir, "drift_analysis", "model_similarity_matrices",
        f"similarity_matrix_{safe_layer}.npy",
    )
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return None


def load_gap_drift(exp_dir: str, layers: List[str]) -> Optional[Dict[str, Tuple[List[int], List[float]]]]:
    json_path = os.path.join(exp_dir, "drift_analysis", "gap_drift", "gap_drift_metrics.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[str, Tuple[List[int], List[float]]] = {}
    for layer in layers:
        if layer in data:
            entry = data[layer]
            spv = entry.get("Sample-PV", entry)
            gaps = [int(g) for g in spv["gaps"]]
            means = [float(m) for m in spv["means"]]
            result[layer] = (gaps, means)
    return result if result else None


def remove_stale_plot(output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        os.remove(path)
        print(f"  Removed stale {filename}")


def plot_avg_accuracy_matrix(
    matrices: List[np.ndarray],
    method: str,
    output_dir: str,
    vmax: Optional[float] = None,
):
    stacked = np.stack(matrices, axis=0)
    mean_matrix = np.nanmean(stacked, axis=0)
    n_tasks = mean_matrix.shape[0]

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.imshow(mean_matrix, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
    ax.set_box_aspect(1)
    sp, sl = sparse_ticks(n_tasks)
    ax.set_xticks(sp)
    ax.set_xticklabels(sl)
    ax.set_xlabel("After Training on Task")
    if method == "normal":
        ax.set_yticks(sp)
        ax.set_yticklabels(sl)
        ax.set_ylabel("Evaluated Task")
    else:
        ax.set_yticks([])
    apply_paper_axis_style(ax)
    path = os.path.join(output_dir, "accuracy_matrix.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] accuracy_matrix.pdf  ({len(matrices)} seeds)")


def plot_avg_similarity_matrix(
    sim_matrices: List[np.ndarray],
    method: str,
    layer: str,
    output_dir: str,
):
    stacked = np.stack(sim_matrices, axis=0)
    mean_sim = np.nanmean(stacked, axis=0)
    n = mean_sim.shape[0]
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.imshow(mean_sim, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    sp, sl = sparse_ticks(n)
    ax.set_xticks(sp)
    ax.set_xticklabels(sl)
    ax.set_xlabel("Model after Task")
    if method == "normal":
        ax.set_yticks(sp)
        ax.set_yticklabels(sl)
        ax.set_ylabel("Model after Task")
    else:
        ax.set_yticks([])
    apply_paper_axis_style(ax)
    safe_layer = layer.replace(".", "_").replace("/", "_")
    path = os.path.join(output_dir, f"similarity_matrix_{safe_layer}.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] similarity_matrix_{safe_layer}.pdf  ({len(sim_matrices)} seeds)")


def plot_avg_gap_drift(
    all_seed_results: List[Dict[str, Tuple[List[int], List[float]]]],
    method: str,
    output_dir: str,
):
    layer_names = list(all_seed_results[0].keys())
    fig, ax = plt.subplots(figsize=(8.8, 7.5))
    n_layers = len(layer_names)
    blue_shades = plt.cm.Blues(
        np.linspace(0.35, 0.85, n_layers) if n_layers > 1 else np.array([0.7])
    )
    all_gaps_union: List[int] = []

    for layer, color in zip(layer_names, blue_shades):
        gap_to_values: Dict[int, List[float]] = defaultdict(list)
        for seed_result in all_seed_results:
            if layer not in seed_result:
                continue
            gaps, means = seed_result[layer]
            for g, m in zip(gaps, means):
                gap_to_values[g].append(m)

        gaps_sorted = sorted(gap_to_values.keys())
        all_gaps_union.extend(gaps_sorted)
        avg = [np.mean(gap_to_values[g]) for g in gaps_sorted]
        std = [np.std(gap_to_values[g]) for g in gaps_sorted]
        ax.errorbar(
            gaps_sorted, avg, yerr=std, label=layer,
            **{
                **layer_errorbar_kwargs(color, "o"),
                "linewidth": 5.0,
                "markersize": 11,
                "markeredgecolor": "none",
                "markeredgewidth": 0,
                "elinewidth": 2.5,
                "capthick": 2.5,
                "capsize": 5,
            },
        )

    ax.set_xlabel("Task Gap")
    ax.set_ylim(-0.1, 1.05)
    if method == "normal":
        ax.set_ylabel("Pearson Correlation")
    else:
        ax.set_yticks([])
    apply_paper_axis_style(
        ax, legend=(method == "normal"),
        legend_kwargs={
            "loc": "upper right",
            "fontsize": LEGEND_FONT_SIZE,
            "title_fontsize": LEGEND_TITLE_SIZE,
        },
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    if all_gaps_union:
        ticks, labels = sparse_value_ticks(all_gaps_union)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    path = os.path.join(output_dir, "gap_drift_sample_pv.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] gap_drift_sample_pv.pdf  ({len(all_seed_results)} seeds)")


def load_sample_similarity_evolution(
    exp_dir: str, layers: List[str],
) -> Optional[Dict[str, List[dict]]]:
    path = os.path.join(
        exp_dir, "drift_analysis", "sample_similarity_evolution",
        "similarity_evolution_metrics.json",
    )
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {layer: data[layer] for layer in layers if layer in data} or None


def load_sample_similarity_gap(
    exp_dir: str, layers: List[str],
) -> Optional[Dict[str, dict]]:
    path = os.path.join(
        exp_dir, "drift_analysis", "sample_similarity_evolution",
        "similarity_evolution_gap_metrics.json",
    )
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {layer: data[layer] for layer in layers if layer in data} or None


def load_sample_sim_cka_matrix(exp_dir: str, layer: str) -> Optional[np.ndarray]:
    safe_layer = layer.replace(".", "_").replace("/", "_")
    path = os.path.join(
        exp_dir, "drift_analysis", "sample_similarity_evolution",
        f"sample_sim_cka_matrix_{safe_layer}.npy",
    )
    if os.path.exists(path):
        return np.load(path)
    return None


def load_sample_similarity_matrix(
    exp_dir: str, layer: str, task_idx: int,
) -> Optional[np.ndarray]:
    safe_layer = layer.replace(".", "_").replace("/", "_")
    path = os.path.join(
        exp_dir, "drift_analysis", "sample_similarity_matrices", safe_layer,
        f"sample_sim_task{task_idx}_{safe_layer}.npy",
    )
    if os.path.exists(path):
        return np.load(path)
    return None


def load_class_boundaries(exp_dir: str) -> Optional[List[int]]:
    path = os.path.join(
        exp_dir, "drift_analysis", "sample_similarity_matrices",
        "class_boundaries.json",
    )
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("boundaries")


def plot_avg_sample_sim_cka_matrix(
    matrices: List[np.ndarray], method: str, layer: str, output_dir: str,
):
    mean_mat = np.nanmean(np.stack(matrices, axis=0), axis=0)
    n = mean_mat.shape[0]
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.imshow(mean_mat, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    sp, sl = sparse_ticks(n)
    ax.set_xticks(sp); ax.set_xticklabels(sl)
    ax.set_yticks(sp); ax.set_yticklabels(sl)
    ax.set_xlabel("Model after Task")
    ax.set_ylabel("Model after Task")
    apply_paper_axis_style(ax)
    safe_layer = layer.replace(".", "_").replace("/", "_")
    path = os.path.join(output_dir, f"sample_sim_cka_matrix_{safe_layer}.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] sample_sim_cka_matrix_{safe_layer}.pdf  ({len(matrices)} seeds)")


def plot_avg_sample_similarity_matrix(
    matrices: List[np.ndarray],
    method: str,
    layer: str,
    task_idx: int,
    output_dir: str,
    class_boundaries: Optional[List[int]] = None,
):
    mean_mat = np.nanmean(np.stack(matrices, axis=0), axis=0)
    safe_layer = layer.replace(".", "_").replace("/", "_")
    out_dir = os.path.join(output_dir, "sample_similarity_matrices", safe_layer)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.imshow(mean_mat, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    if class_boundaries:
        for boundary in class_boundaries:
            ax.axhline(y=boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(x=boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    apply_paper_axis_style(ax)
    path = os.path.join(out_dir, f"sample_sim_task{task_idx}_{safe_layer}.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(
        f"  [{method}] sample_sim_task{task_idx}_{safe_layer}.pdf  ({len(matrices)} seeds)"
    )


def plot_avg_sample_similarity_evolution(
    all_seed_metrics: List[Dict[str, List[dict]]],
    layers: List[str],
    method: str,
    output_dir: str,
):
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for seed_metrics in all_seed_metrics:
        for layer, entries in seed_metrics.items():
            for entry in entries:
                grouped[layer][int(entry["task"])].append(float(entry["cka"]))

    plotted = [layer for layer in layers if layer in grouped]
    colors = layer_color_map(plotted)
    markers = layer_marker_map(plotted)
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    all_tasks: List[int] = []
    for layer in plotted:
        tasks = sorted(grouped[layer])
        means = [np.mean(grouped[layer][t]) for t in tasks]
        stds = [np.std(grouped[layer][t]) for t in tasks]
        all_tasks.extend(tasks)
        ax.errorbar(
            tasks, means, yerr=stds, label=layer_display_name(layer),
            **layer_errorbar_kwargs(colors[layer], markers[layer]),
        )
    ax.set_xlabel("Task")
    ax.set_ylabel("Sample similarity CKA")
    ax.set_ylim(0, 1.05)
    if all_tasks:
        ticks, labels = sparse_value_ticks(all_tasks)
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
    apply_paper_axis_style(ax, legend=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    path = os.path.join(output_dir, "sample_similarity_cka.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] sample_similarity_cka.pdf  ({len(all_seed_metrics)} seeds)")


def plot_avg_sample_similarity_gap(
    all_seed_results: List[Dict[str, dict]],
    layers: List[str],
    method: str,
    output_dir: str,
):
    plotted = [layer for layer in layers if any(layer in r for r in all_seed_results)]
    colors = layer_color_map(plotted)
    markers = layer_marker_map(plotted)
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    all_gaps: List[int] = []
    for layer in plotted:
        gap_to_vals: Dict[int, List[float]] = defaultdict(list)
        for seed_result in all_seed_results:
            if layer not in seed_result:
                continue
            data = seed_result[layer]
            for g, m in zip(data["gaps"], data["cka_means"]):
                gap_to_vals[int(g)].append(float(m))
        gaps_sorted = sorted(gap_to_vals)
        all_gaps.extend(gaps_sorted)
        ax.errorbar(
            gaps_sorted,
            [np.mean(gap_to_vals[g]) for g in gaps_sorted],
            yerr=[np.std(gap_to_vals[g]) for g in gaps_sorted],
            label=layer_display_name(layer),
            **layer_errorbar_kwargs(colors[layer], markers[layer]),
        )
    ax.set_xlabel("Task Gap")
    ax.set_ylabel("Sample similarity CKA")
    ax.set_ylim(-0.1, 1.05)
    if all_gaps:
        ticks, labels = sparse_value_ticks(all_gaps)
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
    apply_paper_axis_style(ax, legend=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    path = os.path.join(output_dir, "sample_similarity_gap_cka.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] sample_similarity_gap_cka.pdf  ({len(all_seed_results)} seeds)")


def plot_sample_similarity_figure_s1(
    seed_dirs: List[str], layers: List[str], method: str, output_dir: str,
):
    """Replay Fig. S1 panels: Gram heatmaps, CKA matrix, CKA vs task / gap."""
    evo = [load_sample_similarity_evolution(sd, layers) for sd in seed_dirs]
    evo = [x for x in evo if x is not None]
    if evo:
        plot_avg_sample_similarity_evolution(evo, layers, method, output_dir)
    else:
        print(f"  [{method}] No CKA-vs-task metrics, skipping.")

    gaps = [load_sample_similarity_gap(sd, layers) for sd in seed_dirs]
    gaps = [x for x in gaps if x is not None]
    if gaps:
        plot_avg_sample_similarity_gap(gaps, layers, method, output_dir)
    else:
        print(f"  [{method}] No CKA-vs-gap metrics, skipping.")

    cka_mats = [load_sample_sim_cka_matrix(sd, PAPER_HEATMAP_LAYER) for sd in seed_dirs]
    cka_mats = [m for m in cka_mats if m is not None]
    if cka_mats:
        plot_avg_sample_sim_cka_matrix(cka_mats, method, PAPER_HEATMAP_LAYER, output_dir)
    else:
        print(f"  [{method}] No CKA matrix .npy for {PAPER_HEATMAP_LAYER}, skipping.")

    boundaries = None
    for sd in seed_dirs:
        boundaries = load_class_boundaries(sd)
        if boundaries is not None:
            break
    for task_idx in PAPER_HEATMAP_TASKS:
        mats = [
            load_sample_similarity_matrix(sd, PAPER_HEATMAP_LAYER, task_idx)
            for sd in seed_dirs
        ]
        mats = [m for m in mats if m is not None]
        if mats:
            plot_avg_sample_similarity_matrix(
                mats, method, PAPER_HEATMAP_LAYER, task_idx, output_dir, boundaries,
            )
        else:
            print(
                f"  [{method}] No sample-sim heatmap for "
                f"{PAPER_HEATMAP_LAYER} task {task_idx}, skipping."
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed CNN results")
    parser.add_argument("--exp_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--methods", type=str, default="normal,replay,ewc,lwf")
    parser.add_argument("--layers", type=str, default="layer1,layer2,layer3,layer4")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    layers = [l.strip() for l in args.layers.split(",")]

    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_root, "aggregate_report")

    print(f"Methods: {methods}")
    print(f"Layers for sim/gap: {layers}")
    print(f"Output: {args.output_dir}")
    print()

    global_acc_vmax: Optional[float] = None
    method_acc_matrices: Dict[str, List[np.ndarray]] = {}
    for method in methods:
        seed_dirs = discover_seed_dirs(args.exp_root, args.prefix, method)
        mats = []
        for sd in seed_dirs:
            m = load_accuracy_matrix(sd)
            if m is not None:
                mats.append(m)
        method_acc_matrices[method] = mats
        if mats:
            stacked = np.stack(mats, axis=0)
            mean_mat = np.nanmax(stacked)
            if math.isfinite(mean_mat):
                if global_acc_vmax is None or mean_mat > global_acc_vmax:
                    global_acc_vmax = float(mean_mat)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "accuracy_matrix_vmax.txt"), "w") as f:
        f.write(f"Global accuracy matrix vmax: {global_acc_vmax}\n")
    print(f"Global accuracy matrix vmax: {global_acc_vmax}")
    print()

    for method in methods:
        seed_dirs = discover_seed_dirs(args.exp_root, args.prefix, method)
        if not seed_dirs:
            print(f"[{method}] No seed directories found, skipping.")
            continue

        print(f"[{method}] Found {len(seed_dirs)} seed(s): "
              f"{[os.path.basename(d) for d in seed_dirs]}")

        method_out = os.path.join(args.output_dir, method)
        os.makedirs(method_out, exist_ok=True)

        acc_matrices = method_acc_matrices.get(method, [])
        if acc_matrices:
            plot_avg_accuracy_matrix(acc_matrices, method, method_out, vmax=global_acc_vmax)
        else:
            print(f"  [{method}] No performance_history.json found, skipping accuracy plot.")

        for ln in [FIGURE2_SIMILARITY_LAYER]:
            sim_mats = []
            for sd in seed_dirs:
                s = load_similarity_matrix(sd, ln)
                if s is not None:
                    sim_mats.append(s)
            if sim_mats:
                plot_avg_similarity_matrix(sim_mats, method, ln, method_out)
            else:
                print(f"  [{method}] No similarity .npy for {ln}, skipping.")

        gap_results: List[Dict[str, Tuple[List[int], List[float]]]] = []
        for sd in seed_dirs:
            g = load_gap_drift(sd, layers)
            if g is not None:
                gap_results.append(g)
        if gap_results:
            plot_avg_gap_drift(gap_results, method, method_out)
        else:
            remove_stale_plot(method_out, "gap_drift_sample_pv.pdf")
            print(f"  [{method}] No gap drift metrics found, skipping.")

        if method == "replay":
            plot_sample_similarity_figure_s1(seed_dirs, layers, method, method_out)

        print()

    print("Aggregation complete.")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
