"""Cross-checkpoint Population Vector (PV) similarity for the appendix figure.

Builds a (checkpoint × time-step) Pearson matrix. Diagonal blocks show
within-checkpoint temporal structure; off-diagonal blocks show drift.
The paper figure splits the probe trial into fixation vs stimulus/response.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.analysis._plot_utils import apply_paper_axis_style, sparse_ticks


def _compute_cross_similarity(
    reps_a: torch.Tensor, reps_b: torch.Tensor, metric: str = "pearson"
) -> np.ndarray:
    """Similarity between every (timestep_i in a, timestep_j in b), averaged over batch."""
    a, b = reps_a, reps_b
    if metric == "pearson":
        a = a - a.mean(dim=2, keepdim=True)
        b = b - b.mean(dim=2, keepdim=True)
    a_norm = F.normalize(a, p=2, dim=2)
    b_norm = F.normalize(b, p=2, dim=2)
    sim = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim.mean(dim=0).numpy()


def _plot_full_matrix(
    full_matrix: np.ndarray,
    seq_len: int,
    n_checkpoints: int,
    task_names: List[str],
    probe_task: str,
    output_path: str,
    metric_label: str = "Pearson Correlation",
):
    """Plot the full (N_checkpoints*Seq_len) x (N_checkpoints*Seq_len) matrix."""
    total = full_matrix.shape[0]
    fig_size = min(max(10, total / 40), 30)
    fig, ax = plt.subplots(figsize=(fig_size + 2, fig_size))

    full_matrix = np.clip(full_matrix, 0, 1)
    ax.imshow(
        full_matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal",
        interpolation="none",
    )

    for k in range(1, n_checkpoints):
        pos = k * seq_len - 0.5
        ax.axhline(y=pos, color="white", linewidth=0.5, alpha=0.7)
        ax.axvline(x=pos, color="white", linewidth=0.5, alpha=0.7)

    sp, sl = sparse_ticks(n_checkpoints)
    sparse_centres = [(k * seq_len + seq_len / 2) for k in sp]
    ax.set_xticks(sparse_centres)
    ax.set_xticklabels(sl, ha="center")
    ax.set_yticks(sparse_centres)
    ax.set_yticklabels(sl)

    ax.set_xlabel("Checkpoint / Time step")
    ax.set_ylabel("Checkpoint / Time step")
    apply_paper_axis_style(ax)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _build_full_matrix(reps_3d_list: List[torch.Tensor], metric: str) -> np.ndarray:
    """Build the full (N*T) x (N*T) cross-checkpoint similarity matrix."""
    n_checkpoints = len(reps_3d_list)
    seq_len = reps_3d_list[0].shape[1]
    total_len = n_checkpoints * seq_len
    full_matrix = np.zeros((total_len, total_len), dtype=np.float32)
    for i in range(n_checkpoints):
        for j in range(i, n_checkpoints):
            block = _compute_cross_similarity(reps_3d_list[i], reps_3d_list[j], metric=metric)
            ri, rj = i * seq_len, j * seq_len
            full_matrix[ri:ri + seq_len, rj:rj + seq_len] = block
            if i != j:
                full_matrix[rj:rj + seq_len, ri:ri + seq_len] = block.T
    return full_matrix


def _get_epoch_boundaries(
    probe_task: str, seq_len: int, batch_size: int = 200, seed: int = 42,
) -> Optional[Dict[str, Tuple[int, int]]]:
    """Regenerate the probe trial to recover epoch boundaries (time-step indices)."""
    try:
        from datasets import get_default_config, get_task_generator
    except ImportError:
        return None

    try:
        gen_fn = get_task_generator(probe_task)
    except ValueError:
        return None

    config = get_default_config()
    config["rng"] = np.random.RandomState(seed)
    trial = gen_fn(config, batch_size, mode="random")

    if not hasattr(trial, "epochs") or not trial.epochs:
        return None

    boundaries: Dict[str, Tuple[int, int]] = {}
    for name, (start, end) in trial.epochs.items():
        s = 0 if start is None else int(start)
        e = seq_len if end is None else int(end)
        boundaries[name] = (s, e)
    return boundaries


def paper_epoch_splits(probe_task: str, seq_len: int) -> List[Tuple[str, int, int]]:
    """Fixation vs stimulus+response splits used in the appendix figure."""
    epochs = _get_epoch_boundaries(probe_task, seq_len)
    if epochs is None:
        raise RuntimeError(f"Cannot recover epoch boundaries for probe '{probe_task}'")

    splits: List[Tuple[str, int, int]] = []
    if "fix1" in epochs:
        s, e = epochs["fix1"]
        splits.append(("fix1", s, e))
    stim_start = None
    if "stim1" in epochs:
        stim_start = epochs["stim1"][0]
    elif "go1" in epochs:
        stim_start = epochs["go1"][0]
    if stim_start is not None:
        splits.append(("stim1_go1", stim_start, seq_len))
    if not splits:
        raise RuntimeError(f"No epoch splits for probe '{probe_task}'")
    return splits
