"""Sample-wise cosine Gram matrices and CKA (Fig. S1).

Per-seed analysis saves numeric outputs only; paper PDFs are produced by
``aggregate_seeds.py``. Heatmaps are written for residual stage 4 at tasks
1, 10, and 20. CKA (vs task 1, vs gap, and the full checkpoint matrix) is
computed for every requested layer.
"""
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PAPER_HEATMAP_LAYER = "layer4"
PAPER_HEATMAP_TASKS = (1, 10, 20)


def compute_sample_similarity_matrix(reps: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity matrix between samples, shape [N, N]."""
    reps_norm = F.normalize(reps, p=2, dim=1)
    return torch.mm(reps_norm, reps_norm.t())


def _hsic(K: torch.Tensor, L: torch.Tensor) -> float:
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - 1.0 / n
    HKH = H @ K @ H
    HLH = H @ L @ H
    return (HKH * HLH).sum().item() / ((n - 1) ** 2)


def linear_cka(S1: torch.Tensor, S2: torch.Tensor) -> float:
    """Linear CKA between two Gram / similarity matrices."""
    hsic_12 = _hsic(S1, S2)
    hsic_11 = _hsic(S1, S1)
    hsic_22 = _hsic(S2, S2)
    denom = (hsic_11 * hsic_22) ** 0.5
    if denom < 1e-12:
        return 0.0
    return hsic_12 / denom


def _pairwise_cka(
    S_by_task: Dict[int, torch.Tensor],
    sorted_task_indices: List[int],
) -> np.ndarray:
    n = len(sorted_task_indices)
    cka_mat = np.eye(n, dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            r = linear_cka(
                S_by_task[sorted_task_indices[a]],
                S_by_task[sorted_task_indices[b]],
            )
            cka_mat[a, b] = cka_mat[b, a] = r
    return cka_mat


def _gap_cka(
    S_by_task: Dict[int, torch.Tensor],
    sorted_task_indices: List[int],
) -> Tuple[List[int], List[float], List[float]]:
    gap_to_vals: Dict[int, List[float]] = {}
    for a_i, i in enumerate(sorted_task_indices):
        for b_i in range(a_i + 1, len(sorted_task_indices)):
            j = sorted_task_indices[b_i]
            gap_to_vals.setdefault(j - i, []).append(
                linear_cka(S_by_task[i], S_by_task[j])
            )
    gaps = sorted(gap_to_vals)
    means = [round(float(np.mean(gap_to_vals[g])), 6) for g in gaps]
    stds = [round(float(np.std(gap_to_vals[g])), 6) for g in gaps]
    return gaps, means, stds


def _class_boundaries(labels: torch.Tensor) -> List[int]:
    sort_indices = torch.argsort(labels)
    sorted_labels = labels[sort_indices]
    unique_labels = sorted_labels.unique()
    boundaries = []
    for lbl in unique_labels[1:]:
        boundary_idx = (sorted_labels == lbl).nonzero(as_tuple=True)[0][0].item()
        boundaries.append(int(boundary_idx))
    return boundaries


def run_sample_similarity(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_names: List[str],
    output_dir: str,
) -> None:
    """Compute sample-similarity Gram matrices and CKA for Fig. S1."""
    print("\n" + "=" * 60)
    print("SAMPLE SIMILARITY / CKA")
    print("=" * 60)

    gram_dir = os.path.join(output_dir, "sample_similarity_matrices")
    evo_dir = os.path.join(output_dir, "sample_similarity_evolution")
    os.makedirs(gram_dir, exist_ok=True)
    os.makedirs(evo_dir, exist_ok=True)

    sorted_task_indices = sorted(reps_cache.keys())
    sort_indices = torch.argsort(labels)
    boundaries = _class_boundaries(labels)
    with open(os.path.join(gram_dir, "class_boundaries.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "boundaries": boundaries,
                "n_samples": int(len(labels)),
                "n_classes": int(len(labels.unique())),
            },
            f,
            indent=2,
        )

    metrics: Dict[str, List[dict]] = {ln: [] for ln in layer_names}
    gap_metrics: Dict[str, dict] = {}

    for layer in layer_names:
        print(f"  Layer: {layer}")
        S_by_task = {
            t: compute_sample_similarity_matrix(reps_cache[t][layer][sort_indices])
            for t in sorted_task_indices
        }
        ref_task = sorted_task_indices[0]
        S_ref = S_by_task[ref_task]
        for task_idx in sorted_task_indices:
            cka_val = 1.0 if task_idx == ref_task else linear_cka(S_ref, S_by_task[task_idx])
            metrics[layer].append({"task": int(task_idx), "cka": round(float(cka_val), 6)})

        gaps, cka_means, cka_stds = _gap_cka(S_by_task, sorted_task_indices)
        gap_metrics[layer] = {
            "gaps": gaps,
            "cka_means": cka_means,
            "cka_stds": cka_stds,
        }

        safe_layer = layer.replace(".", "_").replace("/", "_")
        cka_mat = _pairwise_cka(S_by_task, sorted_task_indices)
        np.save(
            os.path.join(evo_dir, f"sample_sim_cka_matrix_{safe_layer}.npy"),
            cka_mat.astype(np.float32),
        )

        if safe_layer == PAPER_HEATMAP_LAYER or layer == PAPER_HEATMAP_LAYER:
            layer_dir = os.path.join(gram_dir, safe_layer)
            os.makedirs(layer_dir, exist_ok=True)
            for task_idx in PAPER_HEATMAP_TASKS:
                if task_idx not in S_by_task:
                    continue
                stem = f"sample_sim_task{task_idx}_{safe_layer}"
                np.save(
                    os.path.join(layer_dir, f"{stem}.npy"),
                    S_by_task[task_idx].detach().cpu().numpy().astype(np.float32),
                )

        if gaps:
            print(f"    gap=1 CKA={cka_means[0]:.4f}, gap={gaps[-1]} CKA={cka_means[-1]:.4f}")

    with open(os.path.join(evo_dir, "similarity_evolution_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(evo_dir, "similarity_evolution_gap_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(gap_metrics, f, ensure_ascii=False, indent=2)
    print(f"  CKA metrics saved to {evo_dir}")
