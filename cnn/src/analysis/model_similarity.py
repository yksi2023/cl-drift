"""Model pairwise cosine similarity matrices for Fig. 2 (residual stage 3)."""
import os
from typing import Dict, List

import numpy as np
import torch

from src.analysis.drift_metrics import compute_pairwise_similarity_matrix

# Fig. 2 middle row: pairwise checkpoint similarity at residual stage 3.
FIGURE2_SIMILARITY_LAYER = "layer3"


def _reshape_by_layer(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    layer_names: List[str],
) -> Dict[str, Dict[int, torch.Tensor]]:
    sorted_tasks = sorted(reps_cache.keys())
    return {ln: {t: reps_cache[t][ln] for t in sorted_tasks} for ln in layer_names}


def run_model_similarity(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    layer_names: List[str],
    output_dir: str,
):
    """Save the Fig. 2 pairwise cosine matrix as .npy (no per-seed PDFs)."""
    if FIGURE2_SIMILARITY_LAYER not in layer_names:
        print(f"  Skipping model similarity ({FIGURE2_SIMILARITY_LAYER} not in layers)")
        return

    print("\nSaving model similarity matrix (npy, Fig. 2 stage 3)...")
    matrix_dir = os.path.join(output_dir, "model_similarity_matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    sorted_task_indices = sorted(reps_cache.keys())
    all_reps = _reshape_by_layer(reps_cache, [FIGURE2_SIMILARITY_LAYER])
    reps_list = [all_reps[FIGURE2_SIMILARITY_LAYER][t] for t in sorted_task_indices]
    sim_matrix = compute_pairwise_similarity_matrix(reps_list)
    npy_path = os.path.join(matrix_dir, f"similarity_matrix_{FIGURE2_SIMILARITY_LAYER}.npy")
    np.save(npy_path, sim_matrix.numpy())
    print(f"  Saved {npy_path}")
