"""Load saved probe representations from an experiment directory."""
import os
from typing import Dict

import numpy as np


def _load_reps_from_npz(reps_dir: str, probe_task: str) -> Dict[int, np.ndarray]:
    """Load STPVs for a probe task from a saved .npz file.

    Returns:
        Dict mapping task_idx -> np.ndarray of shape [N, D].
    """
    npz_path = os.path.join(reps_dir, f"{probe_task}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Representations not found: {npz_path}")
    data = np.load(npz_path)
    reps = {}
    for key in sorted(data.files, key=lambda k: int(k.split("_")[-1])):
        idx = int(key.split("_")[-1])
        reps[idx] = data[key]
    return reps
