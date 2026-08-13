import torch
import torch.nn.functional as F
from typing import List


def compute_pairwise_pearson_matrix(stpv_list: List[torch.Tensor]) -> torch.Tensor:
    """Pairwise Pearson correlation matrix between checkpoint STPVs.

    Pearson correlation = cosine similarity of mean-centred vectors.
    """
    num_tasks = len(stpv_list)
    corr_matrix = torch.zeros(num_tasks, num_tasks)

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                a = stpv_list[i] - stpv_list[i].mean(dim=1, keepdim=True)
                b = stpv_list[j] - stpv_list[j].mean(dim=1, keepdim=True)
                corr = F.cosine_similarity(a, b, dim=1, eps=1e-8)
                corr_matrix[i, j] = corr.mean().item()

    return corr_matrix
