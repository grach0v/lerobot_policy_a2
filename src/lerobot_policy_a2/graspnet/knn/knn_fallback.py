"""
Pure PyTorch fallback for knn_pytorch CUDA extension.
Works on CPU and MPS (Mac).
"""

import torch


def knn(ref: torch.Tensor, query: torch.Tensor, inds: torch.Tensor) -> None:
    """
    K-nearest neighbors matching knn_pytorch.knn interface.
    Modifies inds in-place.

    Args:
        ref: (B, C, N) reference points
        query: (B, C, M) query points
        inds: (B, k, M) output tensor for indices (modified in-place)
    """
    k = inds.shape[1]

    # Transpose to (B, N, C) and (B, M, C)
    ref_t = ref.permute(0, 2, 1)
    query_t = query.permute(0, 2, 1)

    # Compute squared distances: (B, M, N)
    # dist[b, m, n] = ||query[b, m] - ref[b, n]||^2
    diff = query_t.unsqueeze(2) - ref_t.unsqueeze(1)  # (B, M, N, C)
    dist = torch.sum(diff**2, dim=-1)  # (B, M, N)

    # Get k smallest distances
    _, idx = torch.topk(dist, k, dim=-1, largest=False)  # (B, M, k)

    # Transpose to (B, k, M) and copy to output
    inds.copy_(idx.permute(0, 2, 1))


class KNNModule:
    """Wrapper class matching knn_pytorch module interface."""

    @staticmethod
    def knn(ref: torch.Tensor, query: torch.Tensor, inds: torch.Tensor) -> None:
        return knn(ref, query, inds)


# Create module-level knn function
knn_pytorch = KNNModule()
