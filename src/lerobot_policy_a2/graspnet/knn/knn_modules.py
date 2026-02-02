import torch

# Try to import CUDA knn, fall back to pure PyTorch
try:
    from knn_pytorch import knn_pytorch
except ImportError:
    # Use fallback
    try:
        from . import knn_fallback
    except ImportError:
        import knn_fallback
    knn_pytorch = knn_fallback


def knn(ref, query, k=1):
    """Compute k nearest neighbors for each query point."""
    device = ref.device
    ref = ref.float().to(device)
    query = query.float().to(device)
    inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
    knn_pytorch.knn(ref, query, inds)
    return inds
