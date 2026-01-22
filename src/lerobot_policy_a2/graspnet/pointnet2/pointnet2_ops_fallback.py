"""
Pure PyTorch fallback implementations for PointNet2 CUDA operations.
These work on CPU and MPS (Mac) but are slower than CUDA versions.

Based on:
- https://github.com/yanx27/Pointnet_Pointnet2_pytorch (pure Python PointNet2)
- PyTorch3D ops (sample_farthest_points, ball_query)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared Euclidean distance between each pair of points.

    Args:
        src: (B, N, C) source points
        dst: (B, M, C) target points

    Returns:
        dist: (B, N, M) squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling.

    Args:
        xyz: (B, N, 3) point cloud
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Start from random point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index points by indices.

    Args:
        points: (B, N, C) input points
        idx: (B, S) or (B, S, K) indices

    Returns:
        indexed_points: (B, S, C) or (B, S, K, C) indexed points
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query - find points within radius.

    Args:
        radius: query radius
        nsample: max number of points to return
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points

    Returns:
        group_idx: (B, S, nsample) indices of grouped points
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # Fill with first point if not enough neighbors
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def knn_query(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-nearest neighbors query.

    Args:
        k: number of neighbors
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points

    Returns:
        dist: (B, S, k) squared distances
        idx: (B, S, k) indices
    """
    sqrdists = square_distance(new_xyz, xyz)
    dist, idx = torch.topk(sqrdists, k, dim=-1, largest=False)
    return dist, idx


def three_nn(unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find 3 nearest neighbors.

    Args:
        unknown: (B, N, 3) query points
        known: (B, M, 3) reference points

    Returns:
        dist: (B, N, 3) distances to 3 nearest neighbors
        idx: (B, N, 3) indices of 3 nearest neighbors
    """
    dist2, idx = knn_query(3, known, unknown)
    return torch.sqrt(dist2), idx


def three_interpolate(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Interpolate features using 3 nearest neighbors.

    Args:
        features: (B, C, M) features
        idx: (B, N, 3) indices of 3 nearest neighbors
        weight: (B, N, 3) interpolation weights

    Returns:
        interpolated: (B, C, N) interpolated features
    """
    B, C, M = features.shape
    N = idx.shape[1]

    # (B, N, 3, C)
    indexed = index_points(features.permute(0, 2, 1), idx)
    # (B, N, C)
    interpolated = torch.sum(indexed * weight.unsqueeze(-1), dim=2)

    return interpolated.permute(0, 2, 1)


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather features by indices.

    Args:
        features: (B, C, N) features
        idx: (B, npoint) or (B, npoint, nsample) indices

    Returns:
        gathered: (B, C, npoint) or (B, C, npoint, nsample) gathered features
    """
    features_t = features.permute(0, 2, 1)  # (B, N, C)
    gathered = index_points(features_t, idx)  # (B, npoint, C) or (B, npoint, nsample, C)

    if gathered.dim() == 3:
        return gathered.permute(0, 2, 1)  # (B, C, npoint)
    else:
        return gathered.permute(0, 3, 1, 2)  # (B, C, npoint, nsample)


def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Group features by indices.

    Args:
        features: (B, C, N) features
        idx: (B, npoint, nsample) indices

    Returns:
        grouped: (B, C, npoint, nsample) grouped features
    """
    return gather_operation(features, idx)


# KNN wrapper matching knn_pytorch interface
def knn(ref: torch.Tensor, query: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    K-nearest neighbors matching knn_pytorch interface.

    Args:
        ref: (B, C, N) reference points
        query: (B, C, M) query points
        k: number of neighbors

    Returns:
        inds: (B, k, M) indices of k nearest neighbors
    """
    # Transpose to (B, N, C) format
    ref_t = ref.permute(0, 2, 1)
    query_t = query.permute(0, 2, 1)

    _, idx = knn_query(k, ref_t, query_t)  # (B, M, k)

    return idx.permute(0, 2, 1)  # (B, k, M)


# Check if CUDA extensions are available
def cuda_available() -> bool:
    """Check if CUDA PointNet2 extensions are available."""
    try:
        import pointnet2._ext
        return True
    except ImportError:
        return False


def get_device_type(tensor: torch.Tensor) -> str:
    """Get device type string."""
    if tensor.is_cuda:
        return "cuda"
    elif hasattr(tensor, 'is_mps') and tensor.is_mps:
        return "mps"
    else:
        return "cpu"


def cylinder_query(
    radius: float,
    hmin: float,
    hmax: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    rot: torch.Tensor
) -> torch.Tensor:
    """
    Cylinder query - find points within cylinder.

    Args:
        radius: cylinder radius
        hmin: min height along axis
        hmax: max height along axis
        nsample: max points to return
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query centers
        rot: (B, S, 9) flattened rotation matrices

    Returns:
        group_idx: (B, S, nsample) indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    # Reshape rotation matrices
    rot = rot.view(B, S, 3, 3)

    group_idx = torch.zeros(B, S, nsample, dtype=torch.long, device=device)

    for b in range(B):
        for s in range(S):
            # Transform points to cylinder frame
            center = new_xyz[b, s]  # (3,)
            R = rot[b, s]  # (3, 3)

            # Translate and rotate points
            pts = xyz[b] - center  # (N, 3)
            pts_local = torch.matmul(pts, R.T)  # (N, 3) in cylinder frame

            # Check cylinder constraints
            # Height along x-axis in local frame
            h = pts_local[:, 0]
            # Radius in yz-plane
            r = torch.sqrt(pts_local[:, 1]**2 + pts_local[:, 2]**2)

            mask = (h >= hmin) & (h <= hmax) & (r <= radius)
            valid_idx = torch.where(mask)[0]

            n_valid = min(len(valid_idx), nsample)
            if n_valid > 0:
                group_idx[b, s, :n_valid] = valid_idx[:n_valid]
                # Fill rest with first valid index
                if n_valid < nsample:
                    group_idx[b, s, n_valid:] = valid_idx[0]
            else:
                # No valid points, use index 0
                group_idx[b, s, :] = 0

    return group_idx


# Print warning on import
import sys
if not cuda_available():
    print(
        "Note: Using pure PyTorch fallback for PointNet2 ops. "
        "This is slower than CUDA but works on CPU/MPS.",
        file=sys.stderr
    )
