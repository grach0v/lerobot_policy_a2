"""GraspNet baseline wrapper for inference.

This module wraps the GraspNet model for generating grasp candidates from point clouds.
"""

import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# Compatibility shim for deps expecting np.float
if not hasattr(np, "float"):
    np.float = float

import torch

# Lazy import graspnetAPI (requires separate installation due to sklearn dependency)
GraspGroup = None


def _get_grasp_group():
    """Lazy import GraspGroup from graspnetAPI."""
    global GraspGroup
    if GraspGroup is None:
        try:
            from graspnetAPI import GraspGroup as _GraspGroup

            GraspGroup = _GraspGroup
        except ImportError as err:
            raise ImportError(
                "graspnetAPI is not installed. Install with:\n"
                "  SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install graspnetAPI\n"
                "Or: pip install lerobot_policy_a2[graspnet]"
            ) from err
    return GraspGroup


# Add local paths for imports
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "models"))
sys.path.insert(0, str(_ROOT / "pointnet2"))
sys.path.insert(0, str(_ROOT / "utils"))

from collision_detector import ModelFreeCollisionDetector
from graspnet import GraspNet, pred_decode

# Default HuggingFace repo for GraspNet checkpoint
DEFAULT_GRASPNET_REPO = "dgrachev/graspnet_checkpoint"
DEFAULT_CHECKPOINT_FILENAME = "checkpoint-rs.tar"


def get_graspnet_checkpoint(
    checkpoint_path: str | None = None,
    repo_id: str = DEFAULT_GRASPNET_REPO,
    filename: str = DEFAULT_CHECKPOINT_FILENAME,
) -> str:
    """Get path to GraspNet checkpoint, downloading if necessary.

    Args:
        checkpoint_path: Local path to checkpoint. If provided and exists, returns it.
        repo_id: HuggingFace repo ID to download from
        filename: Checkpoint filename in the repo

    Returns:
        Path to the checkpoint file
    """
    # If explicit path provided and exists, use it
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path

    # Download from HuggingFace
    from huggingface_hub import hf_hub_download

    cache_dir = Path.home() / ".cache" / "graspnet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            repo_type="model",
        )
        print(f"GraspNet checkpoint downloaded to {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"Warning: Could not download GraspNet checkpoint: {e}")
        # Try alternative locations
        alt_paths = [
            Path.home() / ".cache" / "graspnet" / filename,
            _ROOT / filename,
            Path.cwd() / "models" / "graspnet_new" / filename,
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                return str(alt_path)
        raise FileNotFoundError(
            f"GraspNet checkpoint not found. Please download manually or upload to {repo_id}"
        ) from e


class GraspNetBaseLine:
    """GraspNet baseline model for grasp pose generation.

    Uses PointNet++ backbone with grasp proposal network for
    generating 6-DOF grasp candidates from point cloud input.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_point: int = 20000,
        num_view: int = 300,
        collision_thresh: float = 0.001,
        empty_thresh: float = 0.15,
        voxel_size: float = 0.01,
    ):
        self.checkpoint_path = checkpoint_path
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.empty_thresh = empty_thresh
        self.voxel_size = voxel_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.get_net()

    def get_net(self):
        """Load and initialize the GraspNet model."""
        net = GraspNet(
            input_feature_dim=0,
            num_view=self.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        net.to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"-> loaded GraspNet checkpoint {self.checkpoint_path} (epoch: {start_epoch})")
        net.eval()
        return net

    def get_and_process_data(self, cloud):
        """Process point cloud for inference."""
        cloud = cloud.voxel_down_sample(0.001)

        cloud_masked = np.asarray(cloud.points)
        color_masked = np.asarray(cloud.colors)

        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        end_points = {}
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        cloud_sampled = cloud_sampled.to(self.device)
        end_points["point_clouds"] = cloud_sampled
        end_points["cloud_colors"] = color_sampled

        return end_points, cloud

    def get_grasps(self, end_points):
        """Generate grasp proposals from processed point cloud."""
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)

        gg_array = grasp_preds[0].detach().cpu().numpy()
        GraspGroupCls = _get_grasp_group()
        gg = GraspGroupCls(gg_array)
        return gg

    def inference(self, o3d_pcd):
        """Run full inference pipeline on a point cloud.

        Args:
            o3d_pcd: Open3D PointCloud object

        Returns:
            GraspGroup with detected grasp poses
        """
        end_points, cloud = self.get_and_process_data(o3d_pcd)
        gg = self.get_grasps(end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        return gg

    def collision_detection(self, gg, cloud):
        """Filter grasps based on collision detection."""
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(
            gg,
            approach_dist=0.05,
            collision_thresh=self.collision_thresh,
            empty_thresh=self.empty_thresh,
        )
        gg = gg[~collision_mask]
        return gg


def vis_grasps(gg, cloud):
    """Visualize grasp proposals on a point cloud."""
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])
