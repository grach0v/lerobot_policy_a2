"""Grasp pose generation using GraspNet.

This module wraps GraspNet for generating candidate grasp poses from point clouds.
"""
import copy
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Compatibility shim for numpy
if not hasattr(np, "float"):
    np.float = float


class GraspGenerator:
    """GraspNet-based grasp pose generator.

    Generates candidate 6-DOF grasp poses from point cloud observations.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        num_point: int = 20000,
        num_view: int = 300,
        collision_thresh: float = 0.001,
        empty_thresh: float = 0.15,
        voxel_size: float = 0.01,
        dist_thresh: float = 0.15,
        angle_thresh: float = 70.0,
        refine_approach_dist: float = 0.04,
        device: str = "cuda",
    ):
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.empty_thresh = empty_thresh
        self.voxel_size = voxel_size
        self.dist_thresh = dist_thresh
        self.angle_thresh = angle_thresh
        self.refine_approach_dist = refine_approach_dist
        self.device = device
        self.checkpoint_path = checkpoint_path

        self._graspnet = None
        if checkpoint_path is not None:
            self._init_graspnet()

    def _init_graspnet(self):
        """Initialize GraspNet model."""
        try:
            import torch
            from graspnetAPI import GraspGroup

            # Try to import from the original A2 package
            import sys

            a2_path = Path(__file__).parent.parent.parent.parent.parent / "A2_new"
            if a2_path.exists() and str(a2_path) not in sys.path:
                sys.path.insert(0, str(a2_path))

            from models.graspnet_new_wrapper import GraspNetBaseLine

            self._graspnet = GraspNetBaseLine(
                checkpoint_path=self.checkpoint_path,
                num_point=self.num_point,
                num_view=self.num_view,
                collision_thresh=self.collision_thresh,
                empty_thresh=self.empty_thresh,
                voxel_size=self.voxel_size,
            )
        except ImportError as e:
            print(f"Warning: Could not import GraspNet: {e}")
            print("Grasp generation will use random sampling instead.")
            self._graspnet = None

    def generate_grasps(
        self, point_cloud: np.ndarray, colors: Optional[np.ndarray] = None
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate candidate grasp poses from point cloud.

        Args:
            point_cloud: Point cloud array (N, 3)
            colors: Optional color array (N, 3)

        Returns:
            grasp_poses: List of 7D grasp poses (xyz + quaternion)
            grasp_array: Raw grasp array for visualization
        """
        if self._graspnet is None:
            # Fallback to random sampling
            return self._generate_random_grasps(point_cloud)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.ones_like(point_cloud) * 0.5)

        # Generate grasps
        grasp_pcd = copy.deepcopy(pcd)
        grasp_pcd.points = o3d.utility.Vector3dVector(-point_cloud)

        gg = self._graspnet.inference(grasp_pcd)
        gg.translations = -gg.translations
        gg.rotation_matrices = -gg.rotation_matrices
        gg.translations = gg.translations + gg.rotation_matrices[:, :, 0] * self.refine_approach_dist

        # Collision detection
        if self.collision_thresh > 0:
            gg = self._graspnet.collision_detection(gg, point_cloud)

        # Convert to pose format
        grasp_poses = self._convert_to_poses(gg)

        return grasp_poses, gg

    def _convert_to_poses(self, gg) -> list[np.ndarray]:
        """Convert GraspGroup to list of 7D poses."""
        grasp_poses = []

        ts = gg.translations
        rs = gg.rotation_matrices
        depths = gg.depths

        # Move center to ee_link frame
        ts = ts + rs[:, :, 0] * (np.vstack((depths, depths, depths)).T)

        # Convert coordinate system
        eelink_rs = np.zeros(shape=(len(rs), 3, 3), dtype=np.float32)
        eelink_rs[:, :, 0] = rs[:, :, 2]
        eelink_rs[:, :, 1] = -rs[:, :, 1]
        eelink_rs[:, :, 2] = rs[:, :, 0]

        for i in range(len(gg)):
            grasp_rotation_matrix = eelink_rs[i]
            # Ensure right-hand coordinate system
            if np.linalg.norm(np.cross(grasp_rotation_matrix[:, 0], grasp_rotation_matrix[:, 1]) - grasp_rotation_matrix[:, 2]) > 0.1:
                grasp_rotation_matrix[:, 0] = -grasp_rotation_matrix[:, 0]

            grasp_pose = np.zeros(7)
            grasp_pose[:3] = ts[i]
            r = R.from_matrix(grasp_rotation_matrix)
            grasp_pose[3:7] = r.as_quat()
            grasp_poses.append(grasp_pose)

        return grasp_poses

    def _generate_random_grasps(self, point_cloud: np.ndarray, num_grasps: int = 100) -> tuple[list[np.ndarray], None]:
        """Generate random grasp poses as fallback."""
        grasp_poses = []

        # Sample random points from cloud
        indices = np.random.choice(len(point_cloud), min(num_grasps, len(point_cloud)), replace=False)

        for idx in indices:
            pos = point_cloud[idx]
            # Random top-down orientation with some variation
            angle = np.random.uniform(0, 2 * np.pi)
            r = R.from_euler("xyz", [np.pi, 0, angle])
            quat = r.as_quat()

            grasp_pose = np.zeros(7)
            grasp_pose[:3] = pos
            grasp_pose[:3][2] += 0.02  # Slight offset above point
            grasp_pose[3:7] = quat
            grasp_poses.append(grasp_pose)

        return grasp_poses, None


def create_grasp_generator(checkpoint_path: Optional[str] = None, **kwargs) -> GraspGenerator:
    """Factory function to create grasp generator."""
    return GraspGenerator(checkpoint_path=checkpoint_path, **kwargs)
