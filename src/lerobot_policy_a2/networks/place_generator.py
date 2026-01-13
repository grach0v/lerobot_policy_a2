"""Place pose generation for pick-and-place tasks.

This module generates candidate place poses based on object locations and spatial relations.
"""
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


class PlaceGenerator:
    """Place pose generator for pick-and-place tasks.

    Generates candidate 6-DOF place poses based on:
    - Object centers and sizes
    - Reference object locations
    - Spatial relations (on, near, etc.)
    """

    def __init__(
        self,
        workspace_limits: Optional[np.ndarray] = None,
        pixel_size: float = 0.002,
        image_size: int = 224,
        min_dist: float = 0.02,
        max_dist: float = 0.1,
    ):
        self.workspace_limits = workspace_limits or np.array([[0.25, 0.75], [-0.3, 0.3], [0.0, 0.3]])
        self.pixel_size = pixel_size
        self.image_size = image_size
        self.min_dist = min_dist
        self.max_dist = max_dist

    def generate_places(
        self,
        depth: np.ndarray,
        obj_centers: list[tuple[int, int]],
        obj_sizes: list[tuple[int, int]],
        ref_obj_centers: Optional[list[tuple[int, int]]] = None,
        sample_num_each_object: int = 5,
        topdown_place_rot: bool = True,
    ) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
        """Generate candidate place poses.

        Args:
            depth: Depth image (H, W)
            obj_centers: List of object centers in pixel coordinates
            obj_sizes: List of object bounding box sizes
            ref_obj_centers: Reference object centers for spatial relations
            sample_num_each_object: Number of samples per object
            topdown_place_rot: Use top-down orientation

        Returns:
            sample_inds: Sampled pixel indices (2, N)
            place_poses: List of 7D place poses
            valid_places_list: Indices of valid places
        """
        sample_inds = None
        valid_places_list = []

        for i, (center, size) in enumerate(zip(obj_centers, obj_sizes)):
            # Generate samples inside object region
            in_samples = self._generate_in_region_samples(center, size, sample_num_each_object)
            if in_samples is not None:
                if sample_inds is None:
                    sample_inds = in_samples
                else:
                    sample_inds = np.concatenate((sample_inds, in_samples), axis=0)

                # Mark valid places for reference objects
                if ref_obj_centers is not None and center in ref_obj_centers:
                    valid_places_list.extend(
                        range(sample_inds.shape[0] - len(in_samples), sample_inds.shape[0])
                    )

            # Generate samples around object region
            out_samples = self._generate_out_region_samples(center, size, sample_num_each_object, depth.shape)
            if out_samples is not None:
                if sample_inds is None:
                    sample_inds = out_samples
                else:
                    sample_inds = np.concatenate((sample_inds, out_samples), axis=0)

        if sample_inds is None:
            return np.array([]), [], []

        # Convert to poses
        sample_inds = sample_inds.T
        place_poses = self._convert_to_poses(sample_inds, depth, topdown_place_rot)

        return sample_inds, place_poses, valid_places_list

    def _generate_in_region_samples(
        self, center: tuple[int, int], size: tuple[int, int], num_samples: int
    ) -> Optional[np.ndarray]:
        """Generate samples inside object bounding box."""
        cx, cy = center
        sx, sy = size
        radius = min(sx, sy) // 3

        # Generate points in a circle around center
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        radii = np.random.uniform(0, radius, num_samples)

        x = cx + (radii * np.cos(angles)).astype(int)
        y = cy + (radii * np.sin(angles)).astype(int)

        return np.stack([x, y], axis=1)

    def _generate_out_region_samples(
        self,
        center: tuple[int, int],
        size: tuple[int, int],
        num_samples: int,
        img_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Generate samples around object bounding box."""
        cx, cy = center
        sx, sy = size
        inner_radius = max(sx, sy) // 2 + int(self.min_dist / self.pixel_size)
        outer_radius = inner_radius + int((self.max_dist - self.min_dist) / self.pixel_size)

        # Generate points in annulus
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        radii = np.random.uniform(inner_radius, outer_radius, num_samples)

        x = cx + (radii * np.cos(angles)).astype(int)
        y = cy + (radii * np.sin(angles)).astype(int)

        # Clip to image bounds
        x = np.clip(x, 0, img_shape[0] - 1)
        y = np.clip(y, 0, img_shape[1] - 1)

        return np.stack([x, y], axis=1)

    def _convert_to_poses(
        self, sample_inds: np.ndarray, depth: np.ndarray, topdown_place_rot: bool
    ) -> list[np.ndarray]:
        """Convert pixel indices to 7D poses."""
        place_poses = []

        pos_z = depth[sample_inds[0, :], sample_inds[1, :]]
        pos_x = sample_inds[0, :] * self.pixel_size + self.workspace_limits[0][0]
        pos_y = sample_inds[1, :] * self.pixel_size + self.workspace_limits[1][0]

        for i in range(len(pos_x)):
            pose = np.zeros(7)
            pose[0] = pos_x[i]
            pose[1] = pos_y[i]
            pose[2] = pos_z[i]

            if topdown_place_rot:
                # Top-down orientation (gripper pointing down)
                pose[3:7] = [1, 0, 0, 0]  # w, x, y, z
            else:
                pose[3:7] = [0, 0, 0, 1]

            place_poses.append(pose)

        return place_poses

    def generate_simple_places(
        self,
        point_cloud: np.ndarray,
        num_places: int = 50,
        height_offset: float = 0.02,
    ) -> list[np.ndarray]:
        """Generate simple place poses from point cloud.

        This is a simpler version that samples random positions on the table surface.

        Args:
            point_cloud: Point cloud array (N, 3)
            num_places: Number of place candidates to generate
            height_offset: Height above sampled point

        Returns:
            place_poses: List of 7D place poses
        """
        place_poses = []

        # Filter to table surface (low z values)
        z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
        table_mask = point_cloud[:, 2] < (z_min + 0.1 * (z_max - z_min))
        table_points = point_cloud[table_mask]

        if len(table_points) < num_places:
            table_points = point_cloud

        # Sample random points
        indices = np.random.choice(len(table_points), min(num_places, len(table_points)), replace=False)

        for idx in indices:
            pos = table_points[idx].copy()
            pos[2] += height_offset  # Place slightly above

            # Top-down orientation
            r = R.from_euler("xyz", [np.pi, 0, np.random.uniform(0, 2 * np.pi)])
            quat = r.as_quat()

            pose = np.zeros(7)
            pose[:3] = pos
            pose[3:7] = quat
            place_poses.append(pose)

        return place_poses


def create_place_generator(**kwargs) -> PlaceGenerator:
    """Factory function to create place generator."""
    return PlaceGenerator(**kwargs)
