"""Feature Field for A2 Policy.

This module computes per-point CLIP features by projecting 3D points to camera views
and extracting features at the projected pixel locations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, InterpolationMode


def project_points_to_camera(
    points: np.ndarray,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    image_size: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world points to 2D camera coordinates.

    Args:
        points: (N, 3) world coordinates
        camera_intrinsic: (3, 3) camera intrinsic matrix
        camera_extrinsic: (3, 4) or (4, 4) world-to-camera transform [R|t]
        image_size: (H, W) image dimensions

    Returns:
        coords_2d: (N, 2) pixel coordinates
        valid_mask: (N,) boolean mask for points in front of camera and in image bounds
    """
    N = points.shape[0]

    # Convert to homogeneous coordinates
    points_homo = np.concatenate([points, np.ones((N, 1))], axis=1)  # (N, 4)

    # Ensure extrinsic is 3x4
    if camera_extrinsic.shape == (4, 4):
        camera_extrinsic = camera_extrinsic[:3, :]  # (3, 4)

    # Transform to camera coordinates: P_cam = [R|t] @ P_world
    points_cam = (camera_extrinsic @ points_homo.T).T  # (N, 3)

    # Points in front of camera have positive z
    in_front = points_cam[:, 2] > 0.01

    # Project to image plane
    points_2d_homo = (camera_intrinsic @ points_cam.T).T  # (N, 3)

    # Normalize by depth
    with np.errstate(divide='ignore', invalid='ignore'):
        coords_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # (N, 2)

    # Check bounds
    H, W = image_size
    in_bounds = (
        (coords_2d[:, 0] >= 0) & (coords_2d[:, 0] < W) &
        (coords_2d[:, 1] >= 0) & (coords_2d[:, 1] < H)
    )

    valid_mask = in_front & in_bounds

    # Replace invalid coordinates with zeros
    coords_2d[~valid_mask] = 0

    return coords_2d, valid_mask


def interpolate_features(
    feature_map: torch.Tensor,
    coords_2d: torch.Tensor,
    image_size: tuple,
) -> torch.Tensor:
    """Bilinearly interpolate features at 2D coordinates.

    Args:
        feature_map: (C, H, W) feature map
        coords_2d: (N, 2) pixel coordinates (x, y)
        image_size: (H, W) original image size (for normalization)

    Returns:
        features: (N, C) interpolated features
    """
    C, fH, fW = feature_map.shape
    H, W = image_size
    N = coords_2d.shape[0]

    # Scale coordinates to feature map size
    scale_x = fW / W
    scale_y = fH / H

    # Normalize coordinates to [-1, 1] for grid_sample
    coords_normalized = coords_2d.clone()
    coords_normalized[:, 0] = coords_normalized[:, 0] * scale_x
    coords_normalized[:, 1] = coords_normalized[:, 1] * scale_y
    coords_normalized[:, 0] = 2.0 * coords_normalized[:, 0] / fW - 1.0
    coords_normalized[:, 1] = 2.0 * coords_normalized[:, 1] / fH - 1.0

    # Reshape for grid_sample: (1, 1, N, 2)
    grid = coords_normalized.view(1, 1, N, 2)

    # grid_sample expects (N, C, H, W) input and (N, H, W, 2) grid
    # Reshape feature map: (1, C, fH, fW) -> (1, C, 1, N) via reshape
    feature_map = feature_map.unsqueeze(0)  # (1, C, fH, fW)

    # Sample features
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # (1, C, 1, N)

    features = sampled.squeeze(0).squeeze(1).T  # (N, C)

    return features


class FeatureField:
    """Compute per-point CLIP features from multi-view images.

    This class projects 3D points to camera views and extracts CLIP features
    at the projected pixel locations.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._clip_model = None

    def _load_clip(self):
        """Load CLIP model if not already loaded."""
        if self._clip_model is None:
            from . import clip
            # Use ViT-L/14@336px to match the original A2 (768-dim output)
            # This gives higher quality per-patch features
            self._clip_model, _ = clip.load("ViT-L/14@336px", device=self.device)
            self._clip_model.eval()

    def extract_dense_features(self, image: torch.Tensor, use_multiscale: bool = True) -> torch.Tensor:
        """Extract dense CLIP features from an image.

        Uses get_patch_encodings to get per-patch features with projection applied.
        Supports multi-scale extraction by splitting image into grids for higher resolution.

        Args:
            image: (C, H, W) normalized image tensor (values in [0, 1])
            use_multiscale: If True, split image into grid for higher resolution features

        Returns:
            features: (D, fH, fW) dense feature map
        """
        self._load_clip()

        # Get input resolution for this model
        input_res = self._clip_model.visual.input_resolution  # 336 for ViT-L/14@336px
        feat_dim = 768
        patch_size = self._clip_model.visual.patch_size  # 14

        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

        C, H, W = image.shape

        if use_multiscale and H >= 240 and W >= 320:
            # Multi-scale extraction: split image into grid cells
            # This matches the original A2 approach for higher resolution
            row_size = 240  # Height of each grid cell
            column_size = 320  # Width of each grid cell
            row_grid_num = max(1, H // row_size)
            column_grid_num = max(1, W // column_size)

            # Patches per grid cell after CLIP processing
            patch_h = input_res // patch_size  # 24 patches
            # Adjust patch_w for aspect ratio
            patch_w = int(column_size / row_size * input_res // patch_size)

            grids = []
            for i in range(row_grid_num):
                for j in range(column_grid_num):
                    # Extract grid cell
                    y_start = i * row_size
                    y_end = min((i + 1) * row_size, H)
                    x_start = j * column_size
                    x_end = min((j + 1) * column_size, W)

                    grid = image[:, y_start:y_end, x_start:x_end]

                    # Resize to CLIP input size using BICUBIC (matching PIL BICUBIC from original A2)
                    grid = resize(
                        grid,
                        [input_res, input_res],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True
                    ).unsqueeze(0)

                    # Normalize for CLIP
                    grid = (grid - mean) / std
                    grids.append(grid)

            # Process all grids in batch
            grids_batch = torch.cat(grids, dim=0)  # (n_grids, 3, input_res, input_res)

            with torch.no_grad():
                grid_features = self._clip_model.get_patch_encodings(grids_batch)  # (n_grids, N_patches, D)

            # Reshape each grid's features
            grid_features = grid_features.reshape(len(grids), patch_h, patch_h, feat_dim)

            # Stitch grids back together
            all_row_features = []
            idx = 0
            for i in range(row_grid_num):
                row_features = []
                for j in range(column_grid_num):
                    row_features.append(grid_features[idx])
                    idx += 1
                # Concatenate along width (dim=1)
                row_feature = torch.cat(row_features, dim=1)  # (patch_h, patch_w*n_cols, D)
                all_row_features.append(row_feature)

            # Concatenate along height (dim=0)
            features = torch.cat(all_row_features, dim=0)  # (patch_h*n_rows, patch_w*n_cols, D)
            features = features.permute(2, 0, 1)  # (D, fH, fW)

        else:
            # Simple single-scale extraction using BICUBIC (matching PIL BICUBIC from original A2)
            img = resize(
                image,
                [input_res, input_res],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True
            ).unsqueeze(0)

            # Normalize for CLIP
            img = (img - mean) / std

            # Get patch encodings
            with torch.no_grad():
                patch_features = self._clip_model.get_patch_encodings(img)  # (1, N_patches, D)

            # Compute spatial size
            spatial_size = input_res // patch_size

            # Reshape to spatial grid
            features = patch_features.view(1, spatial_size, spatial_size, -1)  # (1, H, W, D)
            features = features.permute(0, 3, 1, 2).squeeze(0)  # (D, H, W)

        # Normalize features
        features = features / (features.norm(dim=0, keepdim=True) + 1e-6)

        return features.float()

    def compute_text_features(self, text: str) -> torch.Tensor:
        """Compute normalized CLIP text features.

        Args:
            text: Language goal string

        Returns:
            text_features: (D,) normalized text embedding
        """
        self._load_clip()

        from . import clip

        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self._clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.squeeze(0).float()

    def compute_per_point_features(
        self,
        points: np.ndarray,
        images: dict[str, np.ndarray],
        camera_configs: list[dict],
        lang_goal: str,
        depth_images: dict[str, np.ndarray] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-point CLIP features from multi-view images.

        Uses depth-weighted fusion when depth images are available, similar to original A2.

        Args:
            points: (N, 3) point cloud in world coordinates
            images: Dict of camera_name -> (H, W, 3) RGB images (uint8)
            camera_configs: List of camera config dicts with intrinsics, position, rotation
            lang_goal: Language goal string
            depth_images: Optional dict of camera_name -> (H, W) depth images

        Returns:
            pts_feat: (1, N, D) per-point CLIP features
            pts_sim: (1, N, 1) per-point visual-language similarity
        """
        import pybullet as pb

        N = len(points)
        # ViT-L/14@336px outputs 768-dim features (matches the network)
        D = 768  # CLIP ViT-L/14@336px output dimension

        # Compute text features
        text_features = self.compute_text_features(lang_goal)  # (D,)

        # Accumulate weighted features across views
        all_features = torch.zeros(N, D, device=self.device)
        weight_sum = torch.zeros(N, 1, device=self.device)

        # Distance threshold for weighting (mu parameter from original A2)
        mu = 0.02  # 2cm threshold

        cam_names = list(images.keys())

        for i, (cam_name, config) in enumerate(zip(cam_names, camera_configs)):
            if cam_name not in images:
                continue

            img = images[cam_name]
            if img is None:
                continue

            H, W = img.shape[:2]

            # Compute camera extrinsic (world-to-camera)
            position = np.array(config["position"]).reshape(3, 1)
            rotation = np.array(pb.getMatrixFromQuaternion(config["rotation"])).reshape(3, 3)

            # World-to-camera transform
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3:] = position
            camera_to_world = transform
            world_to_camera = np.linalg.inv(camera_to_world)

            K = config["intrinsics"]

            # Project points and get depths
            coords_2d, valid_mask, pts_depth = self._project_points_with_depth(
                points, K, world_to_camera[:3, :], (H, W)
            )

            if not np.any(valid_mask):
                continue

            # Extract dense features from image
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # (3, H, W)
            img_tensor = img_tensor.to(self.device)

            dense_features = self.extract_dense_features(img_tensor)  # (D, fH, fW)

            # Interpolate features at projected points
            coords_tensor = torch.from_numpy(coords_2d).float().to(self.device)
            point_features = interpolate_features(dense_features, coords_tensor, (H, W))  # (N, D)

            # Compute distance-based weights if depth is available
            if depth_images is not None and cam_name in depth_images and depth_images[cam_name] is not None:
                depth_img = depth_images[cam_name]

                # Interpolate surface depth at projected points
                depth_tensor = torch.from_numpy(depth_img).float().unsqueeze(0).to(self.device)  # (1, H, W)
                coords_for_depth = coords_tensor.clone()
                # Normalize coords to [-1, 1]
                coords_for_depth[:, 0] = 2.0 * coords_for_depth[:, 0] / (W - 1) - 1.0
                coords_for_depth[:, 1] = 2.0 * coords_for_depth[:, 1] / (H - 1) - 1.0
                grid = coords_for_depth.view(1, 1, N, 2)

                surface_depth = F.grid_sample(
                    depth_tensor.unsqueeze(0),
                    grid,
                    mode='nearest',
                    padding_mode='zeros',
                    align_corners=True
                ).squeeze()  # (N,)

                pts_depth_tensor = torch.from_numpy(pts_depth).float().to(self.device)

                # Distance to surface
                dist = surface_depth - pts_depth_tensor  # (N,)

                # Valid points: have valid depth and distance within threshold
                depth_valid = (surface_depth > 0.001) & (dist > -mu)
                valid_tensor = torch.from_numpy(valid_mask).to(self.device) & depth_valid

                # Distance-based weight (exponential decay from original A2)
                dist_weight = torch.exp(torch.clamp(mu - torch.abs(dist), max=0) / mu)  # (N,)
                dist_weight = dist_weight * valid_tensor.float()

            else:
                # No depth: use uniform weights for valid points
                valid_tensor = torch.from_numpy(valid_mask).to(self.device)
                dist_weight = valid_tensor.float()

            # Accumulate weighted features
            all_features += point_features * dist_weight.unsqueeze(1)
            weight_sum += dist_weight.unsqueeze(1)

        # Weighted average across views
        weight_sum = torch.clamp(weight_sum, min=1e-6)  # Avoid division by zero
        pts_feat = all_features / weight_sum  # (N, D)

        # Normalize features
        pts_feat = pts_feat / (pts_feat.norm(dim=1, keepdim=True) + 1e-6)

        # Compute similarity to text
        pts_sim = (pts_feat @ text_features.unsqueeze(1)).squeeze(1)  # (N,)

        # Reshape for batch dimension
        pts_feat = pts_feat.unsqueeze(0)  # (1, N, D)
        pts_sim = pts_sim.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)

        return pts_feat, pts_sim

    def _project_points_with_depth(
        self,
        points: np.ndarray,
        camera_intrinsic: np.ndarray,
        camera_extrinsic: np.ndarray,
        image_size: tuple,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D world points to 2D camera coordinates with depth.

        Args:
            points: (N, 3) world coordinates
            camera_intrinsic: (3, 3) camera intrinsic matrix
            camera_extrinsic: (3, 4) world-to-camera transform [R|t]
            image_size: (H, W) image dimensions

        Returns:
            coords_2d: (N, 2) pixel coordinates
            valid_mask: (N,) boolean mask for points in front of camera and in image bounds
            pts_depth: (N,) depth of each point in camera frame
        """
        N = points.shape[0]

        # Convert to homogeneous coordinates
        points_homo = np.concatenate([points, np.ones((N, 1))], axis=1)  # (N, 4)

        # Ensure extrinsic is 3x4
        if camera_extrinsic.shape == (4, 4):
            camera_extrinsic = camera_extrinsic[:3, :]  # (3, 4)

        # Transform to camera coordinates: P_cam = [R|t] @ P_world
        points_cam = (camera_extrinsic @ points_homo.T).T  # (N, 3)

        # Get point depths
        pts_depth = points_cam[:, 2]

        # Points in front of camera have positive z
        in_front = pts_depth > 0.01

        # Project to image plane
        points_2d_homo = (camera_intrinsic @ points_cam.T).T  # (N, 3)

        # Normalize by depth
        with np.errstate(divide='ignore', invalid='ignore'):
            coords_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # (N, 2)

        # Check bounds
        H, W = image_size
        in_bounds = (
            (coords_2d[:, 0] >= 0) & (coords_2d[:, 0] < W) &
            (coords_2d[:, 1] >= 0) & (coords_2d[:, 1] < H)
        )

        valid_mask = in_front & in_bounds

        # Replace invalid coordinates with zeros
        coords_2d[~valid_mask] = 0

        return coords_2d, valid_mask, pts_depth


def create_feature_field(device: str = "cuda") -> FeatureField:
    """Create a FeatureField instance."""
    return FeatureField(device=device)
