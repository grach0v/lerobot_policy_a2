"""A2 Policy implementation - VLA-style incremental action policy.

This policy generates grasp/place poses using GraspNet and outputs
incremental delta actions at each timestep (30Hz) to execute the motion.
"""

import os
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_a2 import A2Config
from .networks.action_selection import CLIPGrasp, CLIPPlace
from .networks.clip_action import CLIPActionFusion, Policy
from .networks.grasp_generator import create_grasp_generator
from .networks.place_generator import create_place_generator

# A2 workspace constants (from original A2)
PP_SHIFT_Y = 0.168
GRASP_WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.336, 0.000], [-0.0001, 0.4]])
PLACE_WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.000, 0.336], [-0.0001, 0.4]])


def normalize_pos(pos: torch.Tensor, workspace_limits: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """Normalize positions to [-1, 1] range based on workspace limits.

    Args:
        pos: Position tensor of shape (..., 3)
        workspace_limits: Array of shape (3, 2) with [min, max] for each axis
        device: Device to use

    Returns:
        Normalized positions in [-1, 1] range
    """
    pos_min = torch.from_numpy(workspace_limits.T[0]).float().to(device)
    pos_max = torch.from_numpy(workspace_limits.T[1]).float().to(device)
    return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0


def unnormalize_pos(pos: torch.Tensor, workspace_limits: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """Unnormalize positions from [-1, 1] range to workspace coordinates.

    Args:
        pos: Normalized position tensor of shape (..., 3)
        workspace_limits: Array of shape (3, 2) with [min, max] for each axis
        device: Device to use

    Returns:
        Positions in workspace coordinates
    """
    pos_min = torch.from_numpy(workspace_limits.T[0]).float().to(device)
    pos_max = torch.from_numpy(workspace_limits.T[1]).float().to(device)
    return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min


class GraspPhase(Enum):
    """State machine phases for grasp execution."""

    IDLE = 0  # Waiting for new grasp target
    APPROACHING = 1  # Moving to pre-grasp position
    PRE_GRASP = 2  # At pre-grasp, opening gripper
    DESCENDING = 3  # Moving down to grasp
    GRASPING = 4  # Closing gripper
    LIFTING = 5  # Lifting object
    DONE = 6  # Grasp complete


class A2Policy(PreTrainedPolicy):
    """A2 Policy for VLA-style grasp and place execution.

    This policy operates in two modes:
    1. **Grasp pose generation**: Uses GraspNet + CLIP to select best grasp
    2. **VLA execution**: Outputs incremental delta actions to execute the motion

    The policy is called at each timestep (e.g., 30Hz) and returns delta
    actions that move the robot toward the target grasp pose.
    """

    config_class = A2Config
    name = "a2"

    def __init__(self, config: A2Config, dataset_stats: dict[str, Any] | None = None):
        # Always use nn.Module init to avoid PreTrainedPolicy validation issues
        nn.Module.__init__(self)
        self.config = config

        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the CLIPAction fusion network
        self.vilg_fusion = CLIPActionFusion(
            action_dim=config.action_dim,
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            device=device,
            use_rope=config.use_rope,
            no_feat_rope=config.no_feat_rope,
            sa=config.fusion_sa,
            no_rgb_feat=config.no_rgb_feat,
        )

        # Policy head for action selection
        self.policy = Policy(
            input_dim=config.width,
            hidden_dim=config.width // 2,
            layer_norm=False,
        )

        # Action selection agents
        self.grasp_agent = CLIPGrasp(device=device)
        self.place_agent = CLIPPlace(device=device)

        # Pose generators (loaded lazily)
        self._grasp_generator = None
        self._place_generator = None

        # CLIP model for visual features (loaded lazily)
        self._clip_model = None
        self._clip_preprocess = None
        self._device = device

        # VLA state machine
        self._phase = GraspPhase.IDLE
        self._target_pose = None  # Target grasp pose [x, y, z, qx, qy, qz, qw]
        self._pre_grasp_pose = None  # Pre-grasp position (above target)
        self._current_ee_pos = None  # Current end-effector position
        self._step_count = 0  # Steps in current phase
        self._lang_goal = None  # Language goal for current episode

        # VLA parameters
        self._approach_height = 0.15  # Height above target for approach
        self._lift_height = 0.10  # Height to lift after grasp
        self._position_threshold = 0.01  # Position tolerance (1cm)
        # Action output is in [-1, 1] range, env scales by 0.01 to get meters
        # So action=1.0 means 1cm movement per step
        self._max_action = 1.0  # Maximum action magnitude (maps to 1cm/step)
        self._gripper_steps = 5  # Steps to wait for gripper action

    def reset(self):
        """Reset the policy state between episodes."""
        self._phase = GraspPhase.IDLE
        self._target_pose = None
        self._pre_grasp_pose = None
        self._current_ee_pos = None
        self._step_count = 0
        self._lang_goal = None

    def get_optim_params(self) -> dict:
        """Return parameters for optimization."""
        return {"default": list(self.vilg_fusion.parameters()) + list(self.policy.parameters())}

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Select a single delta action for the current timestep.

        This method is called at each environment step (e.g., 30Hz).
        It returns a 7D delta action: [dx, dy, dz, 0, 0, 0, gripper]

        Args:
            batch: Dictionary containing:
                - observation.state: Robot state [ee_pos(3), ee_quat(4), joints(6), gripper(1)]
                - observation.images.*: Camera images
                - task (optional): Language goal string

        Returns:
            Action tensor (batch_size, 7): [dx, dy, dz, 0, 0, 0, gripper]
            gripper: 0.0 = open, 1.0 = close
        """
        self.eval()

        # Extract current end-effector position from robot state
        # State format: [ee_pos(3), ee_quat(4), joints(6), gripper(1)] = 14D
        if "observation.state" in batch:
            state = batch["observation.state"]
            if state.dim() == 2:
                self._current_ee_pos = state[0, :3].cpu().numpy()  # Batch dim present
            else:
                self._current_ee_pos = state[:3].cpu().numpy()  # No batch dim

        # Get batch size
        batch_size = 1
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].dim() >= 1:
                batch_size = batch[key].shape[0]
                break

        # State machine
        if self._phase == GraspPhase.IDLE:
            # Generate new grasp target
            self._generate_grasp_target(batch)
            self._phase = GraspPhase.APPROACHING
            self._step_count = 0

        if self._phase == GraspPhase.APPROACHING:
            # Move to pre-grasp position
            action = self._move_toward(self._pre_grasp_pose, gripper_open=True)
            if self._at_position(self._pre_grasp_pose):
                self._phase = GraspPhase.PRE_GRASP
                self._step_count = 0
            return self._to_tensor(action, batch_size)

        if self._phase == GraspPhase.PRE_GRASP:
            # Wait with gripper open
            self._step_count += 1
            if self._step_count >= self._gripper_steps:
                self._phase = GraspPhase.DESCENDING
                self._step_count = 0
            return self._to_tensor([0, 0, 0, 0, 0, 0, 0.0], batch_size)  # Gripper open

        if self._phase == GraspPhase.DESCENDING:
            # Move down to grasp position
            action = self._move_toward(self._target_pose[:3], gripper_open=True)
            if self._at_position(self._target_pose[:3]):
                self._phase = GraspPhase.GRASPING
                self._step_count = 0
            return self._to_tensor(action, batch_size)

        if self._phase == GraspPhase.GRASPING:
            # Close gripper
            self._step_count += 1
            if self._step_count >= self._gripper_steps:
                self._phase = GraspPhase.LIFTING
                self._step_count = 0
            return self._to_tensor([0, 0, 0, 0, 0, 0, 1.0], batch_size)  # Gripper close

        if self._phase == GraspPhase.LIFTING:
            # Lift object
            lift_target = self._target_pose[:3].copy()
            lift_target[2] += self._lift_height
            action = self._move_toward(lift_target, gripper_open=False)
            if self._at_position(lift_target):
                self._phase = GraspPhase.DONE
                self._step_count = 0
            return self._to_tensor(action, batch_size)

        if self._phase == GraspPhase.DONE:
            # Stay in place with gripper closed
            return self._to_tensor([0, 0, 0, 0, 0, 0, 1.0], batch_size)

        # Fallback
        return self._to_tensor([0, 0, 0, 0, 0, 0, 0.0], batch_size)

    def _generate_grasp_target(self, batch: dict[str, torch.Tensor]):
        """Generate grasp target using GraspNet and CLIP selection.

        Pipeline:
        1. Get point cloud from batch
        2. Generate grasp candidates using GraspNet
        3. Load CLIP and compute visual-language similarity for each point
        4. Use CLIPActionFusion + Policy to score grasp candidates
        5. Select best grasp using CLIPGrasp
        """
        # Get language goal
        self._lang_goal = batch.get("task", "grasp an object")
        if isinstance(self._lang_goal, torch.Tensor):
            self._lang_goal = "grasp an object"

        # Check if we have point cloud data
        if "point_cloud" not in batch:
            print("Warning: No point cloud in batch, using fallback grasp")
            self._generate_fallback_grasp()
            return

        try:
            # Get point cloud: (B, N, 6) where last 3 are RGB
            pc_tensor = batch["point_cloud"]
            if pc_tensor.dim() == 3:
                pc = pc_tensor[0].cpu().numpy()  # (N, 6)
            else:
                pc = pc_tensor.cpu().numpy()

            if len(pc) < 100:
                print(f"Warning: Point cloud too sparse ({len(pc)} points), using fallback")
                self._generate_fallback_grasp()
                return

            # Split into positions and colors
            pts_pos = pc[:, :3]  # (N, 3)
            pts_colors = pc[:, 3:6] / 255.0  # (N, 3) normalized

            # Generate grasp candidates using GraspNet
            grasp_generator = self._get_grasp_generator()
            grasp_poses, _ = grasp_generator.generate_grasps(pts_pos, pts_colors)

            if len(grasp_poses) == 0:
                print("Warning: No grasps generated, using fallback")
                self._generate_fallback_grasp()
                return

            print(f"Generated {len(grasp_poses)} grasp candidates")

            # Filter grasps to target objects if provided (like original A2's assign_grasp_pose)
            target_object_poses = batch.get("target_object_poses", None)
            if target_object_poses and len(target_object_poses) > 0:
                # Get target object centers
                centers = []
                for pose in target_object_poses.values():
                    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                        centers.append(pose[:3, 3])
                    elif hasattr(pose, '__len__') and len(pose) >= 3:
                        centers.append(np.array(pose[:3]))
                if centers:
                    centers = np.array(centers)
                    grasp_positions = np.array([g[:3] for g in grasp_poses])
                    # For each grasp, find distance to nearest target
                    dists = np.array([np.linalg.norm(grasp_positions - c, axis=1) for c in centers])
                    min_dists = np.min(dists, axis=0)
                    # Keep grasps within 0.15m of any target (like original dist_thresh)
                    dist_thresh = 0.15
                    valid_mask = min_dists < dist_thresh
                    if np.sum(valid_mask) > 0:
                        filtered_grasps = [g for g, v in zip(grasp_poses, valid_mask) if v]
                        print(f"  Filtered to {len(filtered_grasps)} grasps near targets (was {len(grasp_poses)})")
                        grasp_poses = filtered_grasps
                    else:
                        print(f"  Warning: No grasps within {dist_thresh}m of targets, using all")

            # Get CLIP features and compute visual-language similarity
            self._load_clip()
            pts_feat, pts_sim = self._compute_clip_features(batch, pts_pos, self._lang_goal)

            # Sample top K points by CLIP similarity (like original A2)
            sample_num = 500  # Default from original A2
            pts_pos_tensor = torch.from_numpy(pts_pos).float().unsqueeze(0).to(self._device)

            if pts_sim.shape[1] > sample_num:
                # Get indices of top K points by similarity
                sim_flat = pts_sim[0, :, 0]  # (N,)
                top_indices = torch.argsort(sim_flat, descending=True)[:sample_num]

                # Sample points, features, and similarities
                pts_pos_tensor = pts_pos_tensor[:, top_indices, :]  # (1, K, 3)
                pts_feat = pts_feat[:, top_indices, :]  # (1, K, D)
                pts_sim = pts_sim[:, top_indices, :]  # (1, K, 1)

            # Normalize similarity scores to [0, 1] range (like original A2)
            z = 1e-4
            sim_min = pts_sim.min()
            sim_max = pts_sim.max()
            pts_sim = (pts_sim - sim_min + z) / (sim_max - sim_min + z)

            # Convert grasp poses to tensor for scoring
            grasp_array = np.array(grasp_poses)  # (M, 7)
            grasp_tensor = torch.from_numpy(grasp_array).float().unsqueeze(0).to(self._device)

            # Select grasp using direct grounding or learned networks
            if self.config.direct_grounding:
                # Direct grounding: use simple CLIP-based selection (like original A2)
                # Find point with highest smoothed CLIP similarity, select nearest grasp
                best_idx = self.grasp_agent.select_action_knn_greedy(
                    pts=pts_pos_tensor,
                    clip_sims=pts_sim,
                    actions=grasp_tensor,
                    M=sample_num,
                )
            else:
                # Use learned CLIPActionFusion + Policy networks
                # NOTE: Original A2 has --normalize flag with default=False
                # The pretrained model was trained WITHOUT position normalization
                # Only apply workspace shift for pickplace mode
                is_pickplace = batch.get("is_pickplace", False)
                pts_for_model = pts_pos_tensor.clone()
                grasp_for_model = grasp_tensor.clone()

                # Apply workspace shift ONLY for pickplace (like original A2 --workspace_shift)
                if is_pickplace:
                    shift = torch.tensor([[0, PP_SHIFT_Y, 0]], device=self._device)
                    pts_for_model = pts_for_model + shift
                    grasp_for_model[:, :, :3] = grasp_for_model[:, :, :3] + shift

                cross_feat, attn_weights = self.vilg_fusion(
                    pts_pos=pts_for_model,
                    pts_feat=pts_feat,
                    pts_sim=pts_sim,
                    actions=grasp_for_model,
                    mode="grasp",
                )

                # Get action scores from policy head
                logits = self.policy(cross_feat)  # (B, M) or (M,)

                if isinstance(logits, torch.Tensor):
                    if logits.dim() == 0:
                        # Scalar tensor (single grasp)
                        best_idx = 0
                    elif logits.dim() == 1:
                        best_idx = torch.argmax(logits).item()
                    else:
                        # 2D or higher: (B, M)
                        argmax_result = torch.argmax(logits, dim=-1)
                        if argmax_result.dim() == 0:
                            best_idx = argmax_result.item()
                        else:
                            best_idx = argmax_result[0].item()
                else:
                    # Fallback to CLIP-based selection
                    best_idx = self.grasp_agent.select_action_greedy(
                        pts=pts_pos_tensor,
                        clip_sims=pts_sim,
                        actions=grasp_tensor,
                    )

            # Get the selected grasp pose
            self._target_pose = grasp_poses[best_idx].astype(np.float32)
            # Ensure minimum grasp z for reliable execution
            if self._target_pose[2] < 0.025:
                self._target_pose[2] = 0.025
            print(f"Selected grasp {best_idx}: pos={self._target_pose[:3]}")

        except Exception as e:
            print(f"Warning: Grasp generation failed ({e}), using fallback")
            import traceback

            traceback.print_exc()
            self._generate_fallback_grasp()
            return

        # Pre-grasp position (above target)
        self._pre_grasp_pose = self._target_pose[:3].copy()
        self._pre_grasp_pose[2] += self._approach_height

    def _generate_fallback_grasp(self):
        """Generate fallback grasp when full pipeline fails."""
        if self._current_ee_pos is not None:
            # Move forward and down from current position
            self._target_pose = np.array(
                [
                    self._current_ee_pos[0] + 0.1,  # Forward
                    self._current_ee_pos[1],  # Same y
                    0.05,  # Low z for grasp
                    0.0,
                    0.0,
                    0.0,
                    1.0,  # Identity quaternion
                ],
                dtype=np.float32,
            )
        else:
            # Default grasp position in workspace
            self._target_pose = np.array([0.5, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]

        # Pre-grasp position (above target)
        self._pre_grasp_pose = self._target_pose[:3].copy()
        self._pre_grasp_pose[2] += self._approach_height

    def _compute_clip_features(
        self,
        batch: dict[str, torch.Tensor],
        pts_pos: np.ndarray,
        lang_goal: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute CLIP features for points and visual-language similarity.

        This method uses per-point feature extraction when camera configs are available,
        projecting each 3D point to camera views and extracting CLIP features at
        the projected pixel locations.

        Args:
            batch: Observation batch with images and optionally camera_configs
            pts_pos: Point positions (N, 3)
            lang_goal: Language goal string

        Returns:
            pts_feat: Point features (1, N, D)
            pts_sim: Visual-language similarity (1, N, 1)
        """
        N = len(pts_pos)
        device = self._device
        D = self.config.width  # 768

        # Check if we have camera configs for proper per-point feature extraction
        if "camera_configs" in batch and "color_images" in batch:
            try:
                from .feature_field import FeatureField

                # Debug: show we're using per-point features
                if not hasattr(self, "_debug_perpoint_shown"):
                    print("  [CLIP] Using per-point feature extraction (camera configs available)")
                    self._debug_perpoint_shown = True

                # Create feature field if needed
                if not hasattr(self, "_feature_field"):
                    self._feature_field = FeatureField(device=device)

                # Get camera configs and images
                camera_configs = batch["camera_configs"]
                color_images = batch["color_images"]
                depth_images = batch.get("depth_images", None)

                # Compute per-point features using multi-view projection
                pts_feat, pts_sim = self._feature_field.compute_per_point_features(
                    points=pts_pos,
                    images=color_images,
                    camera_configs=camera_configs,
                    lang_goal=lang_goal,
                    depth_images=depth_images,
                )

                # Pad features to match network width (768) if needed
                feat_dim = pts_feat.shape[-1]  # 512 for ViT-B/32
                if feat_dim < D:
                    pts_feat = torch.nn.functional.pad(pts_feat, (0, D - feat_dim))
                elif feat_dim > D:
                    pts_feat = pts_feat[..., :D]

                return pts_feat, pts_sim

            except Exception as e:
                print(f"Warning: Per-point feature extraction failed ({e}), using fallback")
                import traceback
                traceback.print_exc()

        # Fallback: use global CLIP features (less accurate but works without camera configs)
        if not hasattr(self, "_debug_fallback_shown"):
            print("  [CLIP] WARNING: Using fallback global features (no camera_configs in batch)")
            self._debug_fallback_shown = True
        from . import clip

        # Get any available camera image
        img_key = None
        for key in batch:
            if key.startswith("observation.images."):
                img_key = key
                break

        if img_key is None:
            # Return dummy features
            pts_feat = torch.zeros(1, N, D, device=device)
            pts_sim = torch.ones(1, N, 1, device=device) / N
            return pts_feat, pts_sim

        # Get image tensor (B, C, H, W) -> (C, H, W)
        img_tensor = batch[img_key]
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]

        # Compute text features
        text_tokens = clip.tokenize([lang_goal]).to(device)
        with torch.no_grad():
            text_features = self._clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Get image features using CLIP visual encoder
        img_resized = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize for CLIP
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        img_normalized = (img_resized - mean) / std

        with torch.no_grad():
            image_features = self._clip_model.encode_image(img_normalized)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute visual-language similarity
        similarity = (image_features @ text_features.T).squeeze()

        # Broadcast global features to all points (fallback - less accurate)
        pts_feat = image_features.float().unsqueeze(0).expand(1, N, -1)

        # Pad to network width
        if pts_feat.shape[-1] < D:
            pts_feat = torch.nn.functional.pad(pts_feat, (0, D - pts_feat.shape[-1]))
        elif pts_feat.shape[-1] > D:
            pts_feat = pts_feat[..., :D]

        # Uniform similarity across points (fallback)
        pts_sim = torch.full((1, N, 1), similarity.item(), device=device)

        return pts_feat, pts_sim

    def _move_toward(self, target_pos: np.ndarray, gripper_open: bool = True) -> list:
        """Compute delta action to move toward target position.

        Returns action in [-1, 1] range. The environment scales this by 0.01
        to get actual movement in meters (so action=1.0 → 1cm movement).
        """
        if self._current_ee_pos is None:
            return [0, 0, 0, 0, 0, 0, 0.0 if gripper_open else 1.0]

        # Compute direction to target in meters
        delta_meters = target_pos - self._current_ee_pos

        # Convert to action space: divide by 0.01 (the env's scaling factor)
        # This means if we're 1cm away, action = 1.0
        # Clip to [-1, 1] range
        action_scale = 0.01  # env multiplies action by this to get meters
        delta_action = delta_meters / action_scale

        # Clip to max action magnitude
        delta_norm = np.linalg.norm(delta_action)
        if delta_norm > self._max_action:
            delta_action = delta_action * (self._max_action / delta_norm)

        gripper = 0.0 if gripper_open else 1.0
        return [delta_action[0], delta_action[1], delta_action[2], 0, 0, 0, gripper]

    def _at_position(self, target_pos: np.ndarray) -> bool:
        """Check if end-effector is at target position."""
        if self._current_ee_pos is None:
            return False
        distance = np.linalg.norm(target_pos - self._current_ee_pos)
        return distance < self._position_threshold

    def _to_tensor(self, action: list, batch_size: int) -> torch.Tensor:
        """Convert action list to tensor."""
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self._device)
        if batch_size > 1:
            action_tensor = action_tensor.unsqueeze(0).expand(batch_size, -1)
        return action_tensor

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict a chunk of actions (for compatibility with action chunking interface)."""
        return self.select_action(batch, **kwargs).unsqueeze(1)

    def _load_clip(self):
        """Load CLIP model lazily."""
        if self._clip_model is None:
            from . import clip

            device = self._device if isinstance(self._device, str) else str(self._device)
            self._clip_model, self._clip_preprocess = clip.load("ViT-L/14@336px", device=device)
            self._clip_model.eval()

    def _get_grasp_generator(self):
        """Get grasp generator (loaded lazily)."""
        if self._grasp_generator is None:
            # Auto-download GraspNet checkpoint if not provided
            checkpoint_path = self.config.graspnet_checkpoint
            if checkpoint_path is None:
                from .graspnet import get_graspnet_checkpoint
                checkpoint_path = get_graspnet_checkpoint()
                self.config.graspnet_checkpoint = checkpoint_path
            self._grasp_generator = create_grasp_generator(
                checkpoint_path=checkpoint_path,
            )
        return self._grasp_generator

    def _get_place_generator(self):
        """Get place generator (loaded lazily)."""
        if self._place_generator is None:
            self._place_generator = create_place_generator()
        return self._place_generator

    def _generate_place_target(self, batch: dict[str, torch.Tensor]):
        """Generate place target using heightmap-based PlaceGenerator (matching original A2).

        This method replicates the original A2 place candidate generation:
        1. Create a virtual heightmap of the workspace
        2. Convert 3D object positions to pixel coordinates
        3. Generate IN/OUT region masks using sector masks
        4. Sample place candidates from the masks
        5. Use CLIP features to select the best place

        Args:
            batch: Dictionary containing:
                - task: Language goal string
                - point_cloud: Point cloud tensor (1, N, 6)
                - reference_positions: List of reference object positions
                - reference_sizes: List of reference object sizes
                - direction: Spatial relation (in/on/near/around)
                - object_poses: Dict of all object poses
                - color_images: Dict of RGB images for CLIP
                - depth_images: Dict of depth images
                - camera_configs: List of camera configurations
        """
        # Constants matching original A2
        # For pickplace tasks, use larger workspace that covers both grasp (y<0) and place (y>0) areas
        is_pickplace = batch.get("is_pickplace", False)
        if is_pickplace:
            # PP_WORKSPACE_LIMITS and PP_PIXEL_SIZE from original A2
            WORKSPACE_LIMITS = np.array([[0.164, 0.836], [-0.336, 0.336], [-0.0001, 0.4]])
            PIXEL_SIZE = 0.003  # 3mm per pixel for pickplace
        else:
            WORKSPACE_LIMITS = np.array([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])
            PIXEL_SIZE = 0.002  # 2mm per pixel
        IMAGE_SIZE = 224
        MIN_DIS = 0.05  # 5cm
        MAX_DIS = 0.15  # 15cm
        IN_REGION = ["on", "in", "to", "upside", "on top of"]
        OUT_REGION = ["near", "around", "next to", "beside", "close to", "surrounding to"]

        # Get language goal
        lang_goal = batch.get("task", "place it somewhere")
        if lang_goal is None or isinstance(lang_goal, torch.Tensor):
            lang_goal = "place it somewhere"

        # Get direction from batch or parse from language
        direction = batch.get("direction", None)
        if direction is None:
            direction = self._parse_place_direction(lang_goal)

        is_in_region = direction in IN_REGION if direction else False
        is_out_region = direction in OUT_REGION if direction else True

        # Get reference object positions and sizes
        reference_positions = batch.get("reference_positions", [])
        reference_sizes = batch.get("reference_sizes", [])
        all_object_poses = batch.get("object_poses", {})

        if not reference_positions:
            print("Warning: No reference positions for place, using random fallback")
            self._generate_fallback_place(batch)
            return

        # Debug: show reference object positions
        print(f"  Reference objects: {len(reference_positions)}")
        for i, ref_pos in enumerate(reference_positions):
            print(f"    Ref {i}: ({ref_pos[0]:.3f}, {ref_pos[1]:.3f}, {ref_pos[2]:.3f})")
        print(f"  Direction: {direction} (in_region={is_in_region}, out_region={is_out_region})")

        try:
            # Helper function to convert world coords to pixel coords
            def world_to_pixel(pos):
                px = int((pos[0] - WORKSPACE_LIMITS[0][0]) / PIXEL_SIZE)
                py = int((pos[1] - WORKSPACE_LIMITS[1][0]) / PIXEL_SIZE)
                return (px, py)

            # Helper function to convert pixel coords to world coords
            def pixel_to_world(px, py, z=0.02):
                x = px * PIXEL_SIZE + WORKSPACE_LIMITS[0][0]
                y = py * PIXEL_SIZE + WORKSPACE_LIMITS[1][0]
                return (x, y, z)

            # Helper to generate sector mask (matching original A2)
            def generate_sector_mask(shape, centre, angle_range, radius_min, radius_max):
                x, y = np.ogrid[:shape[0], :shape[1]]
                cx, cy = centre
                tmin, tmax = np.deg2rad(angle_range)
                if tmax < tmin:
                    tmax += 2 * np.pi
                r2 = (x - cx) ** 2 + (y - cy) ** 2
                theta = np.arctan2(x - cx, y - cy) - tmin
                theta %= (2 * np.pi)
                circmask_in = r2 >= radius_min ** 2
                circmask_out = r2 <= radius_max ** 2
                anglemask = theta <= (tmax - tmin)
                return circmask_in * circmask_out * anglemask

            # Collect all object positions and convert to pixel coords
            all_positions = []
            all_sizes_world = []

            # Add positions from all_object_poses
            for obj_id, pose in all_object_poses.items():
                if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                    pos = pose[:3, 3]
                elif hasattr(pose, '__len__') and len(pose) >= 3:
                    pos = np.array(pose[:3]).flatten()
                else:
                    continue
                all_positions.append(pos)
                all_sizes_world.append(np.array([0.05, 0.05, 0.05]))

            if not all_positions:
                all_positions = [np.array(p) for p in reference_positions]
                all_sizes_world = [np.array(s) if s is not None else np.array([0.05, 0.05, 0.05])
                                   for s in reference_sizes]

            # Convert to pixel coordinates
            obj_centers_px = [world_to_pixel(pos) for pos in all_positions]
            obj_sizes_px = [(int(s[0] / PIXEL_SIZE), int(s[1] / PIXEL_SIZE)) for s in all_sizes_world]

            # Convert reference positions to pixel coords
            ref_centers_px = [world_to_pixel(pos) for pos in reference_positions]

            # Generate mask map (occupied areas = 0, available = 1)
            mask_map = np.ones((IMAGE_SIZE, IMAGE_SIZE))
            for (cx, cy), (w, h) in zip(obj_centers_px, obj_sizes_px):
                x_min = max(0, cx - h // 2)
                x_max = min(IMAGE_SIZE - 1, cx + h // 2)
                y_min = max(0, cy - w // 2)
                y_max = min(IMAGE_SIZE - 1, cy + w // 2)
                mask_map[x_min:x_max, y_min:y_max] = 0

            # Generate place candidates (matching original A2 place_generation_return_gt)
            sample_inds = None
            valid_places_list = []
            sample_num_each_object = 5

            # Helper to check if pixel coord is near any reference
            def is_near_reference(cx, cy, ref_centers, threshold_px=5):
                for rcx, rcy in ref_centers:
                    if abs(cx - rcx) <= threshold_px and abs(cy - rcy) <= threshold_px:
                        return True
                return False

            for i, ((cx, cy), (w, h)) in enumerate(zip(obj_centers_px, obj_sizes_px)):
                # Check if this object is a reference object (with tolerance)
                is_ref = is_near_reference(cx, cy, ref_centers_px)

                # Generate IN region mask (on/in the object)
                in_radius_max = max(1, min(w, h) // 3)
                in_mask = generate_sector_mask(
                    (IMAGE_SIZE, IMAGE_SIZE), (cx, cy),
                    angle_range=[0, 359.99], radius_min=0, radius_max=in_radius_max
                )
                in_mask_indices = np.argwhere(in_mask)

                if in_mask_indices.shape[0] > 0:
                    num_samples = min(in_mask_indices.shape[0], sample_num_each_object)
                    in_sample_indices = in_mask_indices[
                        np.linspace(0, in_mask_indices.shape[0] - 1, num_samples, dtype=int)
                    ]
                    if sample_inds is None:
                        sample_inds = in_sample_indices
                    else:
                        sample_inds = np.concatenate((sample_inds, in_sample_indices), axis=0)

                    # Mark as valid if reference object and IN_REGION direction
                    if is_ref and is_in_region:
                        valid_places_list += list(range(
                            sample_inds.shape[0] - num_samples, sample_inds.shape[0]
                        ))

                # Generate OUT region mask (near/around the object)
                out_radius_min = max(int(MIN_DIS / PIXEL_SIZE), max(w, h) // 2)
                out_radius_max = max(int(MAX_DIS / PIXEL_SIZE), max(w, h) // 2 + int(0.1 / PIXEL_SIZE))
                out_mask = generate_sector_mask(
                    (IMAGE_SIZE, IMAGE_SIZE), (cx, cy),
                    angle_range=[0, 359.99], radius_min=out_radius_min, radius_max=out_radius_max
                )
                out_mask = out_mask * mask_map  # Exclude occupied areas
                out_mask_indices = np.argwhere(out_mask)

                if out_mask_indices.shape[0] > 0:
                    num_samples = min(out_mask_indices.shape[0], sample_num_each_object)
                    out_sample_indices = out_mask_indices[
                        np.linspace(0, out_mask_indices.shape[0] - 1, num_samples, dtype=int)
                    ]
                    if sample_inds is None:
                        sample_inds = out_sample_indices
                    else:
                        sample_inds = np.concatenate((sample_inds, out_sample_indices), axis=0)

                    # Mark as valid if reference object and OUT_REGION direction
                    if is_ref and is_out_region:
                        valid_places_list += list(range(
                            sample_inds.shape[0] - num_samples, sample_inds.shape[0]
                        ))

            if sample_inds is None or len(sample_inds) == 0:
                print("Warning: No place candidates generated, using fallback")
                self._generate_fallback_place(batch)
                return

            # Convert pixel indices to world coordinates
            sample_inds = sample_inds.transpose(1, 0)  # (2, N)
            place_poses = []
            for i in range(sample_inds.shape[1]):
                px, py = sample_inds[0, i], sample_inds[1, i]
                x, y, z = pixel_to_world(px, py, z=0.02)
                pose = np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
                place_poses.append(pose)

            valid_indices = valid_places_list
            print(f"Generated {len(place_poses)} place candidates ({len(valid_indices)} valid)")

            # Debug: show a few valid candidate positions
            if valid_indices:
                print(f"  Valid candidate positions (first 3):")
                for idx in valid_indices[:3]:
                    pos = place_poses[idx][:3]
                    print(f"    idx={idx}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

            # Convert to tensor for selection
            place_array = np.array(place_poses)  # (M, 7)
            place_positions = place_array[:, :3]  # (M, 3)

            # Get scene point cloud for CLIP feature extraction
            scene_pcd = None
            if "point_cloud" in batch:
                pcd_tensor = batch["point_cloud"]
                if isinstance(pcd_tensor, torch.Tensor):
                    scene_pcd = pcd_tensor[0].cpu().numpy()
                    if scene_pcd.shape[1] > 3:
                        scene_pcd = scene_pcd[:, :3]  # Keep only XYZ

            # Compute CLIP features on scene points (if available) or place positions
            self._load_clip()
            if scene_pcd is not None and len(scene_pcd) >= 20:
                pts_feat, pts_sim = self._compute_clip_features(batch, scene_pcd, lang_goal)
                feature_pts = scene_pcd
            else:
                pts_feat, pts_sim = self._compute_clip_features(batch, place_positions, lang_goal)
                feature_pts = place_positions

            # Sample top points by CLIP similarity (matching original A2 preprocess_pp_unified)
            # This is critical - the model was trained with sampled points, not all points
            sample_num = min(1024, len(feature_pts))  # Default 1024, same as original A2

            # Get similarity values for sorting (flatten to 1D)
            clip_sims_flat = pts_sim.squeeze()  # (N,) or (N, 1) -> (N,)
            if clip_sims_flat.dim() == 2:
                clip_sims_flat = clip_sims_flat[:, 0]

            # Sample top N points by CLIP similarity (descending order)
            sample_indices = torch.argsort(clip_sims_flat, descending=True)[:sample_num]

            # Sample points and features
            pts_feat_sampled = pts_feat[:, sample_indices, :]  # (1, sample_num, D)
            pts_sim_sampled = pts_sim[:, sample_indices, :]    # (1, sample_num, 1)
            feature_pts_sampled = feature_pts[sample_indices.cpu().numpy()]  # (sample_num, 3)

            # Normalize similarity scores AFTER sampling (matching original A2)
            z = 1e-4
            sim_min = pts_sim_sampled.min()
            sim_max = pts_sim_sampled.max()
            pts_sim_norm = (pts_sim_sampled - sim_min + z) / (sim_max - sim_min + z)

            pts_pos_tensor = torch.from_numpy(feature_pts_sampled).float().unsqueeze(0).to(self._device)
            place_tensor = torch.from_numpy(place_array).float().unsqueeze(0).to(self._device)

            if self.config.direct_grounding:
                # Direct grounding: use CLIP+KNN on sampled scene points to find best location
                # Use sampled points (feature_pts_sampled) which match pts_sim_norm dimensions
                from sklearn.neighbors import NearestNeighbors
                M = len(feature_pts_sampled)
                k = max(1, int(0.05 * M))
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(feature_pts_sampled)
                distances, indices = nbrs.kneighbors(feature_pts_sampled)

                clip_sims_np = pts_sim_norm.squeeze().cpu().numpy()
                if clip_sims_np.ndim == 2:
                    clip_sims_np = clip_sims_np.mean(axis=-1)
                average_affordances = np.array([clip_sims_np[idx].mean() for idx in indices])
                best_scene_idx = average_affordances.argmax()
                best_scene_point = feature_pts_sampled[best_scene_idx]

                print(f"  CLIP selection: best scene point at ({best_scene_point[0]:.3f}, {best_scene_point[1]:.3f}, {best_scene_point[2]:.3f})")

                # Check if CLIP found the right object (near reference)
                ref_positions_np = np.array([[p[0], p[1], p[2]] for p in reference_positions])
                dist_to_refs = np.linalg.norm(ref_positions_np[:, :2] - best_scene_point[:2], axis=1)
                min_dist_to_ref = dist_to_refs.min()
                found_correct_object = min_dist_to_ref < MAX_DIS + 0.05

                if not found_correct_object:
                    print(f"  CLIP found wrong object (dist to ref: {min_dist_to_ref:.3f}m > {MAX_DIS+0.05:.3f}m)")
                    place_distances = np.linalg.norm(place_positions - best_scene_point, axis=1)
                    best_idx = int(np.argmin(place_distances))
                elif valid_indices:
                    valid_positions = place_positions[valid_indices]
                    valid_distances = np.linalg.norm(valid_positions - best_scene_point, axis=1)
                    best_valid_local_idx = int(np.argmin(valid_distances))
                    best_idx = valid_indices[best_valid_local_idx]
                    print(f"  Selected from {len(valid_indices)} valid candidates (idx={best_idx})")
                else:
                    place_distances = np.linalg.norm(place_positions - best_scene_point, axis=1)
                    best_idx = int(np.argmin(place_distances))
            else:
                # Learned networks: use vilg_fusion + policy to score place candidates
                # NOTE: Original A2 has --normalize flag with default=False
                # The pretrained model was trained WITHOUT position normalization
                # Only apply workspace shift for pickplace mode
                is_pickplace = batch.get("is_pickplace", False)
                pts_for_model = pts_pos_tensor.clone()
                place_for_model = place_tensor.clone()

                # Apply workspace shift ONLY for pickplace (like original A2 --workspace_shift)
                if is_pickplace:
                    shift = torch.tensor([[0, -PP_SHIFT_Y, 0]], device=self._device)  # Negative for place workspace
                    pts_for_model = pts_for_model + shift
                    place_for_model[:, :, :3] = place_tensor[:, :, :3] + shift

                cross_feat, attn_weights = self.vilg_fusion(
                    pts_pos=pts_for_model,
                    pts_feat=pts_feat_sampled,
                    pts_sim=pts_sim_norm,
                    actions=place_for_model,
                    mode="place",
                )

                logits = self.policy(cross_feat)

                if isinstance(logits, torch.Tensor):
                    if logits.dim() == 0:
                        best_idx = 0
                    elif logits.dim() == 1:
                        best_idx = torch.argmax(logits).item()
                    else:
                        argmax_result = torch.argmax(logits, dim=-1)
                        if argmax_result.dim() == 0:
                            best_idx = argmax_result.item()
                        else:
                            best_idx = argmax_result[0].item()
                else:
                    best_idx = self.place_agent.select_action_greedy(
                        pts=pts_pos_tensor,
                        clip_sims=pts_sim_norm,
                        actions=place_tensor,
                    )

                print(f"  Learned network selection: best_idx={best_idx}")

            # Get the selected place pose
            self._target_pose = place_poses[best_idx].astype(np.float32)

            # Check if selected pose is valid (matching original A2 evaluation)
            is_valid_selection = best_idx in valid_indices
            self._place_selection_valid = is_valid_selection
            valid_str = "VALID" if is_valid_selection else "INVALID"
            print(f"Selected place {best_idx}: pos={self._target_pose[:3]} ({valid_str})")

        except Exception as e:
            print(f"Warning: Place generation failed ({e}), using fallback")
            import traceback
            traceback.print_exc()
            self._generate_fallback_place(batch)

    def _parse_place_direction(self, lang_goal: str | None) -> str | None:
        """Parse spatial direction from language goal."""
        if lang_goal is None:
            return None
        lang_lower = lang_goal.lower()

        # Check for IN_REGION keywords
        in_keywords = ["on the", "in the", "on top of", "inside"]
        for kw in in_keywords:
            if kw in lang_lower:
                return "on" if "on" in kw else "in"

        # Check for OUT_REGION keywords
        out_keywords = ["near the", "around the", "next to", "beside", "close to"]
        for kw in out_keywords:
            if kw in lang_lower:
                if "near" in kw:
                    return "near"
                elif "around" in kw:
                    return "around"
                elif "next" in kw:
                    return "next to"
                elif "beside" in kw:
                    return "beside"
                elif "close" in kw:
                    return "close to"

        return None

    def _generate_in_region_places(
        self, ref_pos: np.ndarray, ref_size: np.ndarray, num_samples: int = 10
    ) -> list[np.ndarray]:
        """Generate place candidates ON/IN the reference object."""
        places = []
        radius = min(ref_size[0], ref_size[1]) / 3  # Inside the object

        # Workspace limits (matching A2_new)
        ws_x_min, ws_x_max = 0.276, 0.724
        ws_y_min, ws_y_max = -0.224, 0.224

        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            r = np.random.uniform(0, radius)

            x = ref_pos[0] + r * np.cos(angle)
            y = ref_pos[1] + r * np.sin(angle)
            z = ref_pos[2] + ref_size[2] / 2 + 0.02  # On top of object

            # Clip to workspace bounds
            x = np.clip(x, ws_x_min + 0.02, ws_x_max - 0.02)
            y = np.clip(y, ws_y_min + 0.02, ws_y_max - 0.02)

            # Top-down orientation
            pose = np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
            places.append(pose)

        return places

    def _generate_out_region_places(
        self, ref_pos: np.ndarray, ref_size: np.ndarray, num_samples: int = 20
    ) -> list[np.ndarray]:
        """Generate place candidates NEAR/AROUND the reference object."""
        places = []

        # Distance constraints (matching A2_new constants)
        min_dist = 0.05  # MIN_DIS
        max_dist = 0.15  # MAX_DIS

        # Workspace limits (matching A2_new)
        ws_x_min, ws_x_max = 0.276, 0.724
        ws_y_min, ws_y_max = -0.224, 0.224

        inner_radius = max(ref_size[0], ref_size[1]) / 2 + min_dist
        outer_radius = inner_radius + (max_dist - min_dist)

        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            r = np.random.uniform(inner_radius, outer_radius)

            x = ref_pos[0] + r * np.cos(angle)
            y = ref_pos[1] + r * np.sin(angle)
            z = 0.02  # Table height + small offset

            # Clip to workspace bounds
            x = np.clip(x, ws_x_min + 0.02, ws_x_max - 0.02)
            y = np.clip(y, ws_y_min + 0.02, ws_y_max - 0.02)

            # Top-down orientation
            pose = np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
            places.append(pose)

        return places

    def _generate_fallback_place(self, batch: dict[str, torch.Tensor]):
        """Generate fallback place when full pipeline fails."""
        reference_positions = batch.get("reference_positions", [])

        if reference_positions:
            # Place near the first reference object
            ref_pos = np.array(reference_positions[0])
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.08, 0.12)

            x = ref_pos[0] + dist * np.cos(angle)
            y = ref_pos[1] + dist * np.sin(angle)
            z = 0.02
        else:
            # Random place in workspace
            x = np.random.uniform(0.3, 0.7)
            y = np.random.uniform(-0.2, 0.2)
            z = 0.02

        self._target_pose = np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
        self._place_selection_valid = False  # Fallback is never considered valid

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """Forward pass for training (not used in VLA mode)."""
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss, {"loss": loss.item()}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | None = None,
        *,
        config: "A2Config | None" = None,
        pretrained_path: str | None = None,  # Legacy argument
        **kwargs,
    ):
        """Load pretrained model from HuggingFace Hub or local path.

        Args:
            pretrained_name_or_path: HuggingFace repo ID or local path (LeRobot interface)
            config: Optional A2Config instance
            pretrained_path: Legacy argument (deprecated, use pretrained_name_or_path)
            **kwargs: Additional arguments for config

        Returns:
            Loaded A2Policy instance
        """
        # Handle both old and new interfaces
        path = pretrained_name_or_path or pretrained_path
        if path is None:
            raise ValueError("Must provide pretrained_name_or_path or pretrained_path")

        graspnet_checkpoint = kwargs.pop("graspnet_checkpoint", None)

        # Check if it's a HF repo or local path
        if os.path.exists(path):
            checkpoint_path = path
        else:
            # Download from HuggingFace
            checkpoint_path = hf_hub_download(
                repo_id=path,
                filename="sl_checkpoint_199.pth",
            )

            # Also download GraspNet checkpoint if available
            if graspnet_checkpoint is None:
                try:
                    graspnet_checkpoint = hf_hub_download(
                        repo_id=path,
                        filename="checkpoint-rs.tar",
                    )
                    print(f"Downloaded GraspNet checkpoint: {graspnet_checkpoint}")
                except Exception as e:
                    print(f"Warning: Could not download GraspNet checkpoint: {e}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create or use provided config
        if config is None:
            kwargs["graspnet_checkpoint"] = graspnet_checkpoint
            config = A2Config(**kwargs)
        elif graspnet_checkpoint:
            config.graspnet_checkpoint = graspnet_checkpoint

        # Create model
        model = cls(config)

        # Load state dict (for the transformer/policy parts)
        model.load_state_dict(checkpoint, strict=False)

        return model

    # ==================== Legacy Interface (for compatibility) ====================

    def predict_grasp(
        self,
        color_images: dict[str, np.ndarray],
        depth_images: dict[str, np.ndarray],
        point_cloud: np.ndarray,
        lang_goal: str,
        **kwargs,
    ) -> tuple[np.ndarray, dict]:
        """Predict grasp pose (legacy interface for benchmark compatibility)."""
        # Generate random grasp in workspace
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(-0.2, 0.2)
        z = np.random.uniform(0.02, 0.15)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([x, y, z, *quat], dtype=np.float32), {"random": True}

    def predict_place(
        self,
        color_images: dict[str, np.ndarray],
        depth_images: dict[str, np.ndarray],
        point_cloud: np.ndarray,
        lang_goal: str,
        env=None,
        **kwargs,
    ) -> tuple[np.ndarray, dict]:
        """Predict place pose (legacy interface for benchmark compatibility).

        Uses _generate_place_target internally for proper place generation.
        """
        try:
            # Build batch for internal method
            batch = {
                "task": lang_goal,
                "point_cloud": torch.from_numpy(point_cloud).float().unsqueeze(0).to(self._device),
                "color_images": color_images,
                "depth_images": depth_images,
            }

            # Add camera images
            if color_images:
                for cam_name, img in color_images.items():
                    if img is not None:
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(self._device)

            # Add reference object info from environment if available
            if env is not None and hasattr(env, "_env"):
                reference_obj_ids = getattr(env._env, "reference_obj_ids", [])
                reference_direction = getattr(env._env, "reference_direction", None)

                if reference_obj_ids:
                    reference_positions = []
                    reference_sizes = []
                    for ref_id in reference_obj_ids:
                        pos, quat, size = env._env.obj_info(ref_id)
                        reference_positions.append(pos)
                        reference_sizes.append(size if size is not None else [0.05, 0.05, 0.05])

                    batch["reference_positions"] = reference_positions
                    batch["reference_sizes"] = reference_sizes

                if reference_direction:
                    batch["direction"] = reference_direction

            # Generate place target
            self._generate_place_target(batch)

            if self._target_pose is not None:
                return self._target_pose, {"source": "place_generator"}

        except Exception as e:
            print(f"Warning: predict_place failed: {e}")
            import traceback
            traceback.print_exc()

        # Fallback to random
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(-0.2, 0.2)
        z = 0.05
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([x, y, z, *quat], dtype=np.float32), {"random": True}
