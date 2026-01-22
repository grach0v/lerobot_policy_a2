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
from .networks.clip_action import CLIPActionFusion, Policy
from .networks.action_selection import CLIPGrasp, CLIPPlace
from .networks.grasp_generator import create_grasp_generator
from .networks.place_generator import create_place_generator

class GraspPhase(Enum):
    """State machine phases for grasp execution."""
    IDLE = 0           # Waiting for new grasp target
    APPROACHING = 1    # Moving to pre-grasp position
    PRE_GRASP = 2      # At pre-grasp, opening gripper
    DESCENDING = 3     # Moving down to grasp
    GRASPING = 4       # Closing gripper
    LIFTING = 5        # Lifting object
    DONE = 6           # Grasp complete


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
        self._target_pose = None      # Target grasp pose [x, y, z, qx, qy, qz, qw]
        self._pre_grasp_pose = None   # Pre-grasp position (above target)
        self._current_ee_pos = None   # Current end-effector position
        self._step_count = 0          # Steps in current phase
        self._lang_goal = None        # Language goal for current episode

        # VLA parameters
        self._approach_height = 0.15   # Height above target for approach
        self._lift_height = 0.10       # Height to lift after grasp
        self._position_threshold = 0.01  # Position tolerance (1cm)
        # Action output is in [-1, 1] range, env scales by 0.01 to get meters
        # So action=1.0 means 1cm movement per step
        self._max_action = 1.0         # Maximum action magnitude (maps to 1cm/step)
        self._gripper_steps = 5        # Steps to wait for gripper action

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
        return {
            "default": list(self.vilg_fusion.parameters()) + list(self.policy.parameters())
        }

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

            # Get CLIP features and compute visual-language similarity
            self._load_clip()
            pts_feat, pts_sim = self._compute_clip_features(batch, pts_pos, self._lang_goal)

            # Convert grasp poses to tensor for scoring
            grasp_array = np.array(grasp_poses)  # (M, 7)
            grasp_tensor = torch.from_numpy(grasp_array).float().unsqueeze(0).to(self._device)
            pts_pos_tensor = torch.from_numpy(pts_pos).float().unsqueeze(0).to(self._device)

            # Use CLIPActionFusion to score grasps
            cross_feat, attn_weights = self.vilg_fusion(
                pts_pos=pts_pos_tensor,
                pts_feat=pts_feat,
                pts_sim=pts_sim,
                actions=grasp_tensor,
                mode="grasp",
            )

            # Get action scores from policy head
            logits = self.policy(cross_feat)  # (B, M) or (M,)

            # Select best grasp using CLIPGrasp
            best_idx = self.grasp_agent.select_action_greedy(
                pts=pts_pos_tensor,
                clip_sims=pts_sim,
                actions=grasp_tensor,
            )

            # Get the selected grasp pose
            self._target_pose = grasp_poses[best_idx].astype(np.float32)
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
            self._target_pose = np.array([
                self._current_ee_pos[0] + 0.1,  # Forward
                self._current_ee_pos[1],         # Same y
                0.05,                            # Low z for grasp
                0.0, 0.0, 0.0, 1.0               # Identity quaternion
            ], dtype=np.float32)
        else:
            # Default grasp position in workspace
            self._target_pose = np.array([0.5, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

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

        Args:
            batch: Observation batch with images
            pts_pos: Point positions (N, 3)
            lang_goal: Language goal string

        Returns:
            pts_feat: Point features (1, N, D)
            pts_sim: Visual-language similarity (1, N, 1)
        """
        from . import clip

        N = len(pts_pos)
        device = self._device

        # Get front camera image from batch
        img_key = None
        for key in batch:
            if key.startswith("observation.images."):
                img_key = key
                break

        if img_key is None:
            # Return dummy features
            D = self.config.width
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
        # Resize image to CLIP input size
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
        similarity = (image_features @ text_features.T).squeeze()  # scalar

        # For now, use uniform similarity across points
        # (Full implementation would project points to image and get per-point features)
        D = self.config.width
        pts_feat = image_features.unsqueeze(0).expand(1, N, -1)

        # Pad or truncate to width
        if pts_feat.shape[-1] < D:
            pts_feat = torch.nn.functional.pad(pts_feat, (0, D - pts_feat.shape[-1]))
        elif pts_feat.shape[-1] > D:
            pts_feat = pts_feat[..., :D]

        # Similarity: broadcast to all points
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
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=device)
            self._clip_model.eval()

    def _get_grasp_generator(self):
        """Get grasp generator (loaded lazily)."""
        if self._grasp_generator is None:
            self._grasp_generator = create_grasp_generator(
                checkpoint_path=self.config.graspnet_checkpoint,
            )
        return self._grasp_generator

    def _get_place_generator(self):
        """Get place generator (loaded lazily)."""
        if self._place_generator is None:
            self._place_generator = create_place_generator()
        return self._place_generator

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
        **kwargs
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

        graspnet_checkpoint = kwargs.pop('graspnet_checkpoint', None)

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
            kwargs['graspnet_checkpoint'] = graspnet_checkpoint
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
        **kwargs
    ) -> tuple[np.ndarray, dict]:
        """Predict grasp pose (legacy interface for benchmark compatibility)."""
        # Generate random grasp in workspace
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(-0.2, 0.2)
        z = np.random.uniform(0.02, 0.15)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        return np.array([x, y, z, *quat], dtype=np.float32), {"random": True}

    def predict_place(
        self,
        color_images: dict[str, np.ndarray],
        depth_images: dict[str, np.ndarray],
        point_cloud: np.ndarray,
        lang_goal: str,
        **kwargs
    ) -> tuple[np.ndarray, dict]:
        """Predict place pose (legacy interface for benchmark compatibility)."""
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(-0.2, 0.2)
        z = 0.05
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        return np.array([x, y, z, *quat], dtype=np.float32), {"random": True}
