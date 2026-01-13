"""A2 Policy implementation (ViLGP3D with CLIP-based action selection)."""
from typing import Any, Optional

import torch
import torch.nn as nn
import numpy as np

from .configuration_a2 import A2Config
from .networks.clip_action import CLIPActionFusion, Policy
from .networks.action_selection import CLIPGrasp, CLIPPlace
from .networks.grasp_generator import create_grasp_generator
from .networks.place_generator import create_place_generator

# Try to import PreTrainedPolicy from lerobot, fallback to nn.Module
try:
    from lerobot.policies.pretrained import PreTrainedPolicy
    _BASE_CLASS = PreTrainedPolicy
except ImportError:
    _BASE_CLASS = nn.Module


class A2Policy(_BASE_CLASS):
    """A2 Policy for 6-DOF grasp and place pose selection.

    This policy uses:
    1. Point cloud extraction from depth images
    2. CLIP features from RGB images
    3. GraspNet/PlaceNet for candidate pose generation
    4. Cross-attention transformer for pose selection

    The policy selects the best grasp/place pose from candidates
    based on visual features and geometric context.
    """

    config_class = A2Config
    name = "a2"

    def __init__(self, config: A2Config, dataset_stats: dict[str, Any] | None = None):
        if _BASE_CLASS == nn.Module:
            super().__init__()
        else:
            super().__init__(config, dataset_stats)
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

    def _load_clip(self):
        """Load CLIP model lazily."""
        if self._clip_model is None:
            from . import clip
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self._device)
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

    def extract_point_cloud(self, depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """Extract point cloud from depth image.

        Args:
            depth: Depth image (B, H, W) or (B, 1, H, W)
            intrinsics: Camera intrinsics (B, 3, 3)

        Returns:
            Point cloud (B, N, 3) where N = sample_num
        """
        if depth.dim() == 4:
            depth = depth.squeeze(1)

        B, H, W = depth.shape
        device = depth.device

        # Create pixel grid
        v, u = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        u = u.float()
        v = v.float()

        # Unproject to 3D
        fx = intrinsics[:, 0, 0].view(B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)

        z = depth
        x = (u.unsqueeze(0) - cx) * z / fx
        y = (v.unsqueeze(0) - cy) * z / fy

        # Stack to (B, H, W, 3) then reshape to (B, H*W, 3)
        points = torch.stack([x, y, z], dim=-1).view(B, -1, 3)

        # Filter valid points (z > 0)
        valid_mask = points[..., 2] > 0.01

        # Sample points
        sampled_points = []
        for b in range(B):
            valid_pts = points[b][valid_mask[b]]
            if len(valid_pts) > self.config.sample_num:
                indices = torch.randperm(len(valid_pts))[:self.config.sample_num]
                sampled_points.append(valid_pts[indices])
            else:
                # Pad with zeros if not enough points
                padding = torch.zeros(
                    self.config.sample_num - len(valid_pts), 3,
                    device=device
                )
                sampled_points.append(torch.cat([valid_pts, padding], dim=0))

        return torch.stack(sampled_points, dim=0)

    def extract_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP features from images.

        Args:
            images: RGB images (B, C, H, W) in [0, 1] range

        Returns:
            CLIP features (B, N, D) for each image patch
        """
        self._load_clip()

        # CLIP expects (B, 3, 224, 224) normalized images
        if images.shape[-2:] != (224, 224):
            images = torch.nn.functional.interpolate(
                images, size=(224, 224), mode='bilinear', align_corners=False
            )

        # Normalize for CLIP
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device)
        images = (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        with torch.no_grad():
            features = self._clip_model.encode_image(images)

        return features

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict | None]:
        """Forward pass for training.

        Args:
            batch: Dictionary containing:
                - observation.images.*: RGB images
                - observation.images.depth: Depth image
                - action: Ground truth actions
                - action_candidates: Candidate poses from GraspNet/PlaceNet

        Returns:
            Tuple of (loss, info_dict)
        """
        # This is a simplified forward pass
        # In practice, you would:
        # 1. Extract point cloud from depth
        # 2. Extract CLIP features from RGB
        # 3. Encode candidate actions
        # 4. Use cross-attention to select best action
        # 5. Compute loss against ground truth

        # For now, return a placeholder
        device = next(self.parameters()).device
        batch_size = 1

        # Placeholder loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss, {"loss": loss.item()}

    def select_action(
        self,
        pts_pos: torch.Tensor,
        pts_feat: torch.Tensor,
        pts_sim: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select best action from candidates.

        Args:
            pts_pos: Point cloud positions (B, N, 3)
            pts_feat: Point cloud features (B, N, D)
            pts_sim: Similarity scores (B, N, 1)
            actions: Candidate actions (B, M, action_dim)

        Returns:
            Tuple of (logits, selected_action)
        """
        # Encode point cloud positions
        pts_pos_proj = self.vilg_fusion.pos_projection(pts_pos)
        pts_pos_emb = self.vilg_fusion.encode_pts_pos(pts_pos_proj)

        # Fuse point features with similarity
        if not self.config.no_rgb_feat:
            pts_fused = self.vilg_fusion.pts_multi_fusion(pts_feat, pts_sim)
        else:
            pts_fused = pts_feat

        # Encode actions
        action_emb = self.vilg_fusion.encode_action(actions)

        # Cross-attention: actions attend to point cloud
        # action_emb: (B, M, D) -> (M, B, D)
        # pts_emb: (B, N, D) -> (N, B, D)
        action_emb_t = action_emb.permute(1, 0, 2)
        pts_emb = pts_pos_emb + pts_fused
        pts_emb_t = pts_emb.permute(1, 0, 2)

        # Apply RoPE if enabled
        if self.config.use_rope:
            # Add positional encoding
            pass

        # Cross-attention
        attn_output, _ = self.vilg_fusion.cross_attn(
            q=action_emb_t,
            k=pts_emb_t,
            v=pts_emb_t,
        )

        # Policy head for scoring
        logits = self.policy(attn_output).squeeze(-1)  # (B, M)

        # Select best action
        best_idx = logits.argmax(dim=-1)
        batch_idx = torch.arange(actions.shape[0], device=actions.device)
        selected_action = actions[batch_idx, best_idx]

        return logits.detach().cpu().numpy(), selected_action.detach().cpu().numpy()

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        """Load pretrained model from HuggingFace Hub or local path.

        Args:
            pretrained_path: HuggingFace repo ID or local path
            **kwargs: Additional arguments for config

        Returns:
            Loaded A2Policy instance
        """
        from huggingface_hub import hf_hub_download
        import os

        # Check if it's a HF repo or local path
        if os.path.exists(pretrained_path):
            checkpoint_path = pretrained_path
        else:
            # Download from HuggingFace
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_path,
                filename="sl_checkpoint_199.pth",
            )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create config with defaults
        config = A2Config(**kwargs)

        # Create model
        model = cls(config)

        # Load state dict
        # The checkpoint contains the vilg_fusion and policy weights
        model_state = {}
        for key, value in checkpoint.items():
            model_state[key] = value

        model.load_state_dict(model_state, strict=False)

        return model
