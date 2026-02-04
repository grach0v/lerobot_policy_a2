"""Simplified Feature Field - Direct copy of original A2 code.

This module copies the exact functions from Action-Prior-Alignment/models/feature_fusion.py
to ensure identical behavior for debugging.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from . import clip
from .clip import tokenize


# =============================================================================
# COPIED EXACTLY FROM: Action-Prior-Alignment/models/feature_fusion.py
# =============================================================================

def project_points_coords(pts, Rt, K):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4]
    :param K:    [rfn,3,3]
    :return:
        coords:         [rfn,pn,2]
        invalid_mask:   [rfn,pn]
        depth:          [rfn,pn,1]
    """
    pn = pts.shape[0]
    hpts = torch.cat(
        [pts, torch.ones([pn, 1], device=pts.device, dtype=torch.float32)], 1
    )
    srn = Rt.shape[0]
    KRt = K @ Rt  # rfn,3,4
    last_row = torch.zeros([srn, 1, 4], device=pts.device, dtype=torch.float32)
    last_row[:, :, 3] = 1.0
    H = torch.cat([KRt, last_row], 1)  # rfn,4,4
    pts_cam = H[:, None, :, :] @ hpts[None, :, :, None]
    pts_cam = pts_cam[:, :, :3, 0]
    depth = pts_cam[:, :, 2:]
    invalid_mask = torch.abs(depth) < 1e-4
    depth[invalid_mask] = 1e-3
    pts_2d = pts_cam[:, :, :2] / depth
    return pts_2d, ~(invalid_mask[..., 0]), depth


def interpolate_feats(
    feats,
    points,
    h=None,
    w=None,
    padding_mode="zeros",
    align_corners=False,
    inter_mode="bilinear",
):
    """
    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return: feats_inter: b,n,f
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)  # [srn,1,n,2]
    feats_inter = F.grid_sample(
        feats,
        points_norm,
        mode=inter_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).squeeze(2)  # srn,f,n
    feats_inter = feats_inter.permute(0, 2, 1)
    return feats_inter


# =============================================================================
# Simplified Fusion class - mirrors original A2 structure
# =============================================================================

class FusionSimple:
    """Simplified Fusion class that mirrors original A2 exactly."""

    def __init__(self, num_cam, feat_backbone="clip", device="cuda"):
        self.device = device
        self.mu = 0.02  # Distance threshold
        self.curr_obs_torch = {}
        self.H = -1
        self.W = -1
        self.num_cam = num_cam
        self.feat_backbone = feat_backbone

        # Load CLIP model
        if "clip" in self.feat_backbone:
            self.clip_feat_extractor, _ = clip.load(
                "ViT-L/14@336px", device=self.device
            )
        print(f"Loaded feature extractor {feat_backbone}")

    def update(self, obs, text, last_text_feat=False, visualize=False):
        """Update current observation - mirrors original A2 Fusion.update()"""
        # obs contains:
        # - 'color': (K, H, W, 3) numpy array
        # - 'depth': (K, H, W) numpy array
        # - 'pose': (K, 3, 4) numpy array - world-to-camera [R|t]
        # - 'K': (K, 3, 3) numpy array - intrinsics

        imgs = obs['color']
        self.H, self.W = imgs.shape[1], imgs.shape[2]

        # Convert to torch
        self.curr_obs_torch['depth'] = torch.from_numpy(obs['depth']).float().to(self.device)
        self.curr_obs_torch['pose'] = torch.from_numpy(obs['pose']).float().to(self.device)
        self.curr_obs_torch['K'] = torch.from_numpy(obs['K']).float().to(self.device)

        # Extract CLIP features
        params = {
            'row_size': self.H / 3,
            'column_size': self.W / 4,
            'last_text_feat': last_text_feat,
        }

        clip_feats, clip_sims, text_feat = self.extract_multiscale_clip_features(
            imgs, text, params
        )

        self.curr_obs_torch['clip_feats'] = clip_feats
        self.curr_obs_torch['clip_sims'] = clip_sims
        self.curr_obs_torch['text_feat'] = text_feat

    def extract_multiscale_clip_features(self, imgs, text, params):
        """Extract high resolution patch-level CLIP features - copied from original A2."""
        feat_dim = 768  # ViT-L/14@336px

        input_res = self.clip_feat_extractor.visual.input_resolution
        preprocess = T.Compose([
            T.Resize(input_res, interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            T.ToTensor(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        row_size = int(params["row_size"])
        column_size = int(params["column_size"])
        row_grid_num = int(imgs.shape[1] // row_size)
        column_grid_num = int(imgs.shape[2] // column_size)
        patch_h = int(input_res // 14)
        patch_w = int(column_size / row_size * input_res // 14)

        grids_list = []
        for img in imgs:
            grids = None
            for i in range(row_grid_num):
                for j in range(column_grid_num):
                    grid = img[
                        i * row_size : (i + 1) * row_size,
                        j * column_size : (j + 1) * column_size,
                        :,
                    ]
                    grid = Image.fromarray(grid)
                    grid = preprocess(grid)
                    grid = grid.unsqueeze(0)
                    if grids is None:
                        grids = grid
                    else:
                        grids = torch.cat((grids, grid), dim=0)
            grids = grids.to(self.device)
            grids_list.append(grids)

        grids_concat = torch.cat(grids_list, dim=0)
        with torch.no_grad():
            grids_concat_features = self.clip_feat_extractor.get_patch_encodings(
                grids_concat
            )
        grids_concat_features = torch.split(
            grids_concat_features, grids_list[0].shape[0], dim=0
        )

        features = []
        for idx, grids in enumerate(grids_list):
            grid_features = grids_concat_features[idx]
            grid_features = grid_features.reshape(
                (grid_features.shape[0], patch_h, patch_w, feat_dim)
            ).float()

            all_row_features = []
            for i in range(0, grids.shape[0], column_grid_num):
                single_row_features = []
                for j in range(0, column_grid_num):
                    single_row_features.append(grid_features[i + j])
                single_row_feature = torch.cat(single_row_features, dim=1)
                all_row_features.append(single_row_feature)
            feature = torch.cat(all_row_features, dim=0)
            feature = feature.unsqueeze(0)
            features.append(feature)
        features = torch.cat(features, dim=0)

        # Get similarity of color and text
        color_embs = features / features.norm(dim=-1, keepdim=True)

        if text is not None:
            if not params["last_text_feat"]:
                with torch.no_grad():
                    tokens = tokenize(text).to(self.device)
                    text_feature = self.clip_feat_extractor.encode_text(tokens).detach()
            else:
                text_feature = self.curr_obs_torch["text_feat"]

            text_embs = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_embs = text_embs.float()

            # Compute similarities
            sims = color_embs @ text_embs.T
            sims = sims.repeat((1, 1, 1, 3))
        else:
            text_feature = None
            sims = None

        return features, sims, text_feature

    def eval(self, pts, return_names=["clip_feats", "clip_sims"]):
        """Evaluate features at 3D points - copied from original A2 Fusion.eval()"""
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3

        # Transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(
            pts, self.curr_obs_torch["pose"], self.curr_obs_torch["K"]
        )
        pts_depth = pts_depth[..., 0]  # [rfn,pn]

        # Get interpolated depth
        inter_depth = interpolate_feats(
            self.curr_obs_torch["depth"].unsqueeze(1),
            pts_2d,
            h=self.H,
            w=self.W,
            padding_mode="zeros",
            align_corners=True,
            inter_mode="nearest",
        )[..., 0]  # [rfn,pn]

        # Compute distance to surface
        dist = inter_depth - pts_depth  # [rfn,pn]
        dist_valid = (
            (inter_depth > -0.0001) & valid_mask & (dist > -self.mu)
        )  # [rfn,pn]

        # Distance-based weight
        dist_weight = torch.exp(
            torch.clamp(self.mu - torch.abs(dist), max=0) / self.mu
        )  # [rfn,pn]

        dist = torch.clamp(dist, min=-self.mu, max=self.mu)

        # Valid-weighted distance
        dist = (dist * dist_valid.float()).sum(0) / (
            dist_valid.float().sum(0) + 1e-6
        )

        dist_all_invalid = dist_valid.float().sum(0) == 0

        outputs = {"dist": dist, "valid_mask": ~dist_all_invalid}

        for k in return_names:
            inter_k = interpolate_feats(
                self.curr_obs_torch[k].permute(0, 3, 1, 2),
                pts_2d,
                h=self.H,
                w=self.W,
                padding_mode="zeros",
                align_corners=True,
                inter_mode="bilinear",
            )  # [rfn,pn,k_dim]

            # Valid-weighted sum (EXACTLY as in original A2)
            val = (
                inter_k * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)
            ).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6)
            val[dist_all_invalid] = 0.0

            outputs[k] = val

        return outputs


# =============================================================================
# Wrapper to match our interface
# =============================================================================

def compute_per_point_features_simple(
    points: np.ndarray,
    images: dict,
    camera_configs: list,
    lang_goal: str,
    depth_images: dict = None,
    device: str = "cuda",
) -> tuple:
    """Compute per-point CLIP features using simplified Fusion class.

    This function wraps FusionSimple to match our expected interface.

    Args:
        points: (N, 3) numpy array of 3D points
        images: Dict of camera_name -> (H, W, 3) RGB images
        camera_configs: List of camera configs with position, rotation, intrinsics
        lang_goal: Language goal string
        depth_images: Optional dict of camera_name -> (H, W) depth images
        device: Device to use

    Returns:
        pts_feat: (1, N, D) per-point CLIP features
        pts_sim: (1, N, 1) per-point similarity scores
    """
    import pybullet as pb

    # CRITICAL: Original A2 uses exactly 3 cameras: front, left, right
    # Filter to only these cameras to match original A2 exactly
    a2_cam_names = ["front", "left", "right"]
    cam_names = [name for name in a2_cam_names if name in images]
    if len(cam_names) < 3:
        # Fallback: use first 3 available cameras
        cam_names = list(images.keys())[:3]
    num_cam = len(cam_names)

    # Stack images - ensure all have same shape and 3 channels
    # Original A2 uses 480x640, so resize all images to that
    target_H, target_W = 480, 640
    color_list = []
    for name in cam_names:
        img = images[name]
        # Convert RGBA to RGB if needed
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        # Resize if needed
        if img.shape[0] != target_H or img.shape[1] != target_W:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize((target_W, target_H), PILImage.BILINEAR)
            img = np.array(pil_img)
        color_list.append(img)
    color = np.stack(color_list)  # (K, H, W, 3)

    # Stack depth
    if depth_images is not None:
        depth = np.stack([depth_images.get(name, np.zeros_like(depth_images[list(depth_images.keys())[0]])) for name in cam_names])
    else:
        H, W = color.shape[1:3]
        depth = np.ones((num_cam, H, W), dtype=np.float32)

    # Build poses and K - only for the cameras we're using
    # camera_configs should match the order/count of images
    poses = []
    Ks = []
    configs_to_use = camera_configs[:num_cam]  # Only use first num_cam configs
    for config in configs_to_use:
        pos = np.array(config["position"]).reshape(3, 1)
        rot = np.array(pb.getMatrixFromQuaternion(config["rotation"])).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3:] = pos
        world_to_cam = np.linalg.inv(transform)
        poses.append(world_to_cam[:3, :].astype(np.float32))
        Ks.append(config["intrinsics"].astype(np.float32))

    obs = {
        'color': color,
        'depth': depth,
        'pose': np.stack(poses),
        'K': np.stack(Ks),
    }

    # Create Fusion and process
    fusion = FusionSimple(num_cam=num_cam, feat_backbone="clip", device=device)
    fusion.update(obs, lang_goal, last_text_feat=False, visualize=False)

    # Evaluate at points
    pts_torch = torch.from_numpy(points).float().to(device)
    outputs = fusion.eval(pts_torch, return_names=["clip_feats", "clip_sims"])

    pts_feat = outputs["clip_feats"].unsqueeze(0)  # (1, N, D)
    pts_sim = outputs["clip_sims"][:, 0:1].unsqueeze(0)  # (1, N, 1)

    return pts_feat, pts_sim
