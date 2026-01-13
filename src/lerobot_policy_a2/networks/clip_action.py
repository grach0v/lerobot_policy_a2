"""CLIP-based Action Selection Networks for A2 Policy.

This module contains the core network components for the ViLGP3D policy:
- CLIPActionFusion: Cross-attention fusion of point cloud and action features
- Policy: MLP head for action scoring
- Position embeddings: Sinusoidal and Rotary Position Encoding
"""
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    """Positional embedding using sin/cos functions."""

    def __init__(
        self,
        input_dim: int,
        max_freq_log2: float,
        N_freqs: int,
        log_sampling: bool = True,
        include_input: bool = True,
        periodic_fns: tuple = (torch.sin, torch.cos),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.input_dim

        out = []
        if self.include_input:
            out.append(input)

        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)
        assert out.shape[-1] == self.out_dim
        return out


def get_embedder(multires: int, input_dim: int = 3):
    """Get positional embedder."""
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Encoding (RoPE)."""

    def __init__(self, feature_dim: int, pe_type: str = "Rotary1D"):
        super().__init__()
        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position: torch.Tensor) -> torch.Tensor:
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / self.feature_dim)
        )
        div_term = div_term.view(1, 1, -1)

        sinx = torch.sin(x_position * div_term)
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx],
        )
        position_code = torch.stack([cos_pos, sin_pos], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):
    """3D Rotary Position Encoding for point clouds."""

    def __init__(self, feature_dim: int, pe_type: str = "Rotary3D"):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ: torch.Tensor) -> torch.Tensor:
        """Apply 3D RoPE to XYZ coordinates.

        Args:
            XYZ: Point coordinates (B, N, 3)

        Returns:
            Position encoding (B, N, D, 2) containing [cos, sin]
        """
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]

        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)

        sinx = torch.sin(x_position * div_term)
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz],
        )

        position_code = torch.stack(
            [
                torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
                torch.cat([sinx, siny, sinz], dim=-1),  # sin_pos
            ],
            dim=-1,
        )

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class LayerNorm(nn.LayerNorm):
    """LayerNorm that handles fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Fast GELU approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention module."""

    def __init__(self, embed_dim: int = 1024, num_heads: int = 32, output_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, attn_weights = F.multi_head_attention_forward(
            query=q,
            key=k,
            value=v,
            embed_dim_to_check=v.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            need_weights=True,
            attn_mask=attn_mask,
        )
        return x, attn_weights


class CrossResidualAttentionBlock(nn.Module):
    """Cross-attention block with residual connection."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = CrossModalAttention(embed_dim=d_model, num_heads=n_head, output_dim=d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.attn_mask = attn_mask.to(dtype=q.dtype, device=q.device) if attn_mask is not None else None
        attn_output, attn_weights = self.attn(q=q, k=k, v=v, attn_mask=self.attn_mask)
        return attn_output, attn_weights

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v), attn_mask)
        q = q + attn_output
        q = q + self.mlp(self.ln_2(q))
        return q, attn_weights


class CrossTransformer(nn.Module):
    """Multi-layer cross-attention transformer."""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([CrossResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for resblock in self.resblocks:
            q, attn_weights = resblock(q, k, v, attn_mask)

        q = q.permute(1, 0, 2)  # L'ND -> NL'D
        return q, attn_weights


class Transformer(nn.Module):
    """Self-attention transformer."""

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class ResidualAttentionBlock(nn.Module):
    """Self-attention block with residual connection."""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def weights_init_(m):
    """Initialize weights for Policy network."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Policy(nn.Module):
    """Policy network for action scoring."""

    def __init__(self, input_dim: int, hidden_dim: int, layer_norm: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.layer_norm:
            x = F.relu(self.ln1(self.linear1(state)))
            x = F.relu(self.ln2(self.linear2(x)))
            logits = self.linear3(x).squeeze()
        else:
            x = F.relu(self.linear1(state))
            x = F.relu(self.linear2(x))
            logits = self.linear3(x).squeeze()
        return logits


class CLIPActionFusion(nn.Module):
    """CLIP-based action fusion network.

    This is the core network for ViLGP3D policy. It fuses:
    - Point cloud positions (with sinusoidal or RoPE encoding)
    - Point cloud features (from CLIP visual encoder)
    - Candidate action embeddings

    Using cross-attention to produce action-conditioned features for scoring.
    """

    def __init__(
        self,
        action_dim: int,
        width: int,
        layers: int,
        heads: int,
        device: str,
        task_num: int = None,
        use_rope: bool = False,
        no_feat_rope: bool = False,
        sa: bool = False,
        no_rgb_feat: bool = False,
    ):
        super().__init__()

        self.device = device

        # Cross attention
        self.cross_attn = CrossTransformer(width=width, layers=layers, heads=heads)

        self.sa = sa
        if self.sa:
            self.fusion_attn = Transformer(width=width, layers=layers * 2, heads=heads)

        hidden_dim = int(width / 2)

        # Action embedding MLP
        self.action_embbedding = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        # Point position embedding
        self.pos_projection, pos_proj_dim = get_embedder(multires=5, input_dim=3)
        self.point_embbedding = nn.Sequential(
            nn.Linear(pos_proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        # RoPE layer
        self.use_rope = use_rope
        if self.use_rope:
            self.no_feat_rope = no_feat_rope
            self.relative_pe_layer = RotaryPositionEncoding3D(width)

        # Whether to use RGB features
        self.no_rgb_feat = no_rgb_feat

        # Task embedding
        if task_num is not None:
            self.task_embedding = nn.Embedding(task_num + 1, width)

    def encode_action(self, x: torch.Tensor) -> torch.Tensor:
        """Encode action candidates."""
        grasp_emb = self.action_embbedding(x.to(self.device))
        return grasp_emb

    def encode_pts_pos(self, x: torch.Tensor) -> torch.Tensor:
        """Encode point positions."""
        pts_pos_emb = self.point_embbedding(x.to(self.device))
        return pts_pos_emb

    def pts_multi_fusion(self, pts_feat: torch.Tensor, pts_sim: torch.Tensor) -> torch.Tensor:
        """Fuse point features with similarity scores."""
        fusion_feat = pts_feat * pts_sim
        return fusion_feat

    def forward(
        self,
        pts_pos: torch.Tensor,
        pts_feat: torch.Tensor,
        pts_sim: torch.Tensor,
        actions: torch.Tensor,
        mode: str = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            pts_pos: Point positions (B, N, 3)
            pts_feat: Point features from CLIP (B, N, D)
            pts_sim: Visual-language similarity (B, N, 1)
            actions: Candidate actions (B, M, action_dim)
            mode: Task mode ("grasp" or "place")

        Returns:
            cross_feat: Cross-attended features (B, M, D)
            attn_weights: Attention weights (B, M, N)
        """
        # Get point features weighted by visual-language similarity
        pts_sim_feat = self.pts_multi_fusion(pts_feat, pts_sim)

        # Encode point positions
        pts_pos = self.pos_projection(pts_pos)
        pts_pos_feat = self.encode_pts_pos(pts_pos)

        if self.use_rope:
            # Add RoPE
            pos_rotary_pe = self.relative_pe_layer(pts_pos.to(self.device))
            pos_cos, pos_sin = pos_rotary_pe[..., 0], pos_rotary_pe[..., 1]

            if not self.no_feat_rope:
                pts_sim_feat = RotaryPositionEncoding.embed_rotary(pts_sim_feat, pos_cos, pos_sin)
            pts_pos_feat = RotaryPositionEncoding.embed_rotary(pts_pos_feat, pos_cos, pos_sin)

        pts_sim_feat = pts_sim_feat.permute(1, 0, 2)  # NLD -> LND
        pts_pos_feat = pts_pos_feat.permute(1, 0, 2)  # NLD -> LND

        action_feat = self.encode_action(actions)  # (B, M, D)
        action_feat = action_feat.permute(1, 0, 2)  # NMD -> MND

        # Add task embedding
        if mode is not None and hasattr(self, "task_embedding"):
            if mode == "grasp":
                task_feat = self.task_embedding(
                    torch.LongTensor([1]).to(self.device)
                ).repeat(action_feat.shape[0], 1, 1)
            elif mode == "place":
                task_feat = self.task_embedding(
                    torch.LongTensor([2]).to(self.device)
                ).repeat(action_feat.shape[0], 1, 1)
            action_feat = action_feat + task_feat

        # Cross-attention
        if not self.no_rgb_feat:
            cross_feat, attn_weights = self.cross_attn(
                q=action_feat, k=pts_pos_feat, v=pts_sim_feat
            )
        else:
            cross_feat, attn_weights = self.cross_attn(
                q=action_feat, k=pts_pos_feat, v=pts_pos_feat
            )

        # Self-attention
        if self.sa:
            cross_feat = cross_feat.permute(1, 0, 2)  # NMD -> MND
            cross_feat = self.fusion_attn(cross_feat)
            cross_feat = cross_feat.permute(1, 0, 2)  # MND -> NMD

        return cross_feat, attn_weights
