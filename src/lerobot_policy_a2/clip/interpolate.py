"""Positional embedding interpolation for CLIP."""
import numpy as np
import torch


def interpolate_positional_embedding(
    positional_embedding: torch.Tensor, x: torch.Tensor, patch_size: int, w: int, h: int
):
    """Interpolate the positional encoding for CLIP to the number of patches in the image.

    Modified from DINO ViT `interpolate_pos_encoding` method.
    https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L174
    """
    assert positional_embedding.ndim == 2, "pos_encoding must be 2D"

    num_patches = x.shape[1] - 1
    num_og_patches = positional_embedding.shape[0] - 1

    if num_patches == num_og_patches and w == h:
        return positional_embedding.to(x.dtype)

    dim = x.shape[-1]
    class_pos_embed = positional_embedding[:1]
    patch_pos_embed = positional_embedding[1:]

    w0 = w // patch_size
    h0 = h // patch_size
    assert w0 * h0 == num_patches, "Number of patches does not match"

    w0, h0 = w0 + 0.1, h0 + 0.1

    patch_per_ax = int(np.sqrt(num_og_patches))
    patch_pos_embed_interp = torch.nn.functional.interpolate(
        patch_pos_embed.reshape(1, patch_per_ax, patch_per_ax, dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / patch_per_ax, h0 / patch_per_ax),
        mode="bicubic",
        align_corners=False,
        recompute_scale_factor=False,
    )
    assert (
        int(w0) == patch_pos_embed_interp.shape[-2] and int(h0) == patch_pos_embed_interp.shape[-1]
    ), "Interpolation error."

    patch_pos_embed_interp = patch_pos_embed_interp.permute(0, 2, 3, 1).reshape(-1, dim)
    pos_embed_interp = torch.cat([class_pos_embed, patch_pos_embed_interp], dim=0)
    return pos_embed_interp.to(x.dtype)
