import torch
import torch.nn as nn
import numpy as np
from models.layers import TimestepEmbedder, DiTBlock, FinalLayer
from configs.flow_matching_config import FlowMatchingConfig

class PatchEmbed2D(nn.Module):
    """Image to Patch Embedding (2D version without temporal dimension)"""
    def __init__(self, patch_size=2, in_channels=4, hidden_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)
        # x: [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)
        # x: [B, L, D] where L = (H/p) * (W/p)
        return x


class DiT2D(nn.Module):
    """
    Diffusion Transformer for 2D Image Generation with Flow Matching.
    Adapted from 3D video DiT to work with 2D images only.
    """
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads

        self.x_embedder = PatchEmbed2D(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size
        )
        self.t_embedder = TimestepEmbedder(config.hidden_size)

        # Optional: Class conditioning
        if config.num_classes > 0:
            self.y_embedder = nn.Embedding(config.num_classes, config.hidden_size)
            self.num_classes = config.num_classes
        else:
            self.y_embedder = None

        # Positional embedding (2D only)
        h_patches = config.input_size // config.patch_size
        w_patches = config.input_size // config.patch_size
        num_patches = h_patches * w_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_size),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads, config, mlp_ratio=config.mlp_ratio)
            for _ in range(config.depth)
        ])

        self.final_layer = FinalLayer(config.hidden_size, config.patch_size, self.out_channels, is_3d=False)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = self.get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.config.input_size // self.patch_size,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaln modulation layers in blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        """
        2D sin-cos positional embedding.
        grid_size: int of the grid height and width
        return: [grid_size*grid_size, embed_dim]
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def forward(self, x, t, y=None):
        """
        Forward pass for flow matching.
        x: (N, C, H, W) - noisy input
        t: (N,) - timestep in [0, 1]
        y: (N,) - class labels (optional)
        Returns: (N, C, H, W) - predicted velocity field
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, L, D)
        t = self.t_embedder(t)                   # (N, D)

        if self.y_embedder is not None and y is not None:
            y_emb = self.y_embedder(y)
            c = t + y_emb
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)  # (N, L, patch_size**2 * out_channels)
        x = self.unpatchify(x)      # (N, out_channels, H, W)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * out_channels)
        imgs: (N, out_channels, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.config.input_size // p

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        """
        if cfg_scale == 1.0 or y is None:
            return self.forward(x, t, y)

        # Combine conditional and unconditional predictions
        half = x.shape[0] // 2
        combined = torch.cat([x, x], dim=0)
        t_combined = torch.cat([t, t], dim=0)

        # Create y with unconditional (zeros) for second half
        y_combined = torch.cat([y, torch.zeros_like(y)], dim=0)

        model_out = self.forward(combined, t_combined, y_combined)
        cond_out, uncond_out = torch.split(model_out, half, dim=0)

        # Apply guidance
        return uncond_out + cfg_scale * (cond_out - uncond_out)
