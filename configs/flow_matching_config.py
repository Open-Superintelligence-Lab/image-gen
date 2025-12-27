from dataclasses import dataclass

@dataclass
class FlowMatchingConfig:
    """Configuration for Flow Matching Image Generation with DiT."""

    # Model Architecture
    in_channels: int = 4  # Latent channels from VQ-VAE or VAE
    input_size: int = 32  # Spatial size (H, W) for latent space
    patch_size: int = 2   # 2x2 patches (not 2x2x2 since no temporal dimension)
    hidden_size: int = 768  # d_model (smaller than video DiT for efficiency)
    depth: int = 12         # n_layers
    num_heads: int = 12     # n_heads
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 10   # CIFAR-10 classes
    learn_sigma: bool = False  # For flow matching, we typically don't learn sigma

    # Flow Matching Specific
    use_cfg: bool = True    # Classifier-free guidance
    cfg_scale: float = 3.0  # Guidance scale for sampling

    # Training
    train_steps: int = 100000
    batch_size: int = 128    # Global batch size
    micro_batch_size: int = 32  # Per-GPU batch size
    lr: float = 1e-4
    weight_decay: float = 0.0

    # Optimization
    grad_clip: float = 1.0
    warmup_steps: int = 5000
    use_ema: bool = True     # Exponential moving average for better sampling
    ema_decay: float = 0.9999

    # Flow Matching Sampling
    num_sampling_steps: int = 50  # Number of Euler steps for sampling

    # Checkpointing
    save_every: int = 5000
    log_every: int = 100
    sample_every: int = 1000  # Generate samples during training

    # Dataset
    image_size: int = 128    # Original image size (before VAE encoding)
    dataset: str = "cifar10"

    @property
    def d_model(self):
        return self.hidden_size

    @property
    def n_heads(self):
        return self.num_heads

    @property
    def n_layers(self):
        return self.depth

    @property
    def latent_size(self):
        """Size of latent space after patch embedding."""
        return self.input_size // self.patch_size
