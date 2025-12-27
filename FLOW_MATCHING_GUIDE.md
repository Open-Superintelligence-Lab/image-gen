# Flow Matching Image Generation Guide

This guide explains how to use the flow matching components that have been added to this codebase.

## Overview

Flow matching is a generative modeling approach that learns to transform noise into data through velocity field prediction. It's conceptually simpler than diffusion models while achieving comparable quality.

## Quick Start

### 1. Training

Train a flow matching model on CIFAR-10:

```bash
python train_flow_matching.py
```

This will:
- Load CIFAR-10 dataset automatically
- Train a DiT2D model with flow matching
- Save checkpoints every 5000 steps to `checkpoints/`
- Generate sample images every 1000 steps to `samples/`
- Use mixed precision (bfloat16) for efficient training
- Apply Exponential Moving Average (EMA) for stable sampling

### 2. Generation

After training, generate images:

```bash
# Generate 16 random samples
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --num_samples 16

# Generate samples for a specific class
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --num_samples 16 --class_label 5

# Generate a grid showing all classes
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --class_grid --samples_per_class 8

# Adjust sampling quality/speed
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --num_samples 16 --num_steps 100 --cfg_scale 4.0
```

## Architecture

### Model: DiT2D (Diffusion Transformer 2D)

The model is based on the DiT architecture adapted for 2D images:

- **Patch Embedding**: 2x2 patches convert images to tokens
- **Positional Encoding**: 2D sinusoidal position embeddings
- **Transformer Blocks**: Standard attention + MLP with AdaLN conditioning
- **Time Conditioning**: Sinusoidal timestep embeddings fed through MLP
- **Class Conditioning**: Optional label embeddings for conditional generation
- **Output**: Predicts velocity field v(x_t, t)

### Configuration

Edit `configs/flow_matching_config.py` to customize:

```python
@dataclass
class FlowMatchingConfig:
    # Model architecture
    hidden_size: int = 768      # Model width
    depth: int = 12             # Number of layers
    num_heads: int = 12         # Attention heads

    # Training
    batch_size: int = 128
    lr: float = 1e-4
    train_steps: int = 100000

    # Sampling
    num_sampling_steps: int = 50    # Quality/speed tradeoff
    cfg_scale: float = 3.0          # Guidance strength
```

## How Flow Matching Works

### Training

1. **Sample timestep**: t ~ Uniform(0, 1)
2. **Create interpolation**: x_t = (1-t) * noise + t * data
3. **Compute target**: v_target = data - noise
4. **Predict velocity**: v_pred = model(x_t, t, class_label)
5. **Loss**: MSE(v_pred, v_target)

### Sampling

1. **Start from noise**: x_0 ~ N(0, I)
2. **Euler integration**: For t = 0 to 1 with step dt:
   - Predict v_t = model(x_t, t, class_label)
   - Update: x_{t+dt} = x_t + v_t * dt
3. **Result**: x_1 is the generated image

## Key Features

### Classifier-Free Guidance (CFG)

During training, class labels are randomly dropped 10% of the time. During sampling:

```
v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
```

Higher `cfg_scale` (e.g., 4.0) gives more class-specific but potentially less diverse samples.

### Exponential Moving Average (EMA)

The training script maintains an EMA of model parameters with decay 0.9999. The EMA model is used for sampling, which typically produces higher quality results.

### Mixed Precision Training

Uses PyTorch AMP with bfloat16 for faster training and lower memory usage:
- ~2x speedup on modern GPUs
- Maintains numerical stability through gradient scaling

## Reused Components

The following components from the original codebase were reused:

1. **VQ-VAE** (`models/vqvae.py`): Can be used for latent flow matching
2. **DiT Architecture** (`models/dit.py`, `models/layers.py`): Adapted to 2D
3. **Training Infrastructure** (`training/trainer.py`): Mixed precision, optimizers
4. **Data Pipeline** (`data/vqvae_dataset.py`): CIFAR-10 loading
5. **Optimizer** (`optimizers/muon.py`): Can replace AdamW if desired

## Advanced Usage

### Latent Flow Matching

To train in VQ-VAE latent space instead of pixel space:

1. Train a VQ-VAE first:
```bash
python train_vqvae.py
```

2. Load the VQ-VAE encoder in the dataset:
```python
from models.vqvae import VQVAE
vqvae = VQVAE.load_from_checkpoint("checkpoints/vqvae.pt")
encoder = vqvae.encoder

dataloader = get_flow_matching_dataloader(
    vqvae_encoder=encoder,
    ...
)
```

### Custom Datasets

Replace CIFAR-10 with your own dataset:

```python
class CustomFlowMatchingDataset(Dataset):
    def __getitem__(self, idx):
        image = load_your_image(idx)
        label = load_your_label(idx)
        image = preprocess(image)  # Normalize to [-1, 1]
        return image, label
```

### Scaling Up

For larger models:
- Increase `hidden_size`, `depth`, `num_heads` in config
- Use gradient accumulation by reducing `micro_batch_size`
- Consider using the Muon optimizer instead of AdamW
- Enable gradient checkpointing to save memory

## File Structure

```
image-gen/
├── configs/
│   └── flow_matching_config.py    # Configuration
├── models/
│   ├── dit_2d.py                  # 2D DiT model
│   └── layers.py                  # Building blocks
├── data/
│   └── flow_matching_dataset.py   # Dataset loaders
├── train_flow_matching.py         # Training script
├── generate_flow_matching.py      # Inference script
└── FLOW_MATCHING_GUIDE.md         # This file
```

## Tips for Best Results

1. **Training Duration**: 50k-100k steps usually sufficient for CIFAR-10
2. **Sampling Steps**: 50 steps is a good balance, 100+ for best quality
3. **CFG Scale**: Try 2.0-4.0, higher = more adherence to class labels
4. **Batch Size**: Larger is better (up to memory limits)
5. **Learning Rate**: 1e-4 works well, use warmup for stability
6. **EMA**: Always use EMA for sampling, it significantly improves quality

## Troubleshooting

**Q: Loss is not decreasing**
- Check learning rate (try 1e-4 to 5e-4)
- Ensure images are normalized to [-1, 1]
- Verify dataset is loading correctly

**Q: Generated images are blurry**
- Increase number of sampling steps
- Use EMA model for sampling
- Train for more steps

**Q: Out of memory**
- Reduce `micro_batch_size`
- Reduce model size (`hidden_size`, `depth`)
- Use gradient checkpointing

**Q: Training is slow**
- Ensure CUDA is available
- Use smaller `num_workers` if CPU is bottleneck
- Consider reducing image resolution

## References

- Flow Matching: [Lipman et al., 2023](https://arxiv.org/abs/2210.02747)
- DiT: [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)
- Classifier-Free Guidance: [Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)

## License

Same as the main project.
