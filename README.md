# Flow Matching Image Generation

A clean, minimal implementation of Flow Matching for image generation using a Diffusion Transformer (DiT) architecture.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train_flow_matching.py

# Generate
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --num_samples 16
```

See [QUICKSTART.md](QUICKSTART.md) for more details.

---

## Overview

**Flow Matching** for high-quality image generation using a Diffusion Transformer (DiT) architecture.

### What is Flow Matching?

Flow Matching is a continuous normalizing flow approach for generative modeling that learns to transform noise into data through velocity field prediction. It's simpler and more stable than traditional diffusion models.

### Features:

#### 1. **Flow Matching DiT (DiT2D)**
- **Architecture**: 2D Diffusion Transformer adapted from the video DiT
- **Default Size**: 768D hidden size, 12 layers, 12 heads (~50M parameters)
- **Conditioning**: Class-conditional generation with classifier-free guidance (CFG)
- **Training**: Flow matching with velocity field prediction
- **Sampling**: Euler ODE solver for fast, high-quality generation

#### 2. **Training Pipeline**
```bash
python train_flow_matching.py
```
- Flow matching loss: MSE between predicted and target velocity fields
- Mixed precision training (bfloat16) with gradient scaling
- Exponential Moving Average (EMA) for stable sampling
- Classifier-free guidance during training
- Automatic checkpoint saving and sample generation

#### 3. **Generation/Inference**
```bash
# Generate 16 random samples
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --num_samples 16

# Generate samples for a specific class (e.g., class 5)
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --num_samples 16 --class_label 5

# Generate a grid showing all classes
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --class_grid --samples_per_class 8

# Custom sampling settings
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --num_samples 16 --num_steps 100 --cfg_scale 4.0
```

#### 4. **Key Components**
- **DiT2D** ([models/dit_2d.py](models/dit_2d.py)): 2D image flow matching model
- **FlowMatchingConfig** ([configs/flow_matching_config.py](configs/flow_matching_config.py)): Training configuration
- **FlowMatchingDataset** ([data/flow_matching_dataset.py](data/flow_matching_dataset.py)): CIFAR-10 dataloader
- **Training Script** ([train_flow_matching.py](train_flow_matching.py)): Full training loop
- **Generation Script** ([generate_flow_matching.py](generate_flow_matching.py)): Sampling and inference

### How Flow Matching Works:

1. **Linear Interpolation**: For time t ∈ [0,1], interpolate between noise and data:
   ```
   x_t = (1-t) * noise + t * data
   ```

2. **Velocity Prediction**: Model predicts the velocity field v(x_t, t):
   ```
   v_target = data - noise
   ```

3. **Training Loss**: Simple MSE loss between predicted and target velocity:
   ```
   loss = MSE(v_pred, v_target)
   ```

4. **Sampling**: Integrate the ODE from t=0 to t=1 using Euler steps:
   ```
   x_{t+dt} = x_t + v(x_t, t) * dt
   ```

### Configuration Options:

Edit [configs/flow_matching_config.py](configs/flow_matching_config.py) to customize:
- Model architecture (hidden_size, depth, num_heads)
- Training hyperparameters (batch_size, lr, train_steps)
- Sampling settings (num_sampling_steps, cfg_scale)
- EMA settings (use_ema, ema_decay)

---

## Project Structure

```
image-gen/
├── configs/
│   └── flow_matching_config.py    # Training configuration
├── models/
│   ├── dit_2d.py                  # 2D DiT model
│   ├── layers.py                  # Building blocks (attention, MLP, etc.)
│   ├── vqvae.py                   # VQ-VAE (optional, for latent flow matching)
│   └── components.py              # Utility components
├── data/
│   └── flow_matching_dataset.py   # CIFAR-10 dataset loader
├── optimizers/
│   └── muon.py                    # Muon optimizer (optional)
├── utils/
│   ├── logger.py                  # Training logger
│   ├── helpers.py                 # Helper functions
│   └── gpu_monitor.py             # GPU monitoring
├── train_flow_matching.py         # Main training script
├── train_vqvae.py                 # VQ-VAE training (optional)
├── generate_flow_matching.py      # Inference script
└── FLOW_MATCHING_GUIDE.md         # Detailed guide
```

## References

- **Flow Matching**: [Lipman et al., 2023](https://arxiv.org/abs/2210.02747)
- **DiT**: [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)
- **Classifier-Free Guidance**: [Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)

## License

MIT License - See LICENSE file for details.


