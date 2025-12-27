# Quick Start Guide

Get started with Flow Matching image generation in 5 minutes.

## Installation

```bash
pip install -r requirements.txt
```

## Train

Start training on CIFAR-10:

```bash
python train_flow_matching.py
```

The script will:
- Automatically download CIFAR-10
- Train a ~50M parameter DiT model
- Save checkpoints to `checkpoints/` every 5000 steps
- Generate sample images to `samples/` every 1000 steps

## Generate Images

After training (or use a checkpoint):

```bash
# Generate 16 random samples
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --num_samples 16

# Generate specific class (e.g., airplane=0, car=1, bird=2, cat=3, deer=4, dog=5, frog=6, horse=7, ship=8, truck=9)
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --num_samples 16 --class_label 5

# Generate all classes
python generate_flow_matching.py --checkpoint checkpoints/flow_matching_final.pt --class_grid --samples_per_class 8
```

## Customize

Edit `configs/flow_matching_config.py` to change:
- Model size (hidden_size, depth, num_heads)
- Training settings (batch_size, lr, train_steps)
- Sampling quality (num_sampling_steps, cfg_scale)

## What's Next?

See [FLOW_MATCHING_GUIDE.md](FLOW_MATCHING_GUIDE.md) for detailed documentation.
