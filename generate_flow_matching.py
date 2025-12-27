import torch
import torchvision.utils as vutils
import os
import argparse
from models.dit_2d import DiT2D
from configs.flow_matching_config import FlowMatchingConfig

def sample_euler(model, config, device, num_samples=16, class_labels=None, num_steps=None):
    """
    Euler sampling for flow matching.
    Integrates the ODE dx/dt = v_theta(x_t, t) from t=0 to t=1.

    Args:
        model: The trained DiT2D model
        config: FlowMatchingConfig
        device: torch device
        num_samples: Number of samples to generate
        class_labels: Optional class labels for conditional generation
        num_steps: Number of integration steps (default: config.num_sampling_steps)

    Returns:
        Generated images tensor [num_samples, C, H, W]
    """
    model.eval()

    # Start from pure noise (t=0)
    channels = config.in_channels
    size = config.input_size
    x_t = torch.randn(num_samples, channels, size, size, device=device)

    # If no class labels provided, use random ones or all zeros
    if class_labels is None:
        class_labels = torch.randint(0, config.num_classes, (num_samples,), device=device)
    elif isinstance(class_labels, int):
        class_labels = torch.full((num_samples,), class_labels, device=device)

    # Number of integration steps
    if num_steps is None:
        num_steps = config.num_sampling_steps
    dt = 1.0 / num_steps

    print(f"Sampling with {num_steps} steps...")

    with torch.no_grad():
        for i in range(num_steps):
            # Current time
            t = torch.ones(num_samples, device=device) * (i * dt)

            # Predict velocity field
            if config.use_cfg:
                v_pred = model.forward_with_cfg(x_t, t, class_labels, config.cfg_scale)
            else:
                v_pred = model(x_t, t, class_labels)

            # Euler integration step: x_{t+dt} = x_t + v(x_t, t) * dt
            x_t = x_t + v_pred * dt

    model.train()
    return x_t


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("Warning: No config found in checkpoint, using default config")
        config = FlowMatchingConfig()

    # Create model
    model = DiT2D(config).to(device)

    # Load weights
    if 'ema_shadow' in checkpoint:
        print("Loading EMA weights...")
        # Load EMA shadow parameters
        model_state = checkpoint['model_state_dict']
        for name, param in model.named_parameters():
            if name in checkpoint['ema_shadow']:
                model_state[name] = checkpoint['ema_shadow'][name]
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print(f"Model loaded successfully! (Step: {checkpoint.get('step', 'unknown')})")

    return model, config


def generate_samples(
    checkpoint_path,
    num_samples=16,
    class_labels=None,
    num_steps=50,
    output_dir="generated_samples",
    cfg_scale=None
):
    """
    Generate samples from a trained flow matching model.

    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of images to generate
        class_labels: Optional class label(s) for conditional generation
        num_steps: Number of sampling steps
        output_dir: Directory to save generated images
        cfg_scale: Optional classifier-free guidance scale override
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(checkpoint_path, device)

    # Override sampling steps and cfg scale if provided
    if num_steps is not None:
        config.num_sampling_steps = num_steps
    if cfg_scale is not None:
        config.cfg_scale = cfg_scale

    print(f"Generating {num_samples} samples...")
    print(f"CFG Scale: {config.cfg_scale}")
    print(f"Sampling steps: {config.num_sampling_steps}")

    # Generate samples
    samples = sample_euler(
        model,
        config,
        device,
        num_samples=num_samples,
        class_labels=class_labels,
        num_steps=config.num_sampling_steps
    )

    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    # Save samples
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated.png")
    vutils.save_image(
        samples,
        output_path,
        nrow=int(num_samples ** 0.5),
        normalize=False
    )
    print(f"Saved samples to {output_path}")

    # Also save individual images if requested
    if num_samples <= 64:
        individual_dir = os.path.join(output_dir, "individual")
        os.makedirs(individual_dir, exist_ok=True)
        for i, sample in enumerate(samples):
            vutils.save_image(
                sample,
                os.path.join(individual_dir, f"sample_{i:03d}.png"),
                normalize=False
            )
        print(f"Saved individual samples to {individual_dir}/")

    return samples


def generate_class_grid(checkpoint_path, samples_per_class=8, num_steps=50, output_dir="generated_samples"):
    """
    Generate a grid of samples for all classes.

    Args:
        checkpoint_path: Path to model checkpoint
        samples_per_class: Number of samples per class
        num_steps: Number of sampling steps
        output_dir: Directory to save generated images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, config = load_model(checkpoint_path, device)
    config.num_sampling_steps = num_steps

    all_samples = []

    for class_idx in range(config.num_classes):
        print(f"Generating samples for class {class_idx}...")
        samples = sample_euler(
            model,
            config,
            device,
            num_samples=samples_per_class,
            class_labels=class_idx,
            num_steps=num_steps
        )
        all_samples.append(samples)

    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)

    # Denormalize
    all_samples = (all_samples + 1) / 2
    all_samples = torch.clamp(all_samples, 0, 1)

    # Save grid
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "class_grid.png")
    vutils.save_image(
        all_samples,
        output_path,
        nrow=samples_per_class,
        normalize=False
    )
    print(f"Saved class grid to {output_path}")

    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using flow matching")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--class_label", type=int, default=None, help="Optional class label for conditional generation")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Classifier-free guidance scale")
    parser.add_argument("--output_dir", type=str, default="generated_samples", help="Output directory")
    parser.add_argument("--class_grid", action="store_true", help="Generate a grid with all classes")
    parser.add_argument("--samples_per_class", type=int, default=8, help="Samples per class for class grid")

    args = parser.parse_args()

    if args.class_grid:
        generate_class_grid(
            checkpoint_path=args.checkpoint,
            samples_per_class=args.samples_per_class,
            num_steps=args.num_steps,
            output_dir=args.output_dir
        )
    else:
        generate_samples(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            class_labels=args.class_label,
            num_steps=args.num_steps,
            output_dir=args.output_dir,
            cfg_scale=args.cfg_scale
        )
