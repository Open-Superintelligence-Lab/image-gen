import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from models.dit_2d import DiT2D
from configs.flow_matching_config import FlowMatchingConfig
from data.flow_matching_dataset import get_flow_matching_dataloader
import torchvision.utils as vutils

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def sample_euler(model, config, device, num_samples=16, class_labels=None):
    """
    Euler sampling for flow matching.
    Integrates the ODE dx/dt = v_theta(x_t, t) from t=0 to t=1.
    """
    model.eval()

    # Start from pure noise
    channels = config.in_channels
    size = config.input_size
    x_t = torch.randn(num_samples, channels, size, size, device=device)

    # If no class labels provided, use random ones
    if class_labels is None:
        class_labels = torch.randint(0, config.num_classes, (num_samples,), device=device)

    # Number of integration steps
    num_steps = config.num_sampling_steps
    dt = 1.0 / num_steps

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.ones(num_samples, device=device) * (i * dt)

            # Predict velocity
            if config.use_cfg:
                v_pred = model.forward_with_cfg(x_t, t, class_labels, config.cfg_scale)
            else:
                v_pred = model(x_t, t, class_labels)

            # Euler step
            x_t = x_t + v_pred * dt

    model.train()
    return x_t


def train_flow_matching(config: FlowMatchingConfig):
    """
    Train a flow matching model for image generation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = DiT2D(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create EMA model if enabled
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None

    # Create dataloader
    train_loader = get_flow_matching_dataloader(
        dataset_name=config.dataset,
        batch_size=config.micro_batch_size,
        image_size=config.image_size,
        num_workers=4,
        split="train"
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Starting training for {config.train_steps} steps...")
    print(f"Batch size: {config.micro_batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Warmup steps: {config.warmup_steps}")

    step = 0
    model.train()

    pbar = tqdm(total=config.train_steps, desc="Training")

    while step < config.train_steps:
        for images, labels in train_loader:
            if step >= config.train_steps:
                break

            images = images.to(device)
            labels = labels.to(device)

            # Flow Matching Training
            # 1. Sample timestep t uniformly from [0, 1]
            t = torch.rand(images.shape[0], device=device)

            # 2. Sample x_0 (pure noise) and set x_1 (data)
            x_0 = torch.randn_like(images)
            x_1 = images

            # 3. Linear interpolation: x_t = (1-t)*x_0 + t*x_1
            t_broadcast = t.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1

            # 4. Target velocity field: v = x_1 - x_0
            v_target = x_1 - x_0

            # 5. Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                # Classifier-free guidance: randomly drop labels 10% of the time
                if config.use_cfg and torch.rand(1).item() < config.class_dropout_prob:
                    labels_input = torch.zeros_like(labels)
                else:
                    labels_input = labels

                v_pred = model(x_t, t, labels_input)

                # Handle potential sigma prediction (if learn_sigma=True)
                if v_pred.shape[1] != v_target.shape[1]:
                    v_pred, _ = torch.split(v_pred, v_target.shape[1], dim=1)

                # Flow matching loss (MSE between predicted and target velocity)
                loss = F.mse_loss(v_pred, v_target)

            # 6. Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update EMA
            if ema is not None:
                ema.update()

            # Logging
            loss_val = loss.item()
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{current_lr:.6f}"})
            pbar.update(1)

            if step % config.log_every == 0:
                print(f"\nStep {step}/{config.train_steps}, Loss: {loss_val:.4f}, LR: {current_lr:.6f}")

            # Sample images during training
            if step % config.sample_every == 0 and step > 0:
                print(f"\nGenerating samples at step {step}...")

                # Use EMA model for sampling if available
                if ema is not None:
                    ema.apply_shadow()

                samples = sample_euler(model, config, device, num_samples=16)

                if ema is not None:
                    ema.restore()

                # Denormalize samples
                samples = (samples + 1) / 2
                samples = torch.clamp(samples, 0, 1)

                # Save samples
                os.makedirs("samples", exist_ok=True)
                vutils.save_image(
                    samples,
                    f"samples/step_{step:06d}.png",
                    nrow=4,
                    normalize=False
                )
                print(f"Saved samples/step_{step:06d}.png")

            # Save checkpoint
            if step % config.save_every == 0 and step > 0:
                os.makedirs("checkpoints", exist_ok=True)

                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                }

                if ema is not None:
                    checkpoint['ema_shadow'] = ema.shadow

                torch.save(checkpoint, f"checkpoints/flow_matching_{step:06d}.pt")
                print(f"\nSaved checkpoint: checkpoints/flow_matching_{step:06d}.pt")

            step += 1

    pbar.close()
    print("\nTraining completed!")

    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    final_checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }
    if ema is not None:
        final_checkpoint['ema_shadow'] = ema.shadow

    torch.save(final_checkpoint, "checkpoints/flow_matching_final.pt")
    print("Saved final checkpoint: checkpoints/flow_matching_final.pt")


if __name__ == "__main__":
    config = FlowMatchingConfig()
    train_flow_matching(config)
