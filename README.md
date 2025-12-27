# 5-Dollar LLM (Blueberry 88M)

Help us build top 10 LLM in the world while keeping it fully open source, which will accelerate everyone and everything that uses LLMs (science, technology, medicine, startups, businesses, etc.)

> Check out our contributors [leaderboard](docs/LEADERBOARD.md)!

## 🗺️ Open Superintelligence Lab Roadmap

**Our goals:**
1. **GPT-1** Level by Dec 20 2025 ✓ [Watch](https://youtu.be/1nf6mVNN2lo)
2. **GPT-2** Level by Jan 20 2026
3. **GPT-3** Level by Apr 20 2026
4. **Top 150** in LMArena (GPT-4o-mini level) by June 2026
5. **Top 50** by Apr 2027
6. **Top 10** by Dec 2027
7. We could aim for **Top 1** by 2028, TBD

---

Can you make our LLM train faster and better?

👉 **[Full Setup Guide](docs/SETUP_INSTRUCTIONS.md)** | **[Leaderboard](docs/LEADERBOARD.md)** | **[Multimodal Guide](README_Multimodal.md)**

---

## 🎨 Image Generation with Flow Matching

This codebase now includes **Flow Matching** for high-quality image generation using a Diffusion Transformer (DiT) architecture.

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

## 🎨 Multimodal Image Generation (Hard Mode)

We have also implemented **"Hard Mode" Multimodal Image Generation**—building a mini-version of **Google Parti** or **DeepSeek Janus** from ground zero with **zero pre-trained weights**.

### How it works:
1.  **Visual Tokenizer**: A custom **VQ-VAE** compresses 128x128 images into a 32x32 grid of discrete "visual words".
2.  **Multimodal Transformer**: A 40M parameter Llama-style transformer trained to predict both text and visual tokens in a single unified stream.
3.  **Unified Vocabulary**: Text (49k) + Image (1k) tokens interleaved: `[BOS] {text} <seg_start> {visual_tokens} <seg_end> [EOS]`.
4.  **Optimized Training**: Powered by the **Muon optimizer** and **Mixed Precision (Bfloat16)**, allowing for high-quality image synthesis on a single GPU.

### Achievement:
The model has been scaled to **1,000,000 training sequences** on CIFAR-10, demonstrating the ability to generate class-specific images (frogs, birds, cars, etc.) from scratch in an autoregressive fashion.

---

## Acceptance criteria:
0. Once you measure an improvement over the baseline according to the [Setup Guide](docs/SETUP_INSTRUCTIONS.md), submit your code in a GitHub pull request.
1. The LLM must train faster or achieve lower loss on any of the benchmarks (8M, 20M, 100M, 1B tokens).
2. Lower loss takes priority over training speed because pretraining data is limited - if your submission trains slower but achieves better (lower) loss for the same amount of tokens, it will probably be accepted, and vice versa.
3. Add as little code as possible, keep it clean, rewrite AI generated pull request descriptions to increase quality.
4. Submissions are judged case by case, tradeoffs between speed / loss etc. will be taken into account.

---

## 🤝 Partners & Support

**If you want to write a research paper improving this project, or if you or someone you know has extensive research experience and wants to contribute to this open-source initiative, contact me.**

We will partner with compute providers while keeping all research/engineering/code fully open source.

**Potential partners include:** Hugging Face, NVIDIA, Microsoft, Google, Amazon, Meta, IBM, Oracle, Alibaba, Tencent, Huawei, Baidu, CoreWeave, Lambda Labs, Hyperbolic, Stability AI, OpenAI, Anthropic, xAI, Cohere, Mistral AI, Graphcore, Tenstorrent, Intel, AMD, Dell Technologies, ai2, a16z, Sequoia Capital, and more.


