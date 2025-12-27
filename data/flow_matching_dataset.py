import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

class FlowMatchingDataset(Dataset):
    """
    Dataset for Flow Matching training.
    Loads images and returns (image, label) pairs.
    Flow matching will interpolate between noise and image during training.
    """
    def __init__(self, dataset_name="cifar10", split="train", image_size=128):
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]
        label = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)
        return image, label


class LatentFlowMatchingDataset(Dataset):
    """
    Dataset for Flow Matching in latent space.
    Pre-encodes images using a VQ-VAE encoder.
    """
    def __init__(self, dataset_name="cifar10", split="train", image_size=128, vqvae_encoder=None):
        self.dataset = load_dataset(dataset_name, split=split)
        self.image_size = image_size
        self.vqvae_encoder = vqvae_encoder
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]
        label = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)

        # If VQ-VAE encoder is provided, encode to latent space
        if self.vqvae_encoder is not None:
            with torch.no_grad():
                # Add batch dimension
                image = image.unsqueeze(0)
                # Encode to latent space
                latent = self.vqvae_encoder(image)
                # Remove batch dimension
                latent = latent.squeeze(0)
                return latent, label

        return image, label


def get_flow_matching_dataloader(
    dataset_name="cifar10",
    batch_size=32,
    image_size=128,
    num_workers=4,
    vqvae_encoder=None,
    split="train"
):
    """
    Create dataloader for flow matching training.

    Args:
        dataset_name: Name of dataset (default: cifar10)
        batch_size: Batch size
        image_size: Image resolution
        num_workers: Number of dataloader workers
        vqvae_encoder: Optional VQ-VAE encoder for latent flow matching
        split: Dataset split (train/test)

    Returns:
        DataLoader
    """
    if vqvae_encoder is not None:
        dataset = LatentFlowMatchingDataset(
            dataset_name=dataset_name,
            split=split,
            image_size=image_size,
            vqvae_encoder=vqvae_encoder
        )
    else:
        dataset = FlowMatchingDataset(
            dataset_name=dataset_name,
            split=split,
            image_size=image_size
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
