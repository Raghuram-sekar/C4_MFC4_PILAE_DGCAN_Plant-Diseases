import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(dataset_dir, image_size=64, batch_size=128, num_workers=2):
    """
    Loads the dataset using PyTorch ImageFolder.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
        image_size (int): Target image size (resize to image_size x image_size).
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of subprocesses for data loading.
        
    Returns:
        dataloader (DataLoader): PyTorch DataLoader.
        dataset_size (int): Total number of images.
        class_names (list): List of class names.
    """
    
    # Define transforms
    # Resize -> CenterCrop (optional but good for aspect ratio) -> ToTensor -> Normalize
    # Normalize to [-1, 1] for DCGAN (mean=0.5, std=0.5 for 3 channels)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return dataloader, len(dataset), dataset.classes
