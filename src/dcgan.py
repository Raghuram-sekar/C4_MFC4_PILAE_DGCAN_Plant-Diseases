"""
Conditional DCGAN Implementation for Plant Disease Classification
==================================================================
This module implements a Conditional Deep Convolutional GAN with:
- Generator: noise + class label → synthetic 64×64 RGB images
- Discriminator: image + class label → real/fake probability + feature extraction
"""

import torch
import torch.nn as nn


def weights_init(m):
    """
    Custom weights initialization called on netG and netD.
    Initializes Conv, ConvTranspose, and BatchNorm layers with specific distributions.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Conditional Generator Network
    
    Architecture:
        Input: noise vector (nz) + embedded class label (embedding_dim)
        Output: 64×64 RGB image
        
    Process:
        [nz + embedding] → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 3×64×64
    """
    def __init__(self, nz, ngf, nc, n_classes, embedding_dim=100):
        super(Generator, self).__init__()
        self.nz = nz
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        
        # Label embedding layer
        self.label_embed = nn.Embedding(n_classes, embedding_dim)
        
        # Generator architecture (input: nz + embedding_dim)
        self.main = nn.Sequential(
            # Input: (nz + embedding_dim) × 1 × 1
            nn.ConvTranspose2d(nz + embedding_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) × 4 × 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) × 8 × 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) × 16 × 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf) × 32 × 32
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: (nc) × 64 × 64
        )
    
    def forward(self, noise, labels):
        """
        Forward pass of conditional generator.
        
        Args:
            noise: (batch_size, nz, 1, 1) - random noise vector
            labels: (batch_size,) - class labels (integers)
            
        Returns:
            Generated images: (batch_size, 3, 64, 64)
        """
        # Embed labels: (batch_size,) → (batch_size, embedding_dim)
        label_embedding = self.label_embed(labels)
        
        # Reshape to (batch_size, embedding_dim, 1, 1)
        label_embedding = label_embedding.unsqueeze(2).unsqueeze(3)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_embedding], dim=1)
        
        # Generate image
        return self.main(gen_input)


class Discriminator(nn.Module):
    """
    Conditional Discriminator Network with Feature Extraction
    
    Architecture:
        Input: 64×64 RGB image + embedded class label (as additional channels)
        Output: Real/Fake probability + intermediate features for PILAE
        
    Process:
        [(3 + n_classes)×64×64] → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → 1
        
    Feature Extraction:
        Returns 512×4×4 = 8192-dimensional feature vector before final classification
    """
    def __init__(self, nc, ndf, n_classes):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        
        # Label embedding as spatial map (broadcast to image dimensions)
        self.label_embed = nn.Embedding(n_classes, n_classes)
        
        # Main discriminator network
        # Input: (nc + n_classes) × 64 × 64
        self.main = nn.Sequential(
            nn.Conv2d(nc + n_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf) × 32 × 32
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) × 16 × 16
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) × 8 × 8
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) × 4 × 4 → This is our FEATURE LAYER (512×4×4 = 8192)
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 × 1 × 1
        )
    
    def forward(self, img, labels):
        """
        Forward pass of conditional discriminator.
        
        Args:
            img: (batch_size, 3, 64, 64) - input images
            labels: (batch_size,) - class labels (integers)
            
        Returns:
            Probability of being real: (batch_size, 1, 1, 1)
        """
        batch_size = img.size(0)
        
        # Embed labels: (batch_size,) → (batch_size, n_classes)
        label_embedding = self.label_embed(labels)
        
        # Reshape to (batch_size, n_classes, 1, 1) and broadcast to image size
        label_map = label_embedding.unsqueeze(2).unsqueeze(3)
        label_map = label_map.expand(batch_size, self.n_classes, img.size(2), img.size(3))
        
        # Concatenate image with label map
        disc_input = torch.cat([img, label_map], dim=1)
        
        # Extract features and classify
        features = self.main(disc_input)
        output = self.classifier(features)
        
        return output.view(-1, 1).squeeze(1)
    
    def extract_features(self, img, labels):
        """
        Extract 8192-dimensional feature vectors for PILAE classifier.
        
        This method is used AFTER GAN training to extract discriminator features
        for training the PILAE classifier. Features are taken from the last
        convolutional layer (512×4×4 = 8192 dimensions).
        
        Args:
            img: (batch_size, 3, 64, 64) - input images
            labels: (batch_size,) - class labels (integers)
            
        Returns:
            Feature vectors: (batch_size, 8192) - flattened features
        """
        batch_size = img.size(0)
        
        # Embed labels and create label map
        label_embedding = self.label_embed(labels)
        label_map = label_embedding.unsqueeze(2).unsqueeze(3)
        label_map = label_map.expand(batch_size, self.n_classes, img.size(2), img.size(3))
        
        # Concatenate image with label map
        disc_input = torch.cat([img, label_map], dim=1)
        
        # Extract features from last conv layer (512×4×4)
        features = self.main(disc_input)
        
        # Flatten to (batch_size, 8192)
        return features.view(batch_size, -1)


# Model summary for verification
if __name__ == "__main__":
    print("="*70)
    print("Conditional DCGAN Architecture Test")
    print("="*70)
    
    # Test parameters
    batch_size = 4
    nz = 100
    ngf = 64
    ndf = 64
    nc = 3
    n_classes = 38
    
    # Create models
    netG = Generator(nz, ngf, nc, n_classes)
    netD = Discriminator(nc, ndf, n_classes)
    
    # Test Generator
    noise = torch.randn(batch_size, nz, 1, 1)
    labels = torch.randint(0, n_classes, (batch_size,))
    
    print("\nGenerator Test:")
    print(f"  Input noise: {noise.shape}")
    print(f"  Input labels: {labels.shape}")
    
    fake_imgs = netG(noise, labels)
    print(f"  Output images: {fake_imgs.shape}")
    assert fake_imgs.shape == (batch_size, nc, 64, 64), "Generator output shape mismatch!"
    print("  ✓ Generator working correctly")
    
    # Test Discriminator
    print("\nDiscriminator Test:")
    print(f"  Input images: {fake_imgs.shape}")
    print(f"  Input labels: {labels.shape}")
    
    output = netD(fake_imgs, labels)
    print(f"  Output probabilities: {output.shape}")
    assert output.shape == (batch_size,), "Discriminator output shape mismatch!"
    print("  ✓ Discriminator working correctly")
    
    # Test Feature Extraction
    print("\nFeature Extraction Test:")
    features = netD.extract_features(fake_imgs, labels)
    print(f"  Extracted features: {features.shape}")
    assert features.shape == (batch_size, 8192), "Feature shape mismatch!"
    print("  ✓ Feature extraction working correctly")
    
    # Count parameters
    g_params = sum(p.numel() for p in netG.parameters())
    d_params = sum(p.numel() for p in netD.parameters())
    
    print("\n" + "="*70)
    print("Model Statistics:")
    print(f"  Generator parameters:     {g_params:,}")
    print(f"  Discriminator parameters: {d_params:,}")
    print(f"  Total parameters:         {g_params + d_params:,}")
    print("="*70)
    
    print("\n✓ All tests passed! Conditional DCGAN ready for training.")
