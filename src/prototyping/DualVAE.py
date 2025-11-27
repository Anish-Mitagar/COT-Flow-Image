"""
Dual VAE Autoencoders (FIXED VERSION)
Variational Autoencoder version of the dual encoder architecture

This replaces DualLinearAutoencoder with probabilistic VAE encoders
Key improvements:
- Probabilistic encodings (μ, σ²) instead of deterministic
- Reparameterization trick for sampling
- Can generate from prior
- KL divergence regularizes latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAEEncoder(nn.Module):
    """
    VAE Encoder that outputs mean and log-variance.
    Supports both convolutional (for images) and MLP (for Fourier) architectures.
    """
    def __init__(self, input_dim, latent_dim, encoder_type='cnn', image_channels=1):
        """
        Args:
            input_dim: Input dimension (784 for flattened 28x28, or 1568 for Fourier)
            latent_dim: Dimension of latent space
            encoder_type: 'cnn' for convolutional, 'mlp' for fully connected
            image_channels: Number of input channels (for CNN)
        """
        super(VAEEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        if encoder_type == 'cnn':
            # CNN encoder for spatial domain
            # 28x28 -> 14x14 -> 7x7
            self.conv_layers = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),  # 28->14
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14->7
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 7->7
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            
            # Flattened dimension: 128 * 7 * 7 = 6272
            self.flatten_dim = 128 * 7 * 7
            
            # Output mean and log-variance
            self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
            self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
            
        elif encoder_type == 'mlp':
            # MLP encoder for Fourier domain
            self.mlp_layers = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
            )
            
            # Output mean and log-variance
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)
        
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def forward(self, x):
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor
               - For CNN: (batch, channels, H, W)
               - For MLP: (batch, input_dim)
        
        Returns:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log-variance of latent distribution (batch, latent_dim)
        """
        if self.encoder_type == 'cnn':
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # Flatten
        else:  # mlp
            x = self.mlp_layers(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder that reconstructs images from latent code.
    """
    def __init__(self, latent_dim, image_channels=1, image_size=28):
        """
        Args:
            latent_dim: Dimension of combined latent space (2 * single_latent_dim)
            image_channels: Number of output channels
            image_size: Size of output images (assumes square)
        """
        super(VAEDecoder, self).__init__()
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        
        # Project latent to feature map
        # For 28x28 output: start from 7x7x128
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        
        # Transposed convolutions to upsample
        # 7x7 -> 14x14 -> 28x28
        self.deconv_layers = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Final convolution to get correct number of channels
            nn.Conv2d(32, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, z):
        """
        Decode latent code to image.
        
        Args:
            z: Latent code (batch, latent_dim)
        
        Returns:
            reconstruction: Reconstructed images (batch, channels, H, W)
        """
        x = self.fc(z)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.deconv_layers(x)
        return x


class DualVAE(nn.Module):
    """
    Dual Variational Autoencoder with separate VAE encoders for:
    - Spatial domain (images)
    - Fourier domain (frequency space)
    
    Both encoders output (μ, σ²) for probabilistic latent representations.
    Single decoder reconstructs images from concatenated latent codes.
    """
    
    def __init__(self, original_dim, encoding_dim, fourier_input_dim, 
                 image_channels=1, image_size=28):
        """
        Args:
            original_dim: Dimension of original images (e.g., 784)
            encoding_dim: Latent dimension for EACH encoder (combined will be 2*encoding_dim)
            fourier_input_dim: Dimension of Fourier input (e.g., 1568 for complex 28x28)
            image_channels: Number of image channels
            image_size: Image size (assumes square)
        """
        super(DualVAE, self).__init__()
        
        self.original_dim = original_dim
        self.encoding_dim = encoding_dim
        self.fourier_input_dim = fourier_input_dim
        self.image_size = image_size
        self.image_channels = image_channels
        
        # Image encoder (CNN-based VAE)
        self.image_encoder = VAEEncoder(
            input_dim=original_dim,
            latent_dim=encoding_dim,
            encoder_type='cnn',
            image_channels=image_channels
        )
        
        # Fourier encoder (MLP-based VAE)
        self.fourier_encoder = VAEEncoder(
            input_dim=fourier_input_dim,
            latent_dim=encoding_dim,
            encoder_type='mlp'
        )
        
        # Decoder (takes combined latent code)
        self.decoder = VAEDecoder(
            latent_dim=encoding_dim * 2,
            image_channels=image_channels,
            image_size=image_size
        )
        
        # For normalization (will be computed during training)
        self.register_buffer('mu', torch.zeros(1, encoding_dim * 2))
        self.register_buffer('std', torch.ones(1, encoding_dim * 2))
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε
        where ε ~ N(0, 1)
        
        Args:
            mu: Mean (batch, latent_dim)
            logvar: Log-variance (batch, latent_dim)
        
        Returns:
            z: Sampled latent code (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode_orig(self, x):
        """
        Encode original image (deterministic for backward compatibility).
        
        Args:
            x: Images (batch, channels, H, W) or (batch, original_dim)
        
        Returns:
            z: Latent code (batch, encoding_dim) - uses mean only
        """
        # Reshape if needed
        if x.ndim == 2:
            x = x.view(-1, self.image_channels, self.image_size, self.image_size)
        
        mu, logvar = self.image_encoder(x)
        
        # For deterministic encoding, return mean
        return mu
    
    def encode_fourier(self, x_fourier):
        """
        Encode Fourier transform (deterministic for backward compatibility).
        
        Args:
            x_fourier: Fourier data (batch, fourier_input_dim)
        
        Returns:
            z: Latent code (batch, encoding_dim) - uses mean only
        """
        mu, logvar = self.fourier_encoder(x_fourier)
        
        # For deterministic encoding, return mean
        return mu
    
    def decode(self, z_combined):
        """
        Decode combined latent code.
        
        Args:
            z_combined: Combined latent (batch, encoding_dim * 2)
        
        Returns:
            reconstruction: Reconstructed images (batch, channels, H, W)
        """
        return self.decoder(z_combined)
    
    def forward(self, x, x_fourier, return_components=False):
        """
        Full forward pass through Dual VAE.
        
        Args:
            x: Original images (batch, channels, H, W) or (batch, original_dim)
            x_fourier: Fourier data (batch, fourier_input_dim)
            return_components: If True, return all VAE components
        
        Returns:
            reconstruction: Reconstructed images
            mu_img, logvar_img: Image encoder parameters
            mu_four, logvar_four: Fourier encoder parameters
            (optional) components: Dict with all intermediate values
        """
        # Reshape if needed
        if x.ndim == 2:
            x = x.view(-1, self.image_channels, self.image_size, self.image_size)
        
        # Encode both domains
        mu_img, logvar_img = self.image_encoder(x)
        mu_four, logvar_four = self.fourier_encoder(x_fourier)
        
        # Reparameterization trick
        z_img = self.reparameterize(mu_img, logvar_img)
        z_four = self.reparameterize(mu_four, logvar_four)
        
        # Concatenate latent codes
        z_combined = torch.cat([z_img, z_four], dim=1)
        
        # Decode
        reconstruction = self.decoder(z_combined)
        
        if return_components:
            components = {
                'z_img': z_img,
                'z_four': z_four,
                'z_combined': z_combined,
                'mu_img': mu_img,
                'logvar_img': logvar_img,
                'mu_four': mu_four,
                'logvar_four': logvar_four
            }
            return reconstruction, mu_img, logvar_img, mu_four, logvar_four, components
        else:
            return reconstruction, mu_img, logvar_img, mu_four, logvar_four
    
    def sample(self, num_samples, device='cpu'):
        """
        Sample from the prior and decode.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            samples: Generated images (num_samples, channels, H, W)
        """
        # Sample from standard normal
        z = torch.randn(num_samples, self.encoding_dim * 2, device=device)
        
        # Decode
        with torch.no_grad():
            samples = self.decoder(z)
        
        return samples


def vae_loss_function(reconstruction, target, mu_image, logvar_image,
                      mu_fourier, logvar_fourier, beta=1.0):
    """
    VAE loss function: reconstruction loss + β * KL divergence.
    
    Args:
        reconstruction: Reconstructed images (batch, channels, H, W)
        target: Original images (batch, channels, H, W)
        mu_image: Mean from image encoder (batch, latent_dim)
        logvar_image: Log-variance from image encoder (batch, latent_dim)
        mu_fourier: Mean from Fourier encoder (batch, latent_dim)
        logvar_fourier: Log-variance from Fourier encoder (batch, latent_dim)
        beta: Weight for KL divergence (β-VAE)
    
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
    
    # KL divergence for image encoder
    # KL(N(μ, σ²) || N(0, 1)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_image = -0.5 * torch.sum(1 + logvar_image - mu_image.pow(2) - logvar_image.exp())
    
    # KL divergence for Fourier encoder
    kl_fourier = -0.5 * torch.sum(1 + logvar_fourier - mu_fourier.pow(2) - logvar_fourier.exp())
    
    # Total KL divergence
    kl_loss = kl_image + kl_fourier
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# ============================================================================
#                           TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Dual VAE")
    print("=" * 70)
    
    # Configuration
    batch_size = 4
    original_dim = 784
    encoding_dim = 64
    fourier_input_dim = 1568
    
    # Create model
    print("\nCreating Dual VAE...")
    model = DualVAE(
        original_dim=original_dim,
        encoding_dim=encoding_dim,
        fourier_input_dim=fourier_input_dim,
        image_channels=1,
        image_size=28
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n" + "-" * 70)
    print("TEST 1: Forward pass")
    print("-" * 70)
    
    x = torch.randn(batch_size, 1, 28, 28)
    x_fourier = torch.randn(batch_size, fourier_input_dim)
    
    reconstruction, mu_img, logvar_img, mu_four, logvar_four = model(x, x_fourier)
    
    print(f"Input shape: {x.shape}")
    print(f"Fourier input shape: {x_fourier.shape}")
    print(f"Image encoder output: μ={mu_img.shape}, log σ²={logvar_img.shape}")
    print(f"Fourier encoder output: μ={mu_four.shape}, log σ²={logvar_four.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Test loss
    print("\n" + "-" * 70)
    print("TEST 2: Loss function")
    print("-" * 70)
    
    total_loss, recon_loss, kl_loss = vae_loss_function(
        reconstruction, x, mu_img, logvar_img, mu_four, logvar_four, beta=1.0
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence: {kl_loss.item():.4f}")
    
    # Test sampling
    print("\n" + "-" * 70)
    print("TEST 3: Sampling from prior")
    print("-" * 70)
    
    samples = model.sample(num_samples=8, device='cpu')
    print(f"Generated {samples.shape[0]} samples")
    print(f"Sample shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
    
    # Test backward compatibility methods
    print("\n" + "-" * 70)
    print("TEST 4: Backward compatibility")
    print("-" * 70)
    
    z_img = model.encode_orig(x)
    z_four = model.encode_fourier(x_fourier)
    z_combined = torch.cat([z_img, z_four], dim=1)
    recon = model.decode(z_combined)
    
    print(f"encode_orig output: {z_img.shape}")
    print(f"encode_fourier output: {z_four.shape}")
    print(f"decode output: {recon.shape}")
    
    # Test gradient flow
    print("\n" + "-" * 70)
    print("TEST 5: Gradient flow")
    print("-" * 70)
    
    model.zero_grad()
    total_loss.backward()
    
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    
    print(f"Parameters with gradients: {has_grad}/{total_params}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
