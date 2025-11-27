import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import the ProbabilisticMask from the original file
from dual_autoencoder import ProbabilisticMask


class VAEEncoder(nn.Module):
    """
    Encoder that outputs mean and log-variance for VAE.
    """
    def __init__(self, input_dim, latent_dim, is_convolutional=True, image_channels=1):
        """
        Args:
            input_dim: Input dimension (flattened size for MLP, or image size for CNN)
            latent_dim: Dimension of latent space
            is_convolutional: If True, use CNN encoder; if False, use MLP encoder
            image_channels: Number of input channels (for CNN)
        """
        super(VAEEncoder, self).__init__()
        self.is_convolutional = is_convolutional
        self.latent_dim = latent_dim
        
        if is_convolutional:
            # CNN encoder for spatial domain
            # 28x28 -> 14x14 -> 7x7
            self.conv_layers = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            
            # For 28x28 input: after two stride-2 convs -> 7x7
            # 128 * 7 * 7 = 6272
            self.flatten_dim = 128 * 7 * 7
            
            # Output mean and log-variance
            self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
            self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
            
        else:
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
    
    def forward(self, x):
        """
        Forward pass that outputs mean and log-variance.
        
        Args:
            x: Input tensor
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        if self.is_convolutional:
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # Flatten
        else:
            x = self.mlp_layers(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Decoder for VAE that reconstructs images from latent code.
    """
    def __init__(self, latent_dim, image_channels=1, image_size=28):
        """
        Args:
            latent_dim: Dimension of combined latent space (2 * single_latent_dim)
            image_channels: Number of output channels
            image_size: Size of output images
        """
        super(VAEDecoder, self).__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        
        # Project latent to feature map
        # For 28x28 output: start from 7x7x128
        # 7x7 -> 14x14 -> 28x28 (two stride-2 upsamples)
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
    Dual Variational Autoencoder with separate encoders for spatial and Fourier domains.
    Both encoders are VAEs with reparameterization trick.
    """
    def __init__(self, image_channels=1, image_size=28, fourier_input_dim=784*2, latent_dim=64):
        """
        Args:
            image_channels: Number of input image channels
            image_size: Size of input images (assumes square)
            fourier_input_dim: Dimension of flattened Fourier input
            latent_dim: Dimension of EACH encoder's latent space (combined will be 2*latent_dim)
        """
        super(DualVAE, self).__init__()
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.fourier_input_dim = fourier_input_dim
        self.latent_dim = latent_dim
        
        # Image encoder (CNN-based VAE)
        self.image_encoder = VAEEncoder(
            input_dim=image_size * image_size,
            latent_dim=latent_dim,
            is_convolutional=True,
            image_channels=image_channels
        )
        
        # Fourier encoder (MLP-based VAE)
        self.fourier_encoder = VAEEncoder(
            input_dim=fourier_input_dim,
            latent_dim=latent_dim,
            is_convolutional=False
        )
        
        # Decoder takes combined latent code
        self.decoder = VAEDecoder(
            latent_dim=latent_dim * 2,  # Combined from both encoders
            image_channels=image_channels,
            image_size=image_size
        )
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)
        
        Args:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log-variance of latent distribution (batch, latent_dim)
        
        Returns:
            z: Sampled latent code (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x, x_fourier, return_components=False):
        """
        Forward pass through dual VAE.
        
        Args:
            x: Input images (batch, channels, H, W)
            x_fourier: Fourier domain input (batch, fourier_input_dim)
            return_components: If True, return all VAE components (mu, logvar, z)
        
        Returns:
            reconstruction: Reconstructed images
            mu_image, logvar_image: Image encoder parameters
            mu_fourier, logvar_fourier: Fourier encoder parameters
            (optional) components: Dict with all intermediate values
        """
        # Encode spatial domain
        mu_image, logvar_image = self.image_encoder(x)
        z_image = self.reparameterize(mu_image, logvar_image)
        
        # Encode frequency domain
        mu_fourier, logvar_fourier = self.fourier_encoder(x_fourier)
        z_fourier = self.reparameterize(mu_fourier, logvar_fourier)
        
        # Concatenate latent codes
        z_combined = torch.cat([z_image, z_fourier], dim=1)
        
        # Decode
        reconstruction = self.decoder(z_combined)
        
        if return_components:
            components = {
                'mu_image': mu_image,
                'logvar_image': logvar_image,
                'z_image': z_image,
                'mu_fourier': mu_fourier,
                'logvar_fourier': logvar_fourier,
                'z_fourier': z_fourier,
                'z_combined': z_combined
            }
            return reconstruction, mu_image, logvar_image, mu_fourier, logvar_fourier, components
        
        return reconstruction, mu_image, logvar_image, mu_fourier, logvar_fourier
    
    def sample(self, num_samples, device='cpu'):
        """
        Sample from the prior p(z) = N(0, I) and decode.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            samples: Generated images (num_samples, channels, H, W)
        """
        # Sample from prior
        z_image = torch.randn(num_samples, self.latent_dim, device=device)
        z_fourier = torch.randn(num_samples, self.latent_dim, device=device)
        z_combined = torch.cat([z_image, z_fourier], dim=1)
        
        # Decode
        with torch.no_grad():
            samples = self.decoder(z_combined)
        
        return samples


class MaskedDualVAE(nn.Module):
    """
    Dual VAE with learnable LOUPE-style probabilistic mask applied to Fourier domain.
    """
    def __init__(self, image_channels=1, image_size=28, fourier_input_dim=784*2, 
                 latent_dim=64, sparsity=0.1, slope=5.0):
        """
        Args:
            image_channels: Number of input image channels
            image_size: Size of input images (assumes square)
            fourier_input_dim: Dimension of flattened Fourier input
            latent_dim: Dimension of each encoder's latent space
            sparsity: Target sparsity/budget for the mask (e.g., 0.1 for 10%)
            slope: Slope for LOUPE sigmoid function
        """
        super(MaskedDualVAE, self).__init__()
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.fourier_input_dim = fourier_input_dim
        self.latent_dim = latent_dim
        
        # Initialize the dual VAE
        self.vae = DualVAE(
            image_channels=image_channels,
            image_size=image_size,
            fourier_input_dim=fourier_input_dim,
            latent_dim=latent_dim
        )
        
        # Initialize the LOUPE-style probabilistic mask
        self.mask = ProbabilisticMask(
            input_dim=fourier_input_dim,
            convolutional=False,
            sparsity=sparsity,
            slope=slope
        )
    
    def forward(self, x, x_fourier, return_components=False):
        """
        Forward pass with masked Fourier domain.
        
        Args:
            x: Input images (batch, channels, H, W)
            x_fourier: Flattened Fourier domain input (batch, fourier_input_dim)
            return_components: If True, return all VAE components
        
        Returns:
            reconstruction: Reconstructed images
            mask: Binary mask used
            mu_image, logvar_image: Image encoder VAE parameters
            mu_fourier, logvar_fourier: Fourier encoder VAE parameters
            (optional) components: Dict with all intermediate values
        """
        # Apply learned mask to Fourier domain
        x_fourier_masked, mask = self.mask(x_fourier)
        
        # Forward through VAE
        if return_components:
            reconstruction, mu_image, logvar_image, mu_fourier, logvar_fourier, components = \
                self.vae(x, x_fourier_masked, return_components=True)
            components['mask'] = mask
            return reconstruction, mask, mu_image, logvar_image, mu_fourier, logvar_fourier, components
        else:
            reconstruction, mu_image, logvar_image, mu_fourier, logvar_fourier = \
                self.vae(x, x_fourier_masked, return_components=False)
            return reconstruction, mask, mu_image, logvar_image, mu_fourier, logvar_fourier
    
    def sample(self, num_samples, device='cpu'):
        """Sample from the prior and decode."""
        return self.vae.sample(num_samples, device)
    
    def get_learned_mask(self, threshold=None):
        """Convenience method to access learned mask."""
        return self.mask.get_deterministic_mask(threshold)
    
    def get_sparsity(self):
        """Get current actual sparsity of the mask."""
        return self.mask.get_sparsity()
    
    def set_sparsity(self, new_sparsity):
        """Adjust target sparsity without retraining."""
        self.mask.set_sparsity(new_sparsity)


def vae_loss_function(reconstruction, target, mu_image, logvar_image, 
                      mu_fourier, logvar_fourier, beta=1.0):
    """
    VAE loss function: reconstruction loss + KL divergence.
    
    Args:
        reconstruction: Reconstructed images
        target: Original images
        mu_image: Mean from image encoder
        logvar_image: Log-variance from image encoder
        mu_fourier: Mean from Fourier encoder
        logvar_fourier: Log-variance from Fourier encoder
        beta: Weight for KL divergence (beta-VAE)
    
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (MSE or BCE)
    # Using MSE since images are in [0, 1] range
    recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
    
    # KL divergence for image encoder
    # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_image = -0.5 * torch.sum(1 + logvar_image - mu_image.pow(2) - logvar_image.exp())
    
    # KL divergence for Fourier encoder
    kl_fourier = -0.5 * torch.sum(1 + logvar_fourier - mu_fourier.pow(2) - logvar_fourier.exp())
    
    # Total KL divergence
    kl_loss = kl_image + kl_fourier
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# ===== Example Usage =====
if __name__ == "__main__":
    print("=" * 70)
    print("Testing DualVAE")
    print("=" * 70)
    
    # Test configuration
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    x_fourier = torch.randn(batch_size, 784*2, dtype=torch.float32)
    
    # Test DualVAE
    print("\n1. Testing DualVAE (without mask)")
    print("-" * 70)
    dual_vae = DualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=784*2,
        latent_dim=64
    )
    
    reconstruction, mu_img, logvar_img, mu_four, logvar_four = dual_vae(x, x_fourier)
    
    print(f"Input shape: {x.shape}")
    print(f"Fourier input shape: {x_fourier.shape}")
    print(f"Image encoder - mu shape: {mu_img.shape}, logvar shape: {logvar_img.shape}")
    print(f"Fourier encoder - mu shape: {mu_four.shape}, logvar shape: {logvar_four.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Calculate loss
    total_loss, recon_loss, kl_loss = vae_loss_function(
        reconstruction, x, mu_img, logvar_img, mu_four, logvar_four, beta=1.0
    )
    print(f"\nLoss components:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL divergence: {kl_loss.item():.4f}")
    
    # Test sampling
    print(f"\nTesting sampling from prior:")
    samples = dual_vae.sample(num_samples=5, device='cpu')
    print(f"Generated samples shape: {samples.shape}")
    
    # Test MaskedDualVAE
    print("\n" + "=" * 70)
    print("2. Testing MaskedDualVAE (with LOUPE mask)")
    print("-" * 70)
    
    masked_vae = MaskedDualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=784*2,
        latent_dim=64,
        sparsity=0.1,
        slope=5.0
    )
    
    # Training mode
    masked_vae.train()
    reconstruction, mask, mu_img, logvar_img, mu_four, logvar_four = masked_vae(x, x_fourier)
    
    print(f"Input shape: {x.shape}")
    print(f"Fourier input shape: {x_fourier.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sparsity (actual): {mask.float().mean():.4f}")
    print(f"Target sparsity: {masked_vae.get_sparsity():.4f}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Calculate loss
    total_loss, recon_loss, kl_loss = vae_loss_function(
        reconstruction, x, mu_img, logvar_img, mu_four, logvar_four, beta=1.0
    )
    print(f"\nLoss components:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL divergence: {kl_loss.item():.4f}")
    
    # Test sampling from prior
    print(f"\nTesting sampling from prior:")
    samples = masked_vae.sample(num_samples=5, device='cpu')
    print(f"Generated samples shape: {samples.shape}")
    
    # Evaluation mode
    print(f"\nInference mode (deterministic mask):")
    masked_vae.eval()
    with torch.no_grad():
        reconstruction, mask, mu_img, logvar_img, mu_four, logvar_four = masked_vae(x, x_fourier)
    print(f"Mask sparsity (actual): {mask.float().mean():.4f}")
    
    # Get learned mask
    learned_mask = masked_vae.get_learned_mask()
    print(f"Learned mask shape: {learned_mask.shape}")
    print(f"Learned mask min/max: {learned_mask.min():.4f} / {learned_mask.max():.4f}")
    
    # Test parameter count
    total_params = sum(p.numel() for p in masked_vae.parameters())
    trainable_params = sum(p.numel() for p in masked_vae.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
