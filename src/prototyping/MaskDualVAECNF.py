"""
Masked Dual VAE with Conditional Normalizing Flow (FIXED VERSION)
Combines:
- LOUPE-style learnable mask
- Dual VAE (probabilistic encoders)
- OT-Flow (Conditional Normalizing Flow)

This is the main model class that replaces MaskAutoEncCNF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils as utils
from src.prototyping.CustomPlotter import *
from src.OTFlowProblem import *
from src.prototyping.LossTerms import *
from src.mmd import *


class MaskDualVAECNF(nn.Module):
    """
    Complete model combining:
    1. Probabilistic Mask (LOUPE-style) - learns optimal k-space sampling
    2. Dual VAE - probabilistic encoders for spatial and Fourier domains
    3. Conditional Normalizing Flow (CNF) - learns smooth transport between latent distributions
    
    Architecture:
        Input (x, condition)
            ↓
        LOUPE Mask → condition_masked
            ↓
        VAE Encoders → (μ, σ²) for both domains
            ↓
        Reparameterization → z_img, z_four
            ↓
        Concatenate → z_combined
            ↓
        Decoder → reconstruction
            ↓
        Normalize → feed to CNF
            ↓
        OT-Flow → learn transport
    """
    
    def __init__(self, original_dim, encoding_dim, mask, vae, Phi, nt, eps):
        """
        Args:
            original_dim: Dimension of original images (e.g., 784)
            encoding_dim: Latent dimension for each encoder (combined will be 2*encoding_dim)
            mask: LOUPE-style probabilistic mask
            vae: Dual VAE model
            Phi: OT-Flow network (conditional normalizing flow)
            nt: Number of time steps for OT integration
            eps: Small epsilon for numerical stability
        """
        super(MaskDualVAECNF, self).__init__()
        
        self.original_dim = original_dim
        self.encoding_dim = encoding_dim
        self.mask = mask
        self.vae = vae
        self.Phi = Phi
        self.nt = nt
        self.eps = eps
        
        # Track encoder types for data preprocessing
        # This is for backward compatibility with existing data loaders
        self.type_net_1 = "cnn"  # Image encoder
        self.type_net_2 = "linear"  # Fourier encoder
    
    def forward(self, x, condition, return_all=False):
        """
        Forward pass through the complete model.
        
        Args:
            x: Original images
               - Shape: (batch, 1, 28, 28) if CNN
               - Shape: (batch, 784) if linear
            condition: Fourier transform
               - Shape: (batch, fourier_dim) - flattened
            return_all: If True, return all intermediate values
        
        Returns:
            x_decoded: Reconstructed images (batch, 1, 28, 28)
            Jc: OT-Flow loss
            costs: Individual OT cost terms
            mu_combined: Mean of combined latent (for normalization)
            musqrd_combined: Mean of squared combined latent
            mask: Binary mask used
            
            (if return_all=True):
            mu_img, logvar_img: Image encoder VAE parameters
            mu_four, logvar_four: Fourier encoder VAE parameters
        """
        # STEP 1: Apply learned mask to Fourier domain
        masked_condition, mask = self.mask(condition)
        
        # STEP 2 & 3: Encode both domains (VAE encoders)
        # This returns (μ, σ²) for each encoder
        reconstruction, mu_img, logvar_img, mu_four, logvar_four, components = \
            self.vae(x, masked_condition, return_components=True)
        
        # Get the sampled latent codes
        z_img = components['z_img']      # (batch, encoding_dim)
        z_four = components['z_four']    # (batch, encoding_dim)
        z_combined = components['z_combined']  # (batch, 2*encoding_dim)
        
        # STEP 4: Compute statistics for normalization
        # These are used by OT-Flow and also tracked across epochs
        mu_combined = torch.mean(z_combined, dim=0, keepdims=True)
        musqrd_combined = torch.mean(z_combined ** 2, dim=0, keepdims=True)
        
        # STEP 5: Normalize for CNF
        # OT-Flow expects normalized inputs
        std_combined = torch.sqrt(torch.abs(mu_combined ** 2 - musqrd_combined) + self.eps)
        normalized_combined = (z_combined - mu_combined) / (std_combined + self.eps)
        
        # Split into source (x0) and condition (y) for CNF
        x0, y = normalized_combined.chunk(2, dim=1)  # Each: (batch, encoding_dim)
        
        # STEP 6: Pass through CNF (OT-Flow)
        # Learn optimal transport from x0 to N(0,1) conditioned on y
        Jc, costs = OTFlowProblem(x0, y, self.Phi, [0, 1], nt=self.nt, 
                                   stepper="rk4", alph=self.Phi.alph)
        
        if return_all:
            return (reconstruction, Jc, costs, mu_combined, musqrd_combined, mask,
                    mu_img, logvar_img, mu_four, logvar_four)
        else:
            return reconstruction, Jc, costs, mu_combined, musqrd_combined, mask
    
    def generate_from_prior(self, condition, num_samples, device):
        """
        Generate images by sampling from prior and flowing through CNF.
        
        This is used during validation/testing to generate samples.
        
        Args:
            condition: Fourier condition (num_samples, fourier_dim)
            num_samples: Number of samples to generate
            device: Device to generate on
        
        Returns:
            generated_images: Generated images (num_samples, 1, 28, 28)
            mask: Mask used (for visualization)
        """
        with torch.no_grad():
            # Apply mask to condition
            masked_condition, mask = self.mask(condition)
            
            # Encode condition
            mu_four, logvar_four = self.vae.fourier_encoder(masked_condition)
            
            # Use mean for condition (deterministic)
            z_four = mu_four
            
            # Normalize condition
            y = (z_four - self.vae.mu[:, self.encoding_dim:]) / \
                (self.vae.std[:, self.encoding_dim:] + self.eps)
            
            # Sample from standard normal for source
            z0 = torch.randn(num_samples, self.encoding_dim, device=device)
            
            # Flow backward through CNF (from noise to data)
            # [1.0, 0.0] means we integrate from t=1 (noise) to t=0 (data)
            from src.OTFlowProblem import integrate
            z_img_generated = integrate(z0, y, self.Phi, [1.0, 0.0], 
                                       nt=self.nt, stepper="rk4", alph=self.Phi.alph)
            
            # IMPORTANT: integrate returns (batch, dx+3) where last 3 dims are log-det, cost, HJB
            # We only need the first encoding_dim dimensions (the actual latent code)
            z_img_generated = z_img_generated[:, :self.encoding_dim]
            
            # Denormalize
            z_img_generated = z_img_generated * (self.vae.std[:, :self.encoding_dim] + self.eps) + \
                             self.vae.mu[:, :self.encoding_dim]
            z_four_denorm = z_four * (self.vae.std[:, self.encoding_dim:] + self.eps) + \
                           self.vae.mu[:, self.encoding_dim:]
            
            # Concatenate and decode
            z_combined = torch.cat([z_img_generated, z_four_denorm], dim=1)
            generated_images = self.vae.decoder(z_combined)
            
            return generated_images, mask


# ============================================================================
#                           LOSS FUNCTIONS
# ============================================================================

def compute_vae_loss(reconstruction, target, mu_img, logvar_img, 
                     mu_four, logvar_four, beta=1.0):
    """
    Compute VAE loss: reconstruction + KL divergence.
    
    Args:
        reconstruction: Reconstructed images
        target: Original images
        mu_img: Mean from image encoder
        logvar_img: Log-variance from image encoder
        mu_four: Mean from Fourier encoder
        logvar_four: Log-variance from Fourier encoder
        beta: Weight for KL divergence (β-VAE)
    
    Returns:
        total_loss: Combined VAE loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    # Reconstruction loss (MSE)
    batch_size = reconstruction.size(0)
    recon_loss = F.mse_loss(reconstruction, target, reduction='sum') / batch_size
    
    # KL divergence for both encoders
    kl_img = -0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()) / batch_size
    kl_four = -0.5 * torch.sum(1 + logvar_four - mu_four.pow(2) - logvar_four.exp()) / batch_size
    
    kl_loss = kl_img + kl_four
    
    # Total VAE loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def compute_total_loss(reconstruction, target, mu_img, logvar_img, 
                      mu_four, logvar_four, ot_flow_loss, beta=1.0):
    """
    Compute total training loss: VAE loss + OT-Flow loss.
    
    Args:
        reconstruction: Reconstructed images
        target: Original images
        mu_img, logvar_img: Image encoder VAE parameters
        mu_four, logvar_four: Fourier encoder VAE parameters
        ot_flow_loss: Loss from OT-Flow
        beta: Weight for KL divergence
    
    Returns:
        total_loss: Combined loss for backpropagation
        loss_dict: Dictionary with individual loss components
    """
    # VAE loss
    vae_loss, recon_loss, kl_loss = compute_vae_loss(
        reconstruction, target, mu_img, logvar_img, mu_four, logvar_four, beta
    )
    
    # Total loss
    total_loss = vae_loss + ot_flow_loss
    
    # Package loss components
    loss_dict = {
        'total': total_loss.item(),
        'vae': vae_loss.item(),
        'reconstruction': recon_loss.item(),
        'kl': kl_loss.item(),
        'ot_flow': ot_flow_loss.item()
    }
    
    return total_loss, loss_dict


# ============================================================================
#                           TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing MaskDualVAECNF")
    print("=" * 70)
    
    # This requires importing actual components
    # For now, we'll just verify the class structure
    
    print("\nModel class structure:")
    print(f"  Class name: MaskDualVAECNF")
    print(f"  Methods:")
    for method_name in dir(MaskDualVAECNF):
        if not method_name.startswith('_') and callable(getattr(MaskDualVAECNF, method_name)):
            print(f"    - {method_name}")
    
    print("\nLoss functions available:")
    print("  - compute_vae_loss")
    print("  - compute_total_loss")
    
    print("\n" + "=" * 70)
    print("Model definition complete! ✓")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Place this file in: src/prototyping/MaskDualVAECNF.py")
    print("2. Update imports in training script")
    print("3. Initialize model with mask, vae, and Phi")
