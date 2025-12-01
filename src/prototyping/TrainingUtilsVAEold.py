"""
Training Utilities for Masked Dual VAE + CNF

This module provides training and validation functions for the complete
Masked Dual VAE + CNF model. It handles:
- Training loop logic
- Validation loop logic
- KL annealing schedules
- Metrics tracking
- Checkpoint saving/loading

IMPORTANT: This version correctly uses the Fourier-transformed data
from the MNISTWithFourier dataloader instead of computing FFT during training.

Usage:
    from TrainingUtilsVAE import train_epoch, validate_epoch, KLAnnealer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import time


class KLAnnealer:
    """
    KL divergence weight annealer.
    
    Gradually increases β from 0 to target_beta over anneal_epochs.
    This prevents posterior collapse in VAE training.
    
    Usage:
        annealer = KLAnnealer(target_beta=1.0, anneal_epochs=10)
        
        for epoch in range(num_epochs):
            beta = annealer.get_beta(epoch)
            # use beta in loss computation
    """
    
    def __init__(self, target_beta=1.0, anneal_epochs=10, start_epoch=0):
        """
        Args:
            target_beta: Final β value (typically 1.0)
            anneal_epochs: Number of epochs to reach target_beta
            start_epoch: Starting epoch (for resuming training)
        """
        self.target_beta = target_beta
        self.anneal_epochs = anneal_epochs
        self.start_epoch = start_epoch
        
    def get_beta(self, epoch):
        """
        Get β value for current epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            beta: KL weight for current epoch
        """
        if epoch < self.start_epoch:
            return 0.0
        
        relative_epoch = epoch - self.start_epoch
        
        if relative_epoch >= self.anneal_epochs:
            return self.target_beta
        else:
            # Linear annealing
            return self.target_beta * (relative_epoch / self.anneal_epochs)


def compute_psnr(img1, img2, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum pixel value (1.0 for normalized images)
        
    Returns:
        psnr: PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


def prepare_fourier_data(fourier_images):
    """
    Convert complex Fourier images to real-valued concatenated format.
    
    The MNISTWithFourier dataloader returns complex tensors from fft2.
    We need to convert them to [real, imag] concatenated format.
    
    Args:
        fourier_images: Complex tensor from dataloader (batch, 1, 28, 28)
        
    Returns:
        fourier_flat: Flattened real-imag format (batch, 1568)
                      where 1568 = 28*28*2 (real + imag)
    """
    # fourier_images is complex, shape (batch, 1, 28, 28)
    batch_size = fourier_images.size(0)
    
    # Flatten spatial dimensions
    fourier_flat = fourier_images.view(batch_size, -1)  # (batch, 784)
    
    # Separate real and imaginary parts and concatenate
    fourier_real_imag = torch.cat([fourier_flat.real, fourier_flat.imag], dim=1)  # (batch, 1568)
    
    return fourier_real_imag


def train_epoch(model, train_loader, optimizer_vae, optimizer_cnf, optimizer_mask,
                beta, device, epoch, logger=None):
    """
    Train for one epoch.
    
    Args:
        model: MaskDualVAECNF model
        train_loader: DataLoader for training data (returns img, fourier_img, label)
        optimizer_vae: Optimizer for VAE parameters
        optimizer_cnf: Optimizer for CNF parameters
        optimizer_mask: Optimizer for mask parameters
        beta: Current β value for KL weight
        device: Device to train on
        epoch: Current epoch number
        logger: Optional logger for detailed logging
        
    Returns:
        metrics: Dictionary with average metrics over epoch
    """
    model.train()
    
    # Metrics tracking
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_kl_img = 0.0
    total_kl_four = 0.0
    total_ot_loss = 0.0
    total_psnr = 0.0
    total_sparsity = 0.0
    
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (images, fourier_images, labels) in enumerate(pbar):
        # Move to device
        images = images.to(device)
        fourier_images = fourier_images.to(device)
        
        # Prepare inputs based on model type
        if model.type_net_1 == "cnn":
            # CNN expects (batch, 1, 28, 28)
            if images.dim() == 2:
                x_spatial = images.view(-1, 1, 28, 28)
            else:
                x_spatial = images
        else:
            # Linear expects (batch, 784)
            x_spatial = images.view(images.size(0), -1)
        
        # Convert Fourier data to real-imag format
        # fourier_images is already FFT'd by the dataloader!
        x_fourier = prepare_fourier_data(fourier_images)
        
        # Forward pass
        reconstruction, Jc, costs, mu_combined, musqrd_combined, mask, \
            mu_img, logvar_img, mu_four, logvar_four = \
            model(x_spatial, x_fourier, return_all=True)
        
        # Compute losses
        batch_size = images.size(0)
        
        # Reconstruction loss
        if reconstruction.dim() == 4:
            target = x_spatial
        else:
            target = x_spatial.view(-1, 784)
        recon_loss = F.mse_loss(reconstruction.view(batch_size, -1), 
                                target.view(batch_size, -1), 
                                reduction='sum') / batch_size
        
        # KL divergence
        kl_img = -0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()) / batch_size
        kl_four = -0.5 * torch.sum(1 + logvar_four - mu_four.pow(2) - logvar_four.exp()) / batch_size
        kl_loss = kl_img + kl_four
        
        # OT-Flow loss
        ot_loss = Jc
        
        # Total loss
        loss = recon_loss + beta * kl_loss + ot_loss
        
        # Backward pass
        optimizer_vae.zero_grad()
        optimizer_cnf.zero_grad()
        optimizer_mask.zero_grad()
        
        loss.backward()
        
        optimizer_vae.step()
        optimizer_cnf.step()
        optimizer_mask.step()
        
        # Compute metrics
        with torch.no_grad():
            psnr = compute_psnr(reconstruction.view(batch_size, -1), 
                               target.view(batch_size, -1))
            sparsity = model.mask.get_sparsity()
        
        # Update tracking
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_kl_img += kl_img.item()
        total_kl_four += kl_four.item()
        total_ot_loss += ot_loss.item()
        total_psnr += psnr
        total_sparsity += sparsity
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.2f}',
            'Recon': f'{recon_loss.item():.2f}',
            'KL': f'{kl_loss.item():.2f}',
            'OT': f'{ot_loss.item():.2f}',
            'PSNR': f'{psnr:.1f}',
            'Spar': f'{sparsity:.3f}',
            'β': f'{beta:.4f}'
        })
    
    # Average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'kl_img': total_kl_img / num_batches,
        'kl_four': total_kl_four / num_batches,
        'ot_loss': total_ot_loss / num_batches,
        'psnr': total_psnr / num_batches,
        'sparsity': total_sparsity / num_batches,
        'beta': beta
    }
    
    return metrics


def validate_epoch(model, val_loader, beta, device, epoch, logger=None):
    """
    Validate for one epoch.
    
    Args:
        model: MaskDualVAECNF model
        val_loader: DataLoader for validation data (returns img, fourier_img, label)
        beta: Current β value for KL weight
        device: Device to validate on
        epoch: Current epoch number
        logger: Optional logger for detailed logging
        
    Returns:
        metrics: Dictionary with average metrics over epoch
        samples: Tuple of (x, reconstruction, mask) for visualization
    """
    model.eval()
    
    # Metrics tracking
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_kl_img = 0.0
    total_kl_four = 0.0
    total_ot_loss = 0.0
    total_psnr = 0.0
    total_sparsity = 0.0
    
    num_batches = 0
    
    # Save first batch for visualization
    saved_samples = None
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    
    with torch.no_grad():
        for batch_idx, (images, fourier_images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(device)
            fourier_images = fourier_images.to(device)
            
            # Prepare inputs based on model type
            if model.type_net_1 == "cnn":
                if images.dim() == 2:
                    x_spatial = images.view(-1, 1, 28, 28)
                else:
                    x_spatial = images
            else:
                x_spatial = images.view(images.size(0), -1)
            
            # Convert Fourier data to real-imag format
            # fourier_images is already FFT'd by the dataloader!
            x_fourier = prepare_fourier_data(fourier_images)
            
            # Forward pass (deterministic in eval mode)
            reconstruction, Jc, costs, mu_combined, musqrd_combined, mask, \
                mu_img, logvar_img, mu_four, logvar_four = \
                model(x_spatial, x_fourier, return_all=True)
            
            # Compute losses
            batch_size = images.size(0)
            
            if reconstruction.dim() == 4:
                target = x_spatial
            else:
                target = x_spatial.view(-1, 784)
            
            recon_loss = F.mse_loss(reconstruction.view(batch_size, -1), 
                                    target.view(batch_size, -1), 
                                    reduction='sum') / batch_size
            
            kl_img = -0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()) / batch_size
            kl_four = -0.5 * torch.sum(1 + logvar_four - mu_four.pow(2) - logvar_four.exp()) / batch_size
            kl_loss = kl_img + kl_four
            
            ot_loss = Jc
            loss = recon_loss + beta * kl_loss + ot_loss
            
            # Compute metrics
            psnr = compute_psnr(reconstruction.view(batch_size, -1), 
                               target.view(batch_size, -1))
            sparsity = model.mask.get_sparsity()
            
            # Update tracking
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_kl_img += kl_img.item()
            total_kl_four += kl_four.item()
            total_ot_loss += ot_loss.item()
            total_psnr += psnr
            total_sparsity += sparsity
            num_batches += 1
            
            # Save first batch for visualization
            if saved_samples is None:
                saved_samples = (
                    images[:16].cpu(),
                    reconstruction[:16].cpu(),
                    mask[:16].cpu()
                )
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.2f}',
                'Recon': f'{recon_loss.item():.2f}',
                'KL': f'{kl_loss.item():.2f}',
                'OT': f'{ot_loss.item():.2f}',
                'PSNR': f'{psnr:.1f}',
                'Spar': f'{sparsity:.3f}'
            })
    
    # Average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'kl_img': total_kl_img / num_batches,
        'kl_four': total_kl_four / num_batches,
        'ot_loss': total_ot_loss / num_batches,
        'psnr': total_psnr / num_batches,
        'sparsity': total_sparsity / num_batches,
        'beta': beta
    }
    
    return metrics, saved_samples


def save_checkpoint(model, optimizer_vae, optimizer_cnf, optimizer_mask,
                    epoch, metrics, save_path):
    """
    Save training checkpoint.
    
    Args:
        model: MaskDualVAECNF model
        optimizer_vae: VAE optimizer
        optimizer_cnf: CNF optimizer
        optimizer_mask: Mask optimizer
        epoch: Current epoch
        metrics: Metrics dictionary
        save_path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_vae_state_dict': optimizer_vae.state_dict(),
        'optimizer_cnf_state_dict': optimizer_cnf.state_dict(),
        'optimizer_mask_state_dict': optimizer_mask.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer_vae=None, 
                    optimizer_cnf=None, optimizer_mask=None, device='cuda'):
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: MaskDualVAECNF model to load into
        optimizer_vae: Optional VAE optimizer to load state
        optimizer_cnf: Optional CNF optimizer to load state
        optimizer_mask: Optional mask optimizer to load state
        device: Device to load to
        
    Returns:
        epoch: Epoch number from checkpoint
        metrics: Metrics from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer_vae is not None and 'optimizer_vae_state_dict' in checkpoint:
        optimizer_vae.load_state_dict(checkpoint['optimizer_vae_state_dict'])
    
    if optimizer_cnf is not None and 'optimizer_cnf_state_dict' in checkpoint:
        optimizer_cnf.load_state_dict(checkpoint['optimizer_cnf_state_dict'])
    
    if optimizer_mask is not None and 'optimizer_mask_state_dict' in checkpoint:
        optimizer_mask.load_state_dict(checkpoint['optimizer_mask_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics


class MetricsTracker:
    """
    Track metrics across training.
    
    Usage:
        tracker = MetricsTracker()
        
        for epoch in range(num_epochs):
            train_metrics = train_epoch(...)
            val_metrics = validate_epoch(...)
            
            tracker.update('train', train_metrics, epoch)
            tracker.update('val', val_metrics, epoch)
            
        tracker.save('metrics.npz')
    """
    
    def __init__(self):
        self.train_metrics = {}
        self.val_metrics = {}
        self.train_epochs = []
        self.val_epochs = []
        
    def update(self, split, metrics, epoch=None):
        """
        Update metrics for a split (train/val).
        
        Args:
            split: 'train' or 'val'
            metrics: Dictionary of metrics
            epoch: Epoch number (optional, for tracking)
        """
        if split == 'train':
            target = self.train_metrics
            if epoch is not None:
                self.train_epochs.append(epoch)
        elif split == 'val':
            target = self.val_metrics
            if epoch is not None:
                self.val_epochs.append(epoch)
        else:
            raise ValueError(f"Invalid split: {split}")
        
        for key, value in metrics.items():
            if key not in target:
                target[key] = []
            target[key].append(value)
    
    def get(self, split, metric):
        """Get metric history for a split."""
        if split == 'train':
            return self.train_metrics.get(metric, [])
        elif split == 'val':
            return self.val_metrics.get(metric, [])
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def get_epochs(self, split):
        """Get epoch numbers for a split."""
        if split == 'train':
            return self.train_epochs if self.train_epochs else list(range(1, len(self.get('train', 'loss')) + 1))
        elif split == 'val':
            return self.val_epochs if self.val_epochs else list(range(1, len(self.get('val', 'loss')) + 1))
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def save(self, path):
        """Save metrics to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        np.savez(path,
                 train_metrics=self.train_metrics,
                 val_metrics=self.val_metrics,
                 train_epochs=self.train_epochs,
                 val_epochs=self.val_epochs)
    
    def load(self, path):
        """Load metrics from file."""
        data = np.load(path, allow_pickle=True)
        self.train_metrics = data['train_metrics'].item()
        self.val_metrics = data['val_metrics'].item()
        self.train_epochs = data.get('train_epochs', []).tolist() if 'train_epochs' in data else []
        self.val_epochs = data.get('val_epochs', []).tolist() if 'val_epochs' in data else []


# ============================================================================
#                           TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing TrainingUtilsVAE (CORRECTED VERSION)")
    print("=" * 70)
    
    # Test KL Annealer
    print("\nTEST 1: KL Annealer")
    print("-" * 70)
    annealer = KLAnnealer(target_beta=1.0, anneal_epochs=10)
    
    for epoch in [0, 5, 10, 15, 20]:
        beta = annealer.get_beta(epoch)
        print(f"Epoch {epoch}: β = {beta:.4f}")
    
    # Test PSNR computation
    print("\nTEST 2: PSNR Computation")
    print("-" * 70)
    img1 = torch.randn(4, 1, 28, 28)
    img2 = img1 + torch.randn(4, 1, 28, 28) * 0.1
    
    psnr = compute_psnr(img1, img2)
    print(f"PSNR: {psnr:.2f} dB")
    
    # Test Fourier data preparation
    print("\nTEST 3: Fourier Data Preparation")
    print("-" * 70)
    # Simulate FFT'd data from dataloader
    img = torch.randn(4, 1, 28, 28)
    fourier_complex = torch.fft.fft2(img)
    
    fourier_flat = prepare_fourier_data(fourier_complex)
    print(f"Input shape: {fourier_complex.shape}")
    print(f"Output shape: {fourier_flat.shape}")
    print(f"Expected: (4, 1568) for 28*28*2")
    assert fourier_flat.shape == (4, 1568), "Shape mismatch!"
    print("✓ Shape correct!")
    
    # Test MetricsTracker
    print("\nTEST 4: MetricsTracker")
    print("-" * 70)
    tracker = MetricsTracker()
    
    for epoch in range(3):
        train_metrics = {
            'loss': 100.0 - epoch * 10,
            'psnr': 15.0 + epoch * 2
        }
        val_metrics = {
            'loss': 110.0 - epoch * 10,
            'psnr': 14.0 + epoch * 2
        }
        
        tracker.update('train', train_metrics)
        tracker.update('val', val_metrics)
    
    print(f"Train losses: {tracker.get('train', 'loss')}")
    print(f"Val PSNRs: {tracker.get('val', 'psnr')}")
    
    print("\n" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nIMPORTANT: This version uses Fourier data from the dataloader")
    print("instead of computing FFT during training!")
