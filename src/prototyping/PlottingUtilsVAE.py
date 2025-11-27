"""
Plotting Utilities for Masked Dual VAE + CNF

This module provides visualization functions for:
- Loss curves (multiple subplots)
- Reconstruction comparisons
- Mask visualizations (real, imag, magnitude)
- Generated samples from prior
- Latent space analysis

Usage:
    from PlottingUtilsVAE import plot_loss_curves, plot_reconstructions, plot_mask
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib.gridspec import GridSpec


def plot_loss_curves(metrics_tracker, save_path):
    """
    Plot comprehensive loss curves with multiple subplots.
    
    Creates 5 subplots:
    1. Total loss (train + val)
    2. Reconstruction loss
    3. KL divergence (stacked: image + Fourier)
    4. OT-Flow loss
    5. Mask sparsity
    
    Args:
        metrics_tracker: MetricsTracker object with training history
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    train_epochs = metrics_tracker.get_epochs('train')
    val_epochs = metrics_tracker.get_epochs('val')
    
    # 1. Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_epochs, metrics_tracker.get('train', 'loss'), 
             'b-', label='Train', linewidth=2)
    ax1.plot(val_epochs, metrics_tracker.get('val', 'loss'), 
             'r--', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Reconstruction Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(train_epochs, metrics_tracker.get('train', 'recon_loss'), 
             'b-', label='Train', linewidth=2)
    ax2.plot(val_epochs, metrics_tracker.get('val', 'recon_loss'), 
             'r--', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
    ax2.set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. KL Divergence (stacked)
    ax3 = fig.add_subplot(gs[1, 0])
    
    kl_img_train = metrics_tracker.get('train', 'kl_img')
    kl_four_train = metrics_tracker.get('train', 'kl_four')
    kl_img_val = metrics_tracker.get('val', 'kl_img')
    kl_four_val = metrics_tracker.get('val', 'kl_four')
    
    # Stacked area plot
    ax3.fill_between(train_epochs, 0, kl_img_train, 
                     alpha=0.5, label='Train Image KL', color='skyblue')
    ax3.fill_between(train_epochs, kl_img_train, 
                     [img + four for img, four in zip(kl_img_train, kl_four_train)],
                     alpha=0.5, label='Train Fourier KL', color='lightcoral')
    
    # Total KL as line
    ax3.plot(train_epochs, metrics_tracker.get('train', 'kl_loss'), 
             'b-', linewidth=2, label='Train Total')
    ax3.plot(val_epochs, metrics_tracker.get('val', 'kl_loss'), 
             'r--', linewidth=2, label='Val Total')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('KL Divergence', fontsize=12)
    ax3.set_title('KL Divergence (Image + Fourier)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. OT-Flow Loss
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(train_epochs, metrics_tracker.get('train', 'ot_loss'), 
             'b-', label='Train', linewidth=2)
    ax4.plot(val_epochs, metrics_tracker.get('val', 'ot_loss'), 
             'r--', label='Val', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('OT-Flow Loss', fontsize=12)
    ax4.set_title('OT-Flow Loss', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 5. Mask Sparsity
    ax5 = fig.add_subplot(gs[2, 0])
    
    train_sparsity = metrics_tracker.get('train', 'sparsity')
    val_sparsity = metrics_tracker.get('val', 'sparsity')
    
    ax5.plot(train_epochs, train_sparsity, 'b-', label='Train', linewidth=2)
    ax5.plot(val_epochs, val_sparsity, 'r--', label='Val', linewidth=2)
    
    # Target sparsity line (assuming 0.1)
    if len(train_sparsity) > 0:
        target = train_sparsity[-1]  # Use final value as approximate target
        ax5.axhline(y=target, color='g', linestyle=':', linewidth=2, 
                   label=f'Target ({target:.3f})')
    
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Sparsity (Fraction)', fontsize=12)
    ax5.set_title('Mask Sparsity', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 0.5])
    
    # 6. PSNR
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(train_epochs, metrics_tracker.get('train', 'psnr'), 
             'b-', label='Train', linewidth=2)
    ax6.plot(val_epochs, metrics_tracker.get('val', 'psnr'), 
             'r--', label='Val', linewidth=2)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('PSNR (dB)', fontsize=12)
    ax6.set_title('Peak Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress', fontsize=16, fontweight='bold', y=0.995)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstructions(original, reconstructed, mask, save_path, 
                        num_samples=16, fourier_data=None):
    """
    Plot comparison of original vs reconstructed images.
    
    Shows:
    - Row 1: Original images
    - Row 2: Reconstructed images
    - Row 3: Masked Fourier (magnitude)
    - Row 4: Absolute difference
    
    Args:
        original: Original images (N, 1, 28, 28) or (N, 784)
        reconstructed: Reconstructed images (N, 1, 28, 28) or (N, 784)
        mask: Binary mask (N, 1568)
        save_path: Path to save figure
        num_samples: Number of samples to show (must be perfect square)
        fourier_data: Optional Fourier data for better mask visualization
    """
    num_samples = min(num_samples, original.size(0))
    grid_size = int(np.sqrt(num_samples))
    num_samples = grid_size * grid_size  # Ensure perfect square
    
    # Ensure correct shape
    if original.dim() == 4:
        original = original[:num_samples].view(num_samples, 28, 28)
    else:
        original = original[:num_samples].view(num_samples, 28, 28)
    
    if reconstructed.dim() == 4:
        reconstructed = reconstructed[:num_samples].view(num_samples, 28, 28)
    else:
        reconstructed = reconstructed[:num_samples].view(num_samples, 28, 28)
    
    # Convert to numpy
    original_np = original.detach().cpu().numpy()
    reconstructed_np = reconstructed.detach().cpu().numpy()
    
    # Compute difference
    diff = np.abs(original_np - reconstructed_np)
    
    # Process mask for visualization
    mask_np = mask[:num_samples].detach().cpu().numpy()
    
    # Mask is (N, 1568) = (N, 28*28*2) for complex Fourier
    # Reshape to (N, 28, 28, 2) and take magnitude
    mask_vis = mask_np.reshape(num_samples, 28, 28, 2)
    mask_magnitude = np.sqrt(mask_vis[:, :, :, 0]**2 + mask_vis[:, :, :, 1]**2)
    
    # Create figure
    fig, axes = plt.subplots(4, grid_size, figsize=(grid_size*2, 8))
    
    for i in range(grid_size):
        idx = i * grid_size
        
        # Original
        axes[0, i].imshow(original_np[idx], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12, fontweight='bold')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed_np[idx], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12, fontweight='bold')
        
        # Mask (magnitude)
        axes[2, i].imshow(mask_magnitude[idx], cmap='viridis', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Mask (Magnitude)', fontsize=12, fontweight='bold')
        
        # Difference
        axes[3, i].imshow(diff[idx], cmap='hot', vmin=0, vmax=0.5)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_ylabel('Abs Difference', fontsize=12, fontweight='bold')
    
    plt.suptitle('Reconstruction Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mask(mask, save_dir, epoch, image_shape=(28, 28)):
    """
    Plot mask visualization (real, imaginary, magnitude components).
    
    For Fourier masks, creates 3 separate plots:
    - Real component
    - Imaginary component  
    - Magnitude (combined)
    
    Args:
        mask: LOUPEMask object
        save_dir: Directory to save figures
        epoch: Current epoch (for filename)
        image_shape: Shape of original image
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get mask for visualization
    mask_vis = mask.get_mask_for_visualization()
    
    if mask_vis['type'] == 'fourier':
        # Plot real component
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(mask_vis['real'], cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_title(f'Mask - Real Component (Epoch {epoch})', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        save_path = os.path.join(save_dir, f'mask_epoch_{epoch:03d}_real.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot imaginary component
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(mask_vis['imag'], cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_title(f'Mask - Imaginary Component (Epoch {epoch})', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        save_path = os.path.join(save_dir, f'mask_epoch_{epoch:03d}_imag.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot magnitude
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(mask_vis['magnitude'], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Mask - Magnitude (Epoch {epoch})', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        save_path = os.path.join(save_dir, f'mask_epoch_{epoch:03d}_magnitude.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    elif mask_vis['type'] == 'spatial':
        # Plot single spatial mask
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(mask_vis['mask'], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Mask (Epoch {epoch})', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        save_path = os.path.join(save_dir, f'mask_epoch_{epoch:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    else:
        # 1D mask - plot as line
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(mask_vis['mask'], linewidth=1)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Mask Value', fontsize=12)
        ax.set_title(f'Mask (Epoch {epoch})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])
        
        save_path = os.path.join(save_dir, f'mask_epoch_{epoch:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_generated_samples(model, fourier_condition, device, num_samples=16, save_path=None):
    """
    Generate and plot samples from prior distribution.
    
    Args:
        model: MaskDualVAECNF model
        fourier_condition: Fourier-transformed condition data from dataloader
                          Shape: (N, 1, 28, 28) complex tensor
                          This is the actual output from MNISTWithFourier
        device: Device to generate on
        num_samples: Number of samples to generate
        save_path: Path to save figure
    """
    model.eval()
    
    grid_size = int(np.sqrt(num_samples))
    num_samples = grid_size * grid_size
    
    with torch.no_grad():
        # Ensure we have enough condition samples
        if fourier_condition.size(0) < num_samples:
            # Repeat if not enough samples
            repeats = (num_samples // fourier_condition.size(0)) + 1
            fourier_condition = fourier_condition.repeat(repeats, 1, 1, 1)
        
        fourier_condition = fourier_condition[:num_samples].to(device)
        
        # Convert complex Fourier to real-imag format
        from src.prototyping.TrainingUtilsVAE import prepare_fourier_data
        condition_flat = prepare_fourier_data(fourier_condition)
        
        # Generate samples
        generated, _ = model.generate_from_prior(condition_flat, num_samples, device)
        
        # Convert to numpy
        if generated.dim() == 4:
            generated_np = generated.view(num_samples, 28, 28).cpu().numpy()
        else:
            generated_np = generated.view(num_samples, 28, 28).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, 
                            figsize=(grid_size*2, grid_size*2))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(generated_np[idx], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
    
    plt.suptitle('Generated Samples from Prior N(0,1)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_latent_space_2d(model, dataloader, device, save_path, 
                        num_samples=1000, use_image_encoder=True):
    """
    Plot 2D projection of latent space.
    
    Args:
        model: MaskDualVAECNF model
        dataloader: DataLoader with data
        device: Device to use
        save_path: Path to save figure
        num_samples: Number of samples to plot
        use_image_encoder: If True, plot image encoder latents; else Fourier
    """
    model.eval()
    
    latents = []
    labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            if len(latents) * x.size(0) >= num_samples:
                break
            
            x = x.to(device)
            
            # Prepare inputs
            if model.type_net_1 == "cnn":
                if x.dim() == 2:
                    x_spatial = x.view(-1, 1, 28, 28)
                else:
                    x_spatial = x
            else:
                x_spatial = x.view(x.size(0), -1)
            
            x_fourier = torch.fft.fft2(x.view(-1, 1, 28, 28)).view(x.size(0), -1)
            x_fourier = torch.cat([x_fourier.real, x_fourier.imag], dim=1)
            
            # Mask Fourier
            masked_fourier, _ = model.mask(x_fourier)
            
            # Encode
            if use_image_encoder:
                mu, _ = model.vae.image_encoder(x_spatial)
            else:
                mu, _ = model.vae.fourier_encoder(masked_fourier)
            
            latents.append(mu[:, :2].cpu().numpy())  # Take first 2 dims
            labels.append(y.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    scatter = ax.scatter(latents[:, 0], latents[:, 1], 
                        c=labels, cmap='tab10', alpha=0.6, s=10)
    
    encoder_name = "Image" if use_image_encoder else "Fourier"
    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.set_title(f'Latent Space ({encoder_name} Encoder)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Class')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_training_summary(metrics_tracker, save_path, final_metrics):
    """
    Create a summary figure with key information.
    
    Args:
        metrics_tracker: MetricsTracker object
        save_path: Path to save figure
        final_metrics: Dictionary with final validation metrics
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Text summary
    summary_text = f"""
    TRAINING SUMMARY
    ================
    
    Final Validation Metrics:
    -------------------------
    Total Loss:       {final_metrics['loss']:.4f}
    Reconstruction:   {final_metrics['recon_loss']:.4f}
    KL Divergence:    {final_metrics['kl_loss']:.4f}
      - Image KL:     {final_metrics['kl_img']:.4f}
      - Fourier KL:   {final_metrics['kl_four']:.4f}
    OT-Flow Loss:     {final_metrics['ot_loss']:.4f}
    PSNR:             {final_metrics['psnr']:.2f} dB
    Mask Sparsity:    {final_metrics['sparsity']:.4f}
    
    Training Progress:
    ------------------
    Total Epochs:     {len(metrics_tracker.get('train', 'loss'))}
    Best Val Loss:    {min(metrics_tracker.get('val', 'loss')):.4f}
    Best Val PSNR:    {max(metrics_tracker.get('val', 'psnr')):.2f} dB
    
    Improvement:
    ------------
    Loss:  {metrics_tracker.get('val', 'loss')[0]:.2f} → {final_metrics['loss']:.2f}
    PSNR:  {metrics_tracker.get('val', 'psnr')[0]:.2f} → {final_metrics['psnr']:.2f} dB
    """
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=fig.transFigure)
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
#                           TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing PlottingUtilsVAE")
    print("=" * 70)
    
    # Create dummy metrics tracker
    from src.prototyping.TrainingUtilsVAE import MetricsTracker
    
    print("\nCreating dummy metrics...")
    tracker = MetricsTracker()
    
    # Simulate training
    for epoch in range(20):
        train_metrics = {
            'loss': 100.0 - epoch * 3 + np.random.randn() * 2,
            'recon_loss': 80.0 - epoch * 2.5 + np.random.randn() * 1.5,
            'kl_loss': 15.0 - epoch * 0.3 + np.random.randn() * 0.5,
            'kl_img': 7.5 - epoch * 0.15 + np.random.randn() * 0.3,
            'kl_four': 7.5 - epoch * 0.15 + np.random.randn() * 0.3,
            'ot_loss': 5.0 - epoch * 0.2 + np.random.randn() * 0.3,
            'psnr': 15.0 + epoch * 0.4 + np.random.randn() * 0.5,
            'sparsity': 0.1 + np.random.randn() * 0.01
        }
        
        val_metrics = {k: v + np.random.randn() * 0.5 
                      for k, v in train_metrics.items()}
        
        tracker.update('train', train_metrics)
        tracker.update('val', val_metrics)
    
    print("✓ Metrics created")
    
    # Test loss curves
    print("\nTEST 1: Loss curves")
    print("-" * 70)
    plot_loss_curves(tracker, '/home/claude/test_loss_curves.png')
    print("✓ Loss curves saved")
    
    # Test reconstructions
    print("\nTEST 2: Reconstructions")
    print("-" * 70)
    original = torch.randn(16, 1, 28, 28)
    reconstructed = original + torch.randn(16, 1, 28, 28) * 0.1
    mask = torch.randint(0, 2, (16, 1568)).float()
    
    plot_reconstructions(original, reconstructed, mask, 
                        '/home/claude/test_reconstructions.png')
    print("✓ Reconstructions saved")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nGenerated test files:")
    print("  - /home/claude/test_loss_curves.png")
    print("  - /home/claude/test_reconstructions.png")
    print("\nThis file is ready to use in your training script!")
