"""
Complete Training Script for MaskedDualVAE

Handles train, validate, and test with VAE-specific losses.
Includes KL annealing and beta-VAE support.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from dual_vae import MaskedDualVAE, DualVAE, vae_loss_function


# ============================================================================
#                           DATA PREPROCESSING
# ============================================================================

def prepare_fourier_input(fourier_tensor):
    """
    Prepare pre-computed Fourier tensor for model input.
    
    Args:
        fourier_tensor: Pre-computed Fourier from MNISTWithFourier
                       Can be complex, real, or already flattened
    
    Returns:
        fourier_flat: (batch, H*W*2) tensor in float32
    """
    # If already flattened and correct dtype
    if fourier_tensor.ndim == 2 and fourier_tensor.dtype == torch.float32:
        return fourier_tensor
    
    # If complex, convert to real representation
    if torch.is_complex(fourier_tensor):
        fourier_tensor = torch.view_as_real(fourier_tensor)
    
    # Flatten if needed
    if fourier_tensor.ndim > 2:
        fourier_tensor = fourier_tensor.flatten(start_dim=1)
    
    # Keep as float32
    return fourier_tensor.float()


# ============================================================================
#                           KL ANNEALING
# ============================================================================

class KLAnnealer:
    """
    KL annealing scheduler for beta-VAE training.
    Gradually increases KL weight from 0 to target beta.
    """
    def __init__(self, beta_start=0.0, beta_target=1.0, n_epochs=10, 
                 anneal_strategy='linear'):
        """
        Args:
            beta_start: Starting beta value
            beta_target: Target beta value
            n_epochs: Number of epochs to anneal over
            anneal_strategy: 'linear', 'cyclical', or 'monotonic'
        """
        self.beta_start = beta_start
        self.beta_target = beta_target
        self.n_epochs = n_epochs
        self.strategy = anneal_strategy
        self.current_epoch = 0
    
    def get_beta(self):
        """Get current beta value based on annealing strategy."""
        if self.strategy == 'linear':
            if self.current_epoch >= self.n_epochs:
                return self.beta_target
            else:
                return self.beta_start + (self.beta_target - self.beta_start) * \
                       (self.current_epoch / self.n_epochs)
        
        elif self.strategy == 'cyclical':
            # Cyclical annealing (good for avoiding posterior collapse)
            cycle_length = self.n_epochs // 4  # 4 cycles
            position = (self.current_epoch % cycle_length) / cycle_length
            return self.beta_start + (self.beta_target - self.beta_start) * position
        
        elif self.strategy == 'monotonic':
            # Sigmoid-like monotonic increase
            if self.current_epoch >= self.n_epochs:
                return self.beta_target
            x = self.current_epoch / self.n_epochs
            sigmoid = 1.0 / (1.0 + np.exp(-12 * (x - 0.5)))
            return self.beta_start + (self.beta_target - self.beta_start) * sigmoid
        
        else:
            return self.beta_target
    
    def step(self):
        """Increment epoch counter."""
        self.current_epoch += 1


# ============================================================================
#                           TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, kl_annealer=None):
    """
    Train for one epoch with VAE loss.
    
    Args:
        model: MaskedDualVAE or DualVAE instance
        train_loader: Training DataLoader
        optimizer: PyTorch optimizer
        device: Device to train on
        epoch: Current epoch number
        kl_annealer: Optional KL annealing scheduler
    
    Returns:
        avg_total_loss: Average total loss
        avg_recon_loss: Average reconstruction loss
        avg_kl_loss: Average KL divergence
        avg_sparsity: Average mask sparsity (if using MaskedDualVAE)
        beta: Current beta value
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_sparsity = 0.0
    
    # Get current beta
    beta = kl_annealer.get_beta() if kl_annealer else 1.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train] β={beta:.4f}')
    for batch in pbar:
        # Unpack batch
        if len(batch) == 3:
            images, fourier, _ = batch
        elif len(batch) == 2:
            images, fourier = batch
        else:
            raise ValueError(f"Expected batch with 2 or 3 elements, got {len(batch)}")
        
        images = images.to(device)
        fourier_input = prepare_fourier_input(fourier).to(device)
        
        # Forward pass
        if isinstance(model, MaskedDualVAE):
            reconstruction, mask, mu_img, logvar_img, mu_four, logvar_four = \
                model(images, fourier_input)
            sparsity = mask.float().mean().item()
        else:
            reconstruction, mu_img, logvar_img, mu_four, logvar_four = \
                model(images, fourier_input)
            sparsity = 0.0
        
        # Compute VAE loss
        total_loss, recon_loss, kl_loss = vae_loss_function(
            reconstruction, images, mu_img, logvar_img, mu_four, logvar_four, beta=beta
        )
        
        # Normalize by batch size
        batch_size = images.size(0)
        total_loss = total_loss / batch_size
        recon_loss = recon_loss / batch_size
        kl_loss = kl_loss / batch_size
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        total_sparsity += sparsity
        
        # Update progress bar
        pbar.set_postfix({
            'total': f'{total_loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}',
            'sparse': f'{sparsity:.3f}'
        })
    
    n_batches = len(train_loader)
    return (total_loss_sum / n_batches, 
            recon_loss_sum / n_batches, 
            kl_loss_sum / n_batches,
            total_sparsity / n_batches,
            beta)


def validate(model, val_loader, device, epoch, beta=1.0):
    """
    Validate on validation set with VAE loss.
    
    Args:
        model: MaskedDualVAE or DualVAE instance
        val_loader: Validation DataLoader
        device: Device to validate on
        epoch: Current epoch number
        beta: Beta value for KL weight
    
    Returns:
        avg_total_loss: Average total loss
        avg_recon_loss: Average reconstruction loss
        avg_kl_loss: Average KL divergence
        avg_sparsity: Average mask sparsity (if using MaskedDualVAE)
    """
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_sparsity = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            # Unpack batch
            if len(batch) == 3:
                images, fourier, _ = batch
            elif len(batch) == 2:
                images, fourier = batch
            else:
                raise ValueError(f"Expected batch with 2 or 3 elements, got {len(batch)}")
            
            images = images.to(device)
            fourier_input = prepare_fourier_input(fourier).to(device)
            
            # Forward pass
            if isinstance(model, MaskedDualVAE):
                reconstruction, mask, mu_img, logvar_img, mu_four, logvar_four = \
                    model(images, fourier_input)
                sparsity = mask.float().mean().item()
            else:
                reconstruction, mu_img, logvar_img, mu_four, logvar_four = \
                    model(images, fourier_input)
                sparsity = 0.0
            
            # Compute VAE loss
            total_loss, recon_loss, kl_loss = vae_loss_function(
                reconstruction, images, mu_img, logvar_img, mu_four, logvar_four, beta=beta
            )
            
            # Normalize by batch size
            batch_size = images.size(0)
            total_loss = total_loss / batch_size
            recon_loss = recon_loss / batch_size
            kl_loss = kl_loss / batch_size
            
            # Track metrics
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            total_sparsity += sparsity
            
            # Update progress bar
            pbar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
    
    n_batches = len(val_loader)
    return (total_loss_sum / n_batches,
            recon_loss_sum / n_batches,
            kl_loss_sum / n_batches,
            total_sparsity / n_batches)


def test(model, test_loader, device, beta=1.0):
    """
    Test on test set with comprehensive metrics.
    
    Args:
        model: MaskedDualVAE or DualVAE instance
        test_loader: Test DataLoader
        device: Device to test on
        beta: Beta value for KL weight
    
    Returns:
        results: Dictionary with test metrics
    """
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_sparsity = 0.0
    total_psnr = 0.0
    num_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch in pbar:
            # Unpack batch
            if len(batch) == 3:
                images, fourier, _ = batch
            elif len(batch) == 2:
                images, fourier = batch
            else:
                raise ValueError(f"Expected batch with 2 or 3 elements, got {len(batch)}")
            
            images = images.to(device)
            fourier_input = prepare_fourier_input(fourier).to(device)
            batch_size = images.size(0)
            
            # Forward pass
            if isinstance(model, MaskedDualVAE):
                reconstruction, mask, mu_img, logvar_img, mu_four, logvar_four = \
                    model(images, fourier_input)
                sparsity = mask.float().mean().item()
            else:
                reconstruction, mu_img, logvar_img, mu_four, logvar_four = \
                    model(images, fourier_input)
                sparsity = 0.0
            
            # Compute VAE loss
            total_loss, recon_loss, kl_loss = vae_loss_function(
                reconstruction, images, mu_img, logvar_img, mu_four, logvar_four, beta=beta
            )
            
            # Normalize by batch size
            total_loss = total_loss / batch_size
            recon_loss = recon_loss / batch_size
            kl_loss = kl_loss / batch_size
            
            # Compute PSNR
            mse = F.mse_loss(reconstruction, images, reduction='none')
            mse = mse.view(batch_size, -1).mean(dim=1)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            # Track metrics
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            total_sparsity += sparsity
            total_psnr += psnr.sum().item()
            num_samples += batch_size
            
            pbar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'psnr': f'{psnr.mean().item():.2f}dB'
            })
    
    n_batches = len(test_loader)
    return {
        'test_total_loss': total_loss_sum / n_batches,
        'test_recon_loss': recon_loss_sum / n_batches,
        'test_kl_loss': kl_loss_sum / n_batches,
        'test_sparsity': total_sparsity / n_batches,
        'test_psnr': total_psnr / num_samples
    }


# ============================================================================
#                           VISUALIZATION
# ============================================================================

def plot_training_curves(train_losses, val_losses, train_recon, val_recon,
                         train_kl, val_kl, save_path='training_curves_vae.png'):
    """Plot training curves for VAE."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(train_losses, label='Train Total Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (Reconstruction + KL)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(train_recon, label='Train Recon Loss', linewidth=2)
    axes[0, 1].plot(val_recon, label='Val Recon Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss (MSE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[1, 0].plot(train_kl, label='Train KL Div', linewidth=2, color='orange')
    axes[1, 0].plot(val_kl, label='Val KL Div', linewidth=2, color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss components comparison
    epochs = range(len(train_losses))
    axes[1, 1].plot(epochs, train_recon, label='Recon Loss', linewidth=2)
    axes[1, 1].plot(epochs, train_kl, label='KL Div', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Value')
    axes[1, 1].set_title('Training Loss Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved training curves to {save_path}')


def visualize_reconstructions(model, data_loader, device, n_samples=8,
                              save_path='reconstructions_vae.png'):
    """Visualize original images, reconstructions, and samples from prior."""
    model.eval()
    
    # Get one batch
    batch = next(iter(data_loader))
    if len(batch) == 3:
        images, fourier, _ = batch
    else:
        images, fourier = batch
    
    images = images[:n_samples].to(device)
    fourier_input = prepare_fourier_input(fourier[:n_samples]).to(device)
    
    with torch.no_grad():
        # Reconstructions
        if isinstance(model, MaskedDualVAE):
            reconstruction, mask, _, _, _, _ = model(images, fourier_input)
        else:
            reconstruction, _, _, _, _ = model(images, fourier_input)
        
        # Samples from prior
        samples = model.sample(n_samples, device=device)
    
    # Move to CPU for plotting
    images = images.cpu()
    reconstruction = reconstruction.cpu()
    samples = samples.cpu()
    
    # Create figure
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstruction
        axes[1, i].imshow(reconstruction[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction', fontsize=10)
        
        # Sample from prior
        axes[2, i].imshow(samples[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Sample from Prior', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved reconstructions to {save_path}')


def visualize_learned_mask(model, save_path='learned_mask_vae.png', image_size=28):
    """Visualize learned mask for MaskedDualVAE."""
    if not isinstance(model, MaskedDualVAE):
        print("Model is not MaskedDualVAE, skipping mask visualization")
        return
    
    # Get learned mask
    learned_mask = model.get_learned_mask().cpu().numpy()
    
    # The mask is for Fourier domain: image_size × image_size × 2 (real + imaginary)
    # Total size: 28*28*2 = 1568
    expected_size = image_size * image_size * 2
    
    if learned_mask.shape[0] == expected_size:
        # Reshape to (height, width, 2) then take magnitude or just use first channel
        learned_mask_2d = learned_mask.reshape(image_size, image_size, 2)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Real component
        im0 = axes[0].imshow(learned_mask_2d[:, :, 0], cmap='viridis')
        axes[0].set_title('Learned Mask (Real Component)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # Imaginary component
        im1 = axes[1].imshow(learned_mask_2d[:, :, 1], cmap='viridis')
        axes[1].set_title('Learned Mask (Imaginary Component)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Magnitude (combined)
        magnitude = np.sqrt(learned_mask_2d[:, :, 0]**2 + learned_mask_2d[:, :, 1]**2)
        im2 = axes[2].imshow(magnitude, cmap='viridis')
        axes[2].set_title(f'Mask Magnitude (sparsity: {magnitude.mean():.3f})')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
    else:
        # Fallback: try to reshape as square if possible
        size = int(np.sqrt(learned_mask.shape[0]))
        if size * size == learned_mask.shape[0]:
            learned_mask_2d = learned_mask.reshape(size, size)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Probability map
            im0 = axes[0].imshow(learned_mask_2d, cmap='viridis')
            axes[0].set_title('Learned Mask Probabilities')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046)
            
            # Binary mask
            binary_mask = (learned_mask_2d > 0.5).astype(float)
            im1 = axes[1].imshow(binary_mask, cmap='binary')
            axes[1].set_title(f'Binary Mask (sparsity: {binary_mask.mean():.3f})')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
        else:
            # Can't reshape - just plot as 1D
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.plot(learned_mask)
            ax.set_title(f'Learned Mask (size: {learned_mask.shape[0]})')
            ax.set_xlabel('Index')
            ax.set_ylabel('Probability')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved learned mask to {save_path}')


# ============================================================================
#                           MAIN TRAINING LOOP
# ============================================================================

def train(model, train_loader, val_loader, test_loader,
          num_epochs=50, lr=1e-4, lr_mask=1e-3,
          device='cuda', checkpoint_dir='checkpoints_vae',
          use_kl_annealing=True, beta_target=1.0, anneal_epochs=10):
    """
    Complete VAE training pipeline.
    
    Args:
        model: MaskedDualVAE or DualVAE instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        num_epochs: Number of epochs to train
        lr: Learning rate for VAE
        lr_mask: Learning rate for mask (if MaskedDualVAE)
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_kl_annealing: Whether to use KL annealing
        beta_target: Target beta value for KL weight
        anneal_epochs: Number of epochs to anneal KL weight
    
    Returns:
        model: Trained model
        losses: Dictionary with all loss curves
        test_results: Dictionary with test metrics
    """
    # Setup
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    model = model.to(device)
    
    # Optimizer
    if isinstance(model, MaskedDualVAE):
        optimizer = optim.Adam([
            {'params': model.vae.parameters(), 'lr': lr},
            {'params': model.mask.parameters(), 'lr': lr_mask}
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # KL annealing
    kl_annealer = None
    if use_kl_annealing:
        kl_annealer = KLAnnealer(
            beta_start=0.0,
            beta_target=beta_target,
            n_epochs=anneal_epochs,
            anneal_strategy='linear'
        )
    
    # Tracking
    train_total_losses = []
    train_recon_losses = []
    train_kl_losses = []
    val_total_losses = []
    val_recon_losses = []
    val_kl_losses = []
    beta_history = []
    
    best_val_loss = float('inf')
    
    print(f'\nStarting VAE training for {num_epochs} epochs')
    print('=' * 70)
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_total, train_recon, train_kl, train_sparsity, beta = train_epoch(
            model, train_loader, optimizer, device, epoch, kl_annealer
        )
        train_total_losses.append(train_total)
        train_recon_losses.append(train_recon)
        train_kl_losses.append(train_kl)
        beta_history.append(beta)
        
        # Validate
        val_total, val_recon, val_kl, val_sparsity = validate(
            model, val_loader, device, epoch, beta
        )
        val_total_losses.append(val_total)
        val_recon_losses.append(val_recon)
        val_kl_losses.append(val_kl)
        
        # Update scheduler
        scheduler.step(val_total)
        
        # Step KL annealer
        if kl_annealer:
            kl_annealer.step()
        
        # Print summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train - Total: {train_total:.6f} | Recon: {train_recon:.6f} | KL: {train_kl:.6f}')
        print(f'  Val   - Total: {val_total:.6f} | Recon: {val_recon:.6f} | KL: {val_kl:.6f}')
        if isinstance(model, MaskedDualVAE):
            print(f'  Sparsity: Train={train_sparsity:.4f}, Val={val_sparsity:.4f}')
        print(f'  Beta: {beta:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
        print('-' * 70)
        
        # Save checkpoint if best
        if val_total < best_val_loss:
            best_val_loss = val_total
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': {
                    'train_total': train_total_losses,
                    'train_recon': train_recon_losses,
                    'train_kl': train_kl_losses,
                    'val_total': val_total_losses,
                    'val_recon': val_recon_losses,
                    'val_kl': val_kl_losses,
                    'beta_history': beta_history
                },
                'best_val_loss': best_val_loss
            }
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f'✓ Saved best model (val_total_loss: {val_total:.6f})')
        
        # Visualize periodically
        if epoch % 5 == 0:
            visualize_reconstructions(
                model, val_loader, device,
                save_path=checkpoint_dir / f'reconstructions_epoch_{epoch}.png'
            )
            if isinstance(model, MaskedDualVAE):
                visualize_learned_mask(
                    model,
                    save_path=checkpoint_dir / f'learned_mask_epoch_{epoch}.png'
                )
    
    # Plot training curves
    plot_training_curves(
        train_total_losses, val_total_losses,
        train_recon_losses, val_recon_losses,
        train_kl_losses, val_kl_losses,
        save_path=checkpoint_dir / 'training_curves_vae.png'
    )
    
    # Load best model
    print('\n' + '=' * 70)
    print('Training completed! Loading best model for testing...')
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_results = test(model, test_loader, device, beta=beta_target)
    
    print(f'\nTest Results (Best Model):')
    print(f'  Test Total Loss: {test_results["test_total_loss"]:.6f}')
    print(f'  Test Recon Loss: {test_results["test_recon_loss"]:.6f}')
    print(f'  Test KL Loss: {test_results["test_kl_loss"]:.6f}')
    print(f'  Test PSNR: {test_results["test_psnr"]:.2f} dB')
    if isinstance(model, MaskedDualVAE):
        print(f'  Test Sparsity: {test_results["test_sparsity"]:.4f}')
    print('=' * 70)
    
    # Final visualizations
    visualize_reconstructions(
        model, test_loader, device,
        save_path=checkpoint_dir / 'final_reconstructions.png'
    )
    if isinstance(model, MaskedDualVAE):
        visualize_learned_mask(
            model,
            save_path=checkpoint_dir / 'final_mask.png'
        )
    
    losses = {
        'train_total': train_total_losses,
        'train_recon': train_recon_losses,
        'train_kl': train_kl_losses,
        'val_total': val_total_losses,
        'val_recon': val_recon_losses,
        'val_kl': val_kl_losses,
        'beta_history': beta_history
    }
    
    return model, losses, test_results


# ============================================================================
#                           EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Training Script for Dual VAE")
    print("=" * 70)
    
    config = {
        'image_channels': 1,
        'image_size': 28,
        'fourier_input_dim': 28 * 28 * 2,
        'latent_dim': 64,
        'sparsity': 0.1,
        'slope': 5.0,
        'num_epochs': 50,
        'lr': 1e-4,
        'lr_mask': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': 'checkpoints_vae',
        'use_kl_annealing': True,
        'beta_target': 1.0,
        'anneal_epochs': 10
    }
    
    print('\nConfiguration:')
    for key, value in config.items():
        print(f'  {key}: {value}')
    
    print('\nTo start training:')
    print('1. Import your dataloaders')
    print('2. Initialize model (MaskedDualVAE or DualVAE)')
    print('3. Call train() function')
    print('\nExample:')
    print('  model = MaskedDualVAE(...)')
    print('  model, losses, results = train(model, train_loader, val_loader, test_loader, ...)')
