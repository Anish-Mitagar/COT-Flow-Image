"""
Train Masked Dual VAE + CNF for MRI Reconstruction

This script trains the complete model:
- LOUPE-style learnable mask
- Dual VAE (probabilistic encoders)
- Conditional Normalizing Flow (OT-Flow)

Usage:
    python train_masked_vae_cnf.py --data mnist --num_epochs 200

For full options:
    python train_masked_vae_cnf.py --help
"""

import argparse
import os
import time
import datetime
import torch
import torch.optim as optim
import numpy as np
import sys

# Add repository root to path
sys.path.insert(0, os.path.abspath('.'))

# Import project modules
from datasets.mnist import getLoader
from src.prototyping.LOUPEMask import LOUPEMask
from src.prototyping.DualVAE import DualVAE
from src.prototyping.MaskDualVAECNF import MaskDualVAECNF
from src.prototyping.TrainingUtilsVAE import (
    train_epoch, validate_epoch, KLAnnealer, 
    MetricsTracker, save_checkpoint, load_checkpoint
)
from src.prototyping.PlottingUtilsVAE import (
    plot_loss_curves, plot_reconstructions, plot_mask,
    plot_generated_samples, create_training_summary
)
from src.Phi import Phi
import config

cf = config.getconfig()


def makedirs(dirname):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# ============================================================================
#                           ARGUMENT PARSING
# ============================================================================

def get_args():
    """Parse command line arguments."""
    
    # Default values based on GPU availability
    if cf.gpu:
        def_batch = 800
        def_epochs = 200
        def_val_freq = 10
        def_viz_freq = 10
    else:
        def_batch = 200
        def_epochs = 50
        def_val_freq = 5
        def_viz_freq = 5
    
    parser = argparse.ArgumentParser('Train Masked Dual VAE + CNF')
    
    # Data arguments
    parser.add_argument('--data', choices=['mnist'], type=str, default='mnist',
                       help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=def_batch,
                       help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=def_batch,
                       help='Validation batch size')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent dimension for each encoder')
    parser.add_argument('--sparsity', type=float, default=0.1,
                       help='Target mask sparsity (0.1 = 10%)')
    parser.add_argument('--slope', type=float, default=5.0,
                       help='Slope parameter for LOUPE sigmoid')
    
    # CNF arguments
    parser.add_argument('--nt', type=int, default=8,
                       help='Number of time steps for CNF')
    parser.add_argument('--nt_val', type=int, default=16,
                       help='Number of time steps for validation')
    parser.add_argument('--alph', type=str, default='1.0,80.0,500.0',
                       help='Alpha parameters for OT-Flow')
    parser.add_argument('--m', type=int, default=128,
                       help='Hidden dimension for CNF')
    parser.add_argument('--nTh', type=int, default=2,
                       help='Number of threads for CNF')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=def_epochs,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for VAE')
    parser.add_argument('--lr_cnf', type=float, default=1e-4,
                       help='Learning rate for CNF')
    parser.add_argument('--lr_mask', type=float, default=5e-4,
                       help='Learning rate for mask (5x VAE LR)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for optimizers')
    parser.add_argument('--eps', type=float, default=1e-6,
                       help='Epsilon for numerical stability')
    
    # KL annealing arguments
    parser.add_argument('--use_kl_annealing', action='store_true',
                       help='Use KL annealing')
    parser.add_argument('--beta_target', type=float, default=1.0,
                       help='Target beta for KL weight')
    parser.add_argument('--anneal_epochs', type=int, default=10,
                       help='Number of epochs to anneal KL weight')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save', type=str, default='experiments/vae_cnf/large',
                       help='Directory to save checkpoints and figures')
    
    # Logging arguments
    parser.add_argument('--val_freq', type=int, default=def_val_freq,
                       help='Validation frequency (epochs)')
    parser.add_argument('--viz_freq', type=int, default=def_viz_freq,
                       help='Visualization frequency (epochs)')
    parser.add_argument('--save_freq', type=int, default=20,
                       help='Checkpoint save frequency (epochs)')
    
    # Device arguments
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Parse alpha
    args.alph = [float(item) for item in args.alph.split(',')]
    
    return args


# ============================================================================
#                           MAIN TRAINING
# ============================================================================

def main():
    # Parse arguments
    args = get_args()
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create save directories
    start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.save = os.path.join(args.save, start_time)
    
    makedirs(args.save)
    makedirs(os.path.join(args.save, 'checkpoints'))
    makedirs(os.path.join(args.save, 'figs'))
    makedirs(os.path.join(args.save, 'logs'))
    
    # Setup logging
    log_path = os.path.join(args.save, 'logs', 'training.log')
    log_file = open(log_path, 'w')
    
    def log(message):
        """Log message to both console and file."""
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    # Print configuration
    log("=" * 80)
    log(f"Training Masked Dual VAE + CNF")
    log("=" * 80)
    log(f"\nStart time: {start_time}")
    log(f"\nConfiguration:")
    log(f"  Device: {device}")
    log(f"  Dataset: {args.data}")
    log(f"  Batch size: {args.batch_size}")
    log(f"  Epochs: {args.num_epochs}")
    log(f"  Latent dim: {args.latent_dim}")
    log(f"  Sparsity: {args.sparsity}")
    log(f"  VAE LR: {args.lr}")
    log(f"  Mask LR: {args.lr_mask}")
    log(f"  CNF LR: {args.lr_cnf}")
    log(f"  KL annealing: {args.use_kl_annealing}")
    if args.use_kl_annealing:
        log(f"    Beta target: {args.beta_target}")
        log(f"    Anneal epochs: {args.anneal_epochs}")
    log(f"  CNF time steps: {args.nt}")
    log(f"  Save directory: {args.save}")
    log("")
    
    # Load data
    log("Loading data...")
    train_loader, val_loader, test_loader = getLoader(
        args.data, 
        args.batch_size, 
        args.val_batch_size,
        augment=False,
        hasGPU=(args.gpu >= 0)
    )
    log(f"  Train batches: {len(train_loader)}")
    log(f"  Val batches: {len(val_loader)}")
    log(f"  Test batches: {len(test_loader)}")
    log("")
    
    # Create models
    log("Creating models...")
    
    # 1. LOUPE Mask
    image_size = 28
    fourier_input_dim = image_size * image_size * 2  # Complex Fourier
    
    mask = LOUPEMask(
        input_dim=fourier_input_dim,
        sparsity=args.sparsity,
        slope=args.slope,
        image_shape=(image_size, image_size)
    ).to(device)
    
    log(f"  LOUPE Mask: {sum(p.numel() for p in mask.parameters())} parameters")
    
    # 2. Dual VAE
    original_dim = image_size * image_size
    
    vae = DualVAE(
        original_dim=original_dim,
        encoding_dim=args.latent_dim,
        fourier_input_dim=fourier_input_dim,
        image_size=image_size
    ).to(device)
    
    log(f"  Dual VAE: {sum(p.numel() for p in vae.parameters())} parameters")
    
    # 3. CNF (Phi)
    # dx = dimension of x (image latent), dy = dimension of y (Fourier latent)
    dx = args.latent_dim
    dy = args.latent_dim
    
    cnf = Phi(
        nTh=args.nTh,
        m=args.m,
        dx=dx,
        dy=dy,
        alph=args.alph
    ).to(device)
    
    log(f"  CNF (Phi): {sum(p.numel() for p in cnf.parameters())} parameters")
    
    # 4. Complete Model
    model = MaskDualVAECNF(
        original_dim=original_dim,
        encoding_dim=args.latent_dim,
        mask=mask,
        vae=vae,
        Phi=cnf,
        nt=args.nt,
        eps=args.eps
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    log(f"  Total parameters: {total_params:,}")
    log("")
    
    # Create optimizers
    log("Creating optimizers...")
    
    optimizer_vae = optim.Adam(
        vae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    optimizer_cnf = optim.Adam(
        cnf.parameters(),
        lr=args.lr_cnf,
        weight_decay=args.weight_decay
    )
    
    optimizer_mask = optim.Adam(
        mask.parameters(),
        lr=args.lr_mask,
        weight_decay=args.weight_decay
    )
    
    log(f"  VAE optimizer: Adam (lr={args.lr})")
    log(f"  CNF optimizer: Adam (lr={args.lr_cnf})")
    log(f"  Mask optimizer: Adam (lr={args.lr_mask})")
    log("")
    
    # Setup KL annealing
    if args.use_kl_annealing:
        kl_annealer = KLAnnealer(
            target_beta=args.beta_target,
            anneal_epochs=args.anneal_epochs
        )
        log(f"KL annealing enabled: 0 → {args.beta_target} over {args.anneal_epochs} epochs")
    else:
        kl_annealer = KLAnnealer(target_beta=args.beta_target, anneal_epochs=0)
        log(f"KL weight fixed at: {args.beta_target}")
    log("")
    
    # Setup metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        log(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(
            args.resume, model, optimizer_vae, optimizer_cnf, optimizer_mask, device
        )
        log(f"  Resumed from epoch {start_epoch}")
        log("")
    
    # Training loop
    log("=" * 80)
    log("Starting training")
    log("=" * 80)
    log("")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        
        # Get current beta
        beta = kl_annealer.get_beta(epoch)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer_vae, optimizer_cnf, optimizer_mask,
            beta, device, epoch
        )
        
        # Update metrics
        metrics_tracker.update('train', train_metrics, epoch + 1)
        
        # Validate
        if (epoch + 1) % args.val_freq == 0 or epoch == 0:
            val_metrics, val_samples = validate_epoch(
                model, val_loader, beta, device, epoch
            )
            
            # Update metrics
            metrics_tracker.update('val', val_metrics, epoch + 1)
            
            # Check for best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_checkpoint_path = os.path.join(
                    args.save, 'checkpoints', 'best_model.pt'
                )
                save_checkpoint(
                    model, optimizer_vae, optimizer_cnf, optimizer_mask,
                    epoch, val_metrics, best_checkpoint_path
                )
                log(f"  ✓ New best model saved (loss: {best_val_loss:.4f})")
            
            # Log validation results
            log("")
            log(f"Epoch {epoch+1}/{args.num_epochs}")
            log(f"  Train - Loss: {train_metrics['loss']:.4f} | "
                f"Recon: {train_metrics['recon_loss']:.4f} | "
                f"KL: {train_metrics['kl_loss']:.4f} | "
                f"OT: {train_metrics['ot_loss']:.4f} | "
                f"PSNR: {train_metrics['psnr']:.2f} dB | "
                f"Sparsity: {train_metrics['sparsity']:.4f}")
            log(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
                f"Recon: {val_metrics['recon_loss']:.4f} | "
                f"KL: {val_metrics['kl_loss']:.4f} | "
                f"OT: {val_metrics['ot_loss']:.4f} | "
                f"PSNR: {val_metrics['psnr']:.2f} dB | "
                f"Sparsity: {val_metrics['sparsity']:.4f}")
            log(f"  Beta: {beta:.4f} | Time: {time.time() - epoch_start_time:.1f}s")
        
        # Visualize
        if (epoch + 1) % args.viz_freq == 0 or epoch == 0:
            # Validate to get samples if we haven't already
            if (epoch + 1) % args.val_freq != 0 and epoch != 0:
                _, val_samples = validate_epoch(
                    model, val_loader, beta, device, epoch
                )
            
            # Plot reconstructions
            original, reconstructed, mask_vis = val_samples
            plot_reconstructions(
                original, reconstructed, mask_vis,
                os.path.join(args.save, 'figs', f'reconstruction_epoch_{epoch+1:03d}.png')
            )
            
            # Plot mask
            plot_mask(
                model.mask,
                os.path.join(args.save, 'figs'),
                epoch + 1
            )
            
            # Generate samples from prior
            # Get real Fourier samples from validation set (NOT random noise!)
            val_iter = iter(val_loader)
            _, sample_fourier, _ = next(val_iter)  # Get Fourier-transformed images
            
            plot_generated_samples(
                model, sample_fourier,  # Pass the actual Fourier samples
                device, num_samples=16,
                save_path=os.path.join(args.save, 'figs', f'generated_epoch_{epoch+1:03d}.png')
            )
            
            # Plot loss curves
            if len(metrics_tracker.get('train', 'loss')) > 1:
                plot_loss_curves(
                    metrics_tracker,
                    os.path.join(args.save, 'figs', 'loss_curves.png')
                )
            
            log(f"  ✓ Visualizations saved")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.save, 'checkpoints', f'model_epoch_{epoch+1:03d}.pt'
            )
            save_checkpoint(
                model, optimizer_vae, optimizer_cnf, optimizer_mask,
                epoch, train_metrics, checkpoint_path
            )
            log(f"  ✓ Checkpoint saved")
    
    # Final checkpoint
    log("")
    log("=" * 80)
    log("Training complete!")
    log("=" * 80)
    log("")
    
    final_checkpoint_path = os.path.join(args.save, 'checkpoints', 'final_model.pt')
    save_checkpoint(
        model, optimizer_vae, optimizer_cnf, optimizer_mask,
        args.num_epochs - 1, train_metrics, final_checkpoint_path
    )
    log(f"Final model saved to: {final_checkpoint_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.save, 'metrics.npz')
    metrics_tracker.save(metrics_path)
    log(f"Metrics saved to: {metrics_path}")
    
    # Final evaluation
    log("")
    log("Running final evaluation...")
    final_metrics, final_samples = validate_epoch(
        model, test_loader, args.beta_target, device, args.num_epochs - 1
    )
    
    log("")
    log("Final Test Metrics:")
    log(f"  Loss: {final_metrics['loss']:.4f}")
    log(f"  Reconstruction: {final_metrics['recon_loss']:.4f}")
    log(f"  KL: {final_metrics['kl_loss']:.4f}")
    log(f"    - Image KL: {final_metrics['kl_img']:.4f}")
    log(f"    - Fourier KL: {final_metrics['kl_four']:.4f}")
    log(f"  OT-Flow: {final_metrics['ot_loss']:.4f}")
    log(f"  PSNR: {final_metrics['psnr']:.2f} dB")
    log(f"  Sparsity: {final_metrics['sparsity']:.4f}")
    
    # Create summary
    create_training_summary(
        metrics_tracker,
        os.path.join(args.save, 'figs', 'training_summary.png'),
        final_metrics
    )
    
    log("")
    log(f"All outputs saved to: {args.save}")
    log("")
    log("=" * 80)
    
    log_file.close()


if __name__ == "__main__":
    main()
