from dual_vae import MaskedDualVAE, DualVAE
from train_vae import train

import argparse
import torch

from datasets.mnist import getLoader

import config

cf = config.getconfig()

if cf.gpu:
    def_viz_freq = 100
    def_batch    = 800
    def_niters   = 1000  # changed from 50000
    def_m        = 128
    def_val_freq = 20
else:  # if no gpu on platform, assume debugging on a local cpu
    def_viz_freq = 4
    def_batch    = 20
    def_niters   = 40
    def_val_freq = 1
    def_m        = 16

parser = argparse.ArgumentParser('Dual-VAE for MRI Undersampling')
parser.add_argument(
    '--data', choices=['mnist'], type=str, default='mnist'
)
parser.add_argument("--nt"    , type=int, default=8, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=16, help="number of time steps for validation")
parser.add_argument('--alph'  , type=str, default='1.0,80.0,500.0')
parser.add_argument('--m'     , type=int, default=def_m)
parser.add_argument('--latent_dim', type=int, default=64, help='latent dimension for each encoder')

parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr'          , type=float, default=1e-4, help='learning rate for VAE')
parser.add_argument('--lr_mask'     , type=float, default=1e-3, help='learning rate for mask')
parser.add_argument('--drop_freq'   , type=int,   default=5000, help="how often to decrease learning rate")
parser.add_argument('--lr_drop'     , type=float, default=10**0.5, help="how much to decrease learning rate")
parser.add_argument('--eps'         , type=float, default=10**-6)

# VAE-specific arguments
parser.add_argument('--beta_target'  , type=float, default=1.0, help='target beta for KL weight')
parser.add_argument('--use_kl_annealing', action='store_true', default=True, help='use KL annealing')
parser.add_argument('--anneal_epochs', type=int, default=10, help='epochs to anneal KL weight')

# Mask-specific arguments
parser.add_argument('--sparsity'    , type=float, default=0.1, help='target sparsity for mask')
parser.add_argument('--slope'       , type=float, default=5.0, help='slope for LOUPE sigmoid')
parser.add_argument('--use_mask'    , action='store_true', default=True, help='use masked VAE')

parser.add_argument('--niters'     , type=int, default=def_niters)
parser.add_argument('--num_epochs' , type=int, default=200, help='number of training epochs')
parser.add_argument('--batch_size' , type=int, default=def_batch)
parser.add_argument('--val_batch_size', type=int, default=def_batch)
parser.add_argument('--resume'     , type=str, default=None)
parser.add_argument('--save'       , type=str, default='experiments/vae/large')
parser.add_argument('--viz_freq'   , type=int, default=def_viz_freq)
parser.add_argument('--val_freq'   , type=int, default=def_val_freq)
parser.add_argument('--gpu'        , type=int, default=0)
parser.add_argument('--conditional', type=int, default=-1)  # -1 means unconditioned
parser.add_argument('--test_shape' , type=str, default='True')
args = parser.parse_args()

# Device setup
device = 'cuda' if torch.cuda.is_available() and cf.gpu else 'cpu'
print(f'Using device: {device}')

# Your dataloaders
print('\nLoading datasets...')
train_loader, val_loader, test_loader = getLoader(
    args.data, 
    args.batch_size, 
    args.val_batch_size, 
    augment=False, 
    hasGPU=cf.gpu, 
    conditional=args.conditional
)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches: {len(val_loader)}')
print(f'Test batches: {len(test_loader)}')

# Initialize model
print('\nInitializing model...')
if args.use_mask:
    print('Using MaskedDualVAE (with LOUPE mask)')
    model = MaskedDualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=28*28*2,
        latent_dim=args.latent_dim,
        sparsity=args.sparsity,
        slope=args.slope
    )
else:
    print('Using DualVAE (without mask)')
    model = DualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=28*28*2,
        latent_dim=args.latent_dim
    )

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nModel parameters:')
print(f'  Total: {total_params:,}')
print(f'  Trainable: {trainable_params:,}')

print(model)

# Train - ONE function call does everything!
print('\nStarting training...')
model, losses, test_results = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_epochs=args.num_epochs,
    lr=args.lr,
    lr_mask=args.lr_mask,
    device=device,
    checkpoint_dir=args.save,
    use_kl_annealing=args.use_kl_annealing,
    beta_target=args.beta_target,
    anneal_epochs=args.anneal_epochs
)

# Print final results
print('\n' + '=' * 70)
print('FINAL RESULTS')
print('=' * 70)
print(f"Test Total Loss: {test_results['test_total_loss']:.6f}")
print(f"Test Reconstruction Loss: {test_results['test_recon_loss']:.6f}")
print(f"Test KL Divergence: {test_results['test_kl_loss']:.6f}")
print(f"Test PSNR: {test_results['test_psnr']:.2f} dB")
if args.use_mask:
    print(f"Test Sparsity: {test_results['test_sparsity']:.4f}")
    print(f"Target Sparsity: {args.sparsity:.4f}")
print('=' * 70)

# Save final model
final_model_path = f"{args.save}/final_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'config': vars(args),
    'test_results': test_results,
    'losses': losses
}, final_model_path)
print(f'\nSaved final model to {final_model_path}')

# Generate some samples from the prior
print('\nGenerating samples from prior...')
import matplotlib.pyplot as plt
samples = model.sample(num_samples=16, device=device)
samples = samples.cpu()

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{args.save}/samples_from_prior.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved samples to {args.save}/samples_from_prior.png')

print('\nTraining complete!')
