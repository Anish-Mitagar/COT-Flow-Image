"""
Test script to verify the Dual VAE implementation.
Run this to ensure all components work correctly.
"""

import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dual_vae import (
    VAEEncoder, VAEDecoder, DualVAE, MaskedDualVAE, 
    vae_loss_function
)

def test_vae_encoder():
    """Test VAE encoder outputs mean and log-variance."""
    print("\n" + "="*70)
    print("TEST 1: VAE Encoder")
    print("="*70)
    
    # Test CNN encoder
    print("\nTesting CNN encoder (image)...")
    encoder_cnn = VAEEncoder(
        input_dim=784, 
        latent_dim=64, 
        is_convolutional=True,
        image_channels=1
    )
    x = torch.randn(4, 1, 28, 28)
    mu, logvar = encoder_cnn(x)
    
    assert mu.shape == (4, 64), f"Expected mu shape (4, 64), got {mu.shape}"
    assert logvar.shape == (4, 64), f"Expected logvar shape (4, 64), got {logvar.shape}"
    print(f"✓ CNN encoder output shapes correct: mu={mu.shape}, logvar={logvar.shape}")
    
    # Test MLP encoder
    print("\nTesting MLP encoder (Fourier)...")
    encoder_mlp = VAEEncoder(
        input_dim=1568,
        latent_dim=64,
        is_convolutional=False
    )
    x_fourier = torch.randn(4, 1568)
    mu, logvar = encoder_mlp(x_fourier)
    
    assert mu.shape == (4, 64), f"Expected mu shape (4, 64), got {mu.shape}"
    assert logvar.shape == (4, 64), f"Expected logvar shape (4, 64), got {logvar.shape}"
    print(f"✓ MLP encoder output shapes correct: mu={mu.shape}, logvar={logvar.shape}")
    
    print("\n✅ VAE Encoder test PASSED")
    return True


def test_vae_decoder():
    """Test VAE decoder reconstruction."""
    print("\n" + "="*70)
    print("TEST 2: VAE Decoder")
    print("="*70)
    
    decoder = VAEDecoder(latent_dim=128, image_channels=1, image_size=28)
    z = torch.randn(4, 128)
    reconstruction = decoder(z)
    
    assert reconstruction.shape == (4, 1, 28, 28), \
        f"Expected shape (4, 1, 28, 28), got {reconstruction.shape}"
    assert reconstruction.min() >= 0 and reconstruction.max() <= 1, \
        "Decoder output should be in [0, 1] range"
    
    print(f"✓ Decoder output shape: {reconstruction.shape}")
    print(f"✓ Output range: [{reconstruction.min():.4f}, {reconstruction.max():.4f}]")
    print("\n✅ VAE Decoder test PASSED")
    return True


def test_reparameterization():
    """Test reparameterization trick."""
    print("\n" + "="*70)
    print("TEST 3: Reparameterization Trick")
    print("="*70)
    
    model = DualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=1568,
        latent_dim=64
    )
    
    mu = torch.zeros(4, 64)
    logvar = torch.zeros(4, 64)  # log(1) = 0, so sigma = 1
    
    # Multiple samples should be different (stochastic)
    z1 = model.reparameterize(mu, logvar)
    z2 = model.reparameterize(mu, logvar)
    
    assert not torch.allclose(z1, z2), "Reparameterization should be stochastic"
    print("✓ Reparameterization is stochastic")
    
    # Mean should be close to mu
    samples = torch.stack([model.reparameterize(mu, logvar) for _ in range(100)])
    sample_mean = samples.mean(dim=0)
    
    assert torch.allclose(sample_mean, mu, atol=0.5), "Sample mean should approximate mu"
    print(f"✓ Sample mean approximates mu (diff: {(sample_mean - mu).abs().max():.4f})")
    
    print("\n✅ Reparameterization test PASSED")
    return True


def test_dual_vae_forward():
    """Test full DualVAE forward pass."""
    print("\n" + "="*70)
    print("TEST 4: DualVAE Forward Pass")
    print("="*70)
    
    model = DualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=1568,
        latent_dim=64
    )
    
    x = torch.randn(4, 1, 28, 28)
    x_fourier = torch.randn(4, 1568)
    
    # Test without components
    reconstruction, mu_img, logvar_img, mu_four, logvar_four = model(x, x_fourier)
    
    assert reconstruction.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {reconstruction.shape}"
    assert mu_img.shape == (4, 64), f"Expected mu_img shape (4, 64), got {mu_img.shape}"
    assert mu_four.shape == (4, 64), f"Expected mu_four shape (4, 64), got {mu_four.shape}"
    
    print(f"✓ Reconstruction shape: {reconstruction.shape}")
    print(f"✓ Image encoder - mu: {mu_img.shape}, logvar: {logvar_img.shape}")
    print(f"✓ Fourier encoder - mu: {mu_four.shape}, logvar: {logvar_four.shape}")
    
    # Test with components
    reconstruction, mu_img, logvar_img, mu_four, logvar_four, components = \
        model(x, x_fourier, return_components=True)
    
    assert 'z_combined' in components, "Components should contain z_combined"
    assert components['z_combined'].shape == (4, 128), \
        f"Expected z_combined shape (4, 128), got {components['z_combined'].shape}"
    
    print(f"✓ Combined latent: {components['z_combined'].shape}")
    print("\n✅ DualVAE forward pass test PASSED")
    return True


def test_vae_loss():
    """Test VAE loss function."""
    print("\n" + "="*70)
    print("TEST 5: VAE Loss Function")
    print("="*70)
    
    # Create dummy data
    batch_size = 4
    reconstruction = torch.rand(batch_size, 1, 28, 28)
    target = torch.rand(batch_size, 1, 28, 28)
    mu_img = torch.randn(batch_size, 64)
    logvar_img = torch.randn(batch_size, 64)
    mu_four = torch.randn(batch_size, 64)
    logvar_four = torch.randn(batch_size, 64)
    
    # Test with beta=1.0
    total_loss, recon_loss, kl_loss = vae_loss_function(
        reconstruction, target, mu_img, logvar_img, mu_four, logvar_four, beta=1.0
    )
    
    assert total_loss > 0, "Total loss should be positive"
    assert recon_loss > 0, "Reconstruction loss should be positive"
    assert kl_loss >= 0, "KL divergence should be non-negative"
    
    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ Reconstruction loss: {recon_loss.item():.4f}")
    print(f"✓ KL divergence: {kl_loss.item():.4f}")
    
    # Test beta weighting
    total_loss_beta2, _, _ = vae_loss_function(
        reconstruction, target, mu_img, logvar_img, mu_four, logvar_four, beta=2.0
    )
    
    print(f"✓ Total loss with β=2.0: {total_loss_beta2.item():.4f}")
    assert total_loss_beta2 > total_loss, "Higher beta should increase total loss"
    
    print("\n✅ VAE loss function test PASSED")
    return True


def test_masked_dual_vae():
    """Test MaskedDualVAE with LOUPE mask."""
    print("\n" + "="*70)
    print("TEST 6: MaskedDualVAE")
    print("="*70)
    
    model = MaskedDualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=1568,
        latent_dim=64,
        sparsity=0.1,
        slope=5.0
    )
    
    x = torch.randn(4, 1, 28, 28)
    x_fourier = torch.randn(4, 1568)
    
    # Training mode
    model.train()
    reconstruction, mask, mu_img, logvar_img, mu_four, logvar_four = \
        model(x, x_fourier)
    
    assert mask.shape == (4, 1568), f"Expected mask shape (4, 1568), got {mask.shape}"
    assert 0 <= mask.float().mean() <= 1, "Mask values should be in [0, 1]"
    
    print(f"✓ Mask shape: {mask.shape}")
    print(f"✓ Mask sparsity: {mask.float().mean().item():.4f}")
    print(f"✓ Target sparsity: {model.get_sparsity():.4f}")
    
    # Test sparsity adjustment
    model.set_sparsity(0.2)
    new_sparsity = model.get_sparsity()
    assert abs(new_sparsity - 0.2) < 0.05, "Sparsity adjustment failed"
    print(f"✓ Adjusted sparsity to: {new_sparsity:.4f}")
    
    print("\n✅ MaskedDualVAE test PASSED")
    return True


def test_sampling():
    """Test sampling from prior."""
    print("\n" + "="*70)
    print("TEST 7: Sampling from Prior")
    print("="*70)
    
    model = DualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=1568,
        latent_dim=64
    )
    
    # Generate samples
    num_samples = 8
    samples = model.sample(num_samples, device='cpu')
    
    assert samples.shape == (num_samples, 1, 28, 28), \
        f"Expected shape ({num_samples}, 1, 28, 28), got {samples.shape}"
    assert 0 <= samples.min() and samples.max() <= 1, \
        "Samples should be in [0, 1] range"
    
    print(f"✓ Generated {num_samples} samples")
    print(f"✓ Sample shape: {samples.shape}")
    print(f"✓ Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
    
    # Test with MaskedDualVAE
    masked_model = MaskedDualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=1568,
        latent_dim=64,
        sparsity=0.1
    )
    
    samples_masked = masked_model.sample(num_samples, device='cpu')
    assert samples_masked.shape == (num_samples, 1, 28, 28), \
        f"Expected shape ({num_samples}, 1, 28, 28), got {samples_masked.shape}"
    
    print(f"✓ MaskedDualVAE sampling works")
    
    print("\n✅ Sampling test PASSED")
    return True


def test_gradients():
    """Test that gradients flow properly."""
    print("\n" + "="*70)
    print("TEST 8: Gradient Flow")
    print("="*70)
    
    model = MaskedDualVAE(
        image_channels=1,
        image_size=28,
        fourier_input_dim=1568,
        latent_dim=64,
        sparsity=0.1
    )
    
    x = torch.randn(2, 1, 28, 28)
    x_fourier = torch.randn(2, 1568)
    
    # Forward pass
    reconstruction, mask, mu_img, logvar_img, mu_four, logvar_four = \
        model(x, x_fourier)
    
    # Compute loss
    total_loss, recon_loss, kl_loss = vae_loss_function(
        reconstruction, x, mu_img, logvar_img, mu_four, logvar_four, beta=1.0
    )
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients exist
    has_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None and param.grad.abs().sum() > 0
            has_grads.append(has_grad)
            if not has_grad:
                print(f"  ⚠️  No gradient for {name}")
    
    grad_percentage = sum(has_grads) / len(has_grads) * 100
    print(f"✓ {sum(has_grads)}/{len(has_grads)} parameters have gradients ({grad_percentage:.1f}%)")
    
    assert grad_percentage > 95, "Most parameters should have gradients"
    
    print("\n✅ Gradient flow test PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*20 + "DUAL VAE TEST SUITE" + " "*28 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    tests = [
        ("VAE Encoder", test_vae_encoder),
        ("VAE Decoder", test_vae_decoder),
        ("Reparameterization", test_reparameterization),
        ("DualVAE Forward", test_dual_vae_forward),
        ("VAE Loss", test_vae_loss),
        ("MaskedDualVAE", test_masked_dual_vae),
        ("Sampling", test_sampling),
        ("Gradients", test_gradients),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} test FAILED")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*25 + "TEST SUMMARY" + " "*31 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print(f"\n✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    else:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
    print("\n" + "#"*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
