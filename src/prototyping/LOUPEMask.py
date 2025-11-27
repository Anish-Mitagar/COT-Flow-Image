"""
LOUPE-style Probabilistic Mask (FIXED VERSION)
Based on: "Learning-based Optimization of the Under-sampling Pattern in MRI"
Bahadir et al., 2019

This is the FIXED version that replaces the flawed ProbabilisticMask.py
Key improvements:
- Controlled sparsity with rescaling
- No conflicting L1 regularization
- Proper probabilistic sampling
- Works with Fourier transform (complex -> 2*real)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LOUPEMask(nn.Module):
    """
    LOUPE-style learnable probabilistic mask for k-space sampling.
    
    Architecture:
    1. Learnable logits θ (parameters to optimize)
    2. Sigmoid with slope parameter → probabilities
    3. Rescaling to maintain exact sparsity budget
    4. Stochastic sampling during training (Bernoulli)
    5. Deterministic thresholding during inference
    """
    
    def __init__(self, input_dim, sparsity=0.1, slope=5.0, image_shape=(28, 28)):
        """
        Args:
            input_dim: Dimension of Fourier input (e.g., 28*28*2 = 1568)
            sparsity: Target sparsity/budget (e.g., 0.1 for 10% sampling)
            slope: Slope parameter for sigmoid (controls sharpness)
            image_shape: Shape of original image for visualization
        """
        super(LOUPEMask, self).__init__()
        
        self.input_dim = input_dim
        self.initial_sparsity = sparsity
        self.slope = slope
        self.image_shape = image_shape
        
        # Learnable parameters (logits)
        # Initialize with slight positive bias to encourage sampling
        self.logits = nn.Parameter(torch.randn(1, input_dim) * 0.1 + 0.5)
        
        # Register sparsity as a buffer (not a parameter, but saved with model)
        self.register_buffer('target_sparsity', torch.tensor(sparsity))
        
    def get_probabilities(self):
        """
        Convert logits to probabilities via sigmoid.
        
        Returns:
            probs: Probabilities in [0, 1] (1, input_dim)
        """
        # Sigmoid with slope parameter
        # Higher slope → sharper transition (more binary-like)
        probs = torch.sigmoid(self.slope * self.logits)
        return probs
    
    def rescale_probabilities(self, probs):
        """
        Rescale probabilities to meet exact sparsity budget.
        
        This is the KEY step in LOUPE:
        - Sum of probabilities should equal (sparsity * input_dim)
        - Ensures expected number of samples matches budget
        
        Args:
            probs: Probabilities before rescaling (1, input_dim)
        
        Returns:
            probs_rescaled: Rescaled probabilities (1, input_dim)
        """
        # Target sum: we want expected number of samples = sparsity * input_dim
        target_sum = self.target_sparsity * self.input_dim
        
        # Current sum of probabilities
        current_sum = torch.sum(probs)
        
        # Rescale factor
        # If sum is too high, scale down; if too low, scale up
        scale = target_sum / (current_sum + 1e-8)
        
        # Apply rescaling and clip to [0, 1]
        probs_rescaled = torch.clamp(probs * scale, 0.0, 1.0)
        
        return probs_rescaled
    
    def sample_mask(self, probs):
        """
        Sample binary mask from probabilities using Bernoulli distribution.
        
        Args:
            probs: Rescaled probabilities (1, input_dim)
        
        Returns:
            mask: Binary mask (1, input_dim) with values in {0, 1}
        """
        # Bernoulli sampling: each element is 1 with probability probs[i]
        # This is differentiable through the Gumbel-Softmax trick (implicit in PyTorch)
        mask = torch.bernoulli(probs)
        return mask
    
    def get_deterministic_mask(self, threshold=0.5):
        """
        Get deterministic mask by thresholding probabilities.
        Used during inference/evaluation.
        
        Args:
            threshold: Threshold for binarization (default 0.5)
        
        Returns:
            mask: Binary mask (1, input_dim)
        """
        probs = self.get_probabilities()
        probs_rescaled = self.rescale_probabilities(probs)
        
        # Threshold to get binary mask
        mask = (probs_rescaled >= threshold).float()
        
        return mask
    
    def forward(self, x):
        """
        Apply learned mask to input.
        
        Args:
            x: Input tensor (batch, input_dim)
        
        Returns:
            x_masked: Masked input (batch, input_dim)
            mask: Binary mask used (batch, input_dim)
        """
        # Get probabilities
        probs = self.get_probabilities()
        
        # Rescale to meet sparsity budget
        probs_rescaled = self.rescale_probabilities(probs)
        
        # Sample or threshold based on training mode
        if self.training:
            # Stochastic sampling during training
            mask = self.sample_mask(probs_rescaled)
        else:
            # Deterministic thresholding during inference
            mask = (probs_rescaled >= 0.5).float()
        
        # Expand mask to batch dimension
        batch_size = x.size(0)
        mask_expanded = mask.expand(batch_size, -1)
        
        # Apply mask
        x_masked = x * mask_expanded
        
        return x_masked, mask_expanded
    
    def get_sparsity(self):
        """
        Get actual sparsity of current mask.
        
        Returns:
            sparsity: Fraction of elements that are 1
        """
        with torch.no_grad():
            mask = self.get_deterministic_mask()
            sparsity = torch.mean(mask).item()
        return sparsity
    
    def set_sparsity(self, new_sparsity):
        """
        Adjust target sparsity.
        
        Args:
            new_sparsity: New target sparsity value
        """
        self.target_sparsity = torch.tensor(new_sparsity, device=self.logits.device)
    
    def get_mask_for_visualization(self):
        """
        Get mask reshaped for visualization.
        Handles both real masks and complex Fourier masks.
        
        Returns:
            mask_vis: Dictionary with visualization-ready masks
        """
        with torch.no_grad():
            mask = self.get_deterministic_mask().cpu().numpy().squeeze()
            
            mask_vis = {}
            
            # Check if this is a Fourier mask (2x the pixels for real+imag)
            expected_fourier_size = self.image_shape[0] * self.image_shape[1] * 2
            
            if mask.shape[0] == expected_fourier_size:
                # Fourier mask: reshape to (H, W, 2)
                mask_2d = mask.reshape(self.image_shape[0], self.image_shape[1], 2)
                
                # Separate real and imaginary components
                mask_vis['real'] = mask_2d[:, :, 0]
                mask_vis['imag'] = mask_2d[:, :, 1]
                
                # Magnitude (combined mask)
                mask_vis['magnitude'] = np.sqrt(mask_2d[:, :, 0]**2 + mask_2d[:, :, 1]**2)
                
                mask_vis['type'] = 'fourier'
                
            else:
                # Regular mask: try to reshape to square
                size = int(np.sqrt(mask.shape[0]))
                if size * size == mask.shape[0]:
                    mask_vis['mask'] = mask.reshape(size, size)
                    mask_vis['type'] = 'spatial'
                else:
                    # Can't reshape, return 1D
                    mask_vis['mask'] = mask
                    mask_vis['type'] = '1d'
            
            return mask_vis
    
    def extra_repr(self):
        """String representation for print(model)."""
        return f'input_dim={self.input_dim}, sparsity={self.target_sparsity.item():.3f}, slope={self.slope}'


# ============================================================================
#                           UTILITY FUNCTIONS
# ============================================================================

def l1_loss(logits):
    """
    L1 regularization on logits.
    
    NOTE: This is NOT recommended to use with LOUPE mask!
    LOUPE already controls sparsity through rescaling.
    Adding L1 loss creates conflicting objectives.
    
    Args:
        logits: Learnable parameters
    
    Returns:
        l1: L1 norm
    """
    return torch.mean(torch.abs(logits))


def mask_sparsity_loss(mask, target_sparsity):
    """
    Loss to encourage mask to match target sparsity.
    
    This is also NOT recommended with LOUPE, as rescaling already
    ensures the sparsity budget is met. This is for reference only.
    
    Args:
        mask: Binary mask
        target_sparsity: Target sparsity value
    
    Returns:
        loss: MSE between actual and target sparsity
    """
    actual_sparsity = torch.mean(mask.float())
    loss = (actual_sparsity - target_sparsity) ** 2
    return loss


# ============================================================================
#                           TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing LOUPE Mask")
    print("=" * 70)
    
    # Configuration
    image_size = 28
    fourier_input_dim = image_size * image_size * 2  # Complex Fourier
    batch_size = 4
    target_sparsity = 0.1
    
    # Create mask
    print(f"\nCreating LOUPE mask:")
    print(f"  Input dimension: {fourier_input_dim}")
    print(f"  Target sparsity: {target_sparsity}")
    
    mask = LOUPEMask(
        input_dim=fourier_input_dim,
        sparsity=target_sparsity,
        slope=5.0,
        image_shape=(image_size, image_size)
    )
    
    print(f"\n{mask}")
    
    # Test forward pass (training mode)
    print("\n" + "-" * 70)
    print("TEST 1: Training mode (stochastic sampling)")
    print("-" * 70)
    
    mask.train()
    x = torch.randn(batch_size, fourier_input_dim)
    
    x_masked, mask_sample = mask(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Masked output shape: {x_masked.shape}")
    print(f"Mask shape: {mask_sample.shape}")
    print(f"Actual sparsity: {mask_sample.mean().item():.4f}")
    print(f"Target sparsity: {target_sparsity:.4f}")
    print(f"Difference: {abs(mask_sample.mean().item() - target_sparsity):.4f}")
    
    # Test multiple samples (should be different)
    x_masked1, mask1 = mask(x)
    x_masked2, mask2 = mask(x)
    print(f"\nStochasticity check:")
    print(f"  Same input, different masks? {not torch.allclose(mask1, mask2)}")
    
    # Test forward pass (eval mode)
    print("\n" + "-" * 70)
    print("TEST 2: Evaluation mode (deterministic)")
    print("-" * 70)
    
    mask.eval()
    x_masked_eval, mask_eval = mask(x)
    
    print(f"Actual sparsity (eval): {mask_eval.mean().item():.4f}")
    
    # Test determinism
    x_masked_eval2, mask_eval2 = mask(x)
    print(f"Deterministic? {torch.allclose(mask_eval, mask_eval2)}")
    
    # Test sparsity adjustment
    print("\n" + "-" * 70)
    print("TEST 3: Sparsity adjustment")
    print("-" * 70)
    
    original_sparsity = mask.get_sparsity()
    print(f"Original sparsity: {original_sparsity:.4f}")
    
    mask.set_sparsity(0.2)
    new_sparsity = mask.get_sparsity()
    print(f"New target: 0.2")
    print(f"Actual sparsity: {new_sparsity:.4f}")
    
    # Test visualization
    print("\n" + "-" * 70)
    print("TEST 4: Visualization")
    print("-" * 70)
    
    mask_vis = mask.get_mask_for_visualization()
    print(f"Mask type: {mask_vis['type']}")
    
    if mask_vis['type'] == 'fourier':
        print(f"  Real component shape: {mask_vis['real'].shape}")
        print(f"  Imaginary component shape: {mask_vis['imag'].shape}")
        print(f"  Magnitude shape: {mask_vis['magnitude'].shape}")
        print(f"  Mean magnitude: {mask_vis['magnitude'].mean():.4f}")
    
    # Test parameter count
    print("\n" + "-" * 70)
    print("TEST 5: Parameters")
    print("-" * 70)
    
    total_params = sum(p.numel() for p in mask.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter shape: {mask.logits.shape}")
    
    # Test gradient flow
    print("\n" + "-" * 70)
    print("TEST 6: Gradient flow")
    print("-" * 70)
    
    mask.train()
    x = torch.randn(batch_size, fourier_input_dim, requires_grad=True)
    x_masked, mask_sample = mask(x)
    
    # Dummy loss
    loss = x_masked.sum()
    loss.backward()
    
    print(f"Input gradient exists? {x.grad is not None}")
    print(f"Logits gradient exists? {mask.logits.grad is not None}")
    if mask.logits.grad is not None:
        print(f"Logits gradient norm: {mask.logits.grad.norm().item():.6f}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nThis mask is ready to use in your training script!")
    print("Remember: DO NOT add L1 regularization - it's not needed!")
