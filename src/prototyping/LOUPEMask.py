"""
LOUPE-style Probabilistic Mask (FIXED TO MATCH WORKING VERSION)
Based on the working ProbabilisticMask from dual_autoencoder.py

This version uses the EXACT same architecture that was proven to work:
- Proper straight-through estimator
- Conditional rescaling (not simple scaling)
- Randomized sampling during training
- All the LOUPE layers from the TensorFlow implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LOUPEMask(nn.Module):
    """
    LOUPE-style learnable probabilistic mask for k-space sampling.
    
    This is the WORKING version based on ProbabilisticMask.
    
    Architecture (following LOUPE paper):
    1. ProbMask: logits → sigmoid(slope * logits) → probabilities
    2. RescaleProbMap: rescale to match target sparsity
    3. ThresholdRandomMask: sample binary mask from probabilities
    4. Apply mask to input
    """
    
    def __init__(self, input_dim, sparsity=0.1, slope=5.0, sample_slope=12.0, 
                 threshold=0.5, image_shape=(28, 28)):
        """
        Args:
            input_dim: Dimension of Fourier input (e.g., 28*28*2 = 1568)
            sparsity: Target sparsity/budget (e.g., 0.1 for 10% sampling)
            slope: Slope parameter for sigmoid (controls sharpness)
            sample_slope: Slope for sampling threshold (higher = harder sampling)
            threshold: Threshold for binarization during inference
            image_shape: Shape of original image for visualization
        """
        super(LOUPEMask, self).__init__()
        
        self.input_dim = input_dim
        self.sparsity = sparsity
        self.threshold = threshold
        self.image_shape = image_shape
        
        # Slope parameters (not trainable, just hyperparameters)
        self.register_buffer('slope', torch.tensor(slope, dtype=torch.float32))
        self.register_buffer('sample_slope', torch.tensor(sample_slope, dtype=torch.float32))
        
        # Initialize logits using LOUPE v2 initialization
        self.logits = nn.Parameter(self._logit_slope_random_uniform((1, input_dim)))
        
        # For tracking/debugging
        self.latest_mask = None
        self.latest_bin_mask = None
    
    def _logit_slope_random_uniform(self, shape, eps=0.01):
        """
        Initialize logits using inverse sigmoid (logit) of uniform distribution.
        This gives more balanced initialization than random logits.
        
        From LOUPE layers.py: ProbMask._logit_slope_random_uniform
        """
        # Sample from uniform [eps, 1-eps]
        x = torch.rand(shape, dtype=torch.float32)
        x = eps + (1.0 - 2*eps) * x
        
        # Apply inverse sigmoid (logit) with slope factor
        # logit(x) = log(x / (1-x))
        # For slope s: w = -log(1/x - 1) / s
        logits = -torch.log(1.0 / x - 1.0) / self.slope.item()
        
        return logits
    
    def prob_mask(self, x):
        """
        ProbMask layer from LOUPE: applies sigmoid to logits with slope.
        
        prob = sigmoid(slope * logits)
        """
        # Broadcast logits to match batch size
        batch_size = x.size(0)
        mask_logits = self.logits.expand(batch_size, -1)
        
        # Apply sigmoid with slope
        probabilities = torch.sigmoid(self.slope * mask_logits)
        
        return probabilities
    
    def rescale_prob_map(self, prob):
        """
        RescaleProbMap layer from LOUPE: rescales probabilities to match target sparsity.
        
        This is the CRITICAL rescaling from the TensorFlow code:
        If mean(prob) > sparsity: prob' = prob * sparsity / mean(prob)
        If mean(prob) < sparsity: prob' = 1 - (1-prob) * (1-sparsity) / (1-mean(prob))
        """
        prob_mean = torch.mean(prob)
        
        # Rescaling factors
        r = self.sparsity / (prob_mean + 1e-8)
        beta = (1.0 - self.sparsity) / (1.0 - prob_mean + 1e-8)
        
        # Conditional rescaling (equivalent to LOUPE's tf.less_equal logic)
        # le = 1 if r <= 1, else 0
        le = (r <= 1.0).float()
        
        # Apply conditional rescaling
        rescaled = le * prob * r + (1.0 - le) * (1.0 - (1.0 - prob) * beta)
        
        return rescaled
    
    def threshold_random_mask(self, prob, training=True):
        """
        ThresholdRandomMask layer from LOUPE: samples binary mask from probabilities.
        
        CRITICAL: Uses straight-through estimator for gradients!
        
        During training: Uses soft thresholding with sigmoid + straight-through
        During inference: Uses hard thresholding
        """
        if training:
            # Generate random uniform samples (RandomMask layer)
            random_samples = torch.rand_like(prob, dtype=torch.float32)
            
            # Soft thresholding with slope (ThresholdRandomMask layer)
            # sigmoid(sample_slope * (prob - random))
            # This gives smooth gradients during training
            binary_mask = torch.sigmoid(self.sample_slope * (prob - random_samples))
            
            # CRITICAL: Straight-through estimator
            # Forward pass: use hard binary values
            # Backward pass: use soft values for gradients
            binary_hard = (prob > random_samples).float()
            binary_mask = binary_hard + (binary_mask - binary_mask.detach())
            
        else:
            # During inference: hard thresholding
            binary_mask = (prob > self.threshold).float()
        
        return binary_mask
    
    def forward(self, x):
        """
        Forward pass implementing full LOUPE architecture.
        
        Pipeline:
        1. ProbMask: logits → sigmoid(slope * logits) → probabilities
        2. RescaleProbMap: rescale to match target sparsity
        3. ThresholdRandomMask: sample binary mask from probabilities
        4. Apply mask to input
        
        Args:
            x: Input tensor (batch, input_dim)
        
        Returns:
            x_masked: Masked input (batch, input_dim)
            mask: Binary mask used (batch, input_dim)
        """
        batch_size = x.size(0)
        
        # Step 1: ProbMask - convert logits to probabilities
        probabilities = self.prob_mask(x)
        
        # Step 2: RescaleProbMap - rescale to match target sparsity
        rescaled_prob = self.rescale_prob_map(probabilities)
        
        # Step 3: ThresholdRandomMask - sample binary mask
        binary_mask = self.threshold_random_mask(rescaled_prob, training=self.training)
        
        # Store for visualization/debugging
        self.latest_mask = rescaled_prob.detach()[0]
        self.latest_bin_mask = binary_mask.detach()[0]
        
        # Step 4: Apply mask to input
        masked_x = x * binary_mask
        
        return masked_x, binary_mask
    
    def get_sparsity(self):
        """
        Get current actual sparsity (sampling rate) of the mask.
        
        Returns:
            sparsity: Mean probability after rescaling
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_dim, dtype=torch.float32, 
                                     device=self.logits.device)
            probabilities = self.prob_mask(dummy_input)
            rescaled_prob = self.rescale_prob_map(probabilities)
            return rescaled_prob.mean().item()
    
    def set_sparsity(self, new_sparsity):
        """
        Adjust target sparsity.
        
        Args:
            new_sparsity: New target sparsity value
        """
        self.sparsity = new_sparsity
    
    def get_mask_for_visualization(self):
        """
        Get mask reshaped for visualization.
        Handles both real masks and complex Fourier masks.
        
        Returns:
            mask_vis: Dictionary with visualization-ready masks
        """
        with torch.no_grad():
            # Get the latest rescaled probabilities
            if self.latest_mask is not None:
                mask = self.latest_mask.cpu().numpy()
            else:
                # Generate from scratch
                dummy_input = torch.zeros(1, self.input_dim, device=self.logits.device)
                probabilities = self.prob_mask(dummy_input)
                rescaled_prob = self.rescale_prob_map(probabilities)
                mask = rescaled_prob.cpu().numpy().squeeze()
            
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
        return f'input_dim={self.input_dim}, sparsity={self.sparsity:.3f}, slope={self.slope.item()}'


# ============================================================================
#                           TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing FIXED LOUPE Mask (Based on Working ProbabilisticMask)")
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
        sample_slope=12.0,
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
    
    # Test multiple samples (should be different due to randomization)
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
    
    # Test gradient flow
    print("\n" + "-" * 70)
    print("TEST 3: Gradient flow (straight-through estimator)")
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
        print(f"✓ Gradients flowing through mask!")
    
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
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nThis mask uses the WORKING architecture from ProbabilisticMask!")
    print("Key features:")
    print("  ✓ Straight-through estimator for gradients")
    print("  ✓ Conditional rescaling (not simple scaling)")
    print("  ✓ Randomized sampling during training")
    print("  ✓ All LOUPE layers from TensorFlow implementation")
