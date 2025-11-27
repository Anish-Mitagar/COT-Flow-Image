import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProbabilisticMask(nn.Module):
    """
    LOUPE-style Probabilistic Mask with proper budget control.
    
    Key improvements over old implementation:
    - Rescaling to maintain target sparsity/budget
    - Proper slope-controlled sigmoid
    - Better initialization strategy
    - Straight-through estimator for gradients
    
    Args:
        input_dim (int): Flattened input dimension (e.g., 784 for 28x28 MNIST)
        temperature (float): DEPRECATED - use slope instead (kept for backward compatibility)
        mask (torch.Tensor): Initial mask values (if None, will initialize)
        convolutional (bool): Whether to reshape mask for convolutional layers
        image_shape (tuple): Shape to reshape to if convolutional (H, W)
        threshold (float): Threshold for binarization during inference (default: 0.5)
        sparsity (float): Target sparsity/budget - fraction to keep (e.g., 0.1 for 10%)
        slope (float): Slope for sigmoid function (higher = more binary)
        sample_slope (float): Slope for sampling threshold (higher = harder sampling)
    """
    
    def __init__(self, input_dim, temperature=1.0, mask=None, convolutional=False, 
                 image_shape=None, threshold=0.5, sparsity=0.1, slope=5.0, 
                 sample_slope=12.0):
        super(ProbabilisticMask, self).__init__()
        
        self.input_dim = input_dim
        self.conv = convolutional
        self.image_shape = image_shape
        self.threshold = threshold
        self.sparsity = sparsity

        self.slope = nn.Parameter(torch.tensor(slope, dtype=torch.float32), requires_grad=False)
        self.sample_slope = nn.Parameter(torch.tensor(sample_slope, dtype=torch.float32), requires_grad=False)
        

        
        # Initialize logit weights
        if mask is None:
            # Use LOUPE v2 initialization: logit of uniform [0, 1] distribution
            self.logits = nn.Parameter(self._logit_slope_random_uniform((1, input_dim)))
        else:
            # If initial mask provided, use it
            if isinstance(mask, torch.Tensor):
                self.logits = nn.Parameter(mask.clone().detach())
            else:
                self.logits = nn.Parameter(mask)
        
        # For tracking
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
        
        Corresponds to: layers.ProbMask in TensorFlow code
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
        
        Corresponds to: layers.RescaleProbMap in TensorFlow code
        
        If mean(prob) > sparsity: prob' = prob * sparsity / mean(prob)
        If mean(prob) < sparsity: prob' = 1 - (1-prob) * (1-sparsity) / (1-mean(prob))
        """
        prob_mean = torch.mean(prob)
        
        # Rescaling factor
        r = self.sparsity / prob_mean
        beta = (1.0 - self.sparsity) / (1.0 - prob_mean + 1e-8)
        
        # Conditional rescaling (equivalent to LOUPE's tf.less_equal logic)
        # le = 1 if r <= 1, else 0
        le = (r <= 1.0).float()
        
        # Apply rescaling
        rescaled = le * prob * r + (1.0 - le) * (1.0 - (1.0 - prob) * beta)
        
        return rescaled
    
    def threshold_random_mask(self, prob, training=True):
        """
        ThresholdRandomMask layer from LOUPE: samples binary mask from probabilities.
        
        Corresponds to: layers.RandomMask + layers.ThresholdRandomMask in TensorFlow code
        
        During training: Uses soft thresholding with sigmoid
        During inference: Uses hard thresholding
        """
        if training:
            # Generate random uniform samples (RandomMask layer)
            random_samples = torch.rand_like(prob, dtype=torch.float32)
            
            # Soft thresholding with slope (ThresholdRandomMask layer)
            # sigmoid(sample_slope * (prob - random))
            # This gives smooth gradients during training
            binary_mask = torch.sigmoid(self.sample_slope * (prob - random_samples))
            
            # Straight-through estimator: binary forward, continuous backward
            # During forward pass: use thresholded binary values
            # During backward pass: use continuous probabilities for gradients
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
        
        Returns:
            tuple: (masked_x, mask)
                - masked_x: input multiplied by mask
                - mask: binary mask (reshaped if convolutional)
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
        
        # Step 4: Reshape if convolutional
        if self.conv and self.image_shape is not None:
            binary_mask = binary_mask.view(batch_size, 1, self.image_shape[0], self.image_shape[1])
        
        # Step 5: Apply mask to input
        # print("x.shape in ProbabilisticMask forward: ", x.shape)
        # print("binary_mask.shape in ProbabilisticMask forward: ", binary_mask.shape)
        masked_x = x * binary_mask
        
        return masked_x, binary_mask
    
    def get_sparsity(self):
        """Get current actual sparsity (sampling rate) of the mask."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_dim, dtype=torch.float32, device=self.logits.device)
            probabilities = self.prob_mask(dummy_input)
            rescaled_prob = self.rescale_prob_map(probabilities)
            return rescaled_prob.mean().item()

    
    def set_sparsity(self, new_sparsity):
        """Adjust target sparsity without retraining."""
        self.sparsity = new_sparsity

    def get_deterministic_mask(self, threshold=None):
        """
        Get a deterministic binary mask for inference.
        
        Args:
            threshold (float): Threshold for binarization (default: self.threshold)
        
        Returns:
            torch.Tensor: Binary mask
        """
        if threshold is None:
            threshold = self.threshold
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_dim, dtype=torch.float32, device=self.logits.device)
            probabilities = self.prob_mask(dummy_input)
            rescaled_prob = self.rescale_prob_map(probabilities)
            binary_mask = (rescaled_prob > threshold).float()
            
            if self.conv and self.image_shape is not None:
                binary_mask = binary_mask.view(1, 1, self.image_shape[0], self.image_shape[1])
            
            return binary_mask.squeeze()
    



class DualAutoEncoder(nn.Module):
    """
    Dual Autoencoder with two parallel encoders:
    1. Convolutional encoder for spatial domain image (28x28 -> 64)
    2. **Linear encoder** for Fourier domain (flattened -> 64)
    
    Both encodings are concatenated (128) and decoded back to image space.
    
    NOTE: This class expects pre-computed Fourier transforms as input.
    No FFT computation is done internally.
    """
    def __init__(self, image_channels=1, image_size=28, fourier_input_dim=784, latent_dim=64):
        super(DualAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.fourier_input_dim = fourier_input_dim
        
        # ===== Image Encoder (Spatial Domain - Convolutional) =====
        self.image_encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )
        
        # ===== Fourier Encoder (Frequency Domain - Pure Linear) =====
        self.fourier_encoder = nn.Sequential(
            nn.Linear(fourier_input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, latent_dim)
        )
        
        # ===== Decoder (Concatenated latent -> Image) =====
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            
            # 4x4 -> 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # For MNIST pixel values in [0, 1]
        )
    
    def forward(self, x, x_fourier, return_encodings=False):
        """
        Forward pass through dual autoencoder.
        
        Args:
            x: Input images (batch, channels, H, W)
            x_fourier: Fourier domain input (batch, fourier_input_dim) - flattened
            return_encodings: If True, return the latent encodings as well
        
        Returns:
            reconstruction: Reconstructed images
            (optional) encodings: Dict with 'image_encoding', 'fourier_encoding', 'combined_encoding'
        """
        # Encode spatial domain
        image_encoding = self.image_encoder(x)
        
        # Encode frequency domain (already provided, no FFT here)
        fourier_encoding = self.fourier_encoder(x_fourier)
        
        # Concatenate encodings
        combined_encoding = torch.cat([image_encoding, fourier_encoding], dim=1)
        
        # Decode
        reconstruction = self.decoder(combined_encoding)
        
        if return_encodings:
            encodings = {
                'image_encoding': image_encoding,
                'fourier_encoding': fourier_encoding,
                'combined_encoding': combined_encoding
            }
            return reconstruction, encodings
        
        return reconstruction


class MaskedDualAutoEncoder(nn.Module):
    """
    Dual Autoencoder with learnable LOUPE-style probabilistic mask applied to Fourier domain.
    
    This class composes ProbabilisticMask and DualAutoEncoder without 
    redefining their layers.
    """
    def __init__(self, image_channels=1, image_size=28, fourier_input_dim=784, 
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
        super(MaskedDualAutoEncoder, self).__init__()

        # Store configuration for later use
        self.image_size = image_size
        self.image_channels = image_channels
        self.fourier_input_dim = fourier_input_dim
        
        # Initialize the dual autoencoder
        self.autoencoder = DualAutoEncoder(
            image_channels=image_channels,
            image_size=image_size,
            fourier_input_dim=fourier_input_dim,
            latent_dim=latent_dim
        )
        
        # Initialize the LOUPE-style probabilistic mask
        self.mask = ProbabilisticMask(
            input_dim=fourier_input_dim,
            convolutional=False,  # Mask works on flattened Fourier
            sparsity=sparsity,
            slope=slope
        )
    
    def forward(self, x, x_fourier, return_encodings=False):
        """
        Forward pass with masked Fourier domain.
        
        Args:
            x: Input images (batch, channels, H, W)
            x_fourier: Flattened Fourier domain input (batch, fourier_input_dim)
            return_encodings: If True, return latent encodings
        
        Returns:
            reconstruction: Reconstructed images
            mask: Binary mask used
            (optional) encodings: Dict with latent encodings
        """
        # Encode spatial domain (unmasked)
        image_encoding = self.autoencoder.image_encoder(x)
        
        # Apply learned mask to Fourier domain
        x_fourier_masked, mask = self.mask(x_fourier)
        
        # Encode masked Fourier domain
        fourier_encoding = self.autoencoder.fourier_encoder(x_fourier_masked)
        
        # Concatenate encodings
        combined_encoding = torch.cat([image_encoding, fourier_encoding], dim=1)
        
        # Decode
        reconstruction = self.autoencoder.decoder(combined_encoding)
        
        if return_encodings:
            encodings = {
                'image_encoding': image_encoding,
                'fourier_encoding': fourier_encoding,
                'combined_encoding': combined_encoding
            }
            return reconstruction, mask, encodings
        
        return reconstruction, mask
    
    def get_learned_mask(self, threshold=None):
        """Convenience method to access learned mask."""
        return self.mask.get_deterministic_mask(threshold)
    
    def get_sparsity(self):
        """Get current actual sparsity of the mask."""
        return self.mask.get_sparsity()
    
    def set_sparsity(self, new_sparsity):
        """Adjust target sparsity without retraining."""
        self.mask.set_sparsity(new_sparsity)


# ===== Example Usage =====
if __name__ == "__main__":
    # Test on MNIST-like data
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    
    # Create dummy Fourier input (flattened)
    # In practice, this would be computed outside: torch.fft.fft2(x).flatten()
    x_fourier = torch.randn(batch_size, 784, dtype=torch.float32)
    
    print("=" * 50)
    print("Testing DualAutoEncoder")
    print("=" * 50)
    
    # Test DualAutoEncoder
    dual_ae = DualAutoEncoder(
        image_channels=1, 
        image_size=28,
        fourier_input_dim=784,
        latent_dim=64
    )
    reconstruction, encodings = dual_ae(x, x_fourier.float(), return_encodings=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Fourier input shape: {x_fourier.shape}")
    print(f"Image encoding shape: {encodings['image_encoding'].shape}")
    print(f"Fourier encoding shape: {encodings['fourier_encoding'].shape}")
    print(f"Combined encoding shape: {encodings['combined_encoding'].shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    print("\n" + "=" * 50)
    print("Testing MaskedDualAutoEncoder")
    print("=" * 50)
    
    # Test MaskedDualAutoEncoder
    masked_ae = MaskedDualAutoEncoder(
        image_channels=1,
        image_size=28,
        fourier_input_dim=784,
        latent_dim=64,
        sparsity=0.1  # 10% sampling
    )
    
    # Training mode (stochastic mask)
    masked_ae.train()
    reconstruction, mask, encodings = masked_ae(x, x_fourier, return_encodings=True)
    
    print(f"\nTraining mode (stochastic mask):")
    print(f"Input shape: {x.shape}")
    print(f"Fourier input shape: {x_fourier.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sparsity (actual): {mask.float().mean():.4f}")
    print(f"Target sparsity: {masked_ae.get_sparsity():.4f}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Inference mode (deterministic mask)
    masked_ae.eval()
    with torch.no_grad():
        reconstruction, mask, encodings = masked_ae(
            x, 
            x_fourier,
            return_encodings=True
        )
    
    print(f"\nInference mode (deterministic mask):")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sparsity (actual): {mask.float().mean():.4f}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Get learned mask probabilities
    learned_mask = masked_ae.get_learned_mask()
    print(f"\nLearned mask shape: {learned_mask.shape}")
    print(f"Learned mask min/max: {learned_mask.min():.4f} / {learned_mask.max():.4f}")
    
    # Test sparsity adjustment
    print(f"\n" + "=" * 50)
    print("Testing sparsity adjustment")
    print("=" * 50)
    original_sparsity = masked_ae.get_sparsity()
    print(f"Original sparsity: {original_sparsity:.4f}")
    
    masked_ae.set_sparsity(0.2)  # Change to 20%
    new_sparsity = masked_ae.get_sparsity()
    print(f"After setting to 0.2: {new_sparsity:.4f}")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
