import torch
import numpy as np
import torch.nn.functional as F

def l1_loss(logits):
    return torch.sum(torch.abs(logits))

def l2_loss(logits):
    return torch.sum(logits ** 2)

def entropy_loss(logits):
    """Compute the entropy loss to encourage diversity in the mask."""
    p = torch.sigmoid(logits)
    entropy = -p * torch.log(p + 1e-12) - (1 - p) * torch.log(1 - p + 1e-12)
    return torch.mean(entropy)

def variance_loss(logits):
    """Compute the variance loss to encourage spread in the mask values."""
    p = torch.sigmoid(logits)
    return torch.var(p)

def center_focus_loss(logits, mask_shape):
    """Encourage the mask values to be higher in the center and lower near the borders."""
    p = torch.sigmoid(logits).reshape(mask_shape)
    center = np.array(mask_shape) // 2
    distances = np.sqrt((np.arange(mask_shape[0]) - center[0])[:, None]**2 + (np.arange(mask_shape[1]) - center[1])[None, :]**2)
    distances = distances / np.max(distances)  # Normalize distances to range [0, 1]
    weights = torch.from_numpy(1 - distances).to(logits.device, torch.float64)
    return -torch.mean(p * weights)

def proportion_loss(logits, target_proportion):
    """Compute the loss to enforce a target proportion of 1s in the mask."""
    p = torch.sigmoid(logits)
    return F.binary_cross_entropy(p, torch.full_like(p, target_proportion, dtype=torch.float64))