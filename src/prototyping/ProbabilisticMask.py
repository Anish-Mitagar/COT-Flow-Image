import torch
import torch.nn as nn

class ProbabilisticMask(nn.Module):
    def __init__(self, input_dim, temperature=1.0, mask=None, convolutional=False, image_shape=None, threshold = 0.5):
        super(ProbabilisticMask, self).__init__()
        self.input_dim = input_dim
        self.temperature = temperature
        if mask is None:
            self.logits = nn.Parameter(torch.rand((1, input_dim), dtype=torch.float64))
        else:
            self.logits = nn.Parameter(mask)
        self.conv = convolutional
        self.image_shape = image_shape
        self.threshold = threshold
        self.latest_mask = None
        self.latest_bin_mask = None

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape, dtype=torch.float64).to(self.logits.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        gumbel_noise = self.sample_gumbel(logits.shape)
        y = logits + gumbel_noise
        return torch.sigmoid(y / self.temperature)

    def forward(self, x):
        batch_size = x.size(0)
        mask_logits = self.logits.repeat(batch_size, 1)
        mask = self.gumbel_softmax_sample(mask_logits)
        self.latest_mask = mask.detach()[0]
        mask = (mask > self.threshold).float()  
        if self.conv:
            mask = mask.view(batch_size, 1, self.image_shape[0], self.image_shape[1]) 
        self.latest_bin_mask = mask.detach()[0]
        return x * mask, mask