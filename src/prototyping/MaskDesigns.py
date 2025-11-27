import torch

def RandomMask(input_dim):
    return torch.rand((1, input_dim), dtype=torch.float64)

def DiamondMask(input_dim):
    size = input_dim
    tensor = torch.zeros((size, size), dtype=torch.float64)
    center = size // 2
    max_value_center = 0.99
    min_value_center = 0.5
    max_value_outer = 0.49
    min_value_outer = 0.01

    for i in range(size):
        for j in range(size):
            distance = abs(i - center) + abs(j - center)
            if distance <= center:
                value = min_value_center + (max_value_center - min_value_center) * (1 - distance / center)
                tensor[i, j] = value
            else:
                outer_distance = distance - center
                max_outer_distance = center  
                value = max_value_outer - (max_value_outer - min_value_outer) * (outer_distance / max_outer_distance)
                tensor[i, j] = value

    flattened_tensor = tensor.view(1, -1)
    return flattened_tensor
    

def MaskDesignTwo(input_dim):
    return torch.rand((1, input_dim), dtype=torch.float64)

def MaskDesignThree(input_dim):
    return torch.rand((1, input_dim), dtype=torch.float64)