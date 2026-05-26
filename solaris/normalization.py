import torch

def transform(input_data, coeff_1, coeff_2, scale_factors):
    """
    Apply normalization transform to solar data.
    
    Args:
        input_data: Input tensor to normalize
        coeff_1: First normalization coefficient
        coeff_2: Second normalization coefficient  
        scale_factors: Scaling factors for each channel
        
    Returns:
        Normalized tensor
    """
    if input_data.dim() == 5:
        scale_shape = (1, 1, -1, 1, 1)
    elif input_data.dim() == 4:
        scale_shape = (1, -1, 1, 1)
    else:
        raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(input_data.shape)}.")

    scale_factors = scale_factors.to(device=input_data.device, dtype=input_data.dtype)
    coeff_1 = coeff_1.to(device=input_data.device, dtype=input_data.dtype)
    coeff_2 = coeff_2.to(device=input_data.device, dtype=input_data.dtype)
    scaled_data = input_data / scale_factors.view(*scale_shape)
    
    epsilon = torch.as_tensor(1e-3, device=input_data.device, dtype=input_data.dtype)
    max_value = torch.as_tensor(2.5, device=input_data.device, dtype=input_data.dtype)
    clipped_data = torch.minimum(scaled_data, max_value)
    log_term = (torch.log(torch.maximum(scaled_data, epsilon)) - torch.log(epsilon)) / torch.log(epsilon)
    
    normalized_data = coeff_1 * clipped_data - coeff_2 * log_term
    return normalized_data
