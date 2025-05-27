import torch
import torch.nn.functional as F
import numpy as np

# Apply stochastic resonance with customizable noise and nonlinearity

def apply_stochastic_resonance(image, action_params, noise_type='gaussian', nonlinearity='tanh', normalize=True):
    """
    Apply stochastic resonance to an image using a selected type of noise and nonlinearity,
    with parameters controlled by an RL agent.

    Parameters:
        image (torch.Tensor): Input image tensor (1xHxW or CxHxW).
        action_params (dict): RL agent output, e.g., {'noise_level': float, 'nonlinearity': str}.
        noise_type (str): 'gaussian' or 'poisson'.
        nonlinearity (str): 'tanh', 'relu', or 'identity'.
        normalize (bool): Whether to normalize output to [0, 1] range.

    Returns:
        torch.Tensor: Processed image.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    # Extract RL-controlled parameters
    noise_level = action_params.get('noise_level', 0.01)
    nonlinearity = action_params.get('nonlinearity', nonlinearity)

    # Add noise
    if noise_type == 'gaussian':
        noise = torch.randn_like(image) * noise_level
    elif noise_type == 'poisson':
        noise = torch.poisson(image * noise_level) / noise_level - image
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    noisy_image = image + noise

    # Apply nonlinearity
    if nonlinearity == 'tanh':
        processed = torch.tanh(noisy_image)
    elif nonlinearity == 'relu':
        processed = F.relu(noisy_image)
    elif nonlinearity == 'identity':
        processed = noisy_image
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    # Normalize result
    if normalize:
        min_val = processed.min()
        max_val = processed.max()
        processed = (processed - min_val) / (max_val - min_val + 1e-8)

    return processed