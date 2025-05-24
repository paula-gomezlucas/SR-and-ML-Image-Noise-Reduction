import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np

# Define action space
# - noise_level: continuous in [0, 0.2]
# - nonlinearity: discrete {0: tanh, 1: relu, 2: identity}
action_space = Dict({
    'noise_level': Box(low=0.0, high=0.2, shape=(), dtype=np.float32),
    'nonlinearity': Discrete(3)
})

class RLPolicy(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.noise_head = nn.Linear(32, 1)      # Continuous: noise level
        self.nl_head = nn.Linear(32, 3)         # Categorical: nonlinearity index

    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        noise_level = torch.sigmoid(self.noise_head(features)) * 0.2
        nl_logits = self.nl_head(features)
        return noise_level.squeeze(dim=-1), nl_logits


def sample_action(noise_level, nl_logits):
    """
    Sample an action dictionary from policy outputs.

    Returns:
        dict with 'noise_level' (float) and 'nonlinearity' (str)
    """
    nonlinearity_index = torch.distributions.Categorical(logits=nl_logits).sample()
    return {
        'noise_level': noise_level.item(),
        'nonlinearity': nonlinearity_index.item()  # return index instead of string
    }