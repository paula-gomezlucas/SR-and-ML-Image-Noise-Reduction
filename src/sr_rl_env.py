import numpy as np
import torch
from gymnasium import Env, spaces
from models.rl_policy import RLPolicy, sample_action
from models.stochastic_resonance import apply_stochastic_resonance
from models.detection_head import HeatmapDetector
from utils.evaluation import compute_reward
from utils.image_tools import load_fits_tensor
from utils.sextractor_loader import load_cat_as_array
import random

class SRRLImageEnv(Env):
    """
    Gym-compatible environment for RL-based stochastic resonance enhancement.
    Samples from a dataset of historical FITS images and associated SExtractor catalogs.
    """
    def __init__(self, image_paths: list, gt_paths: list):
        super().__init__()
        assert len(image_paths) == len(gt_paths), "Mismatch between image and GT lists"

        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.idx = 0
        self.detector = HeatmapDetector()

        # Placeholder shape for observation space, updated at reset
        dummy_image = load_fits_tensor(self.image_paths[0])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=dummy_image.shape, dtype=np.float32)
        self.action_space = spaces.Dict({
            'noise_level': spaces.Box(low=0.0, high=0.2, shape=(), dtype=np.float32),
            'nonlinearity': spaces.Discrete(3)
        })

    def reset(self, seed=None, options=None):
        self.idx = random.randint(0, len(self.image_paths) - 1)
        self.image = load_fits_tensor(self.image_paths[self.idx])  # torch.Tensor CxHxW
        self.gt_catalog = load_cat_as_array(self.gt_paths[self.idx])  # numpy array Nx2 or Nx3
        return self.image, {}

    def step(self, action):
        action_params = {
            'noise_level': float(action['noise_level']),
            'nonlinearity': ['tanh', 'relu', 'identity'][int(action['nonlinearity'])]
        }

        processed = apply_stochastic_resonance(self.image, action_params)
        detected_objects = self.detector(processed)  # torch.Tensor Nx2
        reward = compute_reward(detected_objects.numpy(), self.gt_catalog)

        done = True  # Single-step env for now
        info = {"action_params": action_params, "image_path": self.image_paths[self.idx]}

        return processed, reward, done, False, info
    
    def compute_log_prob(self, action, noise_level_tensor, nl_logits):
        """
        Compute the log probability of the sampled action.

        Args:
            action: dict with 'noise_level' (float) and 'nonlinearity' (int index)
            noise_level_tensor: tensor of shape [1, 1] from the policy
            nl_logits: tensor of shape [1, 3] from the policy

        Returns:
            log_prob: scalar tensor with the combined log-probability
        """
        # Gaussian noise log prob
        dist_noise = torch.distributions.Normal(noise_level_tensor.squeeze(), 0.1)
        log_prob_noise = dist_noise.log_prob(torch.tensor(action['noise_level'], device=nl_logits.device))

        # Categorical nonlinearity log prob
        dist_nl = torch.distributions.Categorical(logits=nl_logits.squeeze(0))
        log_prob_nl = dist_nl.log_prob(torch.tensor(action['nonlinearity'], device=nl_logits.device))

        return log_prob_noise + log_prob_nl
    
    def apply_sr(self, image_tensor, action):
        action_params = {
            'noise_level': float(action['noise_level']),
            'nonlinearity': ['tanh', 'relu', 'identity'][int(action['nonlinearity'])]
        }
        return apply_stochastic_resonance(image_tensor, action_params)
    
    def get_ground_truth_in_tile(self, x, y, tile_size):
        """
        Extract GT coords within a tile.
        """
        gt = []
        for gx, gy in self.gt_catalog:
            if x <= gx < x + tile_size and y <= gy < y + tile_size:
                gt.append([gx - x, gy - y])  # Local to tile coords
        return np.array(gt)


