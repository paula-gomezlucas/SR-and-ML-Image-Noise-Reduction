import numpy as np
import torch
from gymnasium import Env, spaces
from src.models.rl_policy import RLPolicy, sample_action
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
