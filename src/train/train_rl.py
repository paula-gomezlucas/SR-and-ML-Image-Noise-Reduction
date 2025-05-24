import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.models.rl_policy import RLPolicy, sample_action
from src.sr_rl_env import SRRLImageEnv
import numpy as np
import random
import os

# Example, needs to scale
image_paths = ["data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.fits"]
gt_paths = ["data/benchmark/test.cat"]

def train_rl(num_steps=5000, log_dir="runs/rl", checkpoint_path="checkpoints/best_model.pt", patience=200):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    env = SRRLImageEnv(image_paths, gt_paths)
    policy = RLPolicy(input_channels=env.observation_space.shape[0])
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir=log_dir)
    obs, _ = env.reset()

    best_reward = float('-inf')
    steps_without_improvement = 0

    for step in range(num_steps):
        noise_logits, nl_logits = policy(obs.unsqueeze(0))
        action = sample_action(noise_logits, nl_logits)

        print(f"Step {step}: Action={action}")  # debugging line

        obs, reward, done, _, info = env.step(action)

        # Compute loss (policy gradient, simplified)
        loss = -reward * torch.log_softmax(nl_logits, dim=-1)[0, action['nonlinearity']]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Reward", reward, step)
        writer.add_scalar("Loss", loss.item(), step)
        writer.add_scalar("NoiseLevel", action['noise_level'], step)
        writer.add_scalar("Nonlinearity", action['nonlinearity'], step)

        if step % 50 == 0:
            print(f"Step {step}: Reward={reward:.4f} Loss={loss.item():.4f}")

        # Check for improvement
        if reward > best_reward:
            best_reward = reward
            steps_without_improvement = 0
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at step {step} with reward {reward:.4f}")
        else:
            steps_without_improvement += 1

        if steps_without_improvement >= patience:
            print(f"Early stopping at step {step} (no improvement for {patience} steps)")
            break

    writer.close()

if __name__ == "__main__":
    train_rl()