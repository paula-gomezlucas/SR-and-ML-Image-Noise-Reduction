import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt
from models.rl_policy import RLPolicy, sample_action
from models.swin_backbone import SwinBackbone
from models.detection_head import HeatmapDetector
from models.stochastic_resonance import apply_stochastic_resonance
from utils.coordinate_decoder import decode_coordinates_from_heatmap
from utils.evaluation import compute_reward
from sr_rl_env import SRRLImageEnv

# --- Config ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FITS_LIST = [os.path.abspath(os.path.join(BASE_DIR, "../../data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.fits"))]
CATALOG_LIST = [os.path.abspath(os.path.join(BASE_DIR, "../../data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.cat"))]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 5  # Increase to 300 for full run
TILE_SIZE = 224
TILE_STRIDE = 112
LR = 1e-4
SAVE_PATH = "../checkpoints/rl_policy_nosr.pt"
CHECKPOINT_DIR = "../checkpoints/"
PLOT_PATH = "output_visualizations/rl_rewards_nosr.png"
os.makedirs("output_visualizations", exist_ok=True)

# --- Initialize ---
env = SRRLImageEnv(FITS_LIST, CATALOG_LIST)
backbone = SwinBackbone(in_chans=1).to(DEVICE)
detector = HeatmapDetector().to(DEVICE)
policy = RLPolicy(input_channels=1).to(DEVICE)
optimizer = optim.Adam(policy.parameters(), lr=LR)

print(f"Training policy on {DEVICE}")

reward_history = []

def save_checkpoint_and_plot(ep):
    torch.save(policy.state_dict(), SAVE_PATH)
    if len(reward_history) >= 5:
        plt.figure()
        plt.plot(np.arange(1, len(reward_history) + 1), reward_history, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("RL Reward Curve")
        plt.grid(True)
        plt.savefig(PLOT_PATH)
        plt.close()
        print(f"[Saved] Episode {ep+1} → checkpoint + reward plot")

try:
    for episode in range(NUM_EPISODES):
        
        all_rewards = []
        print(f"\n[Episode {episode}]")
        env.reset()  # Ensures env.image and env.gt_catalog are initialized

        for y in range(0, env.image.shape[1] - TILE_SIZE + 1, TILE_STRIDE):
            for x in range(0, env.image.shape[2] - TILE_SIZE + 1, TILE_STRIDE):

                tile = env.image[0, y:y+TILE_SIZE, x:x+TILE_SIZE]  # Extract 2D tile from 3D tensor
                tile_tensor = tile.unsqueeze(0).unsqueeze(0).to(DEVICE)  # Shape [1, 1, 224, 224]

                # Compute basic tile image stats to skip blank tiles
                tile_img = tile_tensor.squeeze().cpu().numpy()
                if tile_img.std() < 0.01:
                    continue  # visually empty tile — skip

                # Get action from policy
                # noise_level, nl_logits = policy(tile_tensor)
                # action = sample_action(noise_level, nl_logits)
 
                # nl_index = int(action['nonlinearity'])  # save this BEFORE converting to string
                # nl_map = ['tanh', 'relu', 'identity']
                # action['nonlinearity'] = nl_map[nl_index]

                # No SR nor RL run
                action = {'noise_level': 0.0, 'nonlinearity': 'identity'}
                nl_index = 2  # identity


                # Apply SR
                # sr_tile = apply_stochastic_resonance(tile_tensor.squeeze(0), action).unsqueeze(0)  # [1, 1, 224, 224]
                sr_tile = tile_tensor.clone() # No SR run for comparison

                # TEMP: bypass SR to see if it improves detections
                # sr_tile = tile_tensor.clone()

                # Extract features and run detection head
                with torch.no_grad():
                    assert sr_tile.shape[-1] == 224 and sr_tile.shape[-2] == 224, f"SR tile shape invalid: {sr_tile.shape}"
                    features = backbone(sr_tile)
                    if features.dim() == 3:
                        B, N, C = features.shape
                        H_ = W_ = int(N ** 0.5)
                        features = features.view(B, H_, W_, C).permute(0, 3, 1, 2).contiguous()
                    elif features.dim() == 4 and features.shape[-1] == 768:
                        features = features.permute(0, 3, 1, 2).contiguous()
                    heatmap = detector(features)
                    heatmap = torch.sigmoid(heatmap)  # squash logits to (0, 1)
                    heatmap_np = heatmap.squeeze().cpu().numpy()
                    if heatmap_np.max() < 0.15:
                        plt.imsave(f"output_visualizations/bad_sr_x{x}_y{y}.png", tile_img, cmap='gray')

                # Decode detections
                detections = decode_coordinates_from_heatmap(
                    heatmap_np,
                    threshold=0.1,  # already lowered
                    min_distance=4,
                    max_peaks=500,
                    refine_method="gaussian"
                )

                if heatmap_np.max() < 0.1:
                    import matplotlib.pyplot as plt
                    plt.imsave(f"output_visualizations/blank_heatmap_{x}_{y}.png", heatmap_np, cmap='inferno')
                    plt.imsave(f"output_visualizations/sr_input_{x}_{y}.png", sr_tile.squeeze().cpu().numpy(), cmap='gray')
                
                if np.random.rand() < 0.005:
                    print(f"[Tile x={x}, y={y}] Detections: {len(detections)} | Reward: {reward:.3f}")

                gt_coords = env.get_ground_truth_in_tile(x, y, TILE_SIZE)

                if len(gt_coords) == 0 and len(detections) == 0:
                    continue  # nothing to learn from — skip reward and update

                # Compute reward and update
                reward = compute_reward(detections, gt_coords, radius=6)
                reward = min(reward, 0.5)  # optional clamp
                all_rewards.append(reward)
                
                if np.random.rand() < 0.003:
                    import matplotlib.pyplot as plt
                    img = sr_tile.squeeze().cpu().numpy()
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img, cmap='gray')
                    if len(detections) > 0:
                        plt.scatter(detections[:,1], detections[:,0], color='red', label='Pred', s=10)
                    if len(gt_coords) > 0:
                        plt.scatter(gt_coords[:,0], gt_coords[:,1], color='cyan', label='GT', s=10, alpha=0.6)
                    plt.legend()
                    plt.title(f"x={x}, y={y}, reward={reward:.2f}")
                    plt.savefig(f"output_visualizations/overlay_tile_x{x}_y{y}.png")
                    plt.close()

                log_prob = torch.tensor(0.0, requires_grad=True, device=DEVICE)

                if reward > 0.2:
                    print(f"[POSITIVE] Tile ({x}, {y}) → reward={reward:.3f}, detections={len(detections)}, GT={len(gt_coords)}")


                loss = log_prob # Zero loss
                loss.backward()

                optimizer.step()

        if len(all_rewards) > 0:
            avg_reward = np.mean(all_rewards)
        else:
            avg_reward = 0.0
            print("Warning: No valid detections this episode.")

        print(f"Avg reward: {avg_reward:.4f}")
        reward_history.append(avg_reward)

        if episode % 5 == 0:
            plt.plot(reward_history)
            plt.xlabel("Episode")
            plt.ylabel("Avg Reward")
            plt.savefig(os.path.join(CHECKPOINT_DIR, "reward_curve.png"))
            plt.close()
            torch.save(policy.state_dict(), os.path.join(CHECKPOINT_DIR, f"rl_policy_ep{episode}.pt"))

except Exception as e:
    print(f"[Exception] Training interrupted — saving current state...")
    save_checkpoint_and_plot(episode)
    raise e

# Final save
save_checkpoint_and_plot(NUM_EPISODES - 1)
np.savetxt("output_visualizations/rl_reward_log_nosr.txt", reward_history)
print("Training complete.")