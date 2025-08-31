import numpy as np
import pandas as pd
import time
from stable_baselines3 import PPO
from environments.HypersonicEnv import HypersonicEnv

# --- Constants ---
MODEL_PATH = "models/best_model/best_model"  # Path to the best model (no .zip extension)
NUM_EPISODES = 10                            # Number of test flights to run

# --- Load Model and Environment ---
print(f"Loading trained model from: {MODEL_PATH}.zip")
model = PPO.load(MODEL_PATH)
env = HypersonicEnv()

# --- Evaluation Loop ---
print(f"\n--- Running {NUM_EPISODES} Evaluation Episodes ---")
all_rewards = []
all_lengths = []

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    # store stepwise logs for this episode
    logs = []

    while not (terminated or truncated):
        # Use deterministic action from trained agent
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract flight dynamics variables
        alt = env.fdm['position/h-sl-ft']
        mach = env.fdm['velocities/mach']
        roll = env.fdm['attitude/phi-deg']
        alpha = env.fdm['aero/alpha-deg']

        # Print flight status
        print(f"Alt: {alt:8.2f} ft | Mach: {mach:5.2f} | Roll: {roll:6.2f}° | α: {alpha:6.2f}° | Reward: {reward:+.4f}")

        # Log step info
        logs.append({
            "step": step_count,
            "altitude_ft": alt,
            "mach": mach,
            "roll_deg": roll,
            "alpha_deg": alpha,
            "reward": reward
        })

        total_reward += reward
        step_count += 1

        # Slow down for readability (sim runs ~60 Hz)
        time.sleep(1/60.0)

    # Save episode logs to CSV
    df = pd.DataFrame(logs)
    csv_filename = f"flight_logs_episode_{episode+1}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"📁 Saved log to {csv_filename}")

    # Print episode summary
    print(f"\nEpisode {episode + 1} Finished:")
    print(f"  Duration = {step_count} steps")
    print(f"  Total Reward = {total_reward:.2f}\n")

    all_rewards.append(total_reward)
    all_lengths.append(step_count)

env.close()

# --- Final Statistics ---
print("\n--- ✅ Evaluation Summary ---")
print(f"Average Episode Duration: {np.mean(all_lengths):.2f} steps")
print(f"Average Total Reward: {np.mean(all_rewards):.2f}")
