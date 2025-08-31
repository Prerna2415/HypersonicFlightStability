import os
from environments.HypersonicEnv import HypersonicEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# --- Constants and Configuration ---
LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# PPO Hyperparameters - these are crucial for performance
PPO_PARAMS = {
    "n_steps": 2048,          # Number of steps to run for each environment per update
    "batch_size": 64,         # Number of samples in a mini-batch
    "n_epochs": 10,           # Number of optimization epochs per update
    "gamma": 0.99,            # Discount factor
    "gae_lambda": 0.95,       # Factor for trade-off of bias vs variance for GAE
    "clip_range": 0.2,        # Clipping parameter for PPO
    "ent_coef": 0.0,          # Entropy coefficient (encourages exploration)
    "learning_rate": 3e-4,    # Learning rate for the optimizer
    "verbose": 1,
    "tensorboard_log": LOG_DIR
}

# --- Environment Setup ---
# Create 4 parallel environments for faster training
# This is one of the most effective ways to speed up RL
train_env = make_vec_env(lambda: HypersonicEnv(), n_envs=4)

# Create a separate environment for evaluation
eval_env = HypersonicEnv()

# --- Callback for Saving the Best Model ---
# This callback will evaluate the agent every 5000 steps and save the
# model if it achieves a new best average reward.
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(MODEL_DIR, "best_model"),
    log_path=os.path.join(LOG_DIR, "eval"),
    eval_freq=5000,
    deterministic=True,
    render=False
)

# --- Model Definition and Training ---
# Create the PPO agent with the specified hyperparameters
model = PPO("MlpPolicy", train_env, **PPO_PARAMS)

print("🚀 Starting agent training...")
# Train for a longer duration for better performance
model.learn(total_timesteps=500000, callback=eval_callback)
print("✅ Training complete.")

# --- Final Model Save ---
# The callback already saved the best model, but we can save the final one too
final_model_path = os.path.join(MODEL_DIR, "final_model")
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")

train_env.close()
eval_env.close()