from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environments.HypersonicEnv import HypersonicEnv

# 1. Create environment
env = HypersonicEnv()

# 2. Check environment compliance
check_env(env, warn=True)

# 3. Define RL model
model = PPO("MlpPolicy", env, verbose=1)

# 4. Train
model.learn(total_timesteps=100000)

# 5. Save trained model
model.save("../models/hypersonic_ppo_agent")

print("Training complete, model saved in ../models/")
