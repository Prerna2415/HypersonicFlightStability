import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import jsbsim
import os
import sys

# Add the IICFLIGHTSIM directory to the Python path
iicflightsim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(iicflightsim_dir)

from environments.HypersonicEnv import HypersonicEnv

def visualize_model(model_path, num_steps=500):
    """
    Visualizes the behavior of a trained model in the HypersonicEnv environment.

    Args:
        model_path (str): Path to the trained model.
        num_steps (int): Number of simulation steps to run.
    """

    # Create environment
    env = HypersonicEnv()

    # Load the trained model
    model = PPO.load(model_path)

    # Initialize lists to store state data
    altitudes = []
    machs = []
    alphas = []
    betas = []
    rolls = []
    pitches = []
    yaw_rates = []

    # Reset the environment
    obs, _ = env.reset()

    # Run the simulation
    for _ in range(num_steps):
        # Use the model to predict the action
        action, _states = model.predict(obs, deterministic=True)

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Store the state data (unnormalized)
        altitudes.append(env.fdm['position/h-sl-ft'])
        machs.append(env.fdm['velocities/mach'])
        alphas.append(env.fdm['aero/alpha-deg'])
        betas.append(env.fdm['aero/beta-deg'])
        rolls.append(env.fdm['attitude/phi-deg'])
        pitches.append(env.fdm['attitude/theta-deg'])
        yaw_rates.append(env.fdm['velocities/r-rad_sec'])

        if terminated or truncated:
            obs, _ = env.reset()

    # Close the environment
    env.close()

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # Plot altitude
    axs[0, 0].plot(altitudes)
    axs[0, 0].set_title('Altitude (ft)')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Altitude (ft)')

    # Plot Mach number
    axs[0, 1].plot(machs)
    axs[0, 1].set_title('Mach Number')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Mach')

    # Plot angle of attack
    axs[1, 0].plot(alphas)
    axs[1, 0].set_title('Angle of Attack (deg)')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Alpha (deg)')

    # Plot roll angle
    axs[1, 1].plot(rolls)
    axs[1, 1].set_title('Roll Angle (deg)')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('Roll (deg)')

   # Plot pitch angle
    axs[2, 0].plot(pitches)
    axs[2, 0].set_title('Pitch Angle (deg)')
    axs[2, 0].set_xlabel('Time Step')
    axs[2, 0].set_ylabel('Pitch (deg)')

    # Plot yaw rate
    axs[2, 1].plot(yaw_rates)
    axs[2, 1].set_title('Yaw Rate (rad/s)')
    axs[2, 1].set_xlabel('Time Step')
    axs[2, 1].set_ylabel('Yaw Rate (rad/s)')

    # Adjust layout and display the plot
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = "../models/hypersonic_ppo_agent"  # Replace with the actual path to your model
    visualize_model(model_path, num_steps=500)
