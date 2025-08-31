import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim

class HypersonicEnv(gym.Env):
    """Custom Gym environment for RL control of a hypersonic aircraft in JSBSim."""

    def __init__(self, aircraft_name='x15', dt=1/30.0):
        super(HypersonicEnv, self).__init__()

        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.load_model(aircraft_name)
        self.dt = dt
        # NEW, correct version
        self.fdm.set_property_value('simulation/dt', self.dt)

        # Define Action Space: [aileron, elevator, rudder] normalized between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define Observation Space: [altitude, mach, alpha, roll, pitch, yaw_rate]
        # Normalizing these values helps the RL agent learn more effectively.
        low = np.array([-1.0] * 6)
        high = np.array([1.0] * 6)
        self.observation_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)

        # State and control properties for JSBSim
        self.state_properties = [
            'position/h-sl-ft',      # Altitude (ft)
            'velocities/mach',       # Mach number
            'aero/alpha-deg',        # Angle of attack (deg)
            'attitude/phi-rad',      # Roll angle (rad)
            'attitude/theta-rad',    # Pitch angle (rad)
            'velocities/r-rad_sec'   # Yaw rate (rad/s)
        ]
        self.control_properties = [
            'fcs/aileron-cmd-norm',
            'fcs/elevator-cmd-norm',
            'fcs/rudder-cmd-norm'
        ]

    def _get_state(self):
        """Retrieves and normalizes the state from JSBSim."""
        state = np.array([self.fdm[prop] for prop in self.state_properties], dtype=np.float32)
        
        # Normalize state values to be roughly within [-1, 1]
        state[0] = (state[0] - 70000) / 20000  # Altitude (centered at 70k ft)
        state[1] = (state[1] - 5.0) / 2.0      # Mach (centered at Mach 5)
        state[2] /= 15.0                       # Alpha (deg)
        state[3] /= np.pi                      # Roll (rad)
        state[4] /= (np.pi / 2)                # Pitch (rad)
        state[5] /= (np.pi / 4)                # Yaw rate (rad/s)
        
        return np.clip(state, -1.0, 1.0)

    # This is the NEW, CORRECT version
    def reset(self, seed=None, options=None):
        """Resets the environment to a challenging initial state."""
        # Call the parent class's reset method to handle the seed
        super().reset(seed=seed)
        
        # Set initial flight conditions
        # This is the NEW, correct version
        self.fdm.set_property_value('ic/h-sl-ft', 70000)
        self.fdm.set_property_value('ic/mach', 5.0)
        self.fdm.set_property_value('ic/psi-true-deg', 0)
        self.fdm.set_property_value('ic/phi-deg', np.random.uniform(-5, 5)) 
        self.fdm.set_property_value('ic/theta-deg', np.random.uniform(-2, 2))
        
        self.fdm.run_ic()

        # Set initial controls to neutral
        for prop in self.control_properties:
            self.fdm[prop] = 0.0
            
        # The new standard requires returning the state AND an empty info dict
        return self._get_state(), {}

    # This is the NEW, CORRECT version
    def step(self, action):
        """Executes one time step within the environment."""
        # Set simulator controls from agent's action
        for i, prop in enumerate(self.control_properties):
            self.fdm.set_property_value(prop, action[i])
        
        # Run the simulation for one step
        self.fdm.run()
        
        state = self._get_state()
        
        # --- Reward Calculation ---
        # Get unnormalized values for reward logic
        altitude_ft = self.fdm['position/h-sl-ft']
        roll_rad = self.fdm['attitude/phi-rad']
        alpha_deg = self.fdm['aero/alpha-deg']
        
        # ... (rest of your reward logic is fine) ...
        alt_error = abs(altitude_ft - 70000)
        altitude_reward = max(0, 1.0 - alt_error / 10000)
        roll_error = abs(roll_rad)
        roll_reward = max(0, 1.0 - roll_error / np.deg2rad(45))
        alpha_penalty = max(0, (abs(alpha_deg) - 10) * 0.1)
        action_penalty = np.sum(np.abs(action)) * 0.05
        reward = altitude_reward + roll_reward - alpha_penalty - action_penalty
        
        # --- Termination and Truncation ---
        terminated = False
        truncated = False  # We don't have a time limit, so this is always false

        if alt_error > 20000 or roll_error > np.deg2rad(80) or abs(alpha_deg) > 20:
            reward = -10.0 # Heavy penalty for leaving the flight envelope
            terminated = True
        
        # The new standard requires returning 5 values
        return state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Prints the current state for monitoring."""
        alt = self.fdm['position/h-sl-ft']
        mach = self.fdm['velocities/mach']
        roll = self.fdm['attitude/phi-deg']
        alpha = self.fdm['aero/alpha-deg']
        print(f"Altitude: {alt:8.2f} ft | Mach: {mach:5.2f} | Roll: {roll:6.2f}° | Alpha: {alpha:6.2f}°")