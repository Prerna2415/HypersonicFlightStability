import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim


class HypersonicEnv(gym.Env):
    """Custom Gymnasium environment for controlling a hypersonic aircraft in JSBSim."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, aircraft_name='x15', dt=1/30.0, max_steps=1000):
        super(HypersonicEnv, self).__init__()

        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.load_model(aircraft_name)
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0

        # --- Action space: [aileron, elevator, rudder], normalized -1 to 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # --- Observation space: normalized values
        low = np.array([
            50000 / 100000,  # Altitude
            3.0 / 10,        # Mach
            -15 / 90,        # Alpha
            -15 / 90,        # Beta
            -180 / 180,      # Roll
            -15 / 90,        # Pitch
            -45 / 45         # Yaw rate
        ], dtype=np.float32)

        high = np.array([
            90000 / 100000,
            7.0 / 10,
            15 / 90,
            15 / 90,
            180 / 180,
            15 / 90,
            45 / 45
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, shape=(7,), dtype=np.float32)

        # Properties to read from JSBSim
        self.state_properties = [
            'position/h-sl-ft',      # Altitude (ft)
            'velocities/mach',       # Mach
            'aero/alpha-deg',        # AoA (deg)
            'aero/beta-deg',         # Sideslip (deg)
            'attitude/phi-rad',      # Roll (rad)
            'attitude/theta-rad',    # Pitch (rad)
            'velocities/r-rad_sec'   # Yaw rate (rad/s)
        ]

        # Control properties
        self.control_properties = [
            'fcs/aileron-cmd-norm',
            'fcs/elevator-cmd-norm',
            'fcs/rudder-cmd-norm'
        ]

    def _get_state(self):
        """Retrieve and normalize state from JSBSim."""
        state = np.array(
            [self.fdm.get_property_value(prop) for prop in self.state_properties],
            dtype=np.float32
        )

        # Normalize
        state[0] /= 100000   # Altitude
        state[1] /= 10       # Mach
        state[2] /= 90       # Alpha
        state[3] /= 90       # Beta
        state[4] /= np.pi    # Roll
        state[5] /= (np.pi/2)  # Pitch
        state[6] /= (np.pi/4)  # Yaw rate

        return state

    def reset(self, seed=None, options=None):
        """Reset environment to initial disturbed hypersonic state."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.fdm.set_property_value('ic/h-sl-ft', 70000)
        self.fdm.set_property_value('ic/mach', 5.0)
        self.fdm.set_property_value('ic/psi-true-deg', 0)
        self.fdm.set_property_value('ic/phi-deg', np.random.uniform(-5, 5))

        self.fdm.run_ic()

        for prop in self.control_properties:
            self.fdm.set_property_value(prop, 0.0)

        self.current_step = 0
        return self._get_state(), {}

    def step(self, action):
        """Run one timestep of the simulation."""
        self.current_step += 1

        # Apply controls
        for i, prop in enumerate(self.control_properties):
            self.fdm.set_property_value(prop, float(action[i]))

        self.fdm.run()

        state = self._get_state()

        # Unpack normalized state
        altitude_norm, mach_norm, alpha_norm, _, roll_norm, _, _ = state
        altitude_ft = altitude_norm * 100000
        roll_rad = roll_norm * np.pi
        alpha_deg = alpha_norm * 90

        # --- Reward shaping ---
        reward = 0.0  # ensure float
        alt_error = abs(altitude_ft - 70000)
        reward += max(0.0, 1.0 - alt_error / 5000.0)
        roll_error = abs(roll_rad)
        reward += max(0.0, 1.0 - roll_error / (np.pi/4))
        if abs(alpha_deg) > 10:
            reward -= (abs(alpha_deg) - 10) * 0.1
        reward -= float(np.sum(np.abs(action)) * 0.05)  # cast to float

        # --- Termination / truncation ---
        terminated = False
        truncated = False

        if altitude_ft < 50000 or altitude_ft > 90000:
            reward = -100.0
            terminated = True
        if abs(roll_rad) > np.deg2rad(80):
            reward = -100.0
            terminated = True
        if self.current_step >= self.max_steps:
            truncated = True

        # Explicit casts to Python types
        return state, float(reward), bool(terminated), bool(truncated), {}


    def render(self):
        state_unnormalized = [
            self.fdm.get_property_value('position/h-sl-ft'),
            self.fdm.get_property_value('velocities/mach'),
            self.fdm.get_property_value('aero/alpha-deg'),
            self.fdm.get_property_value('attitude/phi-deg')
        ]
        print(
            f"Alt: {state_unnormalized[0]:.0f} ft | "
            f"Mach: {state_unnormalized[1]:.2f} | "
            f"AoA: {state_unnormalized[2]:.2f}° | "
            f"Roll: {state_unnormalized[3]:.2f}°"
        )

    def close(self):
        pass
