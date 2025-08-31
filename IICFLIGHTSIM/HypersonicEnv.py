import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim


class HypersonicEnv(gym.Env):
    """
    Custom Gymnasium environment for hypersonic vehicle (e.g. X-15) 
    using JSBSim flight dynamics.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 aircraft_name: str = "x15",
                 dt: float = 1/60.0,
                 substeps: int = 5,
                 action_gain: float = 0.1,
                 slew: float = 0.2):
        super().__init__()

        # JSBSim setup
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.load_model(aircraft_name)
        self.fdm.set_property_value("simulation/dt", dt / substeps)

        # time resolution
        self.dt = dt
        self.substeps = substeps

        # action/observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # tracked state variables
        self.state_props = [
            "position/h-sl-ft",      # altitude (ft)
            "velocities/mach",       # Mach number
            "aero/alpha-deg",        # angle of attack
            "aero/beta-deg",         # sideslip
            "attitude/phi-rad",      # roll
            "attitude/theta-rad",    # pitch
            "velocities/p-rad_sec",  # roll rate
            "velocities/q-rad_sec",  # pitch rate
        ]

        # controls
        self.ctrl_props = [
            "fcs/aileron-cmd-norm",
            "fcs/elevator-cmd-norm",
            "fcs/rudder-cmd-norm"
        ]

        # action scaling
        self.action_gain = float(action_gain)  # scale policy → surface motion
        self.slew = float(slew)                # max change per step
        self._u = np.zeros(3, dtype=np.float32)  # last commanded controls

    # ----------------------
    # state normalization
    # ----------------------
    def _get_state(self):
        s = np.array([self.fdm[p] for p in self.state_props], dtype=np.float32)

        # normalize (clip conservative)
        s[0] = (s[0] - 70000.0) / 20000.0     # altitude: 70k ±20k → [-1,1]
        s[1] = (s[1] - 5.0) / 2.0             # Mach: 5 ±2
        s[2] /= 10.0                          # alpha ±10°
        s[3] /= 10.0                          # beta ±10°
        s[4] /= np.pi                         # roll ±π
        s[5] /= (np.pi/2)                     # pitch ±90°
        s[6] /= 1.0                           # roll rate ±1 rad/s
        s[7] /= 1.0                           # pitch rate ±1 rad/s

        return np.clip(s, -1.0, 1.0)

    # ----------------------
    # environment API
    # ----------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # initial conditions
        self.fdm.set_property_value("ic/h-sl-ft", 70000.0)
        self.fdm.set_property_value("ic/mach", 5.0)
        self.fdm.set_property_value("ic/alpha-deg", 2.5)
        self.fdm.set_property_value("ic/beta-deg", 0.0)
        self.fdm.set_property_value("ic/phi-deg", 0.0)
        self.fdm.set_property_value("ic/theta-deg", 0.0)
        self.fdm.set_property_value("ic/psi-true-deg", 0.0)

        # zero body rates
        self.fdm.set_property_value("ic/p-rad_sec", 0.0)
        self.fdm.set_property_value("ic/q-rad_sec", 0.0)
        self.fdm.set_property_value("ic/r-rad_sec", 0.0)

        # solve initial conditions
        self.fdm.run_ic()

        # neutralize controls
        self._u[:] = 0.0
        for i, prop in enumerate(self.ctrl_props):
            self.fdm.set_property_value(prop, float(self._u[i]))

        # settle a few frames
        for _ in range(20):
            self.fdm.run()

        return self._get_state(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)

        # control update with slew limit
        target = self._u + self.action_gain * a
        delta = np.clip(target - self._u, -self.slew, self.slew)
        self._u = np.clip(self._u + delta, -1.0, 1.0)

        for i, prop in enumerate(self.ctrl_props):
            self.fdm.set_property_value(prop, float(self._u[i]))

        # substeps integration
        for _ in range(self.substeps):
            self.fdm.run()

        # state
        s = self._get_state()

        # physics values
        alt_ft = self.fdm["position/h-sl-ft"]
        mach   = self.fdm["velocities/mach"]
        phi    = self.fdm["attitude/phi-rad"]
        theta  = self.fdm["attitude/theta-rad"]
        alpha  = self.fdm["aero/alpha-deg"]
        beta   = self.fdm["aero/beta-deg"]
        p      = self.fdm["velocities/p-rad_sec"]
        q      = self.fdm["velocities/q-rad_sec"]

        # -----------------
        # reward shaping
        # -----------------
        reward = 0.05  # base living reward

        # tracking errors
        reward -= ((alt_ft - 70000.0) / 15000.0) ** 2 * 0.6
        reward -= (phi / np.deg2rad(30)) ** 2 * 0.5
        reward -= (theta / np.deg2rad(15)) ** 2 * 0.3
        reward -= max(0.0, (4.7 - mach)) ** 2 * 0.3

        # dynamic stability penalties
        reward -= (beta / 5.0) ** 2 * 0.2
        reward -= (p / 0.6) ** 2 * 0.15
        reward -= (q / 0.6) ** 2 * 0.15
        reward -= max(0.0, abs(alpha) - 8.0) ** 2 * 0.1

        # safety termination
        terminated = False
        unsafe = (
            abs(alt_ft - 70000.0) > 15000.0 or
            abs(phi) > np.deg2rad(60) or
            abs(theta) > np.deg2rad(25) or
            mach < 3.8 or
            abs(beta) > 12.0 or
            abs(p) > 1.5 or
            abs(q) > 1.5
        )
        if unsafe:
            reward = -10.0
            terminated = True

        return s, reward, terminated, False, {}

    def render(self, mode="human"):
        alt   = self.fdm["position/h-sl-ft"]
        mach  = self.fdm["velocities/mach"]
        roll  = self.fdm["attitude/phi-deg"]
        pitch = self.fdm["attitude/theta-deg"]
        alpha = self.fdm["aero/alpha-deg"]
        beta  = self.fdm["aero/beta-deg"]

        print(f"Alt:{alt:8.1f} ft | M:{mach:4.2f} | "
              f"Roll:{roll:6.2f}° | Pitch:{pitch:6.2f}° | "
              f"α:{alpha:5.2f}° | β:{beta:5.2f}°")
