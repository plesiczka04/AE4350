
from __future__ import annotations

import numpy as np

OBS_DIM = 12
RAW_DIM = 3  # [thrust_cmd, roll_cmd, pitch_cmd]

# --- CF2X inner-loop constants (from gym_pybullet_drones DSLPIDControl) ---
PWM2RPM_SCALE = 0.2685
PWM2RPM_CONST = 4070.3
MIN_PWM = 20000.0
MAX_PWM = 65535.0
P_COEFF_TOR = np.array([70000.0, 70000.0, 60000.0])
D_COEFF_TOR = np.array([20000.0, 20000.0, 12000.0])
MIXER_MATRIX = np.array([[-.5, -.5, -1.0],
                         [-.5,  .5,  1.0],
                         [ .5,  .5, -1.0],
                         [ .5, -.5,  1.0]])
MAX_TORQUE = 3200.0


def _hover_pwm(hover_rpm):
    return (hover_rpm - PWM2RPM_CONST) / PWM2RPM_SCALE


def inner_loop(thrust_pwm, roll_des, pitch_des, rpy, rpy_rates):
    """Fixed CF2X attitude/rate controller: setpoint -> 4 motor RPMs.

    A stiff PD on attitude error with rate damping produces body torques, which the
    mixer distributes to the four motors. Rate damping is what prevents tumbling.
    """
    rpy_des = np.array([roll_des, pitch_des, 0.0])
    torques = P_COEFF_TOR * (rpy_des - rpy) - D_COEFF_TOR * rpy_rates
    torques = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)
    pwm = thrust_pwm + MIXER_MATRIX @ torques
    pwm = np.clip(pwm, MIN_PWM, MAX_PWM)
    return PWM2RPM_SCALE * pwm + PWM2RPM_CONST


class CascadedController:
    """Wrap any raw policy (MLP or NEAT) behind ``act(obs) -> 4 RPMs``.

    The policy's 3 raw outputs are squashed to a thrust delta around hover and to
    desired roll/pitch within +/- ``max_tilt``. The inner loop reads the measured
    attitude (obs[6:9]) and body rates (obs[9:12]) from the same observation.
    """

    def __init__(self, raw_policy, hover_rpm=14468.4, thrust_frac=0.4, max_tilt=0.6):
        self.pol = raw_policy
        self.hover_pwm = _hover_pwm(hover_rpm)
        self.thrust_delta_pwm = thrust_frac * self.hover_pwm
        self.max_tilt = float(max_tilt)

    def act(self, obs):
        # Policies return commands already squashed to [-1, 1] (tanh output), so
        # MLP and NEAT feed the inner loop through identical scaling -- a fair
        # comparison. Here we only apply the physical scale factors.
        obs = np.asarray(obs, dtype=float)
        cmd = np.asarray(self.pol.raw(obs), dtype=float)
        thrust_pwm = self.hover_pwm + cmd[0] * self.thrust_delta_pwm
        roll_des = cmd[1] * self.max_tilt
        pitch_des = cmd[2] * self.max_tilt
        return inner_loop(thrust_pwm, roll_des, pitch_des, obs[6:9], obs[9:12])


class MLPPolicy:
    """Fixed-topology MLP: OBS_DIM -> hidden... -> RAW_DIM, tanh hidden activations.

    Weights+biases are one flat vector -- the CMA-ES genome.
    """

    def __init__(self, hidden=(16,)):
        self.layer_sizes = [OBS_DIM, *hidden, RAW_DIM]
        self.n_params = 0
        for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.n_params += a * b + b
        self._W = None

    def set_params(self, flat):
        flat = np.asarray(flat, dtype=float)
        assert flat.size == self.n_params, f"expected {self.n_params}, got {flat.size}"
        self._W, i = [], 0
        for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            w = flat[i:i + a * b].reshape(a, b); i += a * b
            bias = flat[i:i + b]; i += b
            self._W.append((w, bias))
        return self

    def raw(self, obs):
        x = np.asarray(obs, dtype=float)
        for (w, bias) in self._W:
            x = np.tanh(x @ w + bias)  # tanh on every layer incl. output -> cmd in [-1,1]
        return x


class NEATPolicy:
    """Wrap a neat-python feed-forward network behind ``raw(obs)``."""

    def __init__(self, net):
        self.net = net

    def raw(self, obs):
        return np.asarray(self.net.activate(np.asarray(obs, dtype=float)), dtype=float)
