
from __future__ import annotations

import contextlib
import os
import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from .trajectory import Figure8

# --- termination thresholds (a "crash" / loss of tracking) ---
GROUND_Z = 0.05          # m, below this we've hit the floor
MAX_TILT = 1.0           # rad (~57 deg) roll/pitch before we call it unstable
POS_ERR_FACTOR = 2.0     # multiples of the path radius


@contextlib.contextmanager
def _suppress_stdout():
    """Silence gym-pybullet-drones' chatty URDF prints during many rollouts."""
    with open(os.devnull, "w") as devnull:
        old = os.dup(1)
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(old, 1)
            os.close(old)


class Figure8Task:
    """Owns one headless CtrlAviary and scores controllers by rollout.

    One environment instance is reused across many rollouts via ``reset()`` to
    avoid repeatedly loading the URDF / reconnecting PyBullet.
    """

    def __init__(self, radius=1.0, lap_time=6.0, altitude=1.0, n_laps=2.0,
                 ctrl_freq=48, pyb_freq=240, w_ctrl=0.0, w_rate=0.0, seed=None):
        self.traj = Figure8(radius=radius, lap_time=lap_time, altitude=altitude)
        self.max_pos_err = POS_ERR_FACTOR * float(radius)  # "lost" threshold, scales w/ track
        self.n_laps = float(n_laps)
        self.ctrl_freq = int(ctrl_freq)
        self.dt = 1.0 / self.ctrl_freq
        self.n_steps = int(self.n_laps * lap_time * self.ctrl_freq)
        self.w_ctrl = float(w_ctrl)
        self.w_rate = float(w_rate)

        start = self.traj.start_pos().reshape(1, 3)
        with _suppress_stdout():
            self.env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                                  initial_xyzs=start, physics=Physics.PYB,
                                  pyb_freq=pyb_freq, ctrl_freq=ctrl_freq, gui=False)
        self.HOVER_RPM = float(self.env.HOVER_RPM)
        self.MAX_RPM = float(self.env.MAX_RPM)

    def close(self):
        with _suppress_stdout():
            self.env.close()

    # --- observation ---
    def _observation(self, state, t):
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        angvel = state[13:16]
        ref_pos = self.traj.pos(t)
        ref_vel = self.traj.vel(t)
        pos_err = ref_pos - pos
        vel_err = ref_vel - vel
        return np.concatenate([pos_err, vel_err, rpy, angvel]), pos_err

    def rollout(self, controller, log=False, obs_noise=0.0, act_noise=0.0, rng=None):
        if rng is None:
            rng = np.random
        obs, _ = self.env.reset()
        state = np.asarray(obs)[0]

        err_sum = 0.0          # penalised position error (counts crash steps)
        sq_err_survived = 0.0  # squared error over survived steps only
        ctrl_sum = 0.0
        rate_sum = 0.0
        survived = 0
        crashed = False
        traj_log = [] if log else None

        for k in range(self.n_steps):
            t = k * self.dt
            ob, pos_err = self._observation(state, t)
            if obs_noise > 0.0:
                ob = ob + rng.normal(0.0, obs_noise, size=ob.shape)
            rpm = controller.act(ob)
            if act_noise > 0.0:
                rpm = np.clip(rpm + rng.normal(0.0, act_noise * self.HOVER_RPM, size=4),
                              0.0, self.MAX_RPM)
            action = rpm.reshape(1, 4).astype(np.float32)
            obs, _, _, _, _ = self.env.step(action)
            state = np.asarray(obs)[0]

            e = float(np.linalg.norm(pos_err))
            pos = state[0:3]
            rpy = state[7:10]
            angvel = state[13:16]

            if log:
                # (t, drone pos, ref pos, err, rpy, commanded rpms, drone velocity)
                traj_log.append((t, pos.copy(), self.traj.pos(t), e, rpy.copy(),
                                 rpm.copy(), state[10:13].copy()))

            # control effort: normalised |rpm - hover|
            ctrl_sum += float(np.mean(np.abs(rpm - self.HOVER_RPM)) / (0.35 * self.HOVER_RPM))
            rate_sum += float(np.linalg.norm(angvel))

            # termination check
            if (pos[2] < GROUND_Z or pos[2] > 3.0 * self.traj.altitude
                    or e > self.max_pos_err or abs(rpy[0]) > MAX_TILT or abs(rpy[1]) > MAX_TILT):
                crashed = True
                break

            err_sum += e
            sq_err_survived += e * e
            survived += 1

        # charge remaining (un-flown) steps at the penalty error (= the "lost" threshold)
        missed = self.n_steps - survived
        err_sum += missed * self.max_pos_err
        mean_pos_err = err_sum / self.n_steps
        mean_ctrl = ctrl_sum / max(survived, 1)
        mean_rate = rate_sum / max(survived, 1)

        fitness = -(mean_pos_err + self.w_ctrl * mean_ctrl + self.w_rate * mean_rate)

        metrics = {
            "fitness": fitness,
            "crashed": crashed,
            "completion": survived / self.n_steps,
            "rmse_survived": float(np.sqrt(sq_err_survived / max(survived, 1))),
            "mean_pos_err": mean_pos_err,
            "mean_ctrl": mean_ctrl,
            "mean_rate": mean_rate,
            "peak_speed": self.traj.peak_speed(),
        }
        if log:
            return fitness, metrics, traj_log
        return fitness, metrics

    def rollout_pid(self):
        """Fly the same trajectory with the library's tuned DSL PID controller,
        scored by the identical termination and RMSE logic. Model-based baseline."""
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.utils.enums import DroneModel
        pid = DSLPIDControl(drone_model=DroneModel.CF2X)
        obs, _ = self.env.reset()
        state = np.asarray(obs)[0]
        sq_err, survived, crashed = 0.0, 0, False
        for k in range(self.n_steps):
            t = k * self.dt
            ref_pos, ref_vel = self.traj.pos(t), self.traj.vel(t)
            rpm, _, _ = pid.computeControlFromState(
                control_timestep=self.dt, state=state,
                target_pos=ref_pos, target_vel=ref_vel)
            obs, _, _, _, _ = self.env.step(rpm.reshape(1, 4).astype(np.float32))
            state = np.asarray(obs)[0]
            e = float(np.linalg.norm(ref_pos - state[0:3]))
            pos, rpy = state[0:3], state[7:10]
            if (pos[2] < GROUND_Z or pos[2] > 3.0 * self.traj.altitude
                    or e > self.max_pos_err or abs(rpy[0]) > MAX_TILT or abs(rpy[1]) > MAX_TILT):
                crashed = True
                break
            sq_err += e * e
            survived += 1
        return {"crashed": crashed, "completion": survived / self.n_steps,
                "rmse_survived": float(np.sqrt(sq_err / max(survived, 1))),
                "peak_speed": self.traj.peak_speed()}
