
from __future__ import annotations

import pickle

from .task import Figure8Task
from .experiment import LAP_TIMES, RADIUS, ALTITUDE, N_LAPS
from .results_io import RESULTS_DIR


def run_pid_baseline():
    out = {}
    for lt in LAP_TIMES:
        task = Figure8Task(radius=RADIUS, altitude=ALTITUDE, n_laps=N_LAPS, lap_time=lt)
        m = task.rollout_pid()
        task.close()
        out[lt] = m
        print(f"[pid] lap {lt:>4}: rmse={m['rmse_survived']:.3f} "
              f"crash={int(m['crashed'])} peak={m['peak_speed']:.2f}", flush=True)
    (RESULTS_DIR / "baselines").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "baselines" / "pid.pkl", "wb") as f:
        pickle.dump(out, f)
    print("saved PID baseline", flush=True)
    return out


if __name__ == "__main__":
    run_pid_baseline()
