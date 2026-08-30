"""Robustness study: how do *trained* champions cope with disturbances they never
saw during evolution? This probes generalisation and the reality gap
(cf. Jakobi et al. 1995) without any retraining -- champions are simply re-flown under
(i) actuation noise (a wind/motor-mismatch-like disturbance on the motor commands) and
(ii) sensor noise (a noisy state estimate on the observation), at increasing magnitude.

Fast: it re-flies the ~10 nominal-lap champions under a few disturbance levels with
several noise realisations each, and saves an aggregated summary for plotting.
"""
from __future__ import annotations

import pickle
from collections import defaultdict

import numpy as np

from .results_io import load_all, RESULTS_DIR
from .task import Figure8Task
from .analyze import rebuild_controller

ACT_LEVELS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]   # fraction of hover RPM
OBS_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]   # s.d. on observation channels
REALISATIONS = 8


def _eval_champion(task, ctrl, kind, level, realisations, base_seed):
    """Return (mean RMSE of survivors, crash fraction) over noise realisations."""
    rmses, crashes = [], []
    for r in range(realisations):
        rng = np.random.default_rng(base_seed + 1000 * r + int(level * 1e4))
        kw = {"obs_noise": level} if kind == "obs" else {"act_noise": level}
        _, m = task.rollout(ctrl, rng=rng, **kw)
        crashes.append(m["crashed"])
        if not m["crashed"]:
            rmses.append(m["rmse_survived"])
    return (np.mean(rmses) if rmses else np.nan), float(np.mean(crashes))


def run_robustness(lap=None, realisations=REALISATIONS):
    if lap is None:
        from .experiment import NOMINAL_LAP
        lap = NOMINAL_LAP
    champs = [r for r in load_all("frontier")
              if abs(r["task_kwargs"]["lap_time"] - lap) < 1e-6]
    if not champs:
        raise RuntimeError("no frontier champions found; run the study first")

    # one task per method reused across that method's champions (same lap)
    task = Figure8Task(**champs[0]["task_kwargs"])
    summary = {}
    for method in ("cmaes", "neat"):
        ms = [r for r in champs if r["method"] == method]
        summary[method] = {"act": defaultdict(list), "obs": defaultdict(list)}
        for res in ms:
            ctrl = rebuild_controller(res, task.HOVER_RPM)
            for kind, levels in (("act", ACT_LEVELS), ("obs", OBS_LEVELS)):
                for lv in levels:
                    rmse, crash = _eval_champion(task, ctrl, kind, lv, realisations,
                                                 base_seed=res["seed"])
                    summary[method][kind][lv].append((rmse, crash))
        print(f"[robustness] {method}: evaluated {len(ms)} champions", flush=True)
    task.close()

    # aggregate across champions
    agg = {}
    for method in ("cmaes", "neat"):
        agg[method] = {}
        for kind in ("act", "obs"):
            agg[method][kind] = {}
            for lv, pairs in summary[method][kind].items():
                arr = np.array(pairs)  # (n_champ, 2): rmse, crash
                agg[method][kind][lv] = {
                    "rmse_mean": np.nanmean(arr[:, 0]),
                    "rmse_std": np.nanstd(arr[:, 0]),
                    "crash": np.mean(arr[:, 1]),
                }
    out = RESULTS_DIR / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.pkl", "wb") as f:
        pickle.dump({"lap": lap, "agg": agg,
                     "act_levels": ACT_LEVELS, "obs_levels": OBS_LEVELS}, f)
    print("saved robustness summary", flush=True)
    return agg


if __name__ == "__main__":
    run_robustness()
