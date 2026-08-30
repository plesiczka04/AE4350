"""Persist and reload run results (result dicts from the evolve_* drivers)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _summary(res: dict) -> dict:
    """Small JSON-friendly digest of a run for quick human inspection."""
    m = res.get("champion_metrics", {})
    return {
        "method": res.get("method"),
        "seed": res.get("seed"),
        "best_fitness": float(res.get("best_fitness", float("nan"))),
        "rmse_survived": float(m.get("rmse_survived", float("nan"))),
        "crashed": bool(m.get("crashed", True)),
        "completion": float(m.get("completion", 0.0)),
        "peak_speed": float(m.get("peak_speed", float("nan"))),
        "generations": res.get("generations"),
        "evaluations": res.get("evaluations"),
        "task_kwargs": res.get("task_kwargs", {}),
        "hidden": res.get("hidden"),
        "popsize": res.get("popsize"),
    }


def save_result(res: dict, subdir: str, name: str):
    """Pickle the full result and drop a JSON summary alongside it."""
    out = RESULTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.pkl", "wb") as f:
        pickle.dump(res, f)
    with open(out / f"{name}.json", "w") as f:
        json.dump(_summary(res), f, indent=2)
    return out / f"{name}.pkl"


def load_result(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all(subdir: str):
    """Load every .pkl result under results/<subdir>."""
    d = RESULTS_DIR / subdir
    return [load_result(p) for p in sorted(d.glob("*.pkl"))]
