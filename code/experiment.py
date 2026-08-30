
from __future__ import annotations

import multiprocessing as mp
import time

from .evolve_cmaes import run_cmaes
from .evolve_neat import run_neat
from .results_io import save_result

# --- nominal task settings ---
# 5 m-wide figure-8 (half-width R = 2.5 m): sized so the whole agile range stays within
# the drone's 34 deg tilt / +-40% thrust envelope. Peak speed = R*w*sqrt(2) = 22.21/T,
# so the lap times below give peak speeds of roughly 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 m/s.
RADIUS = 2.5
ALTITUDE = 1.0
N_LAPS = 2.0
NOMINAL_LAP = 7.4           # s, ~3.0 m/s: the "agile" operating point for sensitivity runs
LAP_TIMES = [14.8, 11.1, 8.9, 7.4, 6.35, 5.55]


def _task_kwargs(lap_time):
    return {"radius": RADIUS, "altitude": ALTITUDE, "n_laps": N_LAPS, "lap_time": lap_time}


def run_one(spec: dict):
    """Execute a single evolutionary run described by ``spec`` and save it."""
    t0 = time.time()
    tk = _task_kwargs(spec["lap_time"])
    if spec["method"] == "cmaes":
        res = run_cmaes(task_kwargs=tk, hidden=spec.get("hidden", (16,)),
                        popsize=spec["popsize"], sigma0=spec.get("sigma0", 0.5),
                        max_tilt=spec.get("max_tilt", 0.6), max_gen=spec["max_gen"],
                        seed=spec["seed"], verbose=False)
    else:
        res = run_neat(task_kwargs=tk, popsize=spec["popsize"],
                       max_tilt=spec.get("max_tilt", 0.6), max_gen=spec["max_gen"],
                       seed=spec["seed"], verbose=False)
    res["spec"] = spec
    save_result(res, subdir=spec["tag"], name=spec["name"])
    m = res["champion_metrics"]
    dt = time.time() - t0
    print(f"[done {dt/60:4.1f}m] {spec['tag']}/{spec['name']:<28s} "
          f"fit={res['best_fitness']:+.3f} rmse={m['rmse_survived']:.3f} "
          f"crash={int(m['crashed'])} compl={m['completion']:.2f}", flush=True)
    return spec["name"], res["best_fitness"], m["rmse_survived"], m["crashed"]


def build_frontier_specs(seeds, max_gen, popsize):
    specs = []
    for method in ("cmaes", "neat"):
        for lt in LAP_TIMES:
            for s in seeds:
                specs.append({
                    "method": method, "seed": s, "lap_time": lt, "popsize": popsize,
                    "max_gen": max_gen, "hidden": (16,), "sigma0": 0.5, "max_tilt": 0.6,
                    "tag": "frontier",
                    "name": f"{method}_lap{lt:g}_seed{s}",
                })
    return specs


def build_sensitivity_specs(seeds, max_gen, include_hidden=True):
    specs = []
    lt = NOMINAL_LAP
    # CMA-ES: sigma0 sweep (dense)
    for sg in (0.1, 0.2, 0.35, 0.5, 0.7, 1.0):
        for s in seeds:
            specs.append({"method": "cmaes", "seed": s, "lap_time": lt, "popsize": 24,
                          "max_gen": max_gen, "hidden": (16,), "sigma0": sg, "max_tilt": 0.6,
                          "tag": "sens_sigma", "name": f"cmaes_sigma{sg:g}_seed{s}"})
    # CMA-ES: population size sweep (dense)
    for pop in (8, 16, 24, 32, 48):
        for s in seeds:
            specs.append({"method": "cmaes", "seed": s, "lap_time": lt, "popsize": pop,
                          "max_gen": max_gen, "hidden": (16,), "sigma0": 0.2, "max_tilt": 0.6,
                          "tag": "sens_pop", "name": f"cmaes_pop{pop}_seed{s}"})
    # CMA-ES: hidden layer size sweep (dense)
    if include_hidden:
        for hid in (4, 8, 16, 24, 32):
            for s in seeds:
                specs.append({"method": "cmaes", "seed": s, "lap_time": lt, "popsize": 24,
                              "max_gen": max_gen, "hidden": (hid,), "sigma0": 0.2, "max_tilt": 0.6,
                              "tag": "sens_hidden", "name": f"cmaes_hid{hid}_seed{s}"})
    # NEAT: population size sweep (dense)
    for pop in (8, 16, 24, 32, 48):
        for s in seeds:
            specs.append({"method": "neat", "seed": s, "lap_time": lt, "popsize": pop,
                          "max_gen": max_gen, "max_tilt": 0.6,
                          "tag": "sens_neatpop", "name": f"neat_pop{pop}_seed{s}"})
    return specs


def run_parallel(specs, n_workers=6):
    print(f"launching {len(specs)} runs on {n_workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for _ in pool.imap_unordered(run_one, specs):
            pass
    print(f"all {len(specs)} runs finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["frontier", "sensitivity", "both"], default="both")
    ap.add_argument("--front-seeds", type=int, default=4)
    ap.add_argument("--front-gen", type=int, default=60)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--sens-seeds", type=int, default=3)
    ap.add_argument("--sens-gen", type=int, default=50)
    ap.add_argument("--no-hidden", action="store_true", help="skip CMA hidden-size sweep")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    specs = []
    if args.study in ("frontier", "both"):
        specs += build_frontier_specs(list(range(args.front_seeds)), args.front_gen, args.pop)
    if args.study in ("sensitivity", "both"):
        specs += build_sensitivity_specs(list(range(args.sens_seeds)), args.sens_gen,
                                         include_hidden=not args.no_hidden)
    # frontier first so the headline result lands earliest
    run_parallel(specs, n_workers=args.workers)
