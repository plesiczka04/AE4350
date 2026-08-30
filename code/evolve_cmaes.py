"""CMA-ES driver: evolve the flat weight vector of a fixed-topology MLP.

CMA-ES (Hansen 2016) is a state-of-the-art derivative-free optimiser that adapts
a full covariance matrix over the search distribution -- a form of self-adaptation
in *parameter* space. Here it optimises the ~hundreds of weights of an MLP flight
controller. CMA-ES minimises, so the objective is the (penalised) mean tracking
error; we negate to report fitness.
"""
from __future__ import annotations

import numpy as np
import cma

from .task import Figure8Task
from .controller import MLPPolicy, CascadedController


def run_cmaes(task_kwargs=None, hidden=(16,), popsize=20, sigma0=0.5,
              max_tilt=0.6, max_gen=100, seed=0, verbose=False):
    """Evolve an MLP controller with CMA-ES.

    Returns a result dict with the best genome and per-generation history.
    """
    task_kwargs = dict(task_kwargs or {})
    task = Figure8Task(seed=seed, **task_kwargs)
    pol = MLPPolicy(hidden=hidden)
    ctrl = CascadedController(pol, hover_rpm=task.HOVER_RPM, max_tilt=max_tilt)
    n = pol.n_params

    def cost(genome):
        pol.set_params(genome)
        fit, _ = task.rollout(ctrl)
        return -fit  # CMA minimises

    x0 = np.zeros(n)  # start from the stable hover prior
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {"popsize": popsize, "seed": seed + 1, "maxiter": max_gen,
         "verbose": -9, "verb_disp": 0},
    )

    hist_best, hist_mean, evals = [], [], 0
    best_genome, best_cost = None, np.inf
    gen = 0
    while not es.stop() and gen < max_gen:
        solutions = es.ask()
        costs = [cost(s) for s in solutions]
        es.tell(solutions, costs)
        evals += len(solutions)
        gbest = float(np.min(costs))
        gmean = float(np.mean(costs))
        hist_best.append(-gbest)   # store as fitness (higher better)
        hist_mean.append(-gmean)
        if gbest < best_cost:
            best_cost = gbest
            best_genome = np.array(solutions[int(np.argmin(costs))])
        if verbose and (gen % 10 == 0 or gen == max_gen - 1):
            print(f"[cmaes seed={seed}] gen {gen:3d}  best_fit={-gbest:.4f}  mean_fit={-gmean:.4f}")
        gen += 1

    # final evaluation of the champion with full metrics
    pol.set_params(best_genome)
    fit, metrics = task.rollout(ctrl)
    task.close()

    return {
        "method": "cmaes",
        "seed": seed,
        "hidden": hidden,
        "popsize": popsize,
        "sigma0": sigma0,
        "n_params": n,
        "best_genome": best_genome,
        "best_fitness": fit,
        "champion_metrics": metrics,
        "hist_best": np.array(hist_best),
        "hist_mean": np.array(hist_mean),
        "generations": gen,
        "evaluations": evals,
        "task_kwargs": task_kwargs,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen", type=int, default=60)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--lap-time", type=float, default=7.4)
    ap.add_argument("--radius", type=float, default=2.5, help="figure-8 half-width (m)")
    args = ap.parse_args()
    tk = {"radius": args.radius, "lap_time": args.lap_time}
    res = run_cmaes(task_kwargs=tk, sigma0=args.sigma,
                    popsize=args.pop, max_gen=args.gen, seed=args.seed, verbose=True)
    m = res["champion_metrics"]
    print(f"\nCHAMPION  fit={res['best_fitness']:.4f}  rmse={m['rmse_survived']:.3f} m  "
          f"crashed={m['crashed']}  completion={m['completion']:.2f}")
