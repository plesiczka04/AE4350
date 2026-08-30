
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import neat

from .task import Figure8Task
from .controller import NEATPolicy, CascadedController

CONFIG_PATH = Path(__file__).resolve().parent / "neat_config.ini"


def _make_config(popsize, config_path=CONFIG_PATH):
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
    cfg.pop_size = int(popsize)
    return cfg


def run_neat(task_kwargs=None, popsize=20, max_gen=100, max_tilt=0.6,
             seed=0, verbose=False):
    """Evolve a controller with NEAT. Returns a result dict mirroring run_cmaes."""
    task_kwargs = dict(task_kwargs or {})
    random.seed(seed)
    np.random.seed(seed)

    task = Figure8Task(seed=seed, **task_kwargs)
    cfg = _make_config(popsize)

    hist_best, hist_mean, hist_nodes, hist_conns = [], [], [], []
    state = {"evals": 0}

    def eval_genomes(genomes, config):
        for _gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            ctrl = CascadedController(NEATPolicy(net), hover_rpm=task.HOVER_RPM,
                                      max_tilt=max_tilt)
            fit, _ = task.rollout(ctrl)
            genome.fitness = fit
            state["evals"] += 1

    pop = neat.Population(cfg)

    # capture per-generation statistics via a reporter
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    class _Track(neat.reporting.BaseReporter):
        def post_evaluate(self, config, population, species, best_genome):
            fits = [g.fitness for g in population.values() if g.fitness is not None]
            hist_best.append(float(max(fits)))
            hist_mean.append(float(np.mean(fits)))
            n_nodes = len(best_genome.nodes)
            n_conns = len([c for c in best_genome.connections.values() if c.enabled])
            hist_nodes.append(n_nodes)
            hist_conns.append(n_conns)
            if verbose and (len(hist_best) % 10 == 0 or len(hist_best) == 1):
                print(f"[neat seed={seed}] gen {len(hist_best)-1:3d}  "
                      f"best_fit={hist_best[-1]:.4f}  mean_fit={hist_mean[-1]:.4f}  "
                      f"nodes={n_nodes} conns={n_conns}")
    pop.add_reporter(_Track())

    champion = pop.run(eval_genomes, max_gen)

    # final full-metric evaluation of the champion
    net = neat.nn.FeedForwardNetwork.create(champion, cfg)
    ctrl = CascadedController(NEATPolicy(net), hover_rpm=task.HOVER_RPM, max_tilt=max_tilt)
    fit, metrics = task.rollout(ctrl)
    task.close()

    return {
        "method": "neat",
        "seed": seed,
        "popsize": popsize,
        "best_fitness": fit,
        "champion_metrics": metrics,
        "champion_genome": champion,
        "hist_best": np.array(hist_best),
        "hist_mean": np.array(hist_mean),
        "hist_nodes": np.array(hist_nodes),
        "hist_conns": np.array(hist_conns),
        "generations": len(hist_best),
        "evaluations": state["evals"],
        "task_kwargs": task_kwargs,
        "hidden": None,  # topology is evolved
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen", type=int, default=60)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--lap-time", type=float, default=7.4)
    ap.add_argument("--radius", type=float, default=2.5, help="figure-8 half-width (m)")
    args = ap.parse_args()
    tk = {"radius": args.radius, "lap_time": args.lap_time}
    res = run_neat(task_kwargs=tk,
                   popsize=args.pop, max_gen=args.gen, seed=args.seed, verbose=True)
    m = res["champion_metrics"]
    print(f"\nCHAMPION  fit={res['best_fitness']:.4f}  rmse={m['rmse_survived']:.3f} m  "
          f"crashed={m['crashed']}  completion={m['completion']:.2f}  "
          f"nodes={res['hist_nodes'][-1]} conns={res['hist_conns'][-1]}")
