# Agile Flight Using Neuroevolution: A Comparison of CMA-ES and NEAT

AE4350 (Bio-inspired Intelligence for Aerospace Applications) assignment.

Two gradient-free neuroevolution methods are evolved to control a simulated
Crazyflie 2.X quadrotor tracking an agile **figure-8**, then compared:

* **CMA-ES** — evolves the weights of a fixed-topology network.
* **NEAT** — evolves the network topology *and* its weights.


## Layout

```
code/
  trajectory.py     figure-8 reference (lap time sets agility)
  controller.py     cascaded controller + MLP / NEAT policies
  task.py           CtrlAviary wrapper: observation, reward, rollout
  evolve_cmaes.py   CMA-ES driver (pycma)
  evolve_neat.py    NEAT driver (neat-python)
  neat_config.ini   NEAT configuration
  experiment.py     parallel study harness (frontier + sensitivity)
  analyze.py        figures + summary tables
  results_io.py     save/load results
results/            per-run results (pickles + JSON, ~1 MB, tracked)
figures/            generated figures
```

## Setup (Linux / WSL2, Python 3.10)

`pybullet` builds from a prebuilt wheel on Linux/WSL. In WSL Ubuntu 22.04:

```bash
python3 -m venv ~/ae4350env && . ~/ae4350env/bin/activate
pip install -r requirements.txt
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
pip install -e gym-pybullet-drones
```

## Reproduce

```bash
# single run (5 m track, R = 2.5 m; nominal 7.4 s lap ≈ 3.0 m/s)
python -m code.evolve_cmaes --seed 0 --gen 200 --lap-time 7.4
python -m code.evolve_neat  --seed 0 --gen 200 --lap-time 7.4

# full study: agility frontier + sensitivity sweeps, in parallel
python -m code.experiment --study both --front-seeds 5 --front-gen 200 --pop 24 \
    --sens-seeds 3 --sens-gen 150 --workers 7
python -m code.baselines      # tuned DSL-PID baseline on the same sweep
python -m code.robustness     # disturbance robustness of trained champions

# regenerate every figure + summary table from results/
python -m code.analyze
```

