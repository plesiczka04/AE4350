
from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .results_io import load_all, RESULTS_DIR
from .task import Figure8Task
from .controller import MLPPolicy, NEATPolicy, CascadedController

FIG_DIR = RESULTS_DIR.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

CMA_C = "#1f77b4"
NEAT_C = "#d62728"


G_ACC = 9.81
TILT_CAP = 0.6  # rad, ~34 deg

def tilt_feasible_speed(radius, tilt=TILT_CAP):
    """Peak figure-8 speed above which the reference demands more bank than the cap."""
    return float(np.sqrt(G_ACC * np.tan(tilt) * radius / 1.062))

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
    "legend.fontsize": 11.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ----------------------------------------------------------------------------
# rebuild champions and roll out with logging
# ----------------------------------------------------------------------------
def rebuild_controller(res, hover_rpm):
    max_tilt = res.get("spec", {}).get("max_tilt", 0.6)
    if res["method"] == "cmaes":
        pol = MLPPolicy(hidden=res["hidden"]).set_params(res["best_genome"])
    else:
        import neat
        from .evolve_neat import _make_config
        cfg = _make_config(res["popsize"])
        net = neat.nn.FeedForwardNetwork.create(res["champion_genome"], cfg)
        pol = NEATPolicy(net)
    return CascadedController(pol, hover_rpm=hover_rpm, max_tilt=max_tilt)


def rollout_champion(res):
    """Re-fly a saved champion and return (task, metrics, log)."""
    tk = dict(res["task_kwargs"])
    task = Figure8Task(**tk)
    ctrl = rebuild_controller(res, task.HOVER_RPM)
    _, metrics, log = task.rollout(ctrl, log=True)
    return task, metrics, log


# ----------------------------------------------------------------------------
# figure 1: convergence curves (CMA vs NEAT at nominal agility)
# ----------------------------------------------------------------------------
def fig_convergence(nominal_lap=7.4):
    runs = load_all("frontier")
    by_method = defaultdict(list)
    for r in runs:
        if abs(r["task_kwargs"]["lap_time"] - nominal_lap) < 1e-6:
            by_method[r["method"]].append(r["hist_best"])
    if not by_method:
        print("convergence: no data yet"); return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for method, color, label in [("cmaes", CMA_C, "CMA-ES"), ("neat", NEAT_C, "NEAT")]:
        hs = by_method.get(method, [])
        if not hs:
            continue
        L = min(len(h) for h in hs)
        H = np.stack([h[:L] for h in hs])
        gens = np.arange(L)
        mean, std = H.mean(0), H.std(0)
        ax.plot(gens, mean, color=color, label=f"{label} (n={len(hs)})")
        ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel("Generation"); ax.set_ylabel("Best Fitness (-Mean Tracking Error)")
    ax.legend()
    p = FIG_DIR / "fig1_convergence.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


def _phase_bin(phase, values, nbins=48):
    """Average per-step ``values`` into phase bins to reveal the envelope."""
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    ym = np.array([values[idx == b].mean() if np.any(idx == b) else np.nan
                   for b in range(nbins)])
    return xc, ym


# ----------------------------------------------------------------------------
# figure 2: agility-safety frontier
# ----------------------------------------------------------------------------
def fig_frontier():
    """Agility-accuracy frontier overlaid in one dual-axis plot: tracking RMSE (left axis)
    and control effort (right axis) versus commanded peak speed."""
    import pickle
    runs = load_all("frontier")
    if not runs:
        print("frontier: no data yet"); return
    agg = defaultdict(list)
    for r in runs:
        agg[(r["method"], r["task_kwargs"]["lap_time"])].append(r)
    fig, axL = plt.subplots(figsize=(7.4, 4.3))
    axR = axL.twinx(); axR.grid(False)
    for method, color, label in [("cmaes", CMA_C, "CMA-ES"), ("neat", NEAT_C, "NEAT")]:
        laps = sorted({lt for (m, lt) in agg if m == method})
        sp, rm, rs, ef, es = [], [], [], [], []
        for lt in laps:
            g = agg[(method, lt)]
            sp.append(g[0]["champion_metrics"]["peak_speed"])
            succ = [x["champion_metrics"]["rmse_survived"] for x in g
                    if not x["champion_metrics"]["crashed"]]
            rm.append(np.mean(succ) if succ else np.nan); rs.append(np.std(succ) if succ else 0.0)
            eff = [x["champion_metrics"]["mean_ctrl"] for x in g]
            ef.append(np.mean(eff)); es.append(np.std(eff))
        sp = np.array(sp)
        axL.errorbar(sp, rm, yerr=rs, marker="o", ls="-", color=color, capsize=3,
                     label=f"{label} RMSE")
        axR.errorbar(sp, ef, yerr=es, marker="s", ls=":", color=color, capsize=3,
                     alpha=0.55, label=f"{label} effort")
    pid_p = RESULTS_DIR / "baselines" / "pid.pkl"
    if pid_p.exists():
        pid = pickle.load(open(pid_p, "rb")); laps = sorted(pid)
        axL.plot([pid[l]["peak_speed"] for l in laps],
                 [pid[l]["rmse_survived"] if not pid[l]["crashed"] else np.nan for l in laps],
                 "k--", marker="^", ms=5, label="DSL-PID RMSE")
    radius = runs[0]["task_kwargs"].get("radius", 1.0)
    vf = tilt_feasible_speed(radius); x0, x1 = axL.get_xlim()
    if vf < x1:
        axL.axvspan(vf, x1, color="0.85", alpha=0.6, zorder=0)
        axL.axvline(vf, ls=":", color="0.35", lw=1.6, zorder=1); axL.set_xlim(x0, x1)
    axL.set_xlabel("Commanded Peak Speed (m/s)")
    axL.set_ylabel("Tracking RMSE (m)")
    axR.set_ylabel("Control Effort")
    h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    fig.tight_layout()
    p = FIG_DIR / "fig2_frontier.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


# ----------------------------------------------------------------------------
# figure 3: trajectory tracking of the best champions (per method)
# ----------------------------------------------------------------------------
def fig_speed_trajectory(lap=5.55):
    """LAFR-style figure: flown path coloured by the drone's actual speed, overlaid on the
    reference, with the peak commanded speed annotated."""
    from matplotlib.collections import LineCollection
    runs = [r for r in load_all("frontier")
            if abs(r["task_kwargs"]["lap_time"] - lap) < 1e-6
            and not r["champion_metrics"]["crashed"]]
    if not runs:
        print("speed trajectory: no data yet"); return
    best = min(runs, key=lambda r: r["champion_metrics"]["rmse_survived"])
    task, metrics, log = rollout_champion(best)
    pos = np.array([e[1] for e in log])          # x, y, z
    ref = np.array([e[2] for e in log])
    vel = np.array([e[6] for e in log])          # vx, vy, vz  (added to the log)
    speed = np.linalg.norm(vel, axis=1)
    peak_cmd = task.traj.peak_speed()

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    # reference figure-8
    ax.plot(ref[:, 0], ref[:, 1], "--", color="0.5", lw=2.2, label="reference", zorder=1)
    # flown path coloured by the drone's true speed
    pts = pos[:, :2].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vmax = max(speed.max(), peak_cmd)
    norm = plt.Normalize(0.0, vmax)
    lc = LineCollection(segs, cmap="turbo", norm=norm, zorder=2)
    lc.set_array(speed[:-1]); lc.set_linewidth(4.0)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Drone Speed (m/s)")
    ax.set_xlabel("X (m)", labelpad=2); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal"); ax.autoscale()
    ax.legend(loc="upper right", framealpha=0.9)
    p = FIG_DIR / "fig10_speed_trajectory.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


def fig_trajectories(nominal_lap=7.4):
    runs = [r for r in load_all("frontier")
            if abs(r["task_kwargs"]["lap_time"] - nominal_lap) < 1e-6]
    if not runs:
        print("trajectories: no data yet"); return
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ref_drawn = False
    for method, color, label in [("cmaes", CMA_C, "CMA-ES"), ("neat", NEAT_C, "NEAT")]:
        cand = [r for r in runs if r["method"] == method
                and not r["champion_metrics"]["crashed"]]
        if not cand:
            continue
        best = min(cand, key=lambda r: r["champion_metrics"]["rmse_survived"])
        _, metrics, log = rollout_champion(best)
        drone = np.array([e[1] for e in log])
        ref = np.array([e[2] for e in log])
        if not ref_drawn:
            ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.4, label="Reference", zorder=1)
            ref_drawn = True
        ax.plot(drone[:, 0], drone[:, 1], color=color, lw=1.4, alpha=0.85,
                label=f"{label} (flown)", zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / "fig3_trajectories.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


# ----------------------------------------------------------------------------
# figure 4: sensitivity sweeps
# ----------------------------------------------------------------------------
def _final_cost(r):
    """Crash-aware, always-defined performance metric: penalised mean tracking
    error of the champion (= -fitness). Lower is better; crashes inflate it."""
    return -float(r["best_fitness"])


def _sweep_panel(ax, subdir, key_fn, xlabel, color, label):
    runs = load_all(subdir)
    if not runs:
        return False
    groups = defaultdict(list)
    for r in runs:
        groups[key_fn(r)].append(_final_cost(r))
    xs = sorted(groups)
    means = [np.mean(groups[x]) for x in xs]
    stds = [np.std(groups[x]) for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker="o", color=color, capsize=3, label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel("Champion Cost (m)")
    ax.legend()
    return True


def fig_sensitivity():
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4))
    ok = False
    ok |= _sweep_panel(axes[0, 0], "sens_sigma",
                       lambda r: r["spec"]["sigma0"], "CMA-ES sigma0", CMA_C, "CMA-ES")
    ok |= _sweep_panel(axes[0, 1], "sens_pop",
                       lambda r: r["popsize"], "Population Size", CMA_C, "CMA-ES")
    ok |= _sweep_panel(axes[1, 0], "sens_hidden",
                       lambda r: r["hidden"][0], "Hidden Units", CMA_C, "CMA-ES")
    ok |= _sweep_panel(axes[1, 1], "sens_neatpop",
                       lambda r: r["popsize"], "Population Size", NEAT_C, "NEAT")
    if not ok:
        print("sensitivity: no data yet"); plt.close(fig); return
    fig.tight_layout(pad=1.6)
    p = FIG_DIR / "fig4_sensitivity.png"; fig.savefig(p); plt.close(fig)
    print("saved", p)


def fig_control_strategy(nominal_lap=6.35):
    """Learned flight strategy: commanded bank angles around the lap and altitude hold."""
    runs = [r for r in load_all("frontier")
            if abs(r["task_kwargs"]["lap_time"] - nominal_lap) < 1e-6]
    if not runs:
        print("control strategy: no data yet"); return
    # side-by-side layout: bank-angle strategy (left) and altitude hold (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for method, color, label in [("cmaes", CMA_C, "CMA-ES"), ("neat", NEAT_C, "NEAT")]:
        cand = [r for r in runs if r["method"] == method
                and not r["champion_metrics"]["crashed"]]
        if not cand:
            continue
        best = min(cand, key=lambda r: r["champion_metrics"]["rmse_survived"])
        task, _, log = rollout_champion(best)
        ts = np.array([e[0] for e in log])
        z = np.array([e[1][2] for e in log])
        rpy = np.array([e[4] for e in log])
        phase = (ts % task.traj.lap_time) / task.traj.lap_time
        bank = np.degrees(np.linalg.norm(rpy[:, :2], axis=1))  # total tilt magnitude
        xc, ym = _phase_bin(phase, bank)
        ax1.plot(xc, ym, color=color, lw=1.8, label=label)
        ax2.plot(ts, z, color=color, lw=1.2, alpha=0.85, label=label)
    ax1.set_xlabel("Phase Around Figure-8 (0..1)")
    ax1.set_ylabel("Bank Angle |Tilt| (deg)")
    ax1.legend()
    ax2.axhline(1.0, color="k", ls="--", lw=0.9, label="target altitude")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Altitude (m)")
    ax2.legend()
    fig.tight_layout()
    p = FIG_DIR / "fig7_control_strategy.png"; fig.savefig(p); plt.close(fig)
    print("saved", p)



def fig_robustness():
    """Robustness of trained champions to unseen actuation and sensor disturbances."""
    import pickle
    p = RESULTS_DIR / "robustness" / "summary.pkl"
    if not p.exists():
        print("robustness: no data yet"); return
    with open(p, "rb") as f:
        D = pickle.load(f)
    agg, act_levels, obs_levels = D["agg"], D["act_levels"], D["obs_levels"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4))
    specs = [("act", act_levels, "Actuation Noise (Frac. Hover RPM)"),
             ("obs", obs_levels, "Sensor Noise (S.D. on Observation)")]
    for row, (kind, levels, xlabel) in enumerate(specs):
        axR, axC = axes[row]
        for method, color, label in [("cmaes", CMA_C, "CMA-ES"), ("neat", NEAT_C, "NEAT")]:
            d = agg[method][kind]
            xs = sorted(d)
            rmse = [d[x]["rmse_mean"] for x in xs]
            rerr = [d[x]["rmse_std"] for x in xs]
            crash = [d[x]["crash"] * 100 for x in xs]
            axR.errorbar(xs, rmse, yerr=rerr, marker="o", color=color, capsize=3, label=label)
            axC.plot(xs, crash, marker="s", color=color, label=label)
        axR.set_xlabel(xlabel); axR.set_ylabel("Survivor RMSE (m)"); axR.legend()
        axC.set_xlabel(xlabel); axC.set_ylabel("Crash Rate (%)"); axC.legend()
    fig.tight_layout()
    q = FIG_DIR / "fig8_robustness.png"; fig.savefig(q); plt.close(fig)
    print("saved", q)


def fig_panels(nominal_lap=7.4, agile_lap=6.35):
    """Render every multi-panel figure as individual single-panel PNGs, so each panel can be
    a separately-captioned LaTeX subfigure (referenced as e.g. Figure 7(b))."""
    import pickle
    PS = (5.2, 3.8)

    def _save(fig, name):
        fig.tight_layout()
        p = FIG_DIR / name
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print("saved", p)

    # ---- frontier: (a) tracking RMSE, (b) control effort ----
    runs = load_all("frontier")
    if runs:
        agg = defaultdict(list)
        for r in runs:
            agg[(r["method"], r["task_kwargs"]["lap_time"])].append(r)
        data = {}
        for method in ("cmaes", "neat"):
            laps = sorted({lt for (m, lt) in agg if m == method})
            sp, rm, rs, ef, es = [], [], [], [], []
            for lt in laps:
                g = agg[(method, lt)]
                sp.append(g[0]["champion_metrics"]["peak_speed"])
                succ = [x["champion_metrics"]["rmse_survived"] for x in g
                        if not x["champion_metrics"]["crashed"]]
                rm.append(np.mean(succ) if succ else np.nan); rs.append(np.std(succ) if succ else 0.0)
                eff = [x["champion_metrics"]["mean_ctrl"] for x in g]
                ef.append(np.mean(eff)); es.append(np.std(eff))
            data[method] = (np.array(sp), rm, rs, ef, es)
        fig, ax = plt.subplots(figsize=PS)
        for method, color, label in [("cmaes", CMA_C, "CMA-ES"), ("neat", NEAT_C, "NEAT")]:
            sp, rm, rs, ef, es = data[method]
            ax.errorbar(sp, rm, yerr=rs, marker="o", color=color, label=label, capsize=3)
        pid_p = RESULTS_DIR / "baselines" / "pid.pkl"
        if pid_p.exists():
            pid = pickle.load(open(pid_p, "rb")); laps = sorted(pid)
            ax.plot([pid[l]["peak_speed"] for l in laps],
                    [pid[l]["rmse_survived"] if not pid[l]["crashed"] else np.nan for l in laps],
                    "k--", marker="^", ms=5, label="DSL-PID")
        radius = runs[0]["task_kwargs"].get("radius", 1.0)
        vf = tilt_feasible_speed(radius); x0, x1 = ax.get_xlim()
        if vf < x1:
            ax.axvspan(vf, x1, color="0.85", alpha=0.6, zorder=0)
            ax.axvline(vf, ls=":", color="0.35", lw=1.6, zorder=1); ax.set_xlim(x0, x1)
        ax.set_xlabel("Commanded Peak Speed (m/s)"); ax.set_ylabel("Tracking RMSE (m)")
        ax.legend(loc="upper left"); _save(fig, "fig2a_rmse.png")
        fig, ax = plt.subplots(figsize=PS)
        for method, color, label in [("cmaes", CMA_C, "CMA-ES"), ("neat", NEAT_C, "NEAT")]:
            sp, rm, rs, ef, es = data[method]
            ax.errorbar(sp, ef, yerr=es, marker="s", color=color, label=label, capsize=3)
        ax.set_xlabel("Commanded Peak Speed (m/s)")
        ax.set_ylabel("Control Effort"); ax.legend(); _save(fig, "fig2b_effort.png")

    # ---- best-champion trajectories: (a) CMA-ES, (b) NEAT ----
    trj = [r for r in load_all("frontier")
           if abs(r["task_kwargs"]["lap_time"] - nominal_lap) < 1e-6]
    for method, color, mlabel, tag in [("cmaes", CMA_C, "CMA-ES", "fig3a_cmaes.png"),
                                       ("neat", NEAT_C, "NEAT", "fig3b_neat.png")]:
        cand = [r for r in trj if r["method"] == method and not r["champion_metrics"]["crashed"]]
        if not cand:
            continue
        best = min(cand, key=lambda r: r["champion_metrics"]["rmse_survived"])
        _, _, log = rollout_champion(best)
        drone = np.array([e[1] for e in log]); ref = np.array([e[2] for e in log])
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.2, label="reference")
        ax.plot(drone[:, 0], drone[:, 1], color=color, lw=1.4, label=f"{mlabel} (flown)")
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right", fontsize=8); _save(fig, tag)

    # ---- sensitivity: (a) sigma0, (b) CMA-ES population, (c) hidden units, (d) NEAT population ----
    for subdir, key, xlabel, color, label, tag in [
        ("sens_sigma", lambda r: r["spec"]["sigma0"], "CMA-ES $\\sigma_0$", CMA_C, "CMA-ES", "fig4a_sigma.png"),
        ("sens_pop", lambda r: r["popsize"], "Population Size", CMA_C, "CMA-ES", "fig4b_cmapop.png"),
        ("sens_hidden", lambda r: r["hidden"][0], "Hidden Units", CMA_C, "CMA-ES", "fig4c_hidden.png"),
        ("sens_neatpop", lambda r: r["popsize"], "Population Size", NEAT_C, "NEAT", "fig4d_neatpop.png"),
    ]:
        fig, ax = plt.subplots(figsize=PS)
        if _sweep_panel(ax, subdir, key, xlabel, color, label):
            _save(fig, tag)
        else:
            plt.close(fig)

    # ---- control strategy: (a) bank angle, (b) altitude ----
    cs = [r for r in load_all("frontier")
          if abs(r["task_kwargs"]["lap_time"] - agile_lap) < 1e-6]
    figB, axB = plt.subplots(figsize=PS); figA, axA = plt.subplots(figsize=PS)
    for method, color, label in [("cmaes", CMA_C, "CMA-ES"), ("neat", NEAT_C, "NEAT")]:
        cand = [r for r in cs if r["method"] == method and not r["champion_metrics"]["crashed"]]
        if not cand:
            continue
        best = min(cand, key=lambda r: r["champion_metrics"]["rmse_survived"])
        task, _, log = rollout_champion(best)
        ts = np.array([e[0] for e in log]); z = np.array([e[1][2] for e in log])
        rpy = np.array([e[4] for e in log])
        phase = (ts % task.traj.lap_time) / task.traj.lap_time
        bank = np.degrees(np.linalg.norm(rpy[:, :2], axis=1))
        xc, ym = _phase_bin(phase, bank)
        axB.plot(xc, ym, color=color, lw=1.8, label=label)
        axA.plot(ts, z, color=color, lw=1.2, alpha=0.85, label=label)
    axB.set_xlabel("Phase Around Figure-8 (0..1)"); axB.set_ylabel("Bank Angle |Tilt| (deg)")
    axB.legend(); _save(figB, "fig7a_bank.png")
    axA.axhline(1.0, color="k", ls="--", lw=0.9, label="target altitude")
    axA.set_xlabel("Time (s)"); axA.set_ylabel("Altitude (m)"); axA.legend()
    _save(figA, "fig7b_altitude.png")

    # ---- robustness: (a) act RMSE, (b) act crash, (c) sensor RMSE, (d) sensor crash ----
    rp = RESULTS_DIR / "robustness" / "summary.pkl"
    if rp.exists():
        agg = pickle.load(open(rp, "rb"))["agg"]
        panels = [("act", "rmse", "Actuation Noise (Frac. Hover RPM)", "Survivor RMSE (m)", "fig8a_act_rmse.png"),
                  ("act", "crash", "Actuation Noise (Frac. Hover RPM)", "Crash Rate (%)", "fig8b_act_crash.png"),
                  ("obs", "rmse", "Sensor Noise (S.D. on Observation)", "Survivor RMSE (m)", "fig8c_obs_rmse.png"),
                  ("obs", "crash", "Sensor Noise (S.D. on Observation)", "Crash Rate (%)", "fig8d_obs_crash.png")]
        for kind, metric, xlabel, ylabel, tag in panels:
            fig, ax = plt.subplots(figsize=PS)
            for method, color, label in [("cmaes", CMA_C, "CMA-ES"), ("neat", NEAT_C, "NEAT")]:
                d = agg[method][kind]; xs = sorted(d)
                if metric == "rmse":
                    ax.errorbar(xs, [d[x]["rmse_mean"] for x in xs],
                                yerr=[d[x]["rmse_std"] for x in xs],
                                marker="o", color=color, capsize=3, label=label)
                else:
                    ax.plot(xs, [d[x]["crash"] * 100 for x in xs], marker="s", color=color, label=label)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(); _save(fig, tag)


def summary_table():
    """Print/save a CSV summary of the frontier runs."""
    runs = load_all("frontier")
    if not runs:
        print("summary: no data yet"); return
    lines = ["method,lap_time,peak_speed,mean_rmse,crash_rate,n"]
    agg = defaultdict(list)
    for r in runs:
        agg[(r["method"], r["task_kwargs"]["lap_time"])].append(r)
    for (method, lt) in sorted(agg):
        rs = agg[(method, lt)]
        succ = [r["champion_metrics"]["rmse_survived"] for r in rs
                if not r["champion_metrics"]["crashed"]]
        peak = rs[0]["champion_metrics"]["peak_speed"]
        cr = np.mean([r["champion_metrics"]["crashed"] for r in rs])
        rmse = np.mean(succ) if succ else float("nan")
        lines.append(f"{method},{lt:g},{peak:.3f},{rmse:.4f},{cr:.2f},{len(rs)}")
    csv = "\n".join(lines)
    (FIG_DIR / "summary_frontier.csv").write_text(csv)
    print(csv)


def make_all():
    import traceback
    steps = [
        ("summary", summary_table),
        ("fig1_convergence", fig_convergence),
        ("fig2_frontier", fig_frontier),
        ("fig3_trajectories", fig_trajectories),
        ("fig4_sensitivity", fig_sensitivity),
        ("fig7_control_strategy", fig_control_strategy),
        ("fig8_robustness", fig_robustness),
        ("fig10_speed_trajectory", fig_speed_trajectory),
        ("fig_panels", fig_panels),   # individual single-panel PNGs used as report subfigures
    ]
    for name, fn in steps:
        try:
            fn()
        except Exception:
            print(f"[analyze] {name} FAILED:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    make_all()
