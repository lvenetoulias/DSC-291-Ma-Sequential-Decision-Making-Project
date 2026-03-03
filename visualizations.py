"""
visualize.py

============

PLOTTING SCRIPT FOR BANDIT EVALUATION PIPELINE


This script puts together a collection of functions to plot the bandit evaluation pipeline. The plotting
functions are organized into five groups, matching the analysis plan:

    Group 1 — Primary comparison plots
        plot_cumulative_regret()       : overlaid regret curves with shaded std
        plot_regret_rate()             : per-step regret R_T/T over time
        plot_rolling_reward()          : smoothed learning curves

    Group 2 — Convergence plots
        plot_arm_entropy()             : rolling arm-selection entropy over time
        plot_entropy_reward_joint()    : dual-axis entropy + reward per algorithm
        plot_convergence_steps()       : bar chart of t* across algorithms
        plot_regret_slope_ratio()      : bar chart of early/late slope ratio

    Group 3 — Exploration vs. exploitation
        plot_arm_pull_heatmap()        : arm pull fraction heatmap (algorithms x arms)
        plot_exploration_bonus_decay() : mean exploration bonus / posterior variance

    Group 4 — Subgroup and robustness
        plot_reward_threshold_sensitivity() : performance vs. binarization threshold
        plot_context_dim_sensitivity()      : regret vs. PCA context dimension

    Group 5 — Hyperparameter sensitivity
        plot_hyperparameter_sweep()    : regret/reward vs. one hyperparameter
        plot_all_hyperparam_sweeps()   : grid of sweep plots for all algorithms

    Utilities
        save_all_figures()   : save all open figures to a directory
        set_style()          : apply consistent visual style
        ALGO_COLORS          : canonical color palette (one color per algorithm)


There are a couple of choices made for the visualizations:
- One color per algorithm, consistent across all plots (ALGO_COLORS dict)
- Log scale on x-axis for regret/reward plots (expands early learning phase)
- Horizontal reference line at dataset base reward rate where applicable
- All functions accept the structured dicts from compare.py and sensitivity.py directly, without requiring
  access to agents or environments
- No computation happens here — only rendering


Usage
-----
    from visualize import set_style, plot_cumulative_regret, save_all_figures
    import matplotlib.pyplot as plt

    set_style()
    fig = plot_cumulative_regret(results)
    plt.show()
    save_all_figures("figures/")
"""

# Import dependencies
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import os


# ---------------------------------------------------------------------------
# Color palette and style
# ---------------------------------------------------------------------------
# One color per algorithm, consistent across all algorithms
ALGO_COLORS = {
    "EpsilonGreedy":    "#E69F00",   # orange
    "LinUCB":           "#56B4E9",   # sky blue
    "ThompsonSampling": "#009E73",   # green
    "LogisticUCB":      "#F0E442",   # yellow
    "BootstrapThompson":"#0072B2",   # deep blue
    "NeuralUCB":        "#D55E00",   # vermillion
    "NeuralTS":         "#CC79A7",   # pink/purple
}

# Fallback color for any algorithm not in the palette
_FALLBACK_COLOR = "#999999"


def _get_color(name: str) -> str:
    """Return the canonical color for an algorithm, or fallback grey."""
    return ALGO_COLORS.get(name, _FALLBACK_COLOR)


def set_style() -> None:
    """
    Apply a consistent visual style to all matplotlib figures.
    Call once at the start of any script that uses visualize.py.
    """
    plt.rcParams.update({
        "figure.dpi":        300,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.fontsize":   9,
        "legend.framealpha": 0.8,
        "lines.linewidth":   2.0,
    })


# ---------------------------------------------------------------------------
# Helper functions across all plot groups
# ---------------------------------------------------------------------------
def _add_base_rate_line(ax, base_rate: float, label: bool = True) -> None:
    """
    Add a horizontal dashed reference line at the dataset base reward rate.
    Any algorithm below this line is performing worse than random selection.
    """
    ax.axhline(
        base_rate,
        color="black", linestyle=":", linewidth=1.2, alpha=0.6,
        label=f"Base rate ({base_rate:.3f})" if label else None,
    )


def _matched_xaxis(ax, n_matched: int, log: bool = True) -> None:
    """
    Configure the x-axis for matched-event plots.
    Uses log scale by default to expand the early learning phase.
    """
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Matched events (log scale)" if log else "Matched events")
    ax.set_xlim(left=1)


# ---------------------------------------------------------------------------
# Group 1: Primary comparison plots
# ---------------------------------------------------------------------------
def plot_cumulative_regret(results: dict, log_x: bool = True, figsize: tuple = (10, 6), title: str = "Cumulative Regret",) -> plt.Figure:
    """
    Function to plot cumulative regret curves for all algorithms with shaded ±1 std bands. This is the
    primary comparison figure. Algorithms with lower and flatter curves are better. Shaded bands show
    variance across trials.

    Parameters
    ----------
    * results: output of compare.run_comparison()
    * log_x: use log scale on the x-axis (recommended)
    * figsize: tuple for the size of the figure
    * title: string with the title of the figure

    Returns
    -------
    * matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name in results["algorithm_names"]:
        agg = results["agg_results"][name]
        mean = agg["mean_cumulative_regret"]
        std = agg["std_cumulative_regret"]
        x = np.arange(1, len(mean) + 1)
        color = _get_color(name)

        # Main curve
        ax.plot(x, mean, label=name, color=color)
        # Shaded ±1 std band
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    _matched_xaxis(ax, len(mean), log=log_x)
    ax.set_ylabel("Cumulative Regret $R_T$")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_regret_rate(results: dict, log_x: bool = True, figsize: tuple = (10, 6), title: str = "Per-Step Regret Rate $R_T / T$",) -> plt.Figure:
    """
    Function to plot the per-step regret rate R_T/T over matched events. A converging algorithm's curve
    should decay toward zero. Algorithms that fail to converge will plateau at a positive constant. This
    plot makes convergence of the regret rate directly visible.

    Parameters
    ----------
    * results: output of compare.run_comparison()
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name in results["algorithm_names"]:
        # Recompute regret rate from mean cumulative regret
        mean_regret = results["agg_results"][name]["mean_cumulative_regret"]
        t_axis = np.arange(1, len(mean_regret) + 1, dtype=float)
        rate = mean_regret / t_axis
        x = t_axis
        color = _get_color(name)
        ax.plot(x, rate, label=name, color=color)

    # Reference line at 0 — a perfectly converged algorithm would reach this
    ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    _matched_xaxis(ax, len(mean_regret), log=log_x)
    ax.set_ylabel("Regret rate $R_T / T$")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_rolling_reward(results: dict, base_rate: float = None, log_x: bool = True, figsize: tuple = (10, 6),
    title: str = "Rolling Average Reward (Learning Curves)",) -> plt.Figure:
    """
    Function to plot smoothed learning curves (rolling average reward) for all algorithms. This is the
    most readable figure for a non-specialist audience. A horizontal reference line at the dataset base
    reward rate shows whether algorithms are outperforming random item selection.

    Parameters
    ----------
    * results: output of compare.run_comparison()
    * base_rate: dataset reward rate (fraction of ratings >= threshold); if provided, a reference line is drawn
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name in results["algorithm_names"]:
        agg = results["agg_results"][name]
        mean  = agg["mean_rolling_reward"]
        std = agg["std_rolling_reward"]
        x = np.arange(1, len(mean) + 1)
        color = _get_color(name)

        ax.plot(x, mean, label=name, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    if base_rate is not None:
        _add_base_rate_line(ax, base_rate)

    _matched_xaxis(ax, len(mean), log=log_x)
    ax.set_ylabel("Rolling average reward")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Group 2: Convergence plots
# ---------------------------------------------------------------------------
def plot_arm_entropy(results: dict, log_x: bool = True, figsize: tuple = (10, 6),
    title: str = "Arm Selection Entropy Over Time", ) -> plt.Figure:
    """
    Function to plot rolling arm-selection entropy for all algorithms. Higher entropy means broader
    exploration, while lower, stable entropy means converged policy (in the form of arms). The entropy
    should generally decrease over time as the algorithm learns which arms are good and concentrates its
    selections.

    Parameters
    ----------
    * results: output of compare.run_comparison()
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name in results["algorithm_names"]:
        agg = results["agg_results"][name]
        mean = agg["mean_arm_entropy"]
        std = agg["std_arm_entropy"]
        x = np.arange(1, len(mean) + 1)
        color = _get_color(name)

        ax.plot(x, mean, label=name, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    _matched_xaxis(ax, len(mean), log=log_x)
    ax.set_ylabel("Arm selection entropy $H_t$")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_entropy_reward_joint(results: dict, algorithm_names: list = None, log_x: bool = True, figsize: tuple = (14, 5),) -> plt.Figure:
    """
    For each algorithm, plot rolling reward (left y-axis) and arm entropy (right y-axis, inverted so 'more
    converged' is visually up on both sides) on a shared x-axis. This dual-axis plot makes the relationship
    between falling entropy and rising reward directly visible — the signature of a well-functioning
    contextual bandit.

    Parameters
    ----------
    * results: output of compare.run_comparison()
    * algorithm_names: subset of algorithms to plot; defaults to all algorithms in results
    """
    if algorithm_names is None: algorithm_names = results["algorithm_names"]

    n_algo = len(algorithm_names)
    fig, axes = plt.subplots(1, n_algo, figsize=figsize, sharey=False)

    # Handle single-algorithm case
    if n_algo == 1:
        axes = [axes]

    for ax, name in zip(axes, algorithm_names):
        agg = results["agg_results"][name]
        mean_rew = agg["mean_rolling_reward"]
        mean_ent = agg["mean_arm_entropy"]
        x = np.arange(1, len(mean_rew) + 1)
        color = _get_color(name)

        # Left axis: rolling reward
        ax.plot(x, mean_rew, color=color, linewidth=2, label="Reward")
        ax.set_ylabel("Rolling reward", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel("Matched events")
        ax.set_title(name)

        # Right axis: entropy (inverted — lower entropy is "up")
        ax2 = ax.twinx()
        ax2.plot(x, mean_ent, color="grey", linewidth=1.5, linestyle="--", label="Entropy")
        ax2.invert_yaxis()   # invert so "converged" (low entropy) is visually up
        ax2.set_ylabel("Arm entropy (inverted)", color="grey")
        ax2.tick_params(axis="y", labelcolor="grey")

    fig.suptitle("Reward vs. Entropy: Exploration–Exploitation Dynamics", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_convergence_steps(results: dict, figsize: tuple = (9, 5), title: str = "Convergence Step $t^*$ by Algorithm",) -> plt.Figure:
    """
    Bar chart of the mean convergence step t* for each algorithm, with error bars showing std across trials.
    Lower t* = the algorithm required fewer matched events to converge. This is the single-number answer
    to 'which algorithm learns fastest?'

    Parameters
    ----------
    * results: output of compare.run_comparison()
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = results["algorithm_names"]
    means = [results["conv_reports"][n]["mean_convergence_step"] for n in names]
    stds = [results["conv_reports"][n]["std_convergence_step"]  for n in names]
    colors = [_get_color(n) for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Convergence step $t^*$ (matched events)")
    ax.set_title(title)
    # Annotate bars with their value
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{mean:.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def plot_regret_slope_ratio(results: dict, figsize: tuple = (9, 5), title: str = "Regret Slope Ratio (Late / Early)",) -> plt.Figure:
    """
    Grouped bar chart showing early-window and late-window regret slopes side by side for each algorithm.
    A bar pair where the late bar is much shorter than the early bar indicates strong convergence. A ratio
    close to 1 (equal bars) means the algorithm is still incurring regret at the same rate throughout.

    Parameters
    ----------
    * results: output of compare.run_comparison()
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = results["algorithm_names"]
    early = [results["conv_reports"][n]["mean_slope_ratio"] * 0 +
             results["conv_reports"][n].get("mean_early_slope",
             results["conv_reports"][n]["mean_slope_ratio"])
             for n in names]

    # Recompute early and late slopes from raw trial data for display
    early_slopes = []
    late_slopes  = []
    for name in names:
        all_trials = results["agg_results"][name]["all_trials"]
        e_list, l_list = [], []
        for trial in all_trials:
            from analysis.convergence import compute_regret_slope_ratio
            sr = compute_regret_slope_ratio(trial["cumulative_regret"])
            e_list.append(sr["early_slope"])
            l_list.append(sr["late_slope"])
        early_slopes.append(float(np.mean(e_list)))
        late_slopes.append(float(np.mean(l_list)))

    x      = np.arange(len(names))
    width  = 0.35
    colors = [_get_color(n) for n in names]

    # Early bars (darker shade)
    ax.bar(x - width/2, early_slopes, width, label="Early window", color=colors, alpha=0.9, edgecolor="white")
    # Late bars (lighter shade)
    ax.bar(x + width/2, late_slopes,  width, label="Late window", color=colors, alpha=0.45, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Regret slope (regret per matched step)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Group 3: Exploration vs. exploitation
# ---------------------------------------------------------------------------
def plot_arm_pull_heatmap(results: dict, top_k: int = 20, figsize: tuple = (12, 6), title: str = "Arm Pull Distribution (Top-K Arms)",) -> plt.Figure:
    """
    Heatmap of arm pull fractions: rows = algorithms, columns = top-K arms. Reveals whether any algorithm
    is over-concentrating on a small set of arms (exploitation collapse) vs. maintaining healthy diversity.
    Color scale is shared across all algorithms.

    Parameters
    ----------
    * results: output of compare.run_comparison()
    top_k: number of top arms (by total pulls across all algorithms) to display
    """
    names = results["algorithm_names"]

    # Collect arm counts for each algorithm (averaged across trials)
    all_counts = {}
    for name in names:
        trial_counts = [t["arm_counts"] for t in results["agg_results"][name]["all_trials"]]
        # Mean arm counts across trials
        all_counts[name] = np.mean(trial_counts, axis=0)

    # Identify top-K arms by total pulls summed across all algorithms
    total_pulls = sum(all_counts[n] for n in names)
    top_arms = np.argsort(total_pulls)[::-1][:top_k]

    # Build heatmap matrix: shape (n_algorithms, top_k)
    matrix = np.zeros((len(names), top_k))
    for i, name in enumerate(names):
        counts = all_counts[name][top_arms]
        # Normalise to fraction of total pulls for this algorithm
        total  = all_counts[name].sum()
        matrix[i] = counts / total if total > 0 else counts

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=matrix.max())

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"Arm {top_arms[i]}" for i in range(top_k)], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Arm (movie)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Pull fraction")
    fig.tight_layout()
    return fig


def plot_exploration_bonus_decay(results: dict, figsize: tuple = (10, 6), title: str = "Exploration Signal Decay Over Time",) -> plt.Figure:
    """
    Plot the mean arm-selection entropy over matched events as a proxy for how the exploration signal
    decays across algorithms. Algorithms that reduce their exploration signal more quickly are transitioning
    to exploitation faster — whether this is good or bad depends on whether they have learned a good policy
    by that point.

    This uses the arm entropy series already computed in evaluate.py rather than re-accessing the exploration
    bonus directly (which would require the agent objects). The entropy serves as a model-agnostic proxy
    for exploration intensity that is comparable across all algorithm types.

    Parameters
    ----------
    * results: output of compare.run_comparison()
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name in results["algorithm_names"]:
        agg = results["agg_results"][name]
        mean = agg["mean_arm_entropy"]
        x = np.arange(1, len(mean) + 1)
        color = _get_color(name)
        ax.plot(x, mean, label=name, color=color)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Matched events (log scale)")
    ax.set_ylabel("Mean arm selection entropy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Group 4: Subgroup and robustness
# ---------------------------------------------------------------------------
def plot_reward_threshold_sensitivity(threshold_results: dict, figsize: tuple = (10, 6), title: str = "Performance vs. Reward Threshold",) -> plt.Figure:
    """
    Line plot of final cumulative regret vs. reward binarization threshold, one line per algorithm. A stable
    relative ranking across thresholds means the conclusions from the main comparison are robust to the
    binarization choice.

    Parameters
    ----------
    * threshold_results: output of sensitivity.sweep_reward_threshold()
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    thresholds = threshold_results["thresholds"]
    base_rates = threshold_results["reward_rates"]

    # Left panel: regret vs threshold
    ax = axes[0]
    for name, regrets in threshold_results["mean_regret"].items():
        ax.plot(thresholds, regrets, marker="o", label=name, color=_get_color(name))
    ax.set_xlabel("Reward threshold")
    ax.set_ylabel("Mean final cumulative regret")
    ax.set_title("Regret vs. Reward Threshold")
    ax.legend()

    # Right panel: base reward rate at each threshold (context for interpretation)
    ax2 = axes[1]
    ax2.plot(thresholds, base_rates, marker="s", color="black", linewidth=1.5, label="Base reward rate")
    ax2.set_xlabel("Reward threshold")
    ax2.set_ylabel("Base reward rate")
    ax2.set_title("Base Reward Rate at Each Threshold")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_context_dim_sensitivity(dim_results: dict, figsize: tuple = (10, 6), title: str = "Performance vs. Context Dimensionality (PCA)",) -> plt.Figure:
    """
    Line plot of final cumulative regret vs. PCA context dimensionality. A flat line means the algorithm
    is not using the additional context; a decreasing line means richer features improve performance.

    Parameters
    ----------
    * dim_results: output of sensitivity.sweep_context_dimensionality()
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    dims = dim_results["dims"]

    # Left panel: regret vs dimension
    ax = axes[0]
    for name in dim_results["mean_regret"]:
        means = dim_results["mean_regret"][name]
        stds = dim_results["std_regret"][name]
        color = _get_color(name)
        ax.plot(dims, means, marker="o", label=name, color=color)
        ax.fill_between(dims,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15, color=color)
    ax.set_xlabel("Context dimensionality (PCA components)")
    ax.set_ylabel("Mean final cumulative regret")
    ax.set_title("Regret vs. Context Dim")
    ax.legend()

    # Right panel: reward rate vs dimension
    ax2 = axes[1]
    for name in dim_results["mean_reward"]:
        rewards = dim_results["mean_reward"][name]
        color = _get_color(name)
        ax2.plot(dims, rewards, marker="o", label=name, color=color)
    ax2.set_xlabel("Context dimensionality (PCA components)")
    ax2.set_ylabel("Mean reward rate")
    ax2.set_title("Reward Rate vs. Context Dim")
    ax2.legend()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Group 5: Hyperparameter sensitivity
# ---------------------------------------------------------------------------
def plot_hyperparameter_sweep(sweep_result: dict, metric: str = "regret", figsize: tuple = (8, 5),
    log_x: bool = False,) -> plt.Figure:
    """
    Plot regret or reward rate as a function of one swept hyperparameter for a single algorithm. A steep
    curve indicates high sensitivity — the algorithm requires careful tuning. A flat curve indicates
    robustness.

    Parameters
    ----------
    * sweep_result: output of sensitivity.sweep_hyperparameter()
    * metric: 'regret' or 'reward'
    * log_x: log scale on x-axis (useful for lambda, alpha)
    """
    fig, ax = plt.subplots(figsize=figsize)

    param_values = sweep_result["param_values"]
    color = _get_color(sweep_result["agent_name"])

    if metric == "regret":
        means = sweep_result["mean_final_regret"]
        stds = sweep_result["std_final_regret"]
        ylabel = "Mean final cumulative regret"
        # Mark best value with a star
        best_val = sweep_result["best_value"]
    else:
        means = sweep_result["mean_reward_rate"]
        stds = sweep_result["std_reward_rate"]
        ylabel = "Mean reward rate"
        best_val = sweep_result["best_reward_value"]

    ax.plot(param_values, means, marker="o", color=color, label=sweep_result["agent_name"])
    ax.fill_between(param_values,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.15, color=color)

    # Mark best value
    best_idx = param_values.index(best_val)
    ax.axvline(best_val, color=color, linestyle="--", linewidth=1.2, alpha=0.7, label=f"Best: {best_val}")
    ax.scatter([best_val], [means[best_idx]], color=color, s=120, zorder=5, marker="*")

    if log_x: ax.set_xscale("log")
    ax.set_xlabel(sweep_result["param_name"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{sweep_result['agent_name']}: "
                 f"Sensitivity to {sweep_result['param_name']}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_all_hyperparam_sweeps(all_sweep_results: dict, metric: str = "regret", figsize_per_panel: tuple = (5, 4),) -> plt.Figure:
    """
    Grid of hyperparameter sensitivity plots, one panel per (algorithm, parameter) combination.

    Parameters
    ----------
    * all_sweep_results: mapping algorithm_name -> {param_name -> sweep_result dict} (output of sensitivity per-algorithm functions)
    * metric: 'regret' or 'reward'
    * figsize_per_panel: size of each individual panel
    """
    # Collect all (algo, param) pairs
    panels = []
    for algo_name, param_dict in all_sweep_results.items():
        for param_name, sweep in param_dict.items():
            if isinstance(sweep, dict) and "param_values" in sweep:
                panels.append((algo_name, param_name, sweep))
            elif isinstance(sweep, dict) and "fixed" in sweep:
                # EpsilonGreedy schedule sweeps
                panels.append((algo_name, f"epsilon (fixed)",  sweep["fixed"]))
                panels.append((algo_name, f"epsilon (decay)",  sweep["decay"]))

    n_panels = len(panels)
    if n_panels == 0: return plt.figure()

    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    for ax, (algo_name, param_name, sweep) in zip(axes_flat, panels):
        color = _get_color(algo_name)

        if metric == "regret":
            means = sweep["mean_final_regret"]
            stds = sweep["std_final_regret"]
            best_val = sweep["best_value"]
            ylabel = "Regret"
        else:
            means = sweep["mean_reward_rate"]
            stds = sweep["std_reward_rate"]
            best_val = sweep["best_reward_value"]
            ylabel = "Reward"

        pv = sweep["param_values"]
        ax.plot(pv, means, marker="o", color=color)
        ax.fill_between(pv,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15, color=color)
        ax.axvline(best_val, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"{algo_name}\n{param_name}", fontsize=9)
        ax.set_xlabel(param_name, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide any unused panels
    for ax in axes_flat[n_panels:]: ax.set_visible(False)

    fig.suptitle(f"Hyperparameter Sensitivity ({metric.capitalize()})", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def save_all_figures(output_dir: str, prefix: str = "", fmt: str = "png", dpi: int = 300,) -> None:
    """
    Function to save all currently open matplotlib figures to output_dir. Each figure is saved as <prefix><figure_number>.<fmt>.

    Parameters
    ----------
    * output_dir: directory to save figures into (created if needed)
    * prefix: filename prefix (e.g. 'linucb_')
    * fmt: file format ('png', 'pdf', 'svg')
    * dpi: resolution for raster formats
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, fig_num in enumerate(plt.get_fignums()):
        fig  = plt.figure(fig_num)
        path = os.path.join(output_dir, f"{prefix}fig_{i+1:02d}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {path}")

    print(f"All figures saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import pandas as pd
    from algorithms.lin_ucb import LinUCB
    from algorithms.thompson_sampling import ThompsonSampling
    from analysis.comparison import run_comparison

    print("Building synthetic data for visualize.py smoke test...\n")

    rng = np.random.default_rng(0)
    n = 3000
    n_arms = 10
    ctx_dim = 10
    arm_pool = list(range(n_arms))

    movie_ids = rng.choice(arm_pool, size=n)
    rewards = np.where(movie_ids == 0,
                         (rng.random(n) > 0.3).astype(float),
                         (rng.random(n) > 0.6).astype(float))

    synthetic_df = pd.DataFrame({
        "user_id": rng.integers(1, 30, size=n),
        "movie_id": movie_ids,
        "rating": rng.choice([1,2,3,4,5], size=n).astype(float),
        "timestamp": np.arange(n),
        "reward": rewards,
        "context": [rng.random(ctx_dim) for _ in range(n)],
    })

    test_config = [
        {"name": "LinUCB", "class": LinUCB,
         "kwargs": {"n_arms": n_arms, "context_dim": ctx_dim, "alpha": 1.0}},
        {"name": "ThompsonSampling", "class": ThompsonSampling,
         "kwargs": {"n_arms": n_arms, "context_dim": ctx_dim, "sigma": 1.0}},
    ]

    results = run_comparison(
        df=synthetic_df, context_dim=ctx_dim, config=test_config,
        n_trials=2, window=100,
        convergence_kwargs={"tolerance": 0.02, "sustained_window": 50, "min_step": 30},
        verbose=False,
    )

    set_style()
    print("Generating plots...")

    figs = [
        ("cumulative_regret", plot_cumulative_regret(results)),
        ("regret_rate", plot_regret_rate(results)),
        ("rolling_reward", plot_rolling_reward(results, base_rate=0.5)),
        ("arm_entropy", plot_arm_entropy(results)),
        ("entropy_reward_joint", plot_entropy_reward_joint(results)),
        ("convergence_steps", plot_convergence_steps(results)),
        ("regret_slope_ratio", plot_regret_slope_ratio(results)),
        ("arm_pull_heatmap", plot_arm_pull_heatmap(results, top_k=6)),
        ("exploration_bonus", plot_exploration_bonus_decay(results)),
    ]

    # Verify all figures were created successfully
    for name, fig in figs:
        assert fig is not None, f"Figure '{name}' returned None"
        assert len(fig.axes) > 0, f"Figure '{name}' has no axes"
        print(f"  {name}: OK ({len(fig.axes)} axes)")

    print("\nAll figures created successfully")
    print("\nSmoke test passed.")

    import os
    os.makedirs("figures/smoke_tests", exist_ok=True)
    for i, (name, fig) in enumerate(figs):
        fig.savefig(f"figures/smoke_tests/{name}.png", dpi=120, bbox_inches="tight")
        print(f"  Saved: figures/smoke_tests/{name}.png")

    plt.close("all")