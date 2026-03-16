"""
analysis/convergence.py

==============

Convergence analysis for bandit evaluation results


This convergence analysis script takes the raw results dictionaries, produced by the `evaluate.py` script,
and extracts the three formal convergence measurements, corresponding to the three notions of convergence
described from the project proposal:

    1. Reward convergence: the matched-event index t* at which the rolling average reward stabilizes within
                           a tolerance band.
    2. Policy convergence: arm selection entropy over time; a converged policy has low, stable entropy.
    3. Regret rate convergence: the per-step regret R_T/T over time; a converged algorithm has a decaying
                                rate. Formalized via the ratio of early-window to late-window regret slopes.

All functions operate on the output of `evaluate.run_trial()` or on `evaluate.run_multiple_trials()`, so
they can slot cleanly into the analysis pipeline without requiring access to the agent or to the environment
directly.

Usage
-----
    from convergence import (
        find_convergence_step,
        compute_regret_slope_ratio,
        compute_entropy_stats,
        full_convergence_report,
    )
    from evaluate import run_trial

    result = run_trial(agent, env)
    report = full_convergence_report(result)
    print(report)
"""

# Import dependencies
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Reward convergence — finding t*
# ---------------------------------------------------------------------------
def find_convergence_step(rolling_reward: np.ndarray, tolerance: float = 0.01, sustained_window: int = 200,
    min_step: int = 100, ) -> int:
    """
    Function to find the matched-event index t* at which the rolling reward has converged. Convergence is
    defind as the first index t >= min_step such that the rolling reward stays within [final_value ± tolerance]
    for all subsequent `sustained_window` steps. This formalizes the intuition that the algorithm has "stopped
    improving" once the reward curve flattens and stays flats. 

    Parameters
    ----------
    * rolling_reward: smoothed reward over matched steps (output of evaluate.run_trial)
    * tolerance: half-width of the convergence band around the final rolling reward value; default 0.01
                 means the reward must stay within ±1% of its final value
    * sustained_window: number of consecutive steps that must remain within the tolerance band to declare
                        convergence
    * min_step: earliest matched step to consider as t*; prevents declaring convergence during warmup

    Returns
    -------
    * int: the convergence step t*, or len(rolling_reward) - 1 if the reward never stabilizes (indicating
           the algorithm did not converge in the available data).
    """
    n = len(rolling_reward)
    # Use the final rolling reward value as the reference "converged" level
    final_val = rolling_reward[-1]
    lower = final_val - tolerance
    upper = final_val + tolerance

    for t in range(min_step, n - sustained_window):
        # Check whether all steps in [t, t + sustained_window) stay in band
        window_vals = rolling_reward[t : t + sustained_window]
        if np.all((window_vals >= lower) & (window_vals <= upper)):
            return t   # first step where sustained convergence begins

    # If no convergence found, return the last step as a sentinel value
    return (n - 1)


def convergence_steps_across_trials(all_trials: list[dict], tolerance: float = 0.01, sustained_window: int = 200,
    min_step: int = 100, ) -> np.ndarray:
    """
    Function to compute the convergence step t* for each trial in a multi-trial result. Useful for computing
    mean ± std of convergence steps across seeds, which is reported in the summary table and in the convergence
    step bar chat.

    Parameters
    ----------
    * all_trials: list of single-trial result dicts (from run_multiple_trials)
    * tolerance: passed to find_convergence_step
    * sustained_window: passed to find_convergence_step
    * min_step: passed to find_convergence_step

    Returns
    -------
    * np.ndarray: array of shape (n_trials,) — t* for each trial
    """
    return np.array([
        find_convergence_step(
            trial["rolling_reward"],
            tolerance=tolerance,
            sustained_window=sustained_window,
            min_step=min_step,
        )
        for trial in all_trials
    ])


# ---------------------------------------------------------------------------
# Regret rate convergence (slope ratio)
# ---------------------------------------------------------------------------
def compute_regret_slope(cumulative_regret: np.ndarray, start: int, end: int,) -> float:
    """
    Function to estimate the slope of cumulative regret over the interval [start, end) by fitting a
    straight line via least squares. The slope represents the average per-step regret in that window. A
    well-converged algorithm has a much shallower slope in late windows than in early windows. 

    Parameters
    ----------
    * cumulative_regret: array with cumulative regret over matched steps
    * start: start index of the window (inclusive)
    * end: end index of the window (exclusive)

    Returns
    -------
    * float: estimated slope (regret per matched step) in the window
    """
    segment = cumulative_regret[start:end]
    if len(segment) < 2:
        return 0.0
    # x-axis is just the step indices within the window
    x = np.arange(len(segment), dtype=float)
    # np.polyfit degree 1 returns [slope, intercept]
    slope, _ = np.polyfit(x, segment, deg=1)
    return float(slope)


def compute_regret_slope_ratio(cumulative_regret: np.ndarray, early_fraction: float = 0.25, late_fraction:  float = 0.25,) -> dict:
    """
    Function to compute the ratio of the late-window regret slope to the early-window regret slope. A
    ratio close to 0 means the algorithm has learned aggressively in its early stage and nearly stopped
    incurring regret by the later stages, the equivalent to strong convergence. A ratio close to 1 means
    the per-step regret barely changed, the equivalent to no meaningful convergence.

    Parameters
    ----------
    * cumulative_regret: array containing cumulative regret over matched steps
    * early_fraction: fraction of total steps defining the early window (e.g. 0.25 = first 25% of matched steps)
    * late_fraction: float — fraction of total steps defining the late window (e.g. 0.25 = last 25% of matched steps)

    Returns
    -------
    dict: dictionary with keys early_slope (mean per-step regret in the early window), late_slope (mean-per-step
          regret in the late window), slope_ratio (late_slop / early_slope, where lower is better convergence),
          early_window (tuple of indices indicating the early window), late_window (tuple of indices indicating
          the later window).
    """
    n = len(cumulative_regret)
    early_end = max(2, int(n * early_fraction))
    late_start = min(n - 2, int(n * (1.0 - late_fraction)))

    early_slope = compute_regret_slope(cumulative_regret, 0, early_end)
    late_slope = compute_regret_slope(cumulative_regret, late_start, n)

    # Avoid division by zero if early slope is essentially flat
    if early_slope < 1e-10:
        slope_ratio = 0.0
    else:
        slope_ratio = late_slope / early_slope

    return {
        "early_slope":  early_slope,
        "late_slope":   late_slope,
        "slope_ratio":  slope_ratio,
        "early_window": (0, early_end),
        "late_window":  (late_start, n),
    }


def slope_ratios_across_trials(all_trials: list[dict], early_fraction: float = 0.25, late_fraction:  float = 0.25,) -> np.ndarray:
    """
    Function to compute the regret slope ratio for each trial in a multi-trial result.

    Returns
    -------
    * np.ndarray: array of shape (n_trials,), with the slope ratio for each trial
    """
    return np.array([
        compute_regret_slope_ratio(
            trial["cumulative_regret"],
            early_fraction=early_fraction,
            late_fraction=late_fraction,
        )["slope_ratio"]
        for trial in all_trials
    ])


# ---------------------------------------------------------------------------
# Policy convergence (entropy statistics)
# ---------------------------------------------------------------------------
def compute_entropy_stats(arm_entropy: np.ndarray, late_fraction: float = 0.2,) -> dict:
    """
    Function to compute summary statistics on the arm selection entropy series. The entropy series is
    computed in `evaluate.py` as a rolling Shannon entropy of arm selections of matched events. Here, we
    sumarize:
        - Initial entropy: mean over the first 10% of steps (reflects early exploration breadth)
        - Final entropy: mean over the last `late_fraction` of steps (reflects converged policy spread)
        - Entropy drop: initial - final (larger = more convergence)
        - Entropy at t*: not computed here, done in full_convergence_report

    Parameters
    ----------
    * arm_entropy: array of rolling entropy over matched steps
    * late_fraction: fraction of steps defining the "final" window

    Returns
    -------
    * dict: dictionary with keys initial_entropy, final_entropy, entropy_drop, entropy_min, entropy_max.
    """
    n = len(arm_entropy)
    early_end = max(1, int(n * 0.1))
    late_start = max(0, int(n * (1.0 - late_fraction)))

    initial_entropy = float(arm_entropy[:early_end].mean())
    final_entropy = float(arm_entropy[late_start:].mean())

    return {
        "initial_entropy": initial_entropy,
        "final_entropy":   final_entropy,
        # How much entropy dropped — captures exploration → exploitation shift
        "entropy_drop":    initial_entropy - final_entropy,
        "entropy_min":     float(arm_entropy.min()),
        "entropy_max":     float(arm_entropy.max()),
    }


# ---------------------------------------------------------------------------
# Full convergence report (combines all three analyses)
# ---------------------------------------------------------------------------
def full_convergence_report(result: dict, tolerance: float = 0.01, sustained_window: int = 200, min_step: int = 100,
    early_fraction: float = 0.25, late_fraction:  float = 0.25,) -> dict:
    """
    Function produce a complete convergence report for a single trial result. This function combines all
    three convergence measurements into one dictionary that can be directly used for plotting and table
    generation in the `visualize.py` and `compare.py` scripts. 

    Parameters
    ----------
    * result: single-trial result from evaluate.run_trial()
    * tolerance: reward convergence band half-width
    * sustained_window : steps required to sustain convergence
    * min_step: earliest step to declare convergence
    * early_fraction: early window size for slope computation
    * late_fraction : late window size for slope computation

    Returns
    -------
    dict with keys:
        agent_name        : str
        convergence_step  : int   — t* (reward convergence)
        slope_ratio       : float — late/early regret slope ratio
        early_slope       : float — regret slope in the early window
        late_slope        : float — regret slope in the late window
        initial_entropy   : float — mean entropy in the first 10% of steps
        final_entropy     : float — mean entropy in the last 20% of steps
        entropy_drop      : float — initial_entropy - final_entropy
        final_regret      : float — cumulative regret at the last matched step
        mean_reward       : float — mean reward over all matched steps
        matched_steps     : int
    """
    conv_step = find_convergence_step(result["rolling_reward"], tolerance=tolerance,
        sustained_window=sustained_window, min_step=min_step,)
    slope_info = compute_regret_slope_ratio(result["cumulative_regret"],
        early_fraction=early_fraction, late_fraction=late_fraction,)
    entropy_info = compute_entropy_stats(result["arm_entropy"], late_fraction=late_fraction,)

    return {
        "agent_name":       result["agent_name"],
        "convergence_step": conv_step,
        "slope_ratio":      slope_info["slope_ratio"],
        "early_slope":      slope_info["early_slope"],
        "late_slope":       slope_info["late_slope"],
        "initial_entropy":  entropy_info["initial_entropy"],
        "final_entropy":    entropy_info["final_entropy"],
        "entropy_drop":     entropy_info["entropy_drop"],
        "final_regret":     float(result["cumulative_regret"][-1]),
        "mean_reward":      float(result["rewards"].mean()),
        "matched_steps":    result["matched_steps"],
    }


def full_convergence_report_multi(agg: dict, tolerance: float = 0.01, sustained_window: int = 200, min_step: int = 100,
    early_fraction: float = 0.25, late_fraction:  float = 0.25,) -> dict:
    """
    Function to produce a convergence report aggregated across multiple trials. The function computes
    per-trial convergence statistics and returns mean ± std values intervals for each metric, suitable
    for the final comparison table.

    Parameters
    ----------
    * agg: dictionary containing multi-trial aggregated result from evaluate.run_multiple_trials()

    Returns
    -------
    * dict: dictionary with with mean_ and std_ versions of each convergence metric
    """
    all_trials = agg["all_trials"]

    # Compute per-trial convergence steps
    conv_steps  = convergence_steps_across_trials(
        all_trials, tolerance=tolerance,
        sustained_window=sustained_window, min_step=min_step,
    )
    # Compute per-trial slope ratios
    s_ratios    = slope_ratios_across_trials(
        all_trials, early_fraction=early_fraction, late_fraction=late_fraction,
    )
    # Compute per-trial entropy stats
    entropy_stats = [
        compute_entropy_stats(t["arm_entropy"], late_fraction=late_fraction)
        for t in all_trials
    ]

    return {
        "agent_name":             agg["agent_name"],
        # Reward convergence
        "mean_convergence_step":  float(conv_steps.mean()),
        "std_convergence_step":   float(conv_steps.std()),
        # Regret slope
        "mean_slope_ratio":       round(float(s_ratios.mean()),4),
        "std_slope_ratio":        round(float(s_ratios.std()),4),
        # Entropy
        "mean_initial_entropy":   round(float(np.mean([e["initial_entropy"] for e in entropy_stats])),4),
        "mean_final_entropy":     round(float(np.mean([e["final_entropy"]   for e in entropy_stats])),4),
        "mean_entropy_drop":      round(float(np.mean([e["entropy_drop"]    for e in entropy_stats])),4),
        "std_entropy_drop":       round(float(np.std( [e["entropy_drop"]    for e in entropy_stats])),4),
        # Final regret summary (already in agg, included here for convenience)
        "mean_final_regret":      round(agg["mean_final_regret"],4),
        "std_final_regret":       round(agg["std_final_regret"],4),
        "n_trials":               agg["n_trials"],
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import pandas as pd
    from algorithms.lin_ucb import LinUCB
    from algorithms.thompson_sampling import ThompsonSampling
    from bandits_environment import OfflineBanditEnv
    from evaluation import run_trial, run_multiple_trials

    print("Building synthetic environment for convergence smoke test...\n")
    rng      = np.random.default_rng(0)
    n        = 3000
    n_arms   = 10
    ctx_dim  = 8
    arm_pool = list(range(n_arms))

    # Construct a dataset where arm 0 has consistently higher reward,
    # so algorithms that learn should converge toward it over time.
    movie_ids = rng.choice(arm_pool, size=n)
    rewards   = np.where(movie_ids == 0,
                         rng.random(n) > 0.3,   # arm 0: 70% reward
                         rng.random(n) > 0.6)   # others: 40% reward

    synthetic_df = pd.DataFrame({
        "user_id": rng.integers(1, 50, size=n),
        "movie_id": movie_ids,
        "rating": rng.choice([1,2,3,4,5], size=n).astype(float),
        "timestamp": np.arange(n),
        "reward": rewards.astype(float),
        "context": [rng.random(ctx_dim) for _ in range(n)],
    })

    env = OfflineBanditEnv(synthetic_df, seed=0, arm_pool=arm_pool)

    # --- Single trial convergence report ---
    print("Single trial convergence report (LinUCB):")
    agent  = LinUCB(n_arms=n_arms, context_dim=ctx_dim, alpha=1.0, seed=0)
    result = run_trial(agent, env, window=100)
    report = full_convergence_report(
        result, tolerance=0.02, sustained_window=50, min_step=50
    )
    for k, v in report.items():
        print(f"  {k:<25}: {v}")

    # Sanity checks on report values
    assert 0 <= report["convergence_step"] <= result["matched_steps"]
    assert 0.0 <= report["slope_ratio"]
    expected_drop = report["initial_entropy"] - report["final_entropy"]
    assert abs(report["entropy_drop"] - expected_drop) < 1e-10
    _ = (
        report["initial_entropy"] - report["final_entropy"]
    )
    print("\n  Single trial report checks passed ✓")

    # --- Multi-trial convergence report ---
    print("\nMulti-trial convergence report (ThompsonSampling, 3 trials):")
    agg = run_multiple_trials(
        ThompsonSampling, env, n_trials=3,
        agent_kwargs=dict(n_arms=n_arms, context_dim=ctx_dim, sigma=1.0),
        window=100,
    )
    multi_report = full_convergence_report_multi(
        agg, tolerance=0.02, sustained_window=50, min_step=50
    )
    for k, v in multi_report.items():
        print(f"  {k:<30}: {v:.4f}" if isinstance(v, float) else f"  {k:<30}: {v}")

    assert multi_report["n_trials"] == 3
    assert multi_report["mean_convergence_step"] >= 0
    print("\n  Multi-trial report checks passed")

    print("\nSmoke test passed.")