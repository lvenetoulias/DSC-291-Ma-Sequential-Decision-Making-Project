"""
analysis/evaluate.py

===========

Core evaluation pipeline for offline bandit replay



This evauation script is responsible for running a single algorithm through one pass of the offline bandit
environment, and then collecting all raw metrics needed for downstream analysis. The script provides a
multi-trial wrapper that runs the same algorithm repeatedly across a number of different random seeds and
aggregates the results, necessary for all of the stochastic algorithms (i.e., Thompson Sampling, NeuralTS,
Bootstrap Thompson Sampling), whose single-trial results are guaranteed to have higher variance. 

The signal trial produces a results dictionary:

    rewards          : np.ndarray (n_matched,)   — reward at each matched step
    cumulative_regret: np.ndarray (n_matched,)   — cumulative regret over matched steps
    regret_per_step  : np.ndarray (n_matched,)   — R_T / T at each matched step
    rolling_reward   : np.ndarray (n_matched,)   — smoothed reward rate over time
    arm_selections   : list[int]                 — arm chosen at each total step
    matched_mask     : np.ndarray (n_total,)     — bool, True on matched steps
    match_rate       : float                     — fraction of total steps matched
    total_steps      : int                       — total events streamed
    matched_steps    : int                       — number of matched events
    arm_counts       : np.ndarray (n_arms,)      — total pulls per arm
    arm_entropy      : np.ndarray (n_matched,)   — rolling arm selection entropy

A multi-trial result aggregates across seeds and adds:

    mean_cumulative_regret : np.ndarray — mean R_T across trials (matched axis)
    std_cumulative_regret  : np.ndarray — std  R_T across trials
    mean_rolling_reward    : np.ndarray — mean rolling reward across trials
    std_rolling_reward     : np.ndarray — std  rolling reward across trials
    mean_match_rate        : float
    std_match_rate         : float
    all_trials             : list[dict] — raw per-trial result dicts

Usage
-----
    from evaluate import run_trial, run_multiple_trials
    from algorithms import LinUCB
    from bandit_env import OfflineBanditEnv

    env   = OfflineBanditEnv(train_df, seed=42)
    agent = LinUCB(n_arms=env.n_arms, context_dim=24, alpha=1.0)

    result = run_trial(agent, env)
    agg    = run_multiple_trials(LinUCB, env, n_trials=5,
                                 agent_kwargs=dict(context_dim=24, alpha=1.0))
"""


# Import dependencies
from bandits_environment import OfflineBanditEnv
from algorithms.base import BaseBandit
from typing import Any
import numpy as np
import os



# ---------------------------------------------------------------------------
# Rolling window helpers
# ---------------------------------------------------------------------------
def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """
    Function to compute a causal rolling mean of array x with the given window size. 'Causal' means the
    mean at time step t uses only values x(t-window+1: t+1), so no future information leaks into any
    earlier time steps. 

    For the first (window - 1) steps, where there are fewer steps than the window observations, we use
    all of the available observations up to that point, which is equivalent to expanding a window at the
    start. 

    Parameters
    ----------
    * x: 1-D numpy array of values to smooth
    * window: number of steps in the rolling window

    Returns
    -------
    * np.ndarray: array whose length is the same as context vector x
    """
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        # Use min(i+1, window) observations so early steps don't divide by zero
        start    = max(0, i - window + 1)
        out[i]   = x[start : i + 1].mean()
    return out


def _rolling_entropy(arm_selections: np.ndarray, n_arms: int, window: int) -> np.ndarray:
    """
    Function to compute the rolling Shannon entropy of the arm selection distribution. At each total step
    t, we compute the empirical distribution of all arm selections over the preceding 'window' steps and
    return its entropy:

        H_t = - sum_a p_a * log(p_a + eps)

    High entropy indicates the algorithm is exploring broadly; low entropy indicates it has concentrated
    on a small set of arms (converged policy).

    Parameters
    ----------
    * arm_selections: np.ndarray of int, shape (n_total_steps,) — arm chosen at each step
    * n_arms: total number of arms
    * window: rolling window size in total steps

    Returns
    -------
    * np.ndarray: array shape (n_total_steps,), showing entropy value at each step
    """
    eps = 1e-10   # small constant to avoid log(0)
    n = len(arm_selections)
    entropy = np.empty(n, dtype=float)

    for t in range(n):
        start   = max(0, t - window + 1)
        window_arms = arm_selections[start : t + 1]

        # Count how many times each arm was pulled in this window
        counts  = np.bincount(window_arms, minlength=n_arms).astype(float)
        probs   = counts / counts.sum()

        # Shannon entropy: H = -sum p * log(p), with eps guard
        entropy[t] = -np.sum(probs * np.log(probs + eps))

    return entropy


# ---------------------------------------------------------------------------
# Single-trial evaluation loop
# ---------------------------------------------------------------------------
def run_trial(agent: BaseBandit, env: OfflineBanditEnv, window: int = 500, verbose: bool = False,) -> dict:
    """
    Function to run a single algorithm through one complete pass of the offline bandit environment acorss
    all raw metrics. The evaluation follows the offline replay protocol:
        1. At each step, observe context and candidate arms from the environment.
        2. The agent selects an arm.
        3. If the chosen arm matches the logged arm, reveal the reward and call agent.update(). Otherwise,
           skip — no update is performed.
        4. Advance to the next event regardless of match.

    This function resets both the agent and the environment before running, meaning it is safe to call
    muktiple times for repeated trials.

    Parameters
    ----------
    * agent: BaseBandit, the algorithm to evaluate (will be reset)
    * env: OfflineBanditEnv, the offline replay environment (will be reset)
    * window: rolling window size (in matched events) for reward smoothing and arm entropy computation
    * verbose: boolean flag print progress every 10,000 total steps

    Returns
    -------
    * dict: results dictionary (see module docstring for full key listing)
    """
    # Reset both agent and environment so repeated calls are independent
    agent.reset()
    env.reset()

    # Storage for per-step metrics
    rewards_list = []   # reward at each matched step
    regret_list = []   # instantaneous regret (1 - reward) at each matched step
    arm_sel_list = []   # arm chosen at each *total* step (matched and skipped)
    matched_list = []   # bool: was this total step a match?

    # Main replay loop 
    while not env.is_done():

        # Get current context and the full arm pool from the environment
        context, candidate_movie_ids = env.get_context()
        # Convert movies to arms
        candidate_arms = [env.arm_to_idx(int(m)) for m in candidate_movie_ids]

        # Agent selects an arm index (into the arm pool, not raw movie_id)
        arm_idx = agent.select_arm(context, candidate_arms)

        # Convert arm index back to movie_id for the environment's step()
        chosen_movie_id = env.idx_to_arm(arm_idx)

        # Step the environment: reveals reward only if chosen == logged arm
        result = env.step(chosen_movie_id)

        # Record the arm index (not movie_id) chosen at this total step
        arm_sel_list.append(arm_idx)
        matched_list.append(result.matched)

        if result.matched:
            # Only update the agent and record reward on matched events
            agent.update(arm_idx, result.reward, context)
            rewards_list.append(result.reward)
            # Regret: best possible reward is 1 (binary setting)
            regret_list.append(1.0 - result.reward)

        if verbose and env._step_idx % 10_000 == 0:
            print(f"  [{agent.name}] step {env._step_idx:,} | "
                  f"matched {env._matched_idx:,} | "
                  f"match_rate {env.matched_rate():.4f}")

    # Convert lists to arrays
    rewards = np.array(rewards_list,  dtype=float)
    regret_steps = np.array(regret_list,   dtype=float)
    arm_selections = np.array(arm_sel_list,  dtype=int)
    matched_mask = np.array(matched_list,  dtype=bool)

    # Cumulative regret: R_T = sum_{t=1}^{T} (1 - r_t) over matched steps
    cumulative_regret = np.cumsum(regret_steps)

    # Per-step regret rate: R_T / T — should decay toward zero if converging
    n_matched = len(rewards)
    t_axis = np.arange(1, n_matched + 1, dtype=float)
    regret_per_step = cumulative_regret / t_axis

    # Rolling average reward over matched events (smoothed learning curve)
    rolling_reward = _rolling_mean(rewards, window=window)

    # Rolling arm selection entropy over *total* steps (exploration signal)
    arm_entropy_total = _rolling_entropy(arm_selections, env.n_arms, window=window)
    # Subsample entropy to matched steps only so it aligns with reward arrays
    arm_entropy = arm_entropy_total[matched_mask]

    # Per-arm pull counts (how often each arm was chosen across total steps)
    arm_counts = np.bincount(arm_selections, minlength=env.n_arms)

    return {
        "rewards":           rewards,
        "cumulative_regret": cumulative_regret,
        "regret_per_step":   regret_per_step,
        "rolling_reward":    rolling_reward,
        "arm_selections":    arm_selections,
        "matched_mask":      matched_mask,
        "match_rate":        env.matched_rate(),
        "total_steps":       env._step_idx,
        "matched_steps":     n_matched,
        "arm_counts":        arm_counts,
        "arm_entropy":       arm_entropy,
        "agent_name":        agent.name,
        "window":            window,
    }


# ---------------------------------------------------------------------------
# Multi-trial aggregation
# ---------------------------------------------------------------------------
def _align_to_length(arrays: list[np.ndarray], length: int) -> np.ndarray:
    """
    Function to align a list of 1-D arrays to a common length by truncating longer arrays and padding
    shorter ones with their final value. This is needed because different trials may have different numbers
    of matched events, due to the stochasiticity in the arm selection, so the arrays returned by the
    `run_trial()` method may yield slightly different lengths. 
    arrays and padding shorter ones with their final value.


    Parameters
    ----------
    * arrays: list of 1-D numpy arrays (possibly different lengths)
    * length: target length to align to

    Returns
    -------
    * np.ndarray: array of shape (len(arrays), length)
    """
    aligned = np.empty((len(arrays), length), dtype=float)
    for i, arr in enumerate(arrays):
        if len(arr) >= length:
            # Truncate to target length
            aligned[i] = arr[:length]
        else:
            # Pad with the last value (flat extrapolation)
            aligned[i, :len(arr)]  = arr
            aligned[i, len(arr):]  = arr[-1]
    return aligned


def run_multiple_trials(agent_class: type, env: OfflineBanditEnv, n_trials: int = 5, agent_kwargs: dict = None,
    window: int = 500, seeds: list[int] = None, verbose: bool = False,) -> dict:
    """
    Function to run an algorithm for multiple independent trials and to aggregate results. Each trial uses
    a different random seed (for the agent's RNG), so stochastic algorithms produce different trajectories.
    Deterministic algorithms (LinUCB, EpsilonGreedy with fixed epsilon) will produce identical results across
    trials, which is expected and serves as a useful reproducibility check. 

    The environment interaction order is identical across all trials (env is reset to the same state each
    time) to ensre that performance differences reflect the algorithm's behavior rather than the orer of
    the data.

    Parameters
    ----------
    * agent_class: the BaseBandit subclass to instantiate
    * env: OfflineBanditEnv, the environment (reset between trials)
    * n_trials: number of independent trials to run
    * agent_kwargs : keyword arguments passed to agent_class.__init__(); do NOT include 'seed' here, it
                     is set per trial
    * window: rolling window size passed to run_trial()
    * seeds: explicit seeds for each trial; if None, uses [0, 1, 2, ..., n_trials - 1]
    * verbose: boolean flag to print progress per trial

    Returns
    -------
    * dict: aggregated results (see module docstring for full key listing)
    """
    if agent_kwargs is None:
        agent_kwargs = {}

    if seeds is None:
        seeds = list(range(n_trials))

    if len(seeds) != n_trials:
        raise ValueError(
            f"len(seeds)={len(seeds)} must equal n_trials={n_trials}."
        )

    all_trials = []

    # Iterate through number of seeds the different trials 
    for trial_idx, seed in enumerate(seeds):
        if verbose:
            print(f"  Trial {trial_idx + 1}/{n_trials} (seed={seed})...")

        # Instantiate a fresh agent with this trial's seed
        agent = agent_class(seed=seed, **agent_kwargs)

        # Run one trial and collect results
        trial_result = run_trial(agent, env, window=window, verbose=False)
        all_trials.append(trial_result)

    # --- Determine a common length for alignment ---
    # Use the minimum number of matched steps across all trials so we never
    # extrapolate beyond what any trial actually observed.
    min_matched = min(r["matched_steps"] for r in all_trials)

    # --- Align and stack per-trial arrays ---
    cum_regret_mat   = _align_to_length(
        [r["cumulative_regret"] for r in all_trials], min_matched
    )
    rolling_rew_mat  = _align_to_length(
        [r["rolling_reward"]    for r in all_trials], min_matched
    )
    arm_entropy_mat  = _align_to_length(
        [r["arm_entropy"]       for r in all_trials], min_matched
    )

    # Aggregate statistics
    return {
        # Mean and std of cumulative regret across trials
        "mean_cumulative_regret": cum_regret_mat.mean(axis=0),
        "std_cumulative_regret":  cum_regret_mat.std(axis=0),

        # Mean and std of rolling reward across trials
        "mean_rolling_reward":    rolling_rew_mat.mean(axis=0),
        "std_rolling_reward":     rolling_rew_mat.std(axis=0),

        # Mean and std of arm entropy across trials
        "mean_arm_entropy":       arm_entropy_mat.mean(axis=0),
        "std_arm_entropy":        arm_entropy_mat.std(axis=0),

        # Scalar summaries
        "mean_match_rate":  float(np.mean([r["match_rate"]     for r in all_trials])),
        "std_match_rate":   float(np.std( [r["match_rate"]     for r in all_trials])),
        "mean_final_regret":float(np.mean([r["cumulative_regret"][-1] for r in all_trials])),
        "std_final_regret": float(np.std( [r["cumulative_regret"][-1] for r in all_trials])),

        # Metadata
        "n_trials":         n_trials,
        "min_matched":      min_matched,
        "agent_name":       all_trials[0]["agent_name"],
        "window":           window,

        # Raw per-trial dicts — retained for convergence analysis downstream
        "all_trials":       all_trials,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summarize_single_trial(result: dict) -> dict:
    """
    Function to extract flat summary dictionary from a single-trial result. This function is intended for
    building comparison tables in `compare.py`. 

    Returns
    -------
    * dict: dictionary with keys agent_name, final_regret, mean_reward, match_rate, matched_steps, total_steps,
            mean_arm_entropy.
    """
    return {
        "agent_name":       result["agent_name"],
        "final_regret":     round(float(result["cumulative_regret"][-1]),4),
        "mean_reward":      round(float(result["rewards"].mean()),4),
        "match_rate":       result["match_rate"],
        "matched_steps":    result["matched_steps"],
        "total_steps":      result["total_steps"],
        # Final entropy value — low means converged policy
        "mean_arm_entropy": round(float(result["arm_entropy"][-1]),4),
    }


def summarize_multi_trial(agg: dict) -> dict:
    """
    Function to extract flat summary dictionary from a muti-trial aggregated result. Intended for building
    comparison tables in `compare.py`. 

    Returns
    -------
    * dict: dictionary with agent_name, mean_final_regret, std_final_regret, mean_reward (from last
            rolling window), mean_match_rate, std_match_rate, n_trials.
    """
    return {
    "agent_name":        agg["agent_name"],
    "mean_final_regret": round(agg["mean_final_regret"], 4),
    "std_final_regret":  round(agg["std_final_regret"], 4),
    "mean_reward":       round(float(agg["mean_rolling_reward"][-1]), 4),
    "mean_match_rate":   round(agg["mean_match_rate"], 4),
    "std_match_rate":    round(agg["std_match_rate"], 4),
    "n_trials":          agg["n_trials"],  # keep as int
}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Smoke test using a small synthetic environment and LinUCB.
    Verifies that run_trial() and run_multiple_trials() produce
    correctly structured outputs without requiring the full dataset.
    """
    import sys
    sys.path.insert(0, ".")

    from algorithms.lin_ucb import LinUCB
    from bandits_environment import OfflineBanditEnv

    print("Building synthetic environment...")
    rng = np.random.default_rng(0)
    n = 2000      # total events in the synthetic stream
    n_arms = 10
    ctx_dim = 8

    arm_pool  = list(range(n_arms))

    # Build a small synthetic DataFrame that mimics data_loader output
    import pandas as pd
    synthetic_df = pd.DataFrame({
        "user_id":   rng.integers(1, 50, size=n),
        "movie_id":  rng.choice(arm_pool, size=n),
        "rating":    rng.choice([1, 2, 3, 4, 5], size=n).astype(float),
        "timestamp": np.arange(n),
        "reward":    rng.integers(0, 2, size=n).astype(float),
        "context":   [rng.random(ctx_dim) for _ in range(n)],
    })

    env = OfflineBanditEnv(synthetic_df, seed=0, arm_pool=arm_pool)
    print(f"Environment: {env}\n")

    # --- Single trial ---
    print("Running single trial with LinUCB...")
    agent  = LinUCB(n_arms=n_arms, context_dim=ctx_dim, alpha=1.0, seed=0)
    result = run_trial(agent, env, window=100, verbose=True)

    print(f"\nSingle trial results:")
    print(f"  matched_steps      : {result['matched_steps']}")
    print(f"  total_steps        : {result['total_steps']}")
    print(f"  match_rate         : {result['match_rate']:.4f}")
    print(f"  final regret       : {result['cumulative_regret'][-1]:.4f}")
    print(f"  mean reward        : {result['rewards'].mean():.4f}")
    print(f"  rewards shape      : {result['rewards'].shape}")
    print(f"  arm_entropy shape  : {result['arm_entropy'].shape}")
    print(f"  regret_per_step[-1]: {result['regret_per_step'][-1]:.4f}")

    # Shape consistency checks
    m = result["matched_steps"]
    assert result["cumulative_regret"].shape == (m,), "Regret shape mismatch"
    assert result["rolling_reward"].shape == (m,), "Rolling reward shape mismatch"
    assert result["arm_entropy"].shape == (m,), "Entropy shape mismatch"
    print("\n  Shape checks passed")

    summary = summarize_single_trial(result)
    print(f"\n  Summary: {summary}")

    # --- Multi-trial ---
    print("\nRunning 3-trial aggregation with LinUCB...")
    agg = run_multiple_trials(
        LinUCB, env, n_trials=3,
        agent_kwargs=dict(n_arms=n_arms, context_dim=ctx_dim, alpha=1.0),
        window=100, verbose=True,
    )

    print(f"\nMulti-trial results:")
    print(f"  n_trials               : {agg['n_trials']}")
    print(f"  min_matched            : {agg['min_matched']}")
    print(f"  mean_final_regret      : {agg['mean_final_regret']:.4f} "
          f"± {agg['std_final_regret']:.4f}")
    print(f"  mean_match_rate        : {agg['mean_match_rate']:.4f} "
          f"± {agg['std_match_rate']:.4f}")
    print(f"  mean_rolling_reward[-1]: {agg['mean_rolling_reward'][-1]:.4f}")

    # Check all aggregated arrays have consistent length
    L = agg["min_matched"]
    assert agg["mean_cumulative_regret"].shape == (L,), "Agg regret shape mismatch"
    assert agg["mean_rolling_reward"].shape == (L,), "Agg reward shape mismatch"
    assert agg["mean_arm_entropy"].shape == (L,), "Agg entropy shape mismatch"
    print("\n  Aggregation shape checks passed")

    multi_summary = summarize_multi_trial(agg)
    print(f"\n  Multi-trial summary: {multi_summary}")

    import matplotlib.pyplot as plt
    os.makedirs("figures/smoke_tests", exist_ok=True)
    plt.figure()
    plt.plot(result["cumulative_regret"])
    plt.title("LinUCB cumulative regret (smoke test)")
    plt.xlabel("Matched events")
    plt.ylabel("Cumulative regret")
    plt.savefig("figures/smoke_tests/evaluate_regret.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: figures/smoke_tests/evaluate_regret.png")

    print("\nSmoke test passed.")