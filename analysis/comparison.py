"""
analysis/compare.py

==========

Cross-algorithm comparison orchestration.



This script does the final bit of hte evaluation and analysis pipeline. It instantiates all algorithms
from a configuration, runs the algorithms through identical interaction sequences via `run_multiple_trials()`,
extracts convergence statisics from the `convergence.py` script, and assesmbles everything into a structured
results dictionary and a printable summary table. 

The key requirement for this comparison script is that every algorithm see the same logged interaction
sequence, in the same order. This is enforced by resetting the environment after the updates to the same
state before each algorithm's trials. 


Typical usage
-------------
    python compare.py

Or import and call programmatically:

    from compare import run_comparison, print_summary_table
    results = run_comparison(train_df, context_dim, config=COMPARISON_CONFIG)
    print_summary_table(results)

The results dictionary produced here is passed directly to visualize.py for plotting.
"""

# Import dependencies
from typing import Any
import pandas as pd
import numpy as np
import gc
import torch

from .convergence import full_convergence_report_multi
from .evaluation import run_multiple_trials, summarize_multi_trial
from bandits_environment import OfflineBanditEnv



# ---------------------------------------------------------------------------
# Default comparison configuration
# ---------------------------------------------------------------------------

# Each entry is one algorithm to include in the comparison, where 'class' is algorithm class, 'kwargs'
# are its constructor arguments (excluding 'seed', 'n_arms', set automatically at runtime).
# Add or remove entries here to change which algorithms are compared.
def build_default_config(n_arms: int, context_dim: int) -> list[dict]:
    """
    Function to build the default list of algorithm configurations for comparison. Each entry is a dictionary
    with:
        'name'   : human-readable label used in plots and tables
        'class'  : the BaseBandit subclass to instantiate
        'kwargs' : constructor arguments (n_arms and seed are added automatically; do not include them here)

    Parameters
    ----------
    * n_arms: number of arms in the environment
    * context_dim: dimensionality of the context vectors

    Returns
    -------
    * list: list of algorithm config dicts
    """
    # Import here to avoid circular imports and keep the config self-contained
    from algorithms.epsilon_greedy import EpsilonGreedy
    from algorithms.lin_ucb import LinUCB
    from algorithms.thompson_sampling import ThompsonSampling
    from algorithms.logistic_ucb import LogisticUCB
    from algorithms.bootstrap_thompson import BootstrapThompson

    configs = [
        {
            "name":   "EpsilonGreedy",
            "class":  EpsilonGreedy,
            "kwargs": {
                "n_arms":      n_arms,
                "context_dim": context_dim,
                "epsilon":     0.1,
                "schedule":    "fixed",
                "lambda_reg":  1.0,
            },
        },
        {
            "name":   "LinUCB",
            "class":  LinUCB,
            "kwargs": {
                "n_arms":      n_arms,
                "context_dim": context_dim,
                "alpha":       1.0,
                "lambda_reg":  1.0,
            },
        },
        {
            "name":   "ThompsonSampling",
            "class":  ThompsonSampling,
            "kwargs": {
                "n_arms":      n_arms,
                "context_dim": context_dim,
                "sigma":       1.0,
                "lambda_reg":  1.0,
            },
        },
        {
            "name":   "LogisticUCB",
            "class":  LogisticUCB,
            "kwargs": {
                "n_arms":      n_arms,
                "context_dim": context_dim,
                "alpha":       1.0,
                "lambda_reg":  1.0,
            },
        },
        {
            "name":   "BootstrapThompson",
            "class":  BootstrapThompson,
            "kwargs": {
                "n_arms":       n_arms,
                "context_dim":  context_dim,
                "n_particles":  10,
                "lambda_reg":   1.0,
            },
        },
    ]

    # NeuralUCB and NeuralTS are included only if PyTorch is available
    try:
        from algorithms.neural_ucb import NeuralUCB
        from algorithms.neural_ts  import NeuralTS
        configs += [
            {
                "name":   "NeuralUCB",
                "class":  NeuralUCB,
                "kwargs": {
                    "n_arms":        n_arms,
                    "context_dim":   context_dim,
                    "d_emb":         32,
                    "hidden_dim":    64,
                    "alpha":         1.0,
                    "lambda_reg":    1.0,
                    "warmup_steps":  200,
                    "train_every":   100,
                },
            },
            {
                "name":   "NeuralTS",
                "class":  NeuralTS,
                "kwargs": {
                    "n_arms":        n_arms,
                    "context_dim":   context_dim,
                    "d_emb":         32,
                    "hidden_dim":    64,
                    "sigma":         1.0,
                    "lambda_reg":    1.0,
                    "warmup_steps":  200,
                    "train_every":   100,
                },
            },
        ]
    except ImportError:
        print("  [compare] PyTorch not available; NeuralUCB and NeuralTS excluded.")

    return configs


# ---------------------------------------------------------------------------
# Core comparison runner
# ---------------------------------------------------------------------------
def run_comparison(df: pd.DataFrame, context_dim: int, config: list[dict] = None, n_trials: int = 10,
    window: int = 500, env_seed: int = 42, candidate_size: int = None, convergence_kwargs: dict = None, verbose: bool = True,) -> dict:
    """
    Function to run all algorithms in the config through identical interaction sequences and to collect
    evaluation and convergence results for each algorithm. Every algorithm is run for n_trials independent
    trials. The environment is reset to the same state before every trial of every algorithm, ensuring all
    algorithms see the same logged events, importantly, in the same order. 

    Parameters
    ----------
    * df: interaction DataFrame from data_loader
    * context_dim: context vector dimensionality
    * config: algorithm configurations; if None, uses build_default_config()
    * n_trials: number of trials per algorithm
    * window: rolling window for reward smoothing / entropy
    * env_seed: seed for the OfflineBanditEnv
    * convergence_kwargs : extra kwargs forwarded to full_convergence_report_multi(); e.g. {'tolerance': 0.02,
                           'sustained_window': 300}
    * verbose: print progress per algorithm

    Returns
    -------
    * dict: dictionary with keys
        'algorithm_names': list[str]
        'agg_results': dict mapping algorithm name -> multi-trial agg dict
        'conv_reports': dict mapping algorithm name -> convergence report dict
        'summaries': dict mapping algorithm name -> flat summary dict
        'config': the config list used
        'n_trials': int
        'context_dim': int
    """
    if convergence_kwargs is None:
        convergence_kwargs = {}

    # Build the environment once; it will be reset between algorithms
    env = OfflineBanditEnv(df, seed=env_seed, candidate_size=candidate_size)

    if config is None:
        config = build_default_config(n_arms=env.n_arms, context_dim=context_dim)

    algorithm_names = [cfg["name"] for cfg in config]
    agg_results = {}
    conv_reports = {}
    summaries = {}

    for cfg in config:
        name  = cfg["name"]
        cls   = cfg["class"]
        # Pass the algorithm's constructor kwargs; seed is added per trial
        # inside run_multiple_trials() automatically
        kwargs = cfg["kwargs"]

        if verbose: print(f"\nRunning {name} ({n_trials} trials)...")

        # Run all trials for this algorithm
        agg = run_multiple_trials(
            agent_class = cls,
            env = env,
            n_trials = n_trials,
            # Exclude 'n_arms' from kwargs since it is already set in env, passed in as part of kwargs for clarity but it must match env.n_arms
            agent_kwargs = kwargs,
            window = window,
            verbose = verbose,
        )
        agg_results[name] = agg
        
        # Ensure no memory leaks/issues for neural algorithms
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute convergence report for this algorithm
        conv = full_convergence_report_multi(agg, **convergence_kwargs)
        conv_reports[name] = conv

        # Build flat summary row for the comparison table
        summaries[name] = _build_summary_row(agg, conv)

        if verbose:
            print(f"  {name}: final regret = "
                  f"{agg['mean_final_regret']:.4f} ± {agg['std_final_regret']:.4f} | "
                  f"match rate = {agg['mean_match_rate']:.4f} | "
                  f"t* = {conv['mean_convergence_step']:.0f}")

    return {
        "algorithm_names": algorithm_names,
        "agg_results":     agg_results,
        "conv_reports":    conv_reports,
        "summaries":       summaries,
        "config":          config,
        "n_trials":        n_trials,
        "context_dim":     context_dim,
    }


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------
def _build_summary_row(agg: dict, conv: dict) -> dict:
    """
    Function to merge the multi-trial evaluation summary and convergence report into a single flat dictionary,
    used as one row in the comparison table.

    Parameters
    ----------
    * agg: output of run_multiple_trials()
    * conv: output of full_convergence_report_multi()

    Returns
    -------
    * dict: dictionary with flat row suitable for pd.DataFrame construction
    """
    return {
        "Algorithm":         agg["agent_name"],
        # Primary evaluation metrics
        "Final Regret":      f"{agg['mean_final_regret']:.4f} ± {agg['std_final_regret']:.4f}",
        "Reward Rate":       f"{agg['mean_rolling_reward'][-1]:.4f}",
        "Match Rate":        f"{agg['mean_match_rate']:.4f} ± {agg['std_match_rate']:.4f}",
        # Convergence metrics
        "Conv. Step t*":     f"{conv['mean_convergence_step']:.0f} ± {conv['std_convergence_step']:.0f}",
        "Slope Ratio":       f"{conv['mean_slope_ratio']:.4f} ± {conv['std_slope_ratio']:.4f}",
        "Entropy Drop":      f"{conv['mean_entropy_drop']:.4f} ± {conv['std_entropy_drop']:.4f}",
        # Raw floats retained for sorting / plotting
        "_final_regret":     round(agg["mean_final_regret"],4),
        "_reward_rate":      round(float(agg["mean_rolling_reward"][-1]),4),
        "_conv_step":        conv["mean_convergence_step"],
        "_slope_ratio":      round(conv["mean_slope_ratio"],4),
        "_entropy_drop":     round(conv["mean_entropy_drop"],4),
    }


def print_summary_table(results: dict) -> None:
    """
    Function to print a formatted comparison table to stdout. The columns reported include: Algorithm,
    Final Regret, Reward Rate, Match Rate, Conv. Step t*, Slope Ratio, Entropy Drop. The algorithms are
    sorted by ascneding final regret (i.e., best one first).

    Parameters
    ----------
    * results: dictionary with output of run_comparison()
    """
    rows = list(results["summaries"].values())

    # Sort by final regret (ascending — lower is better)
    rows_sorted = sorted(rows, key=lambda r: r["_final_regret"])

    # Display columns only (strip private _keys)
    display_cols = [
        "Algorithm", "Final Regret", "Reward Rate",
        "Match Rate", "Conv. Step t*", "Slope Ratio", "Entropy Drop",
    ]

    df_table = pd.DataFrame(rows_sorted)[display_cols]

    # Print with a clear header
    sep = "-" * 100
    print(f"\n{'COMPARISON SUMMARY':^100}")
    print(sep)
    print(df_table.to_string(index=False))
    print(sep)
    print(f"n_trials = {results['n_trials']} | context_dim = {results['context_dim']}")
    print(f"Sorted by ascending final cumulative regret (lower = better).\n")


def export_summary_csv(results: dict, path: str) -> None:
    """
    Function to export the comparison summary table to a csv file.

    Parameters
    ----------
    * results: output of run_comparison()
    * path: output file path (e.g. 'results/summary.csv')
    """
    rows = list(results["summaries"].values())
    display_cols = [
        "Algorithm", "Final Regret", "Reward Rate",
        "Match Rate", "Conv. Step t*", "Slope Ratio", "Entropy Drop",
    ]
    df_table = pd.DataFrame(rows)[display_cols]
    df_table.to_csv(path, index=False)
    print(f"Summary table exported to: {path}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Building synthetic data for compare.py smoke test...\n")

    rng = np.random.default_rng(0)
    n = 3000
    n_arms = 8
    ctx_dim = 10
    arm_pool = list(range(n_arms))

    # Arm 0 has higher reward to give algorithms something to learn
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

    # Build a small config with just two fast algorithms for the smoke test
    from algorithms.lin_ucb import LinUCB
    from algorithms.thompson_sampling import ThompsonSampling

    test_config = [
        {
            "name": "LinUCB",
            "class": LinUCB,
            "kwargs": {"n_arms": n_arms, "context_dim": ctx_dim, "alpha": 1.0},
        },
        {
            "name": "ThompsonSampling",
            "class": ThompsonSampling,
            "kwargs": {"n_arms": n_arms, "context_dim": ctx_dim, "sigma": 1.0},
        },
    ]

    results = run_comparison(
        df = synthetic_df,
        context_dim = ctx_dim,
        config = test_config,
        n_trials = 3,
        window = 100,
        convergence_kwargs = {"tolerance": 0.02, "sustained_window": 50, "min_step": 30},
        verbose = True,
    )

    print_summary_table(results)

    # Structure checks
    assert set(results["algorithm_names"]) == {"LinUCB", "ThompsonSampling"}
    assert "LinUCB" in results["agg_results"]
    assert "ThompsonSampling" in results["conv_reports"]
    assert results["n_trials"] == 3

    # Check that summary rows contain all expected keys
    for name, row in results["summaries"].items():
        assert "Final Regret" in row, f"Missing 'Final Regret' in {name} summary"
        assert "_final_regret" in row, f"Missing '_final_regret' in {name} summary"

    print("Structure checks passed")
    print("\nSmoke test passed.")