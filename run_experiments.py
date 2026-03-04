"""
run_experiments.py
=================
Main execution script for the contextual bandit recommendation experiment.

This script orchestrates the entire pipeline in five sequential phases:

    Phase 1 — Data Loading
        Load the MovieLens 10M dataset, engineer context features,
        binarize rewards, and split into train / validation / test sets.

    Phase 2 — Hyperparameter Selection (Sensitivity on Validation)
        Sweep each algorithm's key hyperparameters on the validation split.
        Select the best value per parameter and save sweep results to disk.
        These results are reused for sensitivity figures later.

    Phase 3 — Main Comparison (on Test)
        Run all algorithms with tuned hyperparameters on the test split
        for n_trials independent trials each. Collect evaluation and
        convergence statistics.

    Phase 4 — Visualization
        Generate all figures (primary comparison, convergence, exploration,
        robustness, hyperparameter sensitivity) and save to figures/.

    Phase 5 — Summary
        Print the final comparison table and export it as a CSV.

Usage
-----
    # Full pipeline with default settings
    python run_experiment.py

    # Quick debug run on a subset of the data, fewer trials
    python run_experiment.py --debug

    # Skip the hyperparameter sweep (use defaults) and go straight to comparison
    python run_experiment.py --skip-sweep

    # Run only specific algorithms
    python run_experiment.py --algorithms LinUCB ThompsonSampling EpsilonGreedy

Command-line arguments are defined at the bottom of this file.

Output structure
----------------
    figures/
        primary/          — Group 1 comparison plots
        convergence/      — Group 2 convergence plots
        exploration/      — Group 3 exploration vs exploitation plots
        robustness/       — Group 4 subgroup and robustness plots
        sensitivity/      — Group 5 hyperparameter sensitivity plots
    results/
        sweep_results.pkl — saved sensitivity sweep results (validation)
        comparison_results.pkl — saved comparison results (test)
        summary.csv       — final comparison table
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure subfolders are importable
# ---------------------------------------------------------------------------
# Project root is the directory containing this script.
# Subfolders 'algorithms/' and 'analysis/' are made importable here
# so that all imports in the pipeline scripts resolve correctly.

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "algorithms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------

from data_loading import load_and_prepare
from bandits_environment  import OfflineBanditEnv

from algorithms.epsilon_greedy     import EpsilonGreedy
from algorithms.lin_ucb            import LinUCB
from algorithms.thompson_sampling  import ThompsonSampling
from algorithms.logistic_ucb       import LogisticUCB
from algorithms.bootstrap_thompson import BootstrapThompson

from analysis.sensitivity import (
    sweep_linucb,
    sweep_thompson_sampling,
    sweep_epsilon_greedy,
    sweep_logistic_ucb,
    sweep_bootstrap_thompson,
    sweep_reward_threshold,
    sweep_context_dimensionality,
    select_best_hyperparameters,
    print_sweep_summary,
)
from analysis.comparison     import run_comparison, print_summary_table, export_summary_csv
from analysis.convergence import full_convergence_report_multi
from visualizations import *

# NeuralUCB and NeuralTS are optional (require PyTorch)
try:
    from algorithms.neural_ucb import NeuralUCB
    from algorithms.neural_ts  import NeuralTS
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False
    print("[Warning] PyTorch not available — NeuralUCB and NeuralTS will be excluded.")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt


# ===========================================================================
# Configuration
# ===========================================================================

# --- Data paths ---
DATA_CONFIG = {
    "ratings_path": "data/ml-10M100K/ratings.dat",
    "movies_path":  "data/ml-10M100K/movies.dat",
}

# --- Data split fractions ---
# The dataset is split temporally: train → val → test.
# train is used to compute user/item statistics (context features).
# val   is used exclusively for hyperparameter selection.
# test  is used for the main algorithm comparison.
SPLIT_CONFIG = {
    "val_fraction":  0.10,   # 10% of events for hyperparameter selection
    "test_fraction": 0.20,   # 20% of events for the main comparison
    "reward_threshold": 4.0, # rating >= threshold → reward = 1
    "context_method":   "raw",
    "min_user_ratings": 20,  # filter users with fewer interactions
}

# --- Evaluation settings ---
EVAL_CONFIG = {
    "n_trials":   5,     # independent trials per algorithm
    "window":     500,   # rolling window for reward smoothing
    "env_seed":   42,    # environment random seed
}

# --- Sensitivity sweep settings ---
SWEEP_CONFIG = {
    "n_trials":      3,    # trials per hyperparameter value (fewer than main)
    "window":        500,
    # Hyperparameter grids
    "alpha_values":  [0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    "sigma_values":  [0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    "lambda_values": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
    "epsilon_values":[0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    "particle_values":[3, 5, 10, 20, 30],
    # Context dimensionality sweep
    "pca_dims":      [2, 5, 10, 15, 20],
    # Reward threshold sweep
    "thresholds":    [3.0, 3.5, 4.0, 4.5],
}

# --- Convergence analysis settings ---
CONVERGENCE_CONFIG = {
    "tolerance":        0.01,
    "sustained_window": 300,
    "min_step":         100,
    "early_fraction":   0.25,
    "late_fraction":    0.25,
}

# --- Output directories ---
DIRS = {
    "figures":     "figures",
    "primary":     "figures/primary",
    "convergence": "figures/convergence",
    "exploration": "figures/exploration",
    "robustness":  "figures/robustness",
    "sensitivity": "figures/sensitivity",
    "results":     "results",
}


def _make_dirs():
    """Create all output directories if they do not already exist."""
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)


def _save(obj: object, filename: str) -> None:
    """Pickle-save an object to the results directory."""
    path = os.path.join(DIRS["results"], filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path}")


def _load(filename: str) -> object:
    """Load a pickled object from the results directory."""
    path = os.path.join(DIRS["results"], filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def _section(title: str) -> None:
    """Print a clearly visible section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ===========================================================================
# Phase 1: Data Loading
# ===========================================================================

def phase1_load_data(args):
    """
    Load the MovieLens 10M dataset and produce three temporally ordered
    splits: train, validation, and test.

    The train split is used to compute aggregate user/item statistics that
    form the context features — it is never passed to the bandit environment
    directly. The validation split is used for hyperparameter selection and
    the test split is used for the main algorithm comparison.

    Returns
    -------
    train_df   : pd.DataFrame — training interactions with context and reward
    val_df     : pd.DataFrame — validation interactions
    test_df    : pd.DataFrame — test interactions
    context_dim: int          — dimensionality of the context vectors
    base_rate  : float        — fraction of rewards = 1 in the full dataset
    """
    _section("Phase 1: Data Loading")

    # In debug mode, print a reminder that we are using the full dataset
    # (subsampling is handled at the environment level via subsample_env)
    if args.debug:
        print("[Debug mode] Pipeline will run on a subsample of the data.\n")

    print(f"Loading ratings from : {DATA_CONFIG['ratings_path']}")
    print(f"Loading movies from  : {DATA_CONFIG['movies_path']}\n")

    # load_and_prepare() handles: loading raw files → merging → feature
    # engineering → binarizing rewards → building context vectors →
    # temporal train/test split (here we use it to get train + held-out)
    train_df, held_out_df, context_dim = load_and_prepare(
        ratings_path    = DATA_CONFIG["ratings_path"],
        movies_path     = DATA_CONFIG["movies_path"],
        context_method  = SPLIT_CONFIG["context_method"],
        reward_threshold= SPLIT_CONFIG["reward_threshold"],
        # test_fraction here gives us the combined val + test held-out portion
        test_fraction   = SPLIT_CONFIG["val_fraction"] + SPLIT_CONFIG["test_fraction"],
        verbose         = True,
    )

    # Further split the held-out portion into validation and test.
    # val_fraction as a fraction of the held-out block:
    val_frac = SPLIT_CONFIG["val_fraction"] / (
        SPLIT_CONFIG["val_fraction"] + SPLIT_CONFIG["test_fraction"]
    )
    split_idx  = int(len(held_out_df) * val_frac)
    val_df     = held_out_df.iloc[:split_idx].reset_index(drop=True)
    test_df    = held_out_df.iloc[split_idx:].reset_index(drop=True)

    base_rate = float(train_df["reward"].mean())

    print(f"Train size     : {len(train_df):,}")
    print(f"Validation size: {len(val_df):,}")
    print(f"Test size      : {len(test_df):,}")
    print(f"Context dim    : {context_dim}")
    print(f"Base reward rate (train): {base_rate:.3f}")

    return train_df, val_df, test_df, context_dim, base_rate


# ===========================================================================
# Phase 2: Hyperparameter Selection (Sensitivity on Validation)
# ===========================================================================

def phase2_hyperparameter_selection(
    val_df,
    n_arms: int,
    context_dim: int,
    args,
) -> tuple[dict, dict]:
    """
    Sweep each algorithm's key hyperparameters on the validation split and
    select the best value per parameter.

    Sweep results are saved to disk so they can be reloaded if the script
    is interrupted and restarted, and reused for sensitivity figures later.

    Parameters
    ----------
    val_df      : pd.DataFrame — validation split
    n_arms      : int — number of arms in the arm pool
    context_dim : int — context vector dimensionality
    args        : argparse.Namespace — command-line arguments

    Returns
    -------
    best_params  : dict — {algorithm_name: {param_name: best_value}}
    sweep_results: dict — raw sweep result dicts for all algorithms
    """
    _section("Phase 2: Hyperparameter Selection (Validation Sweep)")

    # If skip-sweep is requested, return placeholder defaults
    if args.skip_sweep:
        print("--skip-sweep flag set. Using default hyperparameters.\n")
        best_params = {
            "LinUCB":            {"alpha": 1.0,  "lambda_reg": 1.0},
            "ThompsonSampling":  {"sigma": 1.0,  "lambda_reg": 1.0},
            "EpsilonGreedy":     {"epsilon": 0.1, "schedule": "fixed"},
            "LogisticUCB":       {"alpha": 1.0,  "lambda_reg": 1.0},
            "BootstrapThompson": {"n_particles": 10, "lambda_reg": 1.0},
            "NeuralUCB":         {"alpha": 1.0,  "lambda_reg": 1.0},
            "NeuralTS":          {"sigma": 1.0,  "lambda_reg": 1.0},
        }
        return best_params, {}

    # Check whether sweep results were already saved from a previous run
    sweep_cache = os.path.join(DIRS["results"], "sweep_results.pkl")
    if os.path.exists(sweep_cache) and not args.force_sweep:
        print(f"Loading cached sweep results from: {sweep_cache}")
        sweep_results = _load("sweep_results.pkl")
        best_params   = select_best_hyperparameters(sweep_results)
        print_sweep_summary(sweep_results)
        return best_params, sweep_results

    n_sw     = SWEEP_CONFIG["n_trials"]
    sweep_results = {}

    # Determine which algorithms to sweep based on --algorithms flag
    algo_filter = set(args.algorithms) if args.algorithms else None

    # --- LinUCB ---
    if algo_filter is None or "LinUCB" in algo_filter:
        print("Sweeping LinUCB...")
        t0 = time.time()
        sweep_results["LinUCB"] = sweep_linucb(
            df            = val_df,
            n_arms        = n_arms,
            context_dim   = context_dim,
            alpha_values  = SWEEP_CONFIG["alpha_values"],
            lambda_values = SWEEP_CONFIG["lambda_values"],
            n_trials      = n_sw,
            verbose       = True,
        )
        print(f"  Done in {time.time()-t0:.1f}s")

    # --- ThompsonSampling ---
    if algo_filter is None or "ThompsonSampling" in algo_filter:
        print("Sweeping ThompsonSampling...")
        t0 = time.time()
        sweep_results["ThompsonSampling"] = sweep_thompson_sampling(
            df            = val_df,
            n_arms        = n_arms,
            context_dim   = context_dim,
            sigma_values  = SWEEP_CONFIG["sigma_values"],
            lambda_values = SWEEP_CONFIG["lambda_values"],
            n_trials      = n_sw,
            verbose       = True,
        )
        print(f"  Done in {time.time()-t0:.1f}s")

    # --- EpsilonGreedy ---
    if algo_filter is None or "EpsilonGreedy" in algo_filter:
        print("Sweeping EpsilonGreedy...")
        t0 = time.time()
        sweep_results["EpsilonGreedy"] = sweep_epsilon_greedy(
            df             = val_df,
            n_arms         = n_arms,
            context_dim    = context_dim,
            epsilon_values = SWEEP_CONFIG["epsilon_values"],
            n_trials       = n_sw,
            verbose        = True,
        )
        print(f"  Done in {time.time()-t0:.1f}s")

    # --- LogisticUCB ---
    if algo_filter is None or "LogisticUCB" in algo_filter:
        print("Sweeping LogisticUCB...")
        t0 = time.time()
        sweep_results["LogisticUCB"] = sweep_logistic_ucb(
            df            = val_df,
            n_arms        = n_arms,
            context_dim   = context_dim,
            alpha_values  = SWEEP_CONFIG["alpha_values"],
            lambda_values = SWEEP_CONFIG["lambda_values"],
            n_trials      = n_sw,
            verbose       = True,
        )
        print(f"  Done in {time.time()-t0:.1f}s")

    # --- BootstrapThompson ---
    if algo_filter is None or "BootstrapThompson" in algo_filter:
        print("Sweeping BootstrapThompson...")
        t0 = time.time()
        sweep_results["BootstrapThompson"] = sweep_bootstrap_thompson(
            df              = val_df,
            n_arms          = n_arms,
            context_dim     = context_dim,
            particle_values = SWEEP_CONFIG["particle_values"],
            lambda_values   = SWEEP_CONFIG["lambda_values"],
            n_trials        = n_sw,
            verbose         = True,
        )
        print(f"  Done in {time.time()-t0:.1f}s")

    # Neural algorithms use the same grids as their linear counterparts;
    # we do not sweep their network architecture hyperparameters here
    # (d_emb, hidden_dim) as those are fixed by the Neural-Linear design.
    if _NEURAL_AVAILABLE:
        if algo_filter is None or "NeuralUCB" in algo_filter:
            print("Sweeping NeuralUCB (alpha only — architecture fixed)...")
            from analysis.sensitivity import sweep_hyperparameter
            t0 = time.time()
            sweep_results["NeuralUCB"] = {
                "alpha": sweep_hyperparameter(
                    NeuralUCB, val_df, "alpha",
                    SWEEP_CONFIG["alpha_values"],
                    fixed_kwargs={
                        "n_arms": n_arms, "context_dim": context_dim,
                        "lambda_reg": 1.0, "d_emb": 32, "hidden_dim": 64,
                        "warmup_steps": 200, "train_every": 100,
                    },
                    n_trials=n_sw, verbose=True,
                )
            }
            print(f"  Done in {time.time()-t0:.1f}s")

        if algo_filter is None or "NeuralTS" in algo_filter:
            print("Sweeping NeuralTS (sigma only — architecture fixed)...")
            from analysis.sensitivity import sweep_hyperparameter
            t0 = time.time()
            sweep_results["NeuralTS"] = {
                "sigma": sweep_hyperparameter(
                    NeuralTS, val_df, "sigma",
                    SWEEP_CONFIG["sigma_values"],
                    fixed_kwargs={
                        "n_arms": n_arms, "context_dim": context_dim,
                        "lambda_reg": 1.0, "d_emb": 32, "hidden_dim": 64,
                        "warmup_steps": 200, "train_every": 100,
                    },
                    n_trials=n_sw, verbose=True,
                )
            }
            print(f"  Done in {time.time()-t0:.1f}s")

    # Save sweep results to disk before proceeding
    _save(sweep_results, "sweep_results.pkl")

    # Extract best hyperparameter values from sweep results
    best_params = select_best_hyperparameters(sweep_results)
    print_sweep_summary(sweep_results)

    print("\nSelected hyperparameters:")
    for algo, params in best_params.items():
        print(f"  {algo:<25}: {params}")

    return best_params, sweep_results


# ===========================================================================
# Phase 3: Main Comparison (on Test)
# ===========================================================================

def phase3_main_comparison(
    test_df,
    n_arms: int,
    context_dim: int,
    best_params: dict,
    args,
) -> dict:
    """
    Run all algorithms with tuned hyperparameters on the test split for
    n_trials independent trials each.

    Builds the algorithm config by merging the default architecture settings
    with the best hyperparameters selected in Phase 2, then calls
    compare.run_comparison() which handles multi-trial evaluation and
    convergence analysis internally.

    Parameters
    ----------
    test_df     : pd.DataFrame — test split
    n_arms      : int
    context_dim : int
    best_params : dict — output of select_best_hyperparameters()
    args        : argparse.Namespace

    Returns
    -------
    results : dict — output of compare.run_comparison()
    """
    _section("Phase 3: Main Algorithm Comparison (Test Split)")

    # Check for cached comparison results
    comp_cache = os.path.join(DIRS["results"], "comparison_results.pkl")
    if os.path.exists(comp_cache) and not args.force_comparison:
        print(f"Loading cached comparison results from: {comp_cache}")
        return _load("comparison_results.pkl")

    # --- Build algorithm config with tuned hyperparameters ---
    # Each entry: name, class, kwargs (n_arms and seed set automatically)
    def _p(algo, param, default):
        """Helper: get best param value or fall back to default."""
        return best_params.get(algo, {}).get(param, default)

    config = []

    # Determine which algorithms to include
    algo_filter = set(args.algorithms) if args.algorithms else None

    if algo_filter is None or "EpsilonGreedy" in algo_filter:
        config.append({
            "name":   "EpsilonGreedy",
            "class":  EpsilonGreedy,
            "kwargs": {
                "n_arms":      n_arms,
                "context_dim": context_dim,
                "epsilon":     _p("EpsilonGreedy", "epsilon",  0.1),
                "schedule":    _p("EpsilonGreedy", "schedule", "fixed"),
                "lambda_reg":  _p("EpsilonGreedy", "lambda_reg", 1.0),
            },
        })

    if algo_filter is None or "LinUCB" in algo_filter:
        config.append({
            "name":   "LinUCB",
            "class":  LinUCB,
            "kwargs": {
                "n_arms":      n_arms,
                "context_dim": context_dim,
                "alpha":       _p("LinUCB", "alpha",      1.0),
                "lambda_reg":  _p("LinUCB", "lambda_reg", 1.0),
            },
        })

    if algo_filter is None or "ThompsonSampling" in algo_filter:
        config.append({
            "name":   "ThompsonSampling",
            "class":  ThompsonSampling,
            "kwargs": {
                "n_arms":      n_arms,
                "context_dim": context_dim,
                "sigma":       _p("ThompsonSampling", "sigma",      1.0),
                "lambda_reg":  _p("ThompsonSampling", "lambda_reg", 1.0),
            },
        })

    if algo_filter is None or "LogisticUCB" in algo_filter:
        config.append({
            "name":   "LogisticUCB",
            "class":  LogisticUCB,
            "kwargs": {
                "n_arms":      n_arms,
                "context_dim": context_dim,
                "alpha":       _p("LogisticUCB", "alpha",      1.0),
                "lambda_reg":  _p("LogisticUCB", "lambda_reg", 1.0),
            },
        })

    if algo_filter is None or "BootstrapThompson" in algo_filter:
        config.append({
            "name":   "BootstrapThompson",
            "class":  BootstrapThompson,
            "kwargs": {
                "n_arms":       n_arms,
                "context_dim":  context_dim,
                "n_particles":  _p("BootstrapThompson", "n_particles", 10),
                "lambda_reg":   _p("BootstrapThompson", "lambda_reg",  1.0),
            },
        })

    if _NEURAL_AVAILABLE:
        if algo_filter is None or "NeuralUCB" in algo_filter:
            config.append({
                "name":   "NeuralUCB",
                "class":  NeuralUCB,
                "kwargs": {
                    "n_arms":        n_arms,
                    "context_dim":   context_dim,
                    "alpha":         _p("NeuralUCB", "alpha",      1.0),
                    "lambda_reg":    _p("NeuralUCB", "lambda_reg", 1.0),
                    "d_emb":         32,
                    "hidden_dim":    64,
                    "warmup_steps":  200,
                    "train_every":   100,
                },
            })

        if algo_filter is None or "NeuralTS" in algo_filter:
            config.append({
                "name":   "NeuralTS",
                "class":  NeuralTS,
                "kwargs": {
                    "n_arms":        n_arms,
                    "context_dim":   context_dim,
                    "sigma":         _p("NeuralTS", "sigma",      1.0),
                    "lambda_reg":    _p("NeuralTS", "lambda_reg", 1.0),
                    "d_emb":         32,
                    "hidden_dim":    64,
                    "warmup_steps":  200,
                    "train_every":   100,
                },
            })

    print(f"Running comparison with {len(config)} algorithms, "
          f"{EVAL_CONFIG['n_trials']} trials each...\n")

    t0 = time.time()
    results = run_comparison(
        df                 = test_df,
        context_dim        = context_dim,
        config             = config,
        n_trials           = EVAL_CONFIG["n_trials"],
        window             = EVAL_CONFIG["window"],
        env_seed           = EVAL_CONFIG["env_seed"],
        convergence_kwargs = CONVERGENCE_CONFIG,
        verbose            = True,
    )
    print(f"\nComparison completed in {time.time()-t0:.1f}s")

    # Save comparison results to disk
    _save(results, "comparison_results.pkl")

    return results


# ===========================================================================
# Phase 4: Visualization
# ===========================================================================

def phase4_visualization(
    results: dict,
    sweep_results: dict,
    test_df,
    n_arms: int,
    context_dim: int,
    base_rate: float,
    args,
) -> None:
    """
    Generate and save all figures.

    Figures are organized into five subdirectories matching the five
    plot groups defined in visualize.py.

    Parameters
    ----------
    results      : dict — output of compare.run_comparison()
    sweep_results: dict — output of the validation hyperparameter sweeps
    test_df      : pd.DataFrame — test split (used for robustness sweeps)
    n_arms       : int
    context_dim  : int
    base_rate    : float — dataset reward rate for reference lines
    args         : argparse.Namespace
    """
    _section("Phase 4: Visualization")
    set_style()

    # --- Group 1: Primary comparison plots ---
    print("Generating Group 1: Primary comparison plots...")

    fig = plot_cumulative_regret(results, log_x=True)
    fig.savefig(f"{DIRS['primary']}/cumulative_regret.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_regret_rate(results, log_x=True)
    fig.savefig(f"{DIRS['primary']}/regret_rate.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_rolling_reward(results, base_rate=base_rate, log_x=True)
    fig.savefig(f"{DIRS['primary']}/rolling_reward.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  Saved: primary/*.png")

    # --- Group 2: Convergence plots ---
    print("Generating Group 2: Convergence plots...")

    fig = plot_arm_entropy(results, log_x=True)
    fig.savefig(f"{DIRS['convergence']}/arm_entropy.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_entropy_reward_joint(results)
    fig.savefig(f"{DIRS['convergence']}/entropy_reward_joint.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_convergence_steps(results)
    fig.savefig(f"{DIRS['convergence']}/convergence_steps.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_regret_slope_ratio(results)
    fig.savefig(f"{DIRS['convergence']}/regret_slope_ratio.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  Saved: convergence/*.png")

    # --- Group 3: Exploration vs. exploitation plots ---
    print("Generating Group 3: Exploration vs. exploitation plots...")

    fig = plot_arm_pull_heatmap(results, top_k=20)
    fig.savefig(f"{DIRS['exploration']}/arm_pull_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_exploration_bonus_decay(results)
    fig.savefig(f"{DIRS['exploration']}/exploration_bonus_decay.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  Saved: exploration/*.png")

    # --- Group 4: Robustness plots ---
    # These require additional sweeps on the test split.
    # Skip in debug mode to save time.
    if not args.debug:
        print("Generating Group 4: Robustness plots...")

        # Reward threshold sensitivity
        # Build a minimal config with the two core algorithms for robustness plots
        robustness_configs = [
            {"name": "LinUCB", "class": LinUCB,
             "kwargs": {"n_arms": n_arms, "context_dim": context_dim,
                        "alpha": 1.0, "lambda_reg": 1.0}},
            {"name": "ThompsonSampling", "class": ThompsonSampling,
             "kwargs": {"n_arms": n_arms, "context_dim": context_dim,
                        "sigma": 1.0, "lambda_reg": 1.0}},
        ]

        # We need the raw merged df (before binarization) for threshold sweep;
        # approximate here by passing test_df with its existing reward column
        # and relying on sweep_reward_threshold's internal re-binarization.
        threshold_results = sweep_reward_threshold(
            df_raw       = test_df,
            n_arms       = n_arms,
            context_dim  = context_dim,
            thresholds   = SWEEP_CONFIG["thresholds"],
            agent_configs= robustness_configs,
            n_trials     = 2,   # fewer trials for robustness sweeps
            verbose      = True,
        )
        fig = plot_reward_threshold_sensitivity(threshold_results)
        fig.savefig(f"{DIRS['robustness']}/reward_threshold.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Context dimensionality sensitivity
        # Only run if context_dim > 5 (otherwise PCA sweep is trivial)
        if context_dim > 5:
            pca_dims = [d for d in SWEEP_CONFIG["pca_dims"] if d <= context_dim]
            dim_results = sweep_context_dimensionality(
                df_full          = test_df,
                n_arms           = n_arms,
                full_context_dim = context_dim,
                dims_to_sweep    = pca_dims,
                agent_configs    = robustness_configs,
                n_trials         = 2,
                verbose          = True,
            )
            fig = plot_context_dim_sensitivity(dim_results)
            fig.savefig(f"{DIRS['robustness']}/context_dim.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        print("  Saved: robustness/*.png")
    else:
        print("  [Debug mode] Skipping Group 4 robustness plots.")

    # --- Group 5: Hyperparameter sensitivity plots ---
    # Use the validation sweep results already computed in Phase 2.
    if sweep_results:
        print("Generating Group 5: Hyperparameter sensitivity plots...")

        fig = plot_all_hyperparam_sweeps(sweep_results, metric="regret")
        fig.savefig(f"{DIRS['sensitivity']}/all_sweeps_regret.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_all_hyperparam_sweeps(sweep_results, metric="reward")
        fig.savefig(f"{DIRS['sensitivity']}/all_sweeps_reward.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print("  Saved: sensitivity/*.png")
    else:
        print("  [Sweep skipped] No sensitivity plots generated.")

    print("\nAll figures saved.")


# ===========================================================================
# Phase 5: Summary
# ===========================================================================

def phase5_summary(results: dict) -> None:
    """
    Print the final comparison table to stdout and export it as a CSV.

    Parameters
    ----------
    results : dict — output of compare.run_comparison()
    """
    _section("Phase 5: Summary")

    # Print formatted table sorted by ascending final regret
    print_summary_table(results)

    # Export to CSV
    csv_path = os.path.join(DIRS["results"], "summary.csv")
    export_summary_csv(results, path=csv_path)
    print(f"Summary CSV saved to: {csv_path}")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    --debug           : run on a reduced subset of data with fewer trials;
                        useful for verifying the pipeline end-to-end quickly
    --skip-sweep      : skip hyperparameter selection and use default values
    --force-sweep     : re-run the validation sweep even if cached results exist
    --force-comparison: re-run the main comparison even if cached results exist
    --algorithms      : space-separated list of algorithm names to include;
                        defaults to all available algorithms
    """
    parser = argparse.ArgumentParser(
        description="Contextual Bandit Recommendation Experiment"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run a fast debug pass on a subset of the data.",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip hyperparameter sweep and use default values.",
    )
    parser.add_argument(
        "--force-sweep", action="store_true",
        help="Re-run sweep even if cached results exist.",
    )
    parser.add_argument(
        "--force-comparison", action="store_true",
        help="Re-run comparison even if cached results exist.",
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        metavar="ALG",
        help=(
            "Space-separated list of algorithms to include. "
            "Options: EpsilonGreedy LinUCB ThompsonSampling "
            "LogisticUCB BootstrapThompson NeuralUCB NeuralTS"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _make_dirs()

    total_start = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Load data
    # ------------------------------------------------------------------
    train_df, val_df, test_df, context_dim, base_rate = phase1_load_data(args)

    # In debug mode, subsample each split to keep runtimes short
    if args.debug:
        from bandits_environment import subsample_env
        DEBUG_N = 50000
        print(f"\n[Debug] Subsampling each split to {DEBUG_N:,} events.")
        train_df = train_df.head(DEBUG_N).reset_index(drop=True)
        val_df   = val_df.head(DEBUG_N // 5).reset_index(drop=True)
        test_df  = test_df.head(DEBUG_N // 5).reset_index(drop=True)
        # Reduce trial count for faster debugging
        EVAL_CONFIG["n_trials"]   = 2
        SWEEP_CONFIG["n_trials"]  = 1

    # Determine arm pool from the full dataset (all unique movie_ids in test)
    n_arms = test_df["movie_id"].nunique()
    print(f"\nArm pool size (unique movies in test): {n_arms:,}")

    # ------------------------------------------------------------------
    # Phase 2: Hyperparameter selection
    # ------------------------------------------------------------------
    best_params, sweep_results = phase2_hyperparameter_selection(
        val_df, n_arms, context_dim, args
    )

    # ------------------------------------------------------------------
    # Phase 3: Main comparison
    # ------------------------------------------------------------------
    results = phase3_main_comparison(
        test_df, n_arms, context_dim, best_params, args
    )

    # ------------------------------------------------------------------
    # Phase 4: Visualization
    # ------------------------------------------------------------------
    phase4_visualization(
        results      = results,
        sweep_results= sweep_results,
        test_df      = test_df,
        n_arms       = n_arms,
        context_dim  = context_dim,
        base_rate    = base_rate,
        args         = args,
    )

    # ------------------------------------------------------------------
    # Phase 5: Summary
    # ------------------------------------------------------------------
    phase5_summary(results)

    total_time = time.time() - total_start
    print(f"\nTotal pipeline time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()