"""
analysis/sensitivity.py

==============

Hyperparameter sensitivity analysis and context dimensionality sweeps


This sensitivity script intends to:

    1. Hyperparameter selection: before running the main comparison, we sweep each algorithm's key hyperparameters
       over a grid of possible values and identify the value that maximizes the reward rate (or, in the
       of regret, minimizes it) on a held-out validation portion of the data. The selected values should
       be plugged into the `compare.build_default_config()` method before the real comparison takes place. 
    
    2. Sensitivity analysis: after the main comparison, quantify how sensitive each algorithm is to its
    hyperparameters. A flat sweep curve across the hyperparameters means the algorithm is robust; a peaked
    curve means it needs careful fine-tuning. We also sweep context dimensionality (via PCA reduction) to
    test whether algorithms are actually using the contextual information correctly. 


The workflow for this sensitivity analsys is: 
    1. Select hyperparameters on validation data, via best_params = select_all_hyperparameters(val_df, context_dim)
    2. Pass best_params into compare.bild_default_config(), via config = build_default_config(n_arms, context_dim, overrides=best_params).
    3. Run sensitivity figures on test data for the paper, via sweep_results = run_all_sweeps(test_df, context_dim, best_params).

All sweep functions return structured result dictionaries consumed by the `visualize.py` script for plotting.
"""

# Import dependencies
from .evaluation import run_trial, run_multiple_trials
from bandits_environment import OfflineBanditEnv
from typing import Any
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Generic single-hyperparameter sweep
# ---------------------------------------------------------------------------
def sweep_hyperparameter(agent_class: type, df: pd.DataFrame, param_name: str, param_values: list, fixed_kwargs: dict,
    n_trials: int = 3, window: int = 500, env_seed: int = 42, verbose: bool = True,) -> dict:
    """
    Function to sweep one hyperparameter of an algorithm over a list of candidate values and record final
    cumulative regret and reward rate at each value. For each candidate value, n_trials independent trials
    are run and the results are averaged.  The environment interaction order is identical across all values
    (same env_seed), so differences are attributable to the hyperparameter alone.

    Parameters
    ----------
    * agent_class: the BaseBandit subclass to sweep
    * df: interaction data as a DataFrame (validation or test split)
    * param_name: name of the hyperparameter to sweep (must be a valid kwarg of agent_class.__init__)
    * param_values: candidate values to evaluate
    * fixed_kwargs: constructor kwargs that are held fixed during the sweep (must include n_arms and context_dim)
    * n_trials: trials per candidate value (for variance estimation)
    * window: rolling window for reward smoothing
    * env_seed: environment random seed
    * verbose: print progress

    Returns
    -------
    * dict: dictionary with keys 
        param_name: str
        param_values: list  — the swept values
        mean_final_regret: list  — mean final cumulative regret per value
        std_final_regret: list  — std  final cumulative regret per value
        mean_reward_rate: list  — mean final rolling reward per value
        std_reward_rate: list  — std  final rolling reward per value
        best_value: the param value with the lowest mean final regret
        best_reward_value: the param value with the highest mean reward rate
        agent_name: str
    """
    env = OfflineBanditEnv(df, seed=env_seed)

    mean_regrets = []
    std_regrets = []
    mean_rewards = []
    std_rewards = []

    for val in param_values:
        if verbose:
            print(f"  [{agent_class.__name__}] {param_name}={val} ...", end=" ")

        # Build kwargs for this candidate value
        kwargs = {**fixed_kwargs, param_name: val}

        # Run n_trials independent trials
        agg = run_multiple_trials(
            agent_class  = agent_class,
            env = env,
            n_trials = n_trials,
            agent_kwargs = kwargs,
            window = window,
            verbose = False,
        )

        mean_regrets.append(agg["mean_final_regret"])
        std_regrets.append(agg["std_final_regret"])
        # Final rolling reward = converged reward rate
        mean_rewards.append(float(agg["mean_rolling_reward"][-1]))
        std_rewards.append(float(agg["std_rolling_reward"][-1]))

        if verbose:
            print(f"regret={agg['mean_final_regret']:.1f} ± {agg['std_final_regret']:.1f} | "
                  f"reward={mean_rewards[-1]:.3f}")

    # Identify best values (lowest regret and highest reward)
    best_idx = int(np.argmin(mean_regrets))
    best_reward_idx = int(np.argmax(mean_rewards))

    return {
        "param_name": param_name,
        "param_values": param_values,
        "mean_final_regret": mean_regrets,
        "std_final_regret": std_regrets,
        "mean_reward_rate": mean_rewards,
        "std_reward_rate": std_rewards,
        "best_value": param_values[best_idx],
        "best_reward_value": param_values[best_reward_idx],
        "agent_name": agent_class.__name__,
    }


# ---------------------------------------------------------------------------
# Per-algorithm hyperparameter sweep definitions
# ---------------------------------------------------------------------------
def sweep_linucb(df: pd.DataFrame, n_arms: int, context_dim: int, alpha_values: list   = None,
    lambda_values: list  = None, n_trials: int = 10, verbose: bool = True,) -> dict:
    """
    Function to sweep alpha and lambda_reg for LinUCB. Alpha controls exploration bonus width, and lambda_reg
    controls the regularization strength. The function returns a dictionary with keys 'alpha' and 'lambda_reg',
    each containing the output of `sweep_hyperparameter()`.
    """
    from algorithms.lin_ucb import LinUCB

    if alpha_values is None:
        # Standard sweep range for UCB exploration parameter
        alpha_values  = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    if lambda_values is None:
        lambda_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    base = {"n_arms": n_arms, "context_dim": context_dim, "lambda_reg": 1.0}

    if verbose: print("Sweeping LinUCB alpha...")
    alpha_sweep = sweep_hyperparameter(
        LinUCB, df, "alpha", alpha_values,
        fixed_kwargs={**base}, n_trials=n_trials, verbose=verbose,
    )

    if verbose:
        print("Sweeping LinUCB lambda_reg...")
    # Use best alpha found above when sweeping lambda
    lambda_sweep = sweep_hyperparameter(LinUCB, df, "lambda_reg", lambda_values, fixed_kwargs={"n_arms": n_arms,
        "context_dim": context_dim, "alpha": alpha_sweep["best_value"]}, n_trials=n_trials, verbose=verbose,)
    
    return {"alpha": alpha_sweep, "lambda_reg": lambda_sweep}


def sweep_thompson_sampling(df: pd.DataFrame, n_arms: int, context_dim: int, sigma_values: list  = None,
    lambda_values: list = None, n_trials: int = 3, verbose: bool = True,) -> dict:
    """
    Function to sweep sigma and lambda_reg for ThompsonSampling. Sigma controls the posterior sampling
    scale (the explortation intensity), and returns a dictionary with keys 'sigma' and 'lambda_reg', each
    containing the output of `sweep_hyperparameter()`.
    """
    from algorithms.thompson_sampling import ThompsonSampling

    if sigma_values is None:
        sigma_values  = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    if lambda_values is None:
        lambda_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    base = {"n_arms": n_arms, "context_dim": context_dim, "lambda_reg": 1.0}

    if verbose: print("Sweeping Thompson Sampling sigma...")
    sigma_sweep = sweep_hyperparameter(ThompsonSampling, df, "sigma", sigma_values, fixed_kwargs={**base},
        n_trials=n_trials, verbose=verbose,
    )

    if verbose:
        print("Sweeping Thompson Sampling lambda_reg...")
    lambda_sweep = sweep_hyperparameter(
        ThompsonSampling, df, "lambda_reg", lambda_values,
        fixed_kwargs={"n_arms": n_arms, "context_dim": context_dim,
                      "sigma": sigma_sweep["best_value"]},
        n_trials=n_trials, verbose=verbose,
    )

    return {"sigma": sigma_sweep, "lambda_reg": lambda_sweep}


def sweep_epsilon_greedy(df: pd.DataFrame, n_arms: int, context_dim: int, epsilon_values: list = None,
    n_trials: int = 3, verbose: bool = True,) -> dict:
    """
    Function to sweep epsilon for Epsilon Greedy algorithm for both fixed and decay schedules. Function
    returns a dictionary with keys 'fixed' and 'decay', the outputs of `sweep_hyperparameters()` for both
    implementations of the algorithm.
    """
    from algorithms.epsilon_greedy import EpsilonGreedy

    if epsilon_values is None:
        epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    base_fixed = {"n_arms": n_arms, "context_dim": context_dim,
                  "schedule": "fixed", "lambda_reg": 1.0}
    base_decay = {"n_arms": n_arms, "context_dim": context_dim,
                  "schedule": "decay", "lambda_reg": 1.0, "decay_rate": 0.01}

    if verbose:
        print("Sweeping EpsilonGreedy epsilon (fixed schedule)...")
    fixed_sweep = sweep_hyperparameter(
        EpsilonGreedy, df, "epsilon", epsilon_values,
        fixed_kwargs=base_fixed, n_trials=n_trials, verbose=verbose,
    )

    if verbose:
        print("Sweeping EpsilonGreedy epsilon (decay schedule)...")
    decay_sweep = sweep_hyperparameter(
        EpsilonGreedy, df, "epsilon", epsilon_values,
        fixed_kwargs=base_decay, n_trials=n_trials, verbose=verbose,
    )

    return {"fixed": fixed_sweep, "decay": decay_sweep}


def sweep_logistic_ucb(df: pd.DataFrame, n_arms: int, context_dim: int, alpha_values: list  = None,
    lambda_values: list = None, n_trials: int = 3, verbose: bool = True,) -> dict:
    """
    Function to sweep alpha and lambda_reg for LogUCB. Alpha controls exploration bonus width, and lambda_reg
    controls the regularization strength. The function returns a dictionary with keys 'alpha' and 'lambda_reg',
    each containing the output of `sweep_hyperparameter()`.
    """
    from algorithms.logistic_ucb import LogisticUCB

    if alpha_values is None:
        alpha_values  = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    if lambda_values is None:
        lambda_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    base = {"n_arms": n_arms, "context_dim": context_dim, "lambda_reg": 1.0}

    if verbose:
        print("Sweeping LogisticUCB alpha...")
    alpha_sweep = sweep_hyperparameter(
        LogisticUCB, df, "alpha", alpha_values,
        fixed_kwargs={**base}, n_trials=n_trials, verbose=verbose,
    )

    if verbose:
        print("Sweeping LogisticUCB lambda_reg...")
    lambda_sweep = sweep_hyperparameter(
        LogisticUCB, df, "lambda_reg", lambda_values,
        fixed_kwargs={"n_arms": n_arms, "context_dim": context_dim,
                      "alpha": alpha_sweep["best_value"]},
        n_trials=n_trials, verbose=verbose,
    )

    return {"alpha": alpha_sweep, "lambda_reg": lambda_sweep}


def sweep_bootstrap_thompson(df: pd.DataFrame, n_arms: int, context_dim: int, particle_values: list = None,
    lambda_values: list = None, n_trials: int = 3, verbose: bool = True,) -> dict:
    """
    Function to sweep sigma and lambda_reg for bootstrap Thompson sampling. Sigma controls the posterior
    sampling scale (the explortation intensity), and returns a dictionary with keys 'sigma' and 'lambda_reg',
    each containing the output of `sweep_hyperparameter()`.
    """
    from algorithms.bootstrap_thompson import BootstrapThompson

    if particle_values is None:
        # Diminishing returns beyond ~20 particles (Eckles & Kaptein 2014)
        particle_values = [3, 5, 10, 20, 30]
    if lambda_values is None:
        lambda_values   = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    base = {"n_arms": n_arms, "context_dim": context_dim, "lambda_reg": 1.0}

    if verbose:
        print("Sweeping BootstrapThompson n_particles...")
    particle_sweep = sweep_hyperparameter(
        BootstrapThompson, df, "n_particles", particle_values,
        fixed_kwargs={**base}, n_trials=n_trials, verbose=verbose,
    )

    if verbose:
        print("Sweeping BootstrapThompson lambda_reg...")
    lambda_sweep = sweep_hyperparameter(
        BootstrapThompson, df, "lambda_reg", lambda_values,
        fixed_kwargs={"n_arms": n_arms, "context_dim": context_dim,
                      "n_particles": particle_sweep["best_value"]},
        n_trials=n_trials, verbose=verbose,
    )

    return {"n_particles": particle_sweep, "lambda_reg": lambda_sweep}


# ---------------------------------------------------------------------------
# Context dimensionality sweep (PCA reduction)
# ---------------------------------------------------------------------------
def sweep_context_dimensionality(df_full: pd.DataFrame, n_arms: int, full_context_dim: int, dims_to_sweep: list = None,
    agent_configs: list = None, n_trials: int = 3, window: int = 500, env_seed: int = 42, verbose: bool = True,) -> dict:
    """
    Function to evaluate all algorithms at different context dimensionalities by applying PCA to reduce
    the context vectors before running trials. This tests whether algorithms actually benefit from richer
    feature representations or whether a small number of principal components captures most of the useful
    structure. The sweep rebuilds the context vectors at each dimensionality using the
    data_loader.build_context_vectors() function with method='pca', producing a new DataFrame at each
    dimension.

    Parameters
    ----------
    * df_full: full merged interaction DataFrame (output of merge_dataset, before build_context_vectors is called)
    * n_arms: number of arms in the arm pool
    * full_context_dim: the dimensionality of the raw context vectors
    * dims_to_sweep: PCA dimensions to evaluate; must all be <= full_context_dim
    * agent_configs: algorithm configs (same format as compare.build_default_config output); if None, uses LinUCB and ThompsonSampling
    * n_trials: trials per algorithm per dimension
    * window: rolling window for reward smoothing
    * env_seed: environment seed
    * verbose: print progress

    Returns
    -------
    * dict: dictionary with keys
        dims: the swept dimensions
        results: dict mapping algorithm_name -> list of per-dim agg dicts
        mean_regret: dict mapping algorithm_name -> list of mean final regrets
        std_regret: dict mapping algorithm_name -> list of std final regrets
        mean_reward: dict mapping algorithm_name -> list of mean reward rates
    """
    from data_loading import build_context_vectors, CONTEXT_FEATURE_COLS
    from sklearn.preprocessing import StandardScaler

    if dims_to_sweep is None:
        # Default: sweep from 2 up to full dim at a few key checkpoints
        dims_to_sweep = [2, 5, 10, 15, full_context_dim]
        dims_to_sweep = [d for d in dims_to_sweep if d <= full_context_dim]

    if agent_configs is None:
        from algorithms.lin_ucb            import LinUCB
        from algorithms.thompson_sampling import ThompsonSampling
        agent_configs = [
            {"name": "LinUCB",
             "class": LinUCB,
             "kwargs": {"n_arms": n_arms, "alpha": 1.0, "lambda_reg": 1.0}},
            {"name": "ThompsonSampling",
             "class": ThompsonSampling,
             "kwargs": {"n_arms": n_arms, "sigma": 1.0, "lambda_reg": 1.0}},
        ]

    # Initialise result storage per algorithm
    dim_results   = {cfg["name"]: [] for cfg in agent_configs}
    mean_regrets  = {cfg["name"]: [] for cfg in agent_configs}
    std_regrets   = {cfg["name"]: [] for cfg in agent_configs}
    mean_rewards  = {cfg["name"]: [] for cfg in agent_configs}

    for dim in dims_to_sweep:
        if verbose:
            print(f"\n  Context dim = {dim}")

        # Rebuild context vectors at this dimensionality
        if dim == full_context_dim:
            # Use raw features without PCA
            df_dim, _, _, _ = build_context_vectors(
                df_full, method="raw", fit=True,
            )
        else:
            df_dim, _, _, _ = build_context_vectors(
                df_full, method="pca", pca_dim=dim, fit=True,
            )

        env = OfflineBanditEnv(df_dim, seed=env_seed)

        for cfg in agent_configs:
            name = cfg["name"]
            cls = cfg["class"]
            # Override context_dim with the current PCA dimension
            kwargs = {**cfg["kwargs"], "context_dim": dim}

            if verbose:
                print(f"    {name} (dim={dim})...", end=" ")

            agg = run_multiple_trials(
                agent_class = cls,
                env = env,
                n_trials = n_trials,
                agent_kwargs = kwargs,
                window = window,
                verbose = False,
            )

            dim_results[name].append(agg)
            mean_regrets[name].append(agg["mean_final_regret"])
            std_regrets[name].append(agg["std_final_regret"])
            mean_rewards[name].append(float(agg["mean_rolling_reward"][-1]))

            if verbose:
                print(f"regret={agg['mean_final_regret']:.1f} | "
                      f"reward={mean_rewards[name][-1]:.3f}")

    return {
        "dims": dims_to_sweep,
        "results": dim_results,
        "mean_regret": mean_regrets,
        "std_regret": std_regrets,
        "mean_reward": mean_rewards,
    }


# ---------------------------------------------------------------------------
# Reward threshold sensitivity
# ---------------------------------------------------------------------------
def sweep_reward_threshold(df_raw: pd.DataFrame, n_arms: int, context_dim: int, thresholds: list = None,
    agent_configs: list = None, n_trials: int = 10, window: int = 500, env_seed: int = 42, verbose: bool = True,) -> dict:
    """
    Function to evaluate algorithm robustness to the choice of reward binarization threshold. The threshold
    converts continuous ratings into binary rewards (1 if rating >= threshold, else 0). This sweep checks
    whether the relative ranking of algorithms is stable across different threshold choices.

    Parameters
    ----------
    * df_raw: merged interaction DataFrame with 'rating' column but WITHOUT a 'reward' column yet (i.e.,
              before binarize_rewards() is called)
    * thresholds: threshold values to evaluate
    * agent_configs: algorithm configs (same format as compare.py)

    Returns
    -------
    * dict: dictionary with keys
        thresholds: list[float]
        reward_rates: list[float] — base reward rate at each threshold
        mean_regret: dict mapping agent_name -> list of mean final regrets
        mean_reward: dict mapping agent_name -> list of mean reward rates
    """
    from data_loading import binarize_rewards, build_context_vectors

    if thresholds is None:
        thresholds = [3.0, 3.5, 4.0, 4.5]

    if agent_configs is None:
        from algorithms.lin_ucb import LinUCB
        from algorithms.thompson_sampling import ThompsonSampling
        agent_configs = [
            {"name": "LinUCB",
             "class": LinUCB,
             "kwargs": {"n_arms": n_arms, "context_dim": context_dim,
                        "alpha": 1.0, "lambda_reg": 1.0}},
            {"name": "ThompsonSampling",
             "class": ThompsonSampling,
             "kwargs": {"n_arms": n_arms, "context_dim": context_dim,
                        "sigma": 1.0, "lambda_reg": 1.0}},
        ]

    base_reward_rates = []
    mean_regrets = {cfg["name"]: [] for cfg in agent_configs}
    mean_rewards = {cfg["name"]: [] for cfg in agent_configs}

    for thresh in thresholds:
        if verbose: print(f"\n  Reward threshold = {thresh}")

        # Binarize rewards at this threshold
        df_thresh = binarize_rewards(df_raw, threshold=thresh)
        # Rebuild context vectors (standardization is threshold-independent)
        df_thresh, _, _, _ = build_context_vectors(df_thresh, method="raw", fit=True)

        # Record base reward rate at this threshold
        base_rate = float(df_thresh["reward"].mean())
        base_reward_rates.append(base_rate)
        if verbose:
            print(f"    Base reward rate: {base_rate:.3f}")

        env = OfflineBanditEnv(df_thresh, seed=env_seed)

        for cfg in agent_configs:
            name = cfg["name"]
            cls = cfg["class"]
            kwargs = cfg["kwargs"]

            agg = run_multiple_trials(
                agent_class = cls,
                env = env,
                n_trials = n_trials,
                agent_kwargs = kwargs,
                window = window,
                verbose = False,
            )

            mean_regrets[name].append(agg["mean_final_regret"])
            mean_rewards[name].append(float(agg["mean_rolling_reward"][-1]))

            if verbose:
                print(f"    {name}: regret={agg['mean_final_regret']:.1f} | "
                      f"reward={mean_rewards[name][-1]:.3f}")

    return {
        "thresholds": thresholds,
        "reward_rates": base_reward_rates,
        "mean_regret": mean_regrets,
        "mean_reward": mean_rewards,
    }


# ---------------------------------------------------------------------------
# Top-level: select best hyperparameters from sweep results
# ---------------------------------------------------------------------------
def select_best_hyperparameters(sweep_results: dict) -> dict:
    """
    Function to extract the best hyperparameter value for each algorithm and parameter from the output
    of the per-algorithm sweep functions. Takes the nested sweep result dict (output of sweep_linucb,
    sweep_thompson_sampling, etc.) and returns a flat dict of selected values.

    Parameters
    ----------
    * sweep_results: dict mapping algorithm_name -> per-algorithm sweep dict (output of sweep_linucb,
                   sweep_thompson_sampling, etc.)

    Returns
    -------
    * dict: dictionary mapping algorithm_name to a dictionary of {param_name: best_value}

    Example output:
        {
            'LinUCB':            {'alpha': 0.5, 'lambda_reg': 1.0},
            'ThompsonSampling':  {'sigma': 1.0, 'lambda_reg': 0.5},
            ...
        }
    """
    best_params = {}

    for algo_name, param_sweeps in sweep_results.items():
        best_params[algo_name] = {}
        for param_name, sweep in param_sweeps.items():
            if isinstance(sweep, dict) and "best_value" in sweep:
                # Standard single-param sweep result
                best_params[algo_name][param_name] = sweep["best_value"]
            elif isinstance(sweep, dict) and "fixed" in sweep:
                # EpsilonGreedy returns {'fixed': sweep, 'decay': sweep}
                # Choose the schedule with lower mean final regret
                fixed_regret = min(sweep["fixed"]["mean_final_regret"])
                decay_regret = min(sweep["decay"]["mean_final_regret"])
                if decay_regret < fixed_regret:
                    best_params[algo_name]["schedule"] = "decay"
                    best_params[algo_name]["epsilon"] = sweep["decay"]["best_value"]
                else:
                    best_params[algo_name]["schedule"] = "fixed"
                    best_params[algo_name]["epsilon"] = sweep["fixed"]["best_value"]

    return best_params


def print_sweep_summary(sweep_results: dict) -> None:
    """
    Print a human-readable summary of all hyperparameter sweep results, showing the best value and
    corresponding regret for each parameter.

    Parameters
    ----------
    * sweep_results: output of per-algorithm sweep functions
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("=" * 70)

    for algo_name, param_sweeps in sweep_results.items():
        print(f"\n{algo_name}:")
        for param_name, sweep in param_sweeps.items():
            if isinstance(sweep, dict) and "best_value" in sweep:
                best_val = sweep["best_value"]
                best_idx = sweep["param_values"].index(best_val)
                best_regret = sweep["mean_final_regret"][best_idx]
                print(f"  {param_name:<20} best={best_val:<8} "
                      f"(regret={best_regret:.1f})")
                # Print full sweep table
                for v, r, s in zip(sweep["param_values"],
                                   sweep["mean_final_regret"],
                                   sweep["std_final_regret"]):
                    marker = " <-- best" if v == best_val else ""
                    print(f"    {param_name}={v:<8} regret={r:.1f} ± {s:.1f}{marker}")
            elif isinstance(sweep, dict) and "fixed" in sweep:
                # EpsilonGreedy special case
                for sched in ("fixed", "decay"):
                    s = sweep[sched]
                    print(f"  epsilon ({sched} schedule) best={s['best_value']}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Building synthetic data for sensitivity smoke test...\n")

    rng = np.random.default_rng(0)
    n = 2000
    n_arms = 6
    ctx_dim = 8
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

    from algorithms.lin_ucb import LinUCB

    # --- Single hyperparameter sweep ---
    print("Testing sweep_hyperparameter (LinUCB alpha)...")
    result = sweep_hyperparameter(
        agent_class = LinUCB,
        df = synthetic_df,
        param_name = "alpha",
        param_values = [0.1, 0.5, 1.0, 2.0],
        fixed_kwargs = {"n_arms": n_arms, "context_dim": ctx_dim, "lambda_reg": 1.0},
        n_trials = 10,
        window = 100,
        verbose = True,
    )

    print(f"\n  Best alpha (by regret)  : {result['best_value']}")
    print(f"  Best alpha (by reward)  : {result['best_reward_value']}")
    assert len(result["mean_final_regret"]) == 4, "Wrong number of sweep results"
    assert result["best_value"] in [0.1, 0.5, 1.0, 2.0], "Best value not in grid"
    print("  Sweep structure checks passed")

    # --- select_best_hyperparameters ---
    print("\nTesting select_best_hyperparameters...")
    mock_sweeps = {
        "LinUCB": {
            "alpha":      result,
            "lambda_reg": {
                "param_name":        "lambda_reg",
                "param_values":      [0.1, 1.0, 5.0],
                "mean_final_regret": [120.0, 100.0, 130.0],
                "std_final_regret":  [5.0,   3.0,   7.0],
                "mean_reward_rate":  [0.5,   0.55,  0.48],
                "std_reward_rate":   [0.02,  0.01,  0.03],
                "best_value":        1.0,
                "best_reward_value": 1.0,
                "agent_name":        "LinUCB",
            },
        }
    }
    best = select_best_hyperparameters(mock_sweeps)
    assert "LinUCB" in best
    assert "alpha" in best["LinUCB"]
    assert "lambda_reg" in best["LinUCB"]
    print(f"  Selected params: {best}")
    print("  select_best_hyperparameters checks passed")

    print("\nSmoke test passed.")