"""
BANDIT ENVIRONMENT SCRIPT

=================

Script implements the offline bandit replay environment. The key idea behind this offline replay, first
proposed in the literature by Li et al. (2010), is to evaluate an online bandit algorithm using only logged
data, without running a live experiment, by replaying potentially observed interactions one at a time and
only counting steps for which the algorithm would have chosen the same arm as the actually logged arm. All
other events are skipped, and no update to the model is performed. This gives an unbiased estimate of the
algorithm's reward rate under a logging policy's arm distribution. 

Typical usage
-------------
    from bandit_env import OfflineBanditEnv

    env = OfflineBanditEnv(train_df, seed=42)

    obs = env.reset()
    while not env.is_done():
        context, candidate_arms = env.get_context()
        chosen_arm = agent.select_arm(context, candidate_arms)
        reward, matched = env.step(chosen_arm)
        if matched:
            agent.update(chosen_arm, reward, context)

    print(env.summary())
"""

# Import dependencies
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Dataclass: result of single step
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """
    Returned by OfflineBanditEnv.step() after each decision.

    Attributes
    ----------
    reward   : float  — observed reward (0 or 1); meaningful only when matched=True
    matched  : bool   — True iff the agent chose the arm that was logged
    arm      : int    — the arm (movie_id) the agent chose
    logged_arm : int  — the arm actually logged in this event
    step_idx : int    — global step counter (counts every event, matched or not)
    matched_idx : int — matched-event counter (counts only matched events)
    """
    reward: float
    matched: bool
    arm: int
    logged_arm: int
    step_idx: int
    matched_idx: int


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------
class OfflineBanditEnv:
    """
    Offline bandit replay environment for the MovieLens 10M dataset.

    The environment streams logged interactions one at a time.  At each step the agent observes a context
    vector and a set of candidate arms (movies), chooses one, and receives a reward only when its choice
    matches the logged arm.  Non-matched events are counted but skipped — no reward is returned and no
    model update should be performed.

    Parameters
    ----------
    * df: interaction DataFrame produced by data_loader.load_and_prepare(); must contain columns: movie_id,
          context, reward, user_id, timestamp
    * seed: random seed for reproducibility (used if shuffling is enabled)
    * shuffle: if True, shuffle the interaction stream before replay; if False, preserve the temporal
               ordering from the DataFrame
    * arm_pool: if provided, restricts candidate arms to this list of movie_ids; defaults to all unique
                movie_ids seen in df
    """

    def __init__(self, df: pd.DataFrame, seed: int = 42, shuffle: bool = False, arm_pool: Optional[list] = None,):
        self._df_original = df.reset_index(drop=True)
        self._seed = seed
        self._shuffle = shuffle

        # Build the canonical arm set
        if arm_pool is not None:
            self.arm_pool = list(arm_pool)
        else:
            self.arm_pool = sorted(df["movie_id"].unique().tolist())

        self.n_arms = len(self.arm_pool)

        # Map movie_id to an arm index and back
        self._arm_to_idx = {arm: idx for idx, arm in enumerate(self.arm_pool)}
        self._idx_to_arm = {idx: arm for idx, arm in enumerate(self.arm_pool)}

        # Internal state (populated by reset())
        self._df: pd.DataFrame = None
        self._cursor: int = 0
        self._step_idx: int = 0
        self._matched_idx: int = 0
        self._total_reward: float = 0.0
        self._rewards_log: list = []       # reward at each *matched* step
        self._matched_log: list = []       # bool at each *total* step
        self._regret_log: list = []        # regret at each *matched* step

        self.reset()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def reset(self) -> "OfflineBanditEnv":
        """
        Reset the environment to the beginning of the interaction stream. Performs a reshuffling if shuffle=True,
        and returns self for method chaining. 
        """
        rng = np.random.default_rng(self._seed)

        if self._shuffle:
            idx = rng.permutation(len(self._df_original))
            self._df = self._df_original.iloc[idx].reset_index(drop=True)
        else:
            self._df = self._df_original.copy()

        # Populate internal state
        self._cursor = 0
        self._step_idx = 0
        self._matched_idx = 0
        self._total_reward = 0.0
        self._rewards_log = []
        self._matched_log = []
        self._regret_log = []

        return self

    def get_context(self) -> tuple[np.ndarray, list[int]]:
        """
        Return the context vector and the list of candidate arms for the current event. In this
        implementation, we return all arms in the pool as candidates, which follows the standard
        disjoint-model setup.

        (If we want to restrict the arms to a per-user candidate subset, we override this method or
        pass arm_pool at construction time.)

        Returns
        -------
        * context: np.ndarray of shape (context_dim,)
        * candidate_arms : list of movie_id integers (the arm pool)
        """
        if self.is_done():
            raise StopIteration("Environment stream is exhausted. Call reset().")

        row = self._df.iloc[self._cursor]
        context = row["context"]  # np.ndarray, stored by data_loader
        return context, self.arm_pool

    def step(self, chosen_arm: int) -> StepResult:
        """
        Advance the environment by one event, returning the movie_id it would have recommended.
        The environment then checks whether this recommendation matches the logged arm:
            - If yes (matched): reward is revealed; matched_idx increments.
            - If no (skipped): reward is None / 0; step is still counted.
        In both cases the cursor advances to the next event.

        Parameters
        ----------
        chosen_arm : int — the movie_id chosen by the agent

        Returns
        -------
        StepResult dataclass (see definition above)
        """
        if self.is_done():
            raise StopIteration("Environment stream is exhausted. Call reset().")

        # Collect logged arm for movie
        row = self._df.iloc[self._cursor]
        logged_arm = int(row["movie_id"])
        logged_reward = float(row["reward"])

        # Check if logged arm matches chosen arm
        matched = (chosen_arm == logged_arm)
        # If matching, reveal reward
        if matched:
            reward = logged_reward
            self._total_reward += reward
            self._rewards_log.append(reward)
            # Regret: 1 - reward (since the best possible reward is 1)
            self._regret_log.append(1.0 - reward)
            self._matched_idx += 1
        else:
            reward = 0.0  # undefined / not observed

        # Update internal state according to result
        self._matched_log.append(matched)
        self._step_idx += 1
        self._cursor += 1
        # Return result of single step
        return StepResult(
            reward=reward,
            matched=matched,
            arm=chosen_arm,
            logged_arm=logged_arm,
            step_idx=self._step_idx,
            matched_idx=self._matched_idx,
        )

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def is_done(self) -> bool:
        """Return True when all logged events have been streamed."""
        return self._cursor >= len(self._df)

    def current_event(self) -> pd.Series:
        """
        Return the raw DataFrame row at the current cursor position, without advancing the cursor.
        Useful for debugging.
        """
        if self.is_done():
            raise StopIteration("Stream is exhausted.")
        return self._df.iloc[self._cursor]

    def matched_rate(self) -> float:
        """
        Fraction of total steps that resulted in a match so far. Returns 0.0 if no steps have been
        taken.
        """
        return self._matched_idx / self._step_idx if self._step_idx != 0 else 0.0

    def cumulative_reward(self) -> float:
        """Total reward accumulated over matched events so far."""
        return self._total_reward

    def cumulative_regret(self) -> np.ndarray:
        """
        Cumulative regret as a 1-D array over matched events. Regret at step t given by
        R_t = sum_{i=1}^{t} (1 - reward_i).
        """
        return np.cumsum(self._regret_log)

    def rewards_array(self) -> np.ndarray:
        """Reward signal at each matched event as a numpy array."""
        return np.array(self._rewards_log, dtype=float)

    def matched_array(self) -> np.ndarray:
        """Boolean array of length n_total_steps indicating match/no-match."""
        return np.array(self._matched_log, dtype=bool)

    def arm_to_idx(self, arm: int) -> int:
        """Convert a movie_id to its index in the arm pool."""
        return self._arm_to_idx[arm]

    def idx_to_arm(self, idx: int) -> int:
        """Convert an arm pool index to its movie_id."""
        return self._idx_to_arm[idx]

    def summary(self) -> dict:
        """
        Return a summary dictionary of the current episode's statistics.

        Keys
        ----
        total_steps    : total number of events streamed (matched + skipped)
        matched_steps  : number of events where the agent matched the log
        matched_rate   : matched_steps / total_steps
        total_reward   : sum of rewards over matched events
        mean_reward    : total_reward / matched_steps  (reward rate)
        cumulative_regret : final cumulative regret value (scalar)
        n_arms         : size of the arm pool
        """
        matched = self._matched_idx
        return {
            "total_steps": self._step_idx,
            "matched_steps": matched,
            "matched_rate": self.matched_rate(),
            "total_reward": self._total_reward,
            "mean_reward": self._total_reward / matched if matched > 0 else 0.0,
            "cumulative_regret": float(self.cumulative_regret()[-1]) if self._regret_log else 0.0,
            "n_arms": self.n_arms,
        }

    def __repr__(self) -> str:
        return (
            f"OfflineBanditEnv("
            f"n_events={len(self._df)}, "
            f"n_arms={self.n_arms}, "
            f"cursor={self._cursor}/{len(self._df)}, "
            f"matched={self._matched_idx}"
            f")"
        )


# ---------------------------------------------------------------------------
# DEBUGGING SETUP (SMALLER ENVIRONMENT)
# ---------------------------------------------------------------------------
def subsample_env(df: pd.DataFrame, n_events: int, seed: int = 42, temporal: bool = True,) -> OfflineBanditEnv:
    """
    Function to build an OfflineBanditEnv from a random (or temporal) subsample of df. This is intended
    to be used for fast debugging without running the full 10M-event stream.

    Parameters
    ----------
    * df: full merged interaction DataFrame
    * n_events: number of events to include in the subsample
    * seed: random seed for reproducibility
    * temporal: if True, take the first n_events by timestamp (preserves order); if False, sample randomly

    Returns
    -------
    * OfflineBanditEnv over the subsampled events
    """
    if temporal: sub = df.sort_values("timestamp").head(n_events).reset_index(drop=True)
    else: sub = df.sample(n=min(n_events, len(df)), random_state=seed).reset_index(drop=True)
    return OfflineBanditEnv(sub, seed=seed)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Minimal end-to-end smoke test using a tiny synthetic DataFrame so the
    environment can be verified without loading the full dataset.
    """
    import sys

    print("Running quick OfflineBanditEnv smoke test with synthetic data...\n")

    rng = np.random.default_rng(0)
    n = 1000
    arm_pool = list(range(10))  # 10 fake movie ids

    synthetic_df = pd.DataFrame({
        "user_id":   rng.integers(1, 50, size=n),
        "movie_id":  rng.choice(arm_pool, size=n),
        "rating":    rng.choice([1, 2, 3, 4, 5], size=n).astype(float),
        "timestamp": np.arange(n),
        "reward":    rng.integers(0, 2, size=n).astype(float),
        "context":   [rng.random(8) for _ in range(n)],
    })

    env = OfflineBanditEnv(synthetic_df, seed=0, arm_pool=arm_pool)
    print(env)

    # Simulate a random agent
    n_steps = 0
    while not env.is_done():
        context, candidates = env.get_context()
        chosen = int(rng.choice(candidates))
        result = env.step(chosen)
        n_steps += 1

    print(f"\nCompleted {n_steps} steps.")
    print("Summary:", env.summary())

    regret = env.cumulative_regret()
    if len(regret) > 0:
        print(f"Final cumulative regret: {regret[-1]:.2f}")
    print("\nTest passed.")