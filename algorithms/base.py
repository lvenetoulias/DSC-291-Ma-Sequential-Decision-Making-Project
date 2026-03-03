"""
algorithms/base.py

==================

Abstract base class that all contextual bandit algorithms must implement. This class allows for a consistent
interface across all algorithms so that the evaluation loop in `evaluate.py` can remain completely algorithm-agnostic.
That is, it only ever selects an arm, updates according to the result, and resets itself, all of which
are the same procedures, regardless of which algorithm is being used/chosen behind the scenes.

We construct the various algorithms for the projects such that they should: 
    1. Inherit from BaseBandit (listed here)
    2. Call super().__init__(), with n_arms, context_dim, and any shared kwargs listed
    3. Implement select_arm(), update(), and reset().


Example usage
-------
    from algorithms.base import BaseBandit

    class MyAlgorithm(BaseBandit):
        def select_arm(self, context):
            ...
        def update(self, arm, reward, context):
            ...
        def reset(self):
            ...
"""

# Import dependencies
from abc import ABC, abstractmethod
import numpy as np


class BaseBandit(ABC):
    """
    Abstract base class for all contextual bandit algorithms.

    Parameters
    ----------
    * n_arms: int — number of arms (movies) in the arm pool
    * context_dim: int — dimensionality of the context vector
    * seed: int — random seed for reproducibility
    * name: str — human-readable algorithm name (used in plots/logs)
    """

    def __init__(self, n_arms: int, context_dim: int, seed: int = 42, name: str = "BaseBandit",):
        if n_arms <= 0:
            raise ValueError(f"n_arms must be a positive integer, got {n_arms}.")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be a positive integer, got {context_dim}.")

        self.n_arms = n_arms
        self.context_dim = context_dim
        self.seed = seed
        self.name = name

        # Shared RNG — subclasses should use self.rng for all random draws (to keep reproducibility of results)
        self.rng = np.random.default_rng(seed)

        # Step counters maintained by the base class (updated automatically for subclasses via wrappers)
        self._t: int = 0                        # total update steps
        self._arm_counts: np.ndarray = np.zeros(n_arms, dtype=int)

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------
    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm given the current context vector.

        Parameters
        ----------
        * context: np.ndarray of shape (context_dim,)

        Returns
        -------
        * int: an index corresponding to the selected arm from the candidate arms pool (NOT the raw
               movie_id)
        """
        ...

    @abstractmethod
    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        """
        Update the algorithm's internal model given an observed reward.

        This should only be called on *matched* events (i.e. when the
        environment confirmed that the chosen arm matched the logged arm).
        The evaluation loop is responsible for enforcing this.

        Parameters
        ----------
        * arm: an integer corresponding to the index of the chosen arm
        * reward: observed reward (0 or 1 in the binary setting)
        * context: np.ndarray of shape (context_dim,)
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all internal state to its initial condition.

        Must also call super().reset() to reset base-class counters and
        re-seed the RNG so repeated trials are reproducible.
        """
        ...

    # ------------------------------------------------------------------
    # Base class helpers — available to all subclasses
    # ------------------------------------------------------------------
    def _base_update(self, arm: int) -> None:
        """
        Increment shared step counters.  Subclasses should call this at
        the top of their update() implementation.
        """
        self._t += 1
        self._arm_counts[arm] += 1

    def _base_reset(self) -> None:
        """
        Reset shared counters and re-seed the RNG.  Subclasses should
        call this inside their reset() implementation via super().reset().
        """
        self._t = 0
        self._arm_counts = np.zeros(self.n_arms, dtype=int)
        self.rng = np.random.default_rng(self.seed)

    def arm_counts(self) -> np.ndarray:
        """Return a copy of the per-arm update count vector."""
        return self._arm_counts.copy()

    def total_updates(self) -> int:
        """Return the total number of update steps performed so far."""
        return self._t

    def __repr__(self) -> str:
        return (
            f"{self.name}("
            f"n_arms={self.n_arms}, "
            f"context_dim={self.context_dim}, "
            f"t={self._t})"
        )