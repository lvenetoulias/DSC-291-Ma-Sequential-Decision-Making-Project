"""
algorithms/epsilon_greedy.py

============================

Contextual epsilon-greedy bandit algorithm. At each step, the agent either:
    - Explores : selects a uniformly random arm with probability epsilon
    - Exploits : selects the arm with the highest predicted reward with probability (1 - epsilon)

We predict rewards via a separate linear model per arm (using ridge regression). Each arm maintains its
own weight vector θ, estimated by incrementally updating the least-squares solution. This is what makes
the algorithm 'contextual': exploitation decisions are conditional on the current feature vector rather
than just tracking per-arm mean rewards.     

Two epsilon schedules are supported:
    - 'fixed'  : epsilon is constant throughout (good baseline)
    - 'decay'  : epsilon = epsilon_0 / (1 + decay_rate * t), which reduces exploration as more data is collected



****LV WARNING: This algorithm is intended to be on the simpler side, offering up an interpretable lower-bound
baseline against which to compare other algorithms. It is probably unreasonable to expect that it is the
best performer, but it is reasonable to expect it to be the easiest to reason about.****
"""


# Import dependencies
from base import BaseBandit
import numpy as np


class EpsilonGreedy(BaseBandit):
    """
    Contextual epsilon-greedy bandit with per-arm ridge regression.

    Parameters
    ----------
    * n_arms: the number of arms in the candidate pool
    * context_dim: dimensionality of context vectors
    * epsilon: initial exploration probability (such that 0 < epsilon <= 1)
    * schedule: string flag to denote kind of epsilon value to have ('fixed' or 'decay')
    * decay_rate: controls how fast epsilon decays (only used when schedule='decay'); larger = faster decay
    * lambda_reg: L2 regularization strength for ridge regression; higher values shrink estimates closer
                  to 0 and improve stability (when data is scarce)
    * seed: random seed
    """
    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        epsilon: float = 0.1,
        schedule: str = "fixed",
        decay_rate: float = 0.01,
        lambda_reg: float = 1.0,
        seed: int = 42,
    ):
        # Call super constructor
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            seed=seed,
            name="EpsilonGreedy",
        )

        if not (0.0 < epsilon <= 1.0):
            raise ValueError(f"epsilon must be in (0, 1], got {epsilon}.")
        if schedule not in ("fixed", "decay"):
            raise ValueError(f"schedule must be 'fixed' or 'decay', got '{schedule}'.")
        if lambda_reg <= 0:
            raise ValueError(f"lambda_reg must be positive, got {lambda_reg}.")

        self.epsilon_0 = epsilon        # initial (or fixed) epsilon
        self.schedule = schedule
        self.decay_rate = decay_rate
        self.lambda_reg = lambda_reg

        # Per-arm ridge regression parameters. For each arm, keep:
        #   A[a]  : (d x d) matrix = lambda * I + sum_t x_t x_t^T
        #   b[a]  : (d,)    vector = sum_t r_t x_t
        # Param weights are: theta[a] = A[a]^{-1} b[a]; predicted reward given context x: x^T theta[a]

        # Store A_inv directly (avoid recomputing it for every select_arm call) via Sherman-Morrison rank-1 updates
        d = context_dim
        self._A_inv: list[np.ndarray] = [
            (1.0 / lambda_reg) * np.eye(d) for _ in range(n_arms)
        ]
        self._b: list[np.ndarray] = [np.zeros(d) for _ in range(n_arms)]

        # Cache weight vectors and updated lazily on each update() call
        self._theta: list[np.ndarray] = [np.zeros(d) for _ in range(n_arms)]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm using the epsilon-greedy rule.
            * With probability epsilon: explore — pick a random arm uniformly.
            * With probability 1 - epsilon: exploit — pick the arm with the highest predicted reward under
                                            the current linear model.

        Parameters
        ----------
        * context: np.ndarray of shape (context_dim,)

        Returns
        -------
        * int: an index corresponding to the selected arm.
        """
        epsilon = self._current_epsilon()

        if self.rng.random() < epsilon:
            # Explore: uniform random arm
            return int(self.rng.integers(0, self.n_arms))
        else:
            # Exploit: arm with highest predicted reward
            return int(self._best_arm(context))

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        """
        Update the ridge regression model for the chosen arm. Use Sherman-Morrison rank-1 update to A_inv
        and an additive update for b, before recomputing theta for chosen arm.

        Parameters
        ----------
        * arm: integer corresponting to chosen arm index
        * reward: observed binary reward (0 or 1) from chosen arm
        * context: np.ndarray of shape (context_dim,)
        """
        self._base_update(arm)
        x = context.reshape(-1)  # ensure 1-D

        # Sherman-Morrison update: A_inv <- A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        self._A_inv[arm] = self._sherman_morrison(self._A_inv[arm], x)
        # b update: b <- b + r * x
        self._b[arm] += reward * x

        # Recompute weight vector for this arm
        self._theta[arm] = self._A_inv[arm] @ self._b[arm]

    def reset(self) -> None:
        """Reset all internal state and re-seed the RNG."""
        super()._base_reset()

        d = self.context_dim
        self._A_inv = [
            (1.0 / self.lambda_reg) * np.eye(d) for _ in range(self.n_arms)
        ]
        self._b = [np.zeros(d) for _ in range(self.n_arms)]
        self._theta = [np.zeros(d) for _ in range(self.n_arms)]

    # ------------------------------------------------------------------
    # Helper (private) functions for class
    # ------------------------------------------------------------------
    def _current_epsilon(self) -> float:
        """
        Return the exploration probability for the current time step.

        'fixed': always returns epsilon_0
        'decay': returns epsilon_0 / (1 + decay_rate * t) where t is the number of updates so far
        """
        if self.schedule == "fixed": return self.epsilon_0
        else: return self.epsilon_0 / (1.0 + self.decay_rate * self._t)

    def _estimate_reward(self, arm: int, context: np.ndarray) -> float:
        """
        Predicted reward for a given arm and context vector.

        Returns the dot product x^T theta[arm], i.e. the linear model's
        point estimate of E[reward | arm, context].
        """
        return float(np.dot(context, self._theta[arm]))

    def _best_arm(self, context: np.ndarray) -> int:
        """
        Return the arm index with the highest predicted reward.
        Ties are broken randomly to avoid systematic bias.
        """
        estimates = np.array([
            self._estimate_reward(a, context) for a in range(self.n_arms)
        ])
        max_val = np.max(estimates)
        best_arms = np.where(estimates == max_val)[0]
        return int(self.rng.choice(best_arms))

    @staticmethod
    def _sherman_morrison(A_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Perform Sherman-Morrison rank-1 update. Given current A_inv and a new observation vector
        x, return updated inverse via:

            A_inv_new = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)

        Benefit is not having to recompute full A_inv from stratch at every update step, keeping the
        cost per-step at O(d^2) instead of O(d^3).
        """
        Ax = A_inv @ x                          # shape (d,)
        denom = 1.0 + x @ Ax                    # scalar
        return A_inv - np.outer(Ax, Ax) / denom

    # ------------------------------------------------------------------
    # Functions to report algorithm features. 
    # ------------------------------------------------------------------
    def current_epsilon(self) -> float:
        """Public accessor for the current exploration probability."""
        return self._current_epsilon()

    def get_theta(self, arm: int) -> np.ndarray:
        """Return a copy of the weight vector for a given arm."""
        return self._theta[arm].copy()

    def predicted_rewards(self, context: np.ndarray) -> np.ndarray:
        """
        Return predicted rewards for all arms given a context vector.
        Useful for debugging and visualisation.

        Returns
        -------
        np.ndarray of shape (n_arms,)
        """
        return np.array([self._estimate_reward(a, context) for a in range(self.n_arms)])

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedy("
            f"n_arms={self.n_arms}, "
            f"context_dim={self.context_dim}, "
            f"epsilon_0={self.epsilon_0}, "
            f"schedule='{self.schedule}', "
            f"lambda_reg={self.lambda_reg}, "
            f"t={self._t})"
        )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running EpsilonGreedy smoke test...\n")

    rng = np.random.default_rng(0)
    n_arms = 10
    context_dim = 8
    n_steps = 1000

    agent_fixed = EpsilonGreedy(n_arms, context_dim, epsilon=0.1, schedule="fixed", seed=0)
    agent_decay = EpsilonGreedy(n_arms, context_dim, epsilon=0.5, schedule="decay",
                                decay_rate=0.01, seed=0)

    for agent in [agent_fixed, agent_decay]:
        total_reward = 0.0
        for t in range(n_steps):
            context = rng.random(context_dim)
            arm = agent.select_arm(context)
            reward = float(rng.random() < 0.3 + 0.1 * arm)  # arm 8 is best
            agent.update(arm, reward, context)
            total_reward += reward

        print(agent)
        print(f"  Total reward over {n_steps} steps : {total_reward:.1f}")
        print(f"  Arm selection counts              : {agent.arm_counts()}")
        print(f"  Current epsilon                   : {agent.current_epsilon():.4f}")
        print()

    print("Smoke test passed.")