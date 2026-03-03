"""
algorithms/lin_ucb.py

====================

Disjoint Linear Upper Confidence Bound (LinUCB) algorithm.

LinUCB maintains a separate ridge regression model per arm and then selects actions by maximizing an upper
confidence bound on the predicted reward. The UCB score for an arm a at a time t is given by:

    UCB(a, x) = x^T θ_hat_a    +  alpha * sqrt(x^T A_a^{-1} x)
               |_____________|   |___________________________|
               predicted reward        exploration bonus

where:
    A_a     = lambda * I + sum_{matched t, arm=a} x_t x_t^T   (design matrix)
    b_a     = sum_{matched t, arm=a} r_t x_t                  (reward vector)
    θ_hat_a = A_a^{-1} b_a                                    (weight estimate)
    alpha   = sigma * sqrt(beta_t)                             (exploration scalar)

Benefit of exploration is larger when x lies in a direction that A_a^{-1} has not been compressed (i.e.,
not properly explored previously). This gives LinUCB its 'optimism in the face of uncertainty' property
specific to UCB.

Reference
---------
Li, L., Chu, W., Langford, J., & Schapire, R. (2010).
A contextual-bandit approach to personalized news article recommendation.
WWW 2010.  https://doi.org/10.1145/1772690.1772758
(Equations 3.1.1 – 3.1.5 in the Methods Report.)
"""


# Import dependencies
from algorithms.base import BaseBandit
import numpy as np


class LinUCB(BaseBandit):
    """
    Disjoint LinUCB with Sherman-Morrison rank-1 inverse updates.

    Parameters
    ----------
    n_arms      : int   — number of arms in the pool
    context_dim : int   — dimensionality of context vectors
    alpha       : float — exploration scalar (sigma * sqrt(beta_t) collapsed
                          into a single tunable parameter); larger values
                          encourage more exploration
    lambda_reg  : float — L2 regularisation strength (lambda in the report);
                          initialises each arm's design matrix as lambda * I
    seed        : int   — random seed for tie-breaking
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            seed=seed,
            name="LinUCB",
        )

        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if lambda_reg <= 0:
            raise ValueError(f"lambda_reg must be positive, got {lambda_reg}.")

        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # Per-arm parameters (disjoint model).
        # A_inv[a] : (d x d) inverse design matrix, initialised to (1/lambda) I
        # b[a]     : (d,)    reward-weighted context accumulator
        # theta[a] : (d,)    current weight estimate A_inv[a] @ b[a]
        self._A_inv, self._b, self._theta = self._init_params()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select the arm with the highest UCB score.

        UCB(a, x) = x^T θ_hat_a  +  alpha * sqrt(x^T A_a^{-1} x)

        Ties (rare in practice) are broken by random selection to avoid
        systematic bias toward low-index arms.

        Parameters
        ----------
        context : np.ndarray of shape (context_dim,)

        Returns
        -------
        arm_idx : int
        """
        x = context.reshape(-1)
        scores = np.array([self._compute_ucb(a, x) for a in range(self.n_arms)])
        max_score = np.max(scores)
        best_arms = np.where(scores == max_score)[0]
        return int(self.rng.choice(best_arms))

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        """
        Update the design matrix and reward vector for the chosen arm,
        then recompute its weight estimate.

        Uses the Sherman-Morrison rank-1 formula to update A_inv in O(d^2)
        rather than recomputing the full inverse in O(d^3).

        Parameters
        ----------
        arm     : int   — arm index that was chosen and matched
        reward  : float — observed binary reward (0 or 1)
        context : np.ndarray of shape (context_dim,)
        """
        self._base_update(arm)
        x = context.reshape(-1)

        # Rank-1 update: A_a <- A_a + x x^T  =>  A_inv via Sherman-Morrison
        self._A_inv[arm] = self._sherman_morrison(self._A_inv[arm], x)

        # b_a <- b_a + r * x
        self._b[arm] += reward * x

        # Recompute weight estimate for this arm
        self._theta[arm] = self._A_inv[arm] @ self._b[arm]

    def reset(self) -> None:
        """Reset all internal state to initial conditions and re-seed RNG."""
        super()._base_reset()
        self._A_inv, self._b, self._theta = self._init_params()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_params(self):
        """Initialise per-arm parameter arrays."""
        d = self.context_dim
        A_inv = [(1.0 / self.lambda_reg) * np.eye(d) for _ in range(self.n_arms)]
        b     = [np.zeros(d) for _ in range(self.n_arms)]
        theta = [np.zeros(d) for _ in range(self.n_arms)]
        return A_inv, b, theta

    def _compute_ucb(self, arm: int, x: np.ndarray) -> float:
        """
        Compute the UCB score for a single arm given context vector x.

            UCB = x^T θ_hat_a  +  alpha * sqrt(x^T A_a^{-1} x)

        The second term is the ellipsoidal uncertainty (bonus) in the
        direction of x under the current posterior geometry of arm a.
        """
        theta = self._theta[arm]
        A_inv = self._A_inv[arm]

        exploit = float(x @ theta)
        bonus   = self.alpha * float(np.sqrt(x @ A_inv @ x))
        return exploit + bonus

    @staticmethod
    def _sherman_morrison(A_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Rank-1 Sherman-Morrison update.

        Updates A_inv to reflect A <- A + x x^T:

            A_inv_new = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)

        Cost: O(d^2) matrix-vector products instead of O(d^3) inversion.
        """
        Ax    = A_inv @ x
        denom = 1.0 + float(x @ Ax)
        return A_inv - np.outer(Ax, Ax) / denom

    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------

    def compute_ucb_scores(self, context: np.ndarray) -> np.ndarray:
        """
        Return UCB scores for all arms given a context vector.
        Useful for debugging and visualisation.

        Returns
        -------
        np.ndarray of shape (n_arms,)
        """
        x = context.reshape(-1)
        return np.array([self._compute_ucb(a, x) for a in range(self.n_arms)])

    def exploration_bonus(self, arm: int, context: np.ndarray) -> float:
        """
        Return only the exploration bonus term for a given arm and context.
        Useful for tracking how uncertainty evolves over time.
        """
        x = context.reshape(-1)
        return self.alpha * float(np.sqrt(x @ self._A_inv[arm] @ x))

    def get_theta(self, arm: int) -> np.ndarray:
        """Return a copy of the weight vector for a given arm."""
        return self._theta[arm].copy()

    def __repr__(self) -> str:
        return (
            f"LinUCB("
            f"n_arms={self.n_arms}, "
            f"context_dim={self.context_dim}, "
            f"alpha={self.alpha}, "
            f"lambda_reg={self.lambda_reg}, "
            f"t={self._t})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("Running LinUCB smoke test...\n")

    rng = np.random.default_rng(0)
    n_arms      = 5
    context_dim = 8
    n_steps     = 600

    # True reward weights per arm — arm 3 is best in this synthetic setting
    true_theta = rng.standard_normal((n_arms, context_dim))
    true_theta[3] += 1.5   # make arm 3 clearly better on average

    agent = LinUCB(n_arms, context_dim, alpha=1.0, lambda_reg=1.0, seed=0)
    print(agent)

    total_reward   = 0.0
    exploit_counts = np.zeros(n_arms, dtype=int)

    for t in range(n_steps):
        context = rng.standard_normal(context_dim)

        # Oracle best arm for this context
        oracle_arm = int(np.argmax([context @ true_theta[a] for a in range(n_arms)]))

        chosen = agent.select_arm(context)

        # Simulate reward: linear model + Bernoulli noise
        p = float(np.clip(0.5 + 0.1 * (context @ true_theta[chosen]), 0.05, 0.95))
        reward = float(rng.random() < p)

        agent.update(chosen, reward, context)
        total_reward += reward
        exploit_counts[chosen] += 1

    print(f"\nAfter {n_steps} steps:")
    print(f"  Total reward     : {total_reward:.1f}  ({total_reward/n_steps:.3f} per step)")
    print(f"  Arm pull counts  : {exploit_counts}")
    print(f"  Arm pull fracs   : {exploit_counts / n_steps}")
    print()

    # Check that exploration bonuses shrink with more data
    test_ctx = rng.standard_normal(context_dim)
    bonuses_before = agent.exploration_bonus(0, test_ctx)
    for _ in range(200):
        agent.update(0, 1.0, rng.standard_normal(context_dim))
    bonuses_after = agent.exploration_bonus(0, test_ctx)

    print(f"  Exploration bonus arm 0 before extra updates : {bonuses_before:.4f}")
    print(f"  Exploration bonus arm 0 after  200 updates   : {bonuses_after:.4f}")
    assert bonuses_after < bonuses_before, "Bonus should shrink as arm is observed more."
    print("\nSmoke test passed.")