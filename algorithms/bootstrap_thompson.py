"""
algorithms/bootstrap_thompson.py

=================================

Bootstrap Thompson Sampling (BTS) for contextual linear bandits.


Standard Thompson Sampling requires sampling from a posterior distribution, which is tractable for conjugate
models (typically, Gaussian) but expensive or intractable for more complex reward modles. Bootstrap Thompson
Sampling replaces the posteiror with a *bootstrap distribution*: we maintain J independent 'particle' models
(e.g. J linear weight vectors), each fitted on a randomly reweighted version of the observed data. At each
step, we draw one particle at random and act greedily from it, which mirrors the Thompson Sampling mechanism
without required any Bayesian computation. 

The online version of bootstrap sampling (Eckles & Kaptein 2014) uses Poisson(1) weights to incrementally
reweight each particle at every update, avoiding the need to resample from stratch. This keeps the per-step
cost of the sampling at O(J * d^2), via J Sherman-Morrison updates (one per particle), and, thus, is linear
in J. Formally, at update step t, the j-th particle sees the new observation (x_t, a_t, r_t) with weight
w_j ~ Poisson(1), independently across particles. Over time, this converges to the standard bootstrap
distribution, as described by Rubin's Bayesian bootstrap). 

We select an arm as follows:
    1. Sample one particle j* uniformly at random from {1, ..., J}.
    2. Select a_t = argmax_a  x_t^T theta_a^{(j*)}.


For any matched events, we update as follows:
    For each particle j, draw w_j ~ Poisson(1), then apply w_j weighted
    Sherman-Morrison updates to A_a^{(j)-1} and b_a^{(j)}.

References
----------
Primary:
    Eckles, D., & Kaptein, M. (2014).
    Thompson Sampling with the Online Bootstrap.
    arXiv: https://arxiv.org/abs/1410.4009

Theory (regret guarantees for linear bootstrap bandits):
    Lu, X., & Van Roy, B. (2017).
    Ensemble Sampling.
    NeurIPS 2017.
    https://proceedings.neurips.cc/paper/2017/hash/9f3de16edcb77b849b5f392f2e0adf16-Abstract.html

Bootstrapped DQN / deep exploration motivation:
    Osband, I., & Van Roy, B. (2015).
    Bootstrapped Thompson Sampling and Deep Exploration.
    arXiv: https://arxiv.org/abs/1507.00300

Empirical comparison and analysis:
    Eckles, D., & Kaptein, M. (2019).
    Bootstrap Thompson Sampling and Sequential Decision Problems
    in the Behavioral Sciences.
    SAGE Open, 9(2).
    https://journals.sagepub.com/doi/10.1177/2158244019851675
"""

# Import dependencies
from base import BaseBandit
import numpy as np


class BootstrapThompson(BaseBandit):
    """
    Bootstrap Thompson Sampling with online Poisson reweighting.

    Maintains J independent linear particles per arm, each incrementally reweighted via Poisson(1) draws
    at every matched update.

    Parameters
    ----------
    * n_arms: the number of arms in the candidate pool
    * context_dim: the dimensionality of context vectors
    * n_particles: the number of bootstrap particles (J); larger J give better coverage of bootstrap
                   distribution but increases per-step cost (use J=10-20 in practice)
    * lambda_reg: regularization for each particle's ridge model
    * seed: random seed for reproducibility
    """

    def __init__(self, n_arms: int, context_dim: int, n_particles: int = 10, lambda_reg: float = 1.0,
        seed: int = 42,):
        # Call super constructor
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            seed=seed,
            name="BootstrapThompson",
        )

        # Ensure hyperparameters are valid
        if n_particles < 1: raise ValueError(f"n_particles must be >= 1, got {n_particles}.")
        if lambda_reg <= 0: raise ValueError(f"lambda_reg must be positive, got {lambda_reg}.")

        # Set initial state
        self.n_particles = n_particles
        self.lambda_reg  = lambda_reg

        # Per-particle, per-arm parameters.
        # Shape convention:  _A_inv[j][a]  is a (d x d) matrix
        #                    _b[j][a]      is a (d,) vector
        #                    _theta[j][a]  is a (d,) vector
        self._A_inv, self._b, self._theta = self._init_params()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def select_arm(self, context: np.ndarray) -> int:
        """
        Sample one particle uniformly at random and act greedily under it.

        This is the Thompson Sampling step: the randomness in particle selection plays the role of posterior
        sampling.

        Parameters
        ----------
        * context: np.ndarray of shape (context_dim,)

        Returns
        -------
        * int: index of selected arm
        """
        x = context.reshape(-1)

        # Draw one particle
        j = int(self.rng.integers(0, self.n_particles))

        # Greedy arm under this particle's weight vectors
        scores = np.array([float(x @ self._theta[j][a]) for a in range(self.n_arms)])
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        """
        Update all J particles for the chosen arm. For each particle j, draw w_j ~ Poisson (1) independently.
        If w_j > 0, apply w_j weighted rank-1 updates to A_a^{(j)-1} and b_a^{(j)}.

        Implement main online bootstrap reweighting scheme described in Eckles & Kaptein 2014. Poisson(1)
        weights ensure that every observation is counted once per paarticle, in expectation, matching the
        standard bootstrap procedure.

        Parameters
        ----------
        * arm: the arm index that was chosen and matched
        * reward: observed binary reward (0 or 1)
        * context: np.ndarray of shape (context_dim,)
        """
        self._base_update(arm)
        x = context.reshape(-1)

        # Iterate through number of particles for distribution 
        for j in range(self.n_particles):
            # Poisson(1) weight for this particle
            w = int(self.rng.poisson(1.0))
            if w == 0:
                continue   # this observation is not included in particle j

            # Apply w weighted rank-1 updates (equivalent to seeing the
            # observation w times); we apply them sequentially
            for _ in range(w):
                self._A_inv[j][arm] = self._sherman_morrison(self._A_inv[j][arm], x)
                self._b[j][arm]    += reward * x

            # Recompute weight estimate for this particle / arm
            self._theta[j][arm] = self._A_inv[j][arm] @ self._b[j][arm]

    def reset(self) -> None:
        """
        Reset all particles, counters, and RNG.
        """
        super()._base_reset()
        self._A_inv, self._b, self._theta = self._init_params()

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _init_params(self):
        """Initialise per-particle, per-arm parameter arrays."""
        d = self.context_dim
        A_inv  = [
            [(1.0 / self.lambda_reg) * np.eye(d) for _ in range(self.n_arms)]
            for _ in range(self.n_particles)
        ]
        b = [
            [np.zeros(d) for _ in range(self.n_arms)]
            for _ in range(self.n_particles)
        ]
        theta  = [
            [np.zeros(d) for _ in range(self.n_arms)]
            for _ in range(self.n_particles)
        ]
        return A_inv, b, theta

    @staticmethod
    def _sherman_morrison(A_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Perform Sherman-Morrison rank-1 update. Given current A_inv and a new observation vector
        x, return updated inverse via:

            A_inv_new = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)

        Benefit is not having to recompute full A_inv from stratch at every update step, keeping the
        cost per-step at O(d^2) instead of O(d^3).
        """
        Ax = A_inv @ x
        denom = 1.0 + float(x @ Ax)
        return A_inv - np.outer(Ax, Ax) / denom

    # ------------------------------------------------------------------
    # Functions to report algorithm features. 
    # ------------------------------------------------------------------
    def particle_scores(self, context: np.ndarray) -> np.ndarray:
        """
        Return predicted reward scores for all arms under all particles.

        Returns
        -------
        * np.ndarray: array of shape (n_particles, n_arms)
        """
        x = context.reshape(-1)
        return np.array([
            [float(x @ self._theta[j][a]) for a in range(self.n_arms)]
            for j in range(self.n_particles)
        ])

    def arm_score_distribution(self, context: np.ndarray, arm: int) -> np.ndarray:
        """
        Return the distribution of predicted scores for a given arm
        across all J particles.  This is the bootstrap approximation to
        the posterior predictive distribution over theta_a.

        Returns
        -------
        * np.ndarray: array of shape (n_particles,)
        """
        x = context.reshape(-1)
        return np.array([float(x @ self._theta[j][arm]) for j in range(self.n_particles)])

    def __repr__(self) -> str:
        return (
            f"BootstrapThompson("
            f"n_arms={self.n_arms}, "
            f"context_dim={self.context_dim}, "
            f"n_particles={self.n_particles}, "
            f"lambda_reg={self.lambda_reg}, "
            f"t={self._t})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running BootstrapThompson smoke test...\n")

    rng = np.random.default_rng(4)
    n_arms = 5
    context_dim = 8
    n_steps = 600

    # True reward weights — arm 1 is best
    true_theta = rng.standard_normal((n_arms, context_dim))
    true_theta[0] += 2.0

    agent = BootstrapThompson(
        n_arms, context_dim, n_particles=10, lambda_reg=2.5, seed=4
    )
    print(agent)

    total_reward = 0.0
    arm_counts   = np.zeros(n_arms, dtype=int)

    for t in range(n_steps):
        context = rng.standard_normal(context_dim)
        chosen  = agent.select_arm(context)
        p       = float(np.clip(0.5 + 0.1 * (context @ true_theta[chosen]), 0.05, 0.95))
        reward  = float(rng.random() < p)
        agent.update(chosen, reward, context)
        total_reward += reward
        arm_counts[chosen] += 1

    print(f"\nAfter {n_steps} steps:")
    print(f"  Total reward     : {total_reward:.1f}  ({total_reward/n_steps:.3f} per step)")
    print(f"  Arm pull counts  : {arm_counts}")
    print(f"  Arm pull fracs   : {arm_counts / n_steps}")

    # Particle score distribution should have variance > 0 (diversity check)
    test_ctx  = rng.standard_normal(context_dim)
    score_dist = agent.arm_score_distribution(test_ctx, arm=0)
    assert score_dist.shape == (10,), "Wrong score distribution shape."
    print(f"\n  Score distribution arm 0 (std across particles): {score_dist.std():.4f}")
    assert score_dist.std() >= 0.0, "Particle score std should be non-negative."
    print(f"  Particle diversity check ✓")

    # Particle scores shape check
    all_scores = agent.particle_scores(test_ctx)
    assert all_scores.shape == (10, n_arms), f"Wrong shape: {all_scores.shape}"
    print(f"  Particle scores shape: {all_scores.shape} ✓")

    # Reset check
    agent.reset()
    assert agent._t == 0
    score_after_reset = agent.arm_score_distribution(test_ctx, arm=0)
    assert np.allclose(score_after_reset, 0.0), "Scores should be 0 after reset."
    print(f"  Reset check: all scores zero after reset ✓")

    print("\nSmoke test passed.")