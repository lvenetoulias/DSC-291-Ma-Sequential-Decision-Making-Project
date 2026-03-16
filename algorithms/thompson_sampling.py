"""
algorithms/thompson_sampling.py
================================
Contextual Thompson Sampling with a linear Gaussian reward model.

The reward model is:
    r_t = x_{t,a}^T theta + xi_t,   xi_t ~ N(0, sigma^2)

with a Gaussian prior over the parameter vector:
    theta ~ N(0, lambda^{-1} I)

Given observed data D_{t-1}, Bayes' rule yields a Gaussian posterior:

    theta | D_{t-1}  ~  N(theta_hat_t,  sigma^2 * V_t^{-1})

where:
    V_t       = lambda * I + sum_{s<t} x_s x_s^T    (design matrix, same as LinUCB)
    theta_hat = V_t^{-1} b_t                        (posterior mean)
    b_t       = sum_{s<t} r_s x_s                   (reward-weighted contexts)

At each step, Thompson Sampling:
    1. Samples  theta_tilde ~ N(theta_hat_t,  sigma^2 * V_t^{-1})  per arm.
    2. Selects  a_t = argmax_a  x_{t,a}^T theta_tilde_a            (eq. 3.2.3)

Exploration arises naturally: high posterior variance (scarce data) leads
to diverse samples and broad exploration; as data accumulates the posterior
concentrates and the algorithm exploits more aggressively.

Design note
-----------
Like LinUCB, this is a *disjoint* model — each arm has its own independent
posterior.  V_t^{-1} is maintained via Sherman-Morrison rank-1 updates
(same helper as LinUCB) to keep per-step cost at O(d^2).
"""

import numpy as np
from base import BaseBandit


class ThompsonSampling(BaseBandit):
    """
    Disjoint contextual Thompson Sampling with Gaussian posterior.

    Parameters
    ----------
    n_arms      : int   — number of arms in the pool
    context_dim : int   — dimensionality of context vectors
    sigma       : float — assumed noise standard deviation of the reward
                          model; scales the posterior covariance as sigma^2 * V^{-1};
                          larger values inflate uncertainty and increase exploration
    lambda_reg  : float — prior precision (lambda in the report); initialises
                          V_a = lambda * I for each arm; larger values impose
                          stronger shrinkage toward zero
    seed        : int   — random seed (used for posterior sampling)
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        sigma: float = 1.0,
        lambda_reg: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            seed=seed,
            name="ThompsonSampling",
        )

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        if lambda_reg <= 0:
            raise ValueError(f"lambda_reg must be positive, got {lambda_reg}.")

        self.sigma      = sigma
        self.sigma2     = sigma ** 2
        self.lambda_reg = lambda_reg

        # Per-arm posterior parameters (disjoint model).
        # V_inv[a] : (d x d) inverse design matrix  — posterior covariance is sigma^2 * V_inv[a]
        # b[a]     : (d,)    reward-weighted context accumulator
        # mu[a]    : (d,)    posterior mean = V_inv[a] @ b[a]   (= theta_hat_a)
        self._V_inv, self._b, self._mu = self._init_params()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def select_arm(self, context: np.ndarray, candidate_arms: list = None) -> int:
        """
        Sample a weight vector from each arm's posterior and select the arm
        with the highest predicted reward under its sample.

        For arm a:
            theta_tilde_a ~ N(mu_a,  sigma^2 * V_inv_a)
            score_a        = x^T theta_tilde_a

        Returns the arm with the highest score.

        Parameters
        ----------
        context : np.ndarray of shape (context_dim,)

        Returns
        -------
        arm_idx : int
        """
        x = context.reshape(-1)
        arms_to_score = candidate_arms if candidate_arms is not None else list(range(self.n_arms))
        scores = np.array([self._sample_score(a, x) for a in arms_to_score])
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        """
        Perform a Bayesian conjugate update to the posterior for the chosen arm.

        V_a   <- V_a + x x^T      (accumulated via Sherman-Morrison on V_inv)
        b_a   <- b_a + r * x
        mu_a  <- V_a^{-1} @ b_a   (recompute posterior mean)

        Parameters
        ----------
        arm     : int   — arm index that was chosen and matched
        reward  : float — observed binary reward (0 or 1)
        context : np.ndarray of shape (context_dim,)
        """
        self._base_update(arm)
        x = context.reshape(-1)

        # Rank-1 update of V_inv via Sherman-Morrison
        self._V_inv[arm] = self._sherman_morrison(self._V_inv[arm], x)

        # Accumulate reward signal
        self._b[arm] += reward * x

        # Recompute posterior mean
        self._mu[arm] = self._V_inv[arm] @ self._b[arm]

    def reset(self) -> None:
        """Reset all internal state to initial conditions and re-seed RNG."""
        super()._base_reset()
        self._V_inv, self._b, self._mu = self._init_params()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_params(self):
        """Initialise per-arm posterior parameter arrays."""
        d     = self.context_dim
        V_inv = [(1.0 / self.lambda_reg) * np.eye(d) for _ in range(self.n_arms)]
        b     = [np.zeros(d) for _ in range(self.n_arms)]
        mu    = [np.zeros(d) for _ in range(self.n_arms)]
        return V_inv, b, mu

    def _sample_weights(self, arm: int) -> np.ndarray:
        """
        Draw one sample from the posterior distribution for arm a:

            theta_tilde ~ N(mu_a,  sigma^2 * V_inv_a)

        Uses a Cholesky decomposition of the covariance matrix for
        numerically stable sampling.  If Cholesky fails (rare, due to
        floating-point near-singularity), falls back to eigendecomposition.

        Returns
        -------
        theta_tilde : np.ndarray of shape (context_dim,)
        """
        mu    = self._mu[arm]
        cov   = self.sigma2 * self._V_inv[arm]

        try:
            L     = np.linalg.cholesky(cov)
            z     = self.rng.standard_normal(self.context_dim)
            return mu + L @ z
        except np.linalg.LinAlgError:
            # Fallback: add a small jitter and retry
            cov_jittered = cov + 1e-8 * np.eye(self.context_dim)
            return self.rng.multivariate_normal(mu, cov_jittered)

    def _sample_score(self, arm: int, x: np.ndarray) -> float:
        """
        Sample a weight vector for arm a and return the predicted reward
        for context x under that sample: score = x^T theta_tilde_a.
        """
        theta_tilde = self._sample_weights(arm)
        return float(x @ theta_tilde)

    @staticmethod
    def _sherman_morrison(V_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Rank-1 Sherman-Morrison update.

        Updates V_inv to reflect V <- V + x x^T:

            V_inv_new = V_inv - (V_inv x x^T V_inv) / (1 + x^T V_inv x)
        """
        Vx    = V_inv @ x
        denom = 1.0 + float(x @ Vx)
        return V_inv - np.outer(Vx, Vx) / denom

    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------

    def posterior_mean(self, arm: int) -> np.ndarray:
        """Return a copy of the posterior mean (theta_hat) for a given arm."""
        return self._mu[arm].copy()

    def posterior_variance(self, arm: int, context: np.ndarray) -> float:
        """
        Return the marginal posterior variance of the predicted reward
        for arm a at context x:

            Var[x^T theta | D] = sigma^2 * x^T V_inv_a x

        This is the same quantity LinUCB uses for its exploration bonus,
        making it straightforward to compare exploration behaviour.
        """
        x = context.reshape(-1)
        return self.sigma2 * float(x @ self._V_inv[arm] @ x)

    def sample_all_weights(self) -> list[np.ndarray]:
        """
        Draw one weight sample per arm from their respective posteriors.
        Returns a list of n_arms arrays, each of shape (context_dim,).
        Useful for visualising posterior spread across arms.
        """
        return [self._sample_weights(a) for a in range(self.n_arms)]

    def __repr__(self) -> str:
        return (
            f"ThompsonSampling("
            f"n_arms={self.n_arms}, "
            f"context_dim={self.context_dim}, "
            f"sigma={self.sigma}, "
            f"lambda_reg={self.lambda_reg}, "
            f"t={self._t})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running ThompsonSampling smoke test...\n")

    rng = np.random.default_rng(1)
    n_arms      = 5
    context_dim = 8
    n_steps     = 1000

    # True reward weights per arm — arm 3 is best (same setup as LinUCB test)
    true_theta = rng.standard_normal((n_arms, context_dim))
    true_theta[3] += 1.5

    agent = ThompsonSampling(n_arms, context_dim, sigma=1.0, lambda_reg=1.0, seed=1)
    print(agent)

    total_reward   = 0.0
    arm_counts     = np.zeros(n_arms, dtype=int)

    for t in range(n_steps):
        context = rng.standard_normal(context_dim)
        chosen  = agent.select_arm(context)

        p      = float(np.clip(0.5 + 0.1 * (context @ true_theta[chosen]), 0.05, 0.95))
        reward = float(rng.random() < p)

        agent.update(chosen, reward, context)
        total_reward += reward
        arm_counts[chosen] += 1

    print(f"\nAfter {n_steps} steps:")
    print(f"  Total reward     : {total_reward:.1f}  ({total_reward/n_steps:.3f} per step)")
    print(f"  Arm pull counts  : {arm_counts}")
    print(f"  Arm pull fracs   : {arm_counts / n_steps}")

    # Check that posterior variance shrinks with more observations
    test_ctx = rng.standard_normal(context_dim)
    var_before = agent.posterior_variance(0, test_ctx)
    for _ in range(300):
        agent.update(0, 1.0, rng.standard_normal(context_dim))
    var_after = agent.posterior_variance(0, test_ctx)

    print(f"\n  Posterior variance arm 0 before extra updates : {var_before:.6f}")
    print(f"  Posterior variance arm 0 after {n_steps} updates   : {var_after:.6f}")
    assert var_after < var_before, "Posterior variance should shrink with more data."

    # Check Cholesky sampling returns correct shape
    sample = agent._sample_weights(0)
    assert sample.shape == (context_dim,), f"Expected shape ({context_dim},), got {sample.shape}"
    print(f"\n  Sampled weight vector shape : {sample.shape}")

    # Check reset restores initial state
    agent.reset()
    assert agent._t == 0, "Step counter should be 0 after reset."
    var_reset = agent.posterior_variance(0, test_ctx)
    assert var_reset > var_after, "Posterior variance should be larger after reset."
    print(f"  Posterior variance after reset : {var_reset:.6f} (restored)")

    print("\nSmoke test passed.")