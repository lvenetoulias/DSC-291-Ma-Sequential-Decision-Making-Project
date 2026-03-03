"""
algorithms/logistic_ucb.py

==========================

Logistic UCB (LogUCB) for contextual bandits


Unlike LinUCB, which models rewards as linear with respect to the context vector x, LogUCB, corresponding
to logistic UCB, models the reward probability via a logistic (sigmoid) function:

    P(r = 1 | x, a) = sigma(x^T theta_a),   sigma(z) = 1 / (1 + exp(-z))

The use of the logistic function is a more natural model for binary rewards (e.g. clicks versus no-click,
like versus no-like) since it respects the constraint on interval [0,1] on probabilities, by construction.
The parameter vector theta_a is estimated via online logisitic regression (formally by gradient descent
on the log-likelihood), and a UCB-styled bonus for worthwhile exploration is applied using the Fisher
information matrix as an estimate for the local curvature of the reward function. 

The UCB score for an arm a at time t is given by:

    UCB(a, x) = sigma(x^T theta_a)  +  alpha * sqrt(x^T H_a^{-1} x),

where H_a is the accumulated (regularised) Hessian of the negative log-likelihood (i.e. the empirical
Fisher information matrix):

    H_a = lambda * I  +  sum_{s : a_s = a} sigma'(x_s^T theta_a) x_s x_s^T

and sigma'(z) = sigma(z)(1 - sigma(z)) is the derivative of the sigmoid, which acts as an adaptive, data-
dependent weight on each outer product.

This LogUCB implementation uses online Newton-step update (i.e., one gradient step with H_a^{-1} as the
preconditioner), which is the computationally efficient variant described in Faury et al. (2020). The
inverse Hessian matrix H_a^{-1} is maintained via Sherman-Morrison rank-1 updates, as was the case with
the epsilon-Greedy algorithm. 


References
----------
Primary:
    Faury, L., Abeille, M., Calauzenes, C., & Fercoq, O. (2020).
    Improved Optimistic Algorithms for Logistic Bandits.
    Proceedings of the 37th ICML, PMLR 119:3052-3060.
    https://proceedings.mlr.press/v119/faury20a.html
    arXiv: https://arxiv.org/abs/2002.07530

Supplementary (efficient / jointly optimal variant):
    Faury, L., Abeille, M., Calauzenes, C., & Jun, K.-S. (2022).
    Jointly Efficient and Optimal Algorithms for Logistic Bandits.
    AISTATS 2022, PMLR 151.
    arXiv: https://arxiv.org/abs/2201.01985

Background (GLM bandits, generalises logistic setting):
    Filippi, S., Cappe, O., Garivier, A., & Szepesvári, C. (2010).
    Parametric Bandits: The Generalized Linear Case.
    NeurIPS 2010.
    https://proceedings.neurips.cc/paper/2010/hash/c2626d850c80ea07e7511bbae4c76f4b-Abstract.html
"""


# Import dependencies
from base import BaseBandit
import numpy as np


class LogisticUCB(BaseBandit):
    """
    Disjoint Logistic UCB with online Newton-step updates.

    Parameters
    ----------
    * n_arms: the number of arms in the pool
    * context_dim: the dimensionality of context vectors
    * alpha: exploration scalar controlling the UCB bonus width; larger values encourage more exploration
    * lambda_reg: regularization strength; initialises the Hessian approximation as lambda * I for each arm
    * learning_rate: step size for the online Newton update of theta; set to 1.0 for a full Newton step
    * clip: clips the logit x^T theta to [-clip, clip] before applying sigmoid, preventing numerical overflow
    * seed: random seed for tie-breaking
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0, lambda_reg: float = 1.0,
        learning_rate: float = 1.0, clip: float = 10.0, seed: int = 42,):
        # Call superconstructor
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            seed=seed,
            name="LogisticUCB",
        )

        # Ensure hyperparameters are non-negative
        if alpha < 0: raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if lambda_reg <= 0: raise ValueError(f"lambda_reg must be positive, got {lambda_reg}.")
        if learning_rate <= 0: raise ValueError(f"learning_rate must be positive, got {learning_rate}.")

        # Establish initial state
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.clip = clip

        # Per-arm parameters.
        # theta[a]  : (d,)    logistic regression weight vector
        # H_inv[a]  : (d x d) inverse of the accumulated (regularised) Hessian = inverse Fisher information matrix
        self._theta, self._H_inv = self._init_params()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select the arm with the highest logistic UCB score.

        UCB(a, x) = sigma(x^T theta_a)  +  alpha * sqrt(x^T H_a^{-1} x)

        The first term is the predicted reward probability; the second is the exploration bonus scaled by
        the inverse Fisher information in the direction of x.

        Parameters
        ----------
        * context: np.ndarray of shape (context_dim,)

        Returns
        -------
        * int: the index of the selected arm
        """
        x = context.reshape(-1)
        scores = np.array([self._compute_ucb(a, x) for a in range(self.n_arms)])
        max_score = np.max(scores)
        best_arms = np.where(scores == max_score)[0]
        return int(self.rng.choice(best_arms))

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        """
        Update theta_a via one online Newton step and update H_a^{-1}
        via a Sherman-Morrison rank-1 update.

        Newton step:
            theta_a <- theta_a + lr * H_a^{-1} * (r - sigma(x^T theta_a)) * x

        Hessian update (rank-1, corresponds to adding sigma'(x^T theta_a) * xx^T):
            H_a^{-1} <- Sherman-Morrison(H_a^{-1}, sqrt(sigma'_a) * x)

        Note: the Hessian is updated *after* the theta step so the exploration bonus reflects post-update
        uncertainty on subsequent steps.

        Parameters
        ----------
        * arm: the index of the arm that was chosen and matched
        * reward: observed binary reward (0 or 1)
        * context: np.ndarray of shape (context_dim,)
        """
        self._base_update(arm)
        x = context.reshape(-1)

        # Current predicted probability
        p = self._sigmoid(float(x @ self._theta[arm]))

        # Newton gradient step: H_inv * grad_log_lik
        grad = (reward - p) * x                              # (r - p) * x
        self._theta[arm] += self.learning_rate * (self._H_inv[arm] @ grad)

        # Hessian update: H <- H + sigma'(z) * x x^T
        # sigma'(z) = p(1-p);  we use the UPDATED p for slightly better stability
        p_new  = self._sigmoid(float(x @ self._theta[arm]))
        dprime = p_new * (1.0 - p_new)                      # sigmoid derivative

        if dprime > 1e-10:                                   # skip if saturated
            # Scale x by sqrt(dprime) so rank-1 update is: H += (sqrt(d)*x)(sqrt(d)*x)^T
            x_scaled = np.sqrt(dprime) * x
            self._H_inv[arm] = self._sherman_morrison(self._H_inv[arm], x_scaled)

    def reset(self) -> None:
        """
        Reset all internal state to initial conditions and re-seed RNG.
        """
        super()._base_reset()
        self._theta, self._H_inv = self._init_params()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _init_params(self):
        """Initialise per-arm parameter arrays."""
        d = self.context_dim
        theta = [np.zeros(d) for _ in range(self.n_arms)]
        H_inv = [(1.0 / self.lambda_reg) * np.eye(d) for _ in range(self.n_arms)]
        return theta, H_inv

    def _sigmoid(self, z: float) -> float:
        """Numerically stable sigmoid with clipping."""
        z = float(np.clip(z, -self.clip, self.clip))
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_ucb(self, arm: int, x: np.ndarray) -> float:
        """
        UCB score for a single arm: sigma(x^T theta_a)  +  alpha * sqrt(x^T H_a^{-1} x)
        """
        logit = float(x @ self._theta[arm])
        exploit = self._sigmoid(logit)
        bonus = self.alpha * float(np.sqrt(np.clip(x @ self._H_inv[arm] @ x, 0, None)))
        return exploit + bonus

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
    def predicted_prob(self, arm: int, context: np.ndarray) -> float:
        """Return the predicted reward probability for arm a given context x."""
        return self._sigmoid(float(context.reshape(-1) @ self._theta[arm]))

    def exploration_bonus(self, arm: int, context: np.ndarray) -> float:
        """Return only the exploration bonus term for arm a given context x."""
        x = context.reshape(-1)
        return self.alpha * float(np.sqrt(np.clip(x @ self._H_inv[arm] @ x, 0, None)))

    def get_theta(self, arm: int) -> np.ndarray:
        """Return a copy of the logistic weight vector for arm a."""
        return self._theta[arm].copy()

    def __repr__(self) -> str:
        return (
            f"LogisticUCB("
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
    print("Running LogisticUCB smoke test...\n")

    rng = np.random.default_rng(2)
    n_arms = 10
    context_dim = 8
    n_steps = 1000

    # True logistic weights per arm — arm 2 is best
    true_theta = rng.standard_normal((n_arms, context_dim))
    true_theta[2] += 1.5

    agent = LogisticUCB(n_arms, context_dim, alpha=1.0, lambda_reg=2.0, seed=42)
    print(agent)

    total_reward = 0.0
    arm_counts = np.zeros(n_arms, dtype=int)

    for t in range(n_steps):
        context = rng.standard_normal(context_dim)
        chosen = agent.select_arm(context)

        # Binary reward via logistic model
        p = 1.0 / (1.0 + np.exp(-float(context @ true_theta[chosen])))
        reward = float(rng.random() < p)

        agent.update(chosen, reward, context)
        total_reward += reward
        arm_counts[chosen] += 1

    print(f"\nAfter {n_steps} steps:")
    print(f"  Total reward     : {total_reward:.1f}  ({total_reward/n_steps:.3f} per step)")
    print(f"  Arm pull counts  : {arm_counts}")
    print(f"  Arm pull fracs   : {arm_counts / n_steps}")

    # Predicted probs should be in [0, 1]
    test_ctx = rng.standard_normal(context_dim)
    for a in range(n_arms):
        p = agent.predicted_prob(a, test_ctx)
        assert 0.0 <= p <= 1.0, f"Predicted prob out of range: {p}"
    print(f"\n  All predicted probabilities in [0,1]")

    # Exploration bonus should decrease after many arm-0 updates
    bonus_before = agent.exploration_bonus(0, test_ctx)
    for _ in range(200):
        agent.update(0, 1.0, rng.standard_normal(context_dim))
    bonus_after = agent.exploration_bonus(0, test_ctx)
    assert bonus_after < bonus_before, "Bonus should shrink with more data."
    print(f"  Exploration bonus arm 0 before/after 200 updates: "
          f"{bonus_before:.4f} → {bonus_after:.4f}")

    print("\nSmoke test passed.")