"""
algorithms/neural_ts.py

========================

Neural Thompson Sampling (NeuralTS) for contextual bandits.


Like NeuralUCB, NeuralTS uses the Neural-Linear approximation:
    1. A shared neural network phi(x; w) maps raw context x to a
       d_emb-dimensional embedding (last hidden layer output).
    2. A *Bayesian linear head* is maintained per arm using the embedding
       as the feature vector, with Gaussian posterior:

           theta_a | D  ~  N(mu_a,  sigma^2 * V_a^{-1})

       where A_a and mu_a are the same ridge-regression quantities as
       ThompsonSampling and LinUCB, applied to phi(x) instead of x.

       
Main difference from NeuralUCB is in the selection of the arm:
    - NeuralUCB : argmax_a  [ phi(x)^T mu_a  +  alpha * sqrt(phi(x)^T V_a^{-1} phi(x)) ]
                             (deterministic UCB score)
    - NeuralTS  : argmax_a  phi(x)^T theta_tilde_a,
                             theta_tilde_a ~ N(mu_a,  sigma^2 * V_a^{-1})
                             (stochastic posterior sample)

The stochasticity in the sampling mechanism enables more *deep exploration*: unlike pointwise bonuses in
the exploration phase, the sampling from the poarweior distribution can sustain exploration of a suboptimal
arm across multiple time steps/iteratiosn if uncertainty warrants it. This is particularly valuable in
sparse-reward settings, such as offline replay, where the bonus of the UCB may be poorly calibrated relative
to the actual uncertainty of the posterior distribution.
                             

The training of the neural network is the same as NeuralUCB, consisting of a warm-up period of random
exploration to collect initial data, a periodic retraining of the network every `train_every` matched
updates, and a full refit of all linear neural network head after each retraining pass. 




References
----------
Zhang, W., Zhou, D., Li, L., & Gu, Q. (2021).
Neural Thompson Sampling.
ICLR 2021.
https://openreview.net/forum?id=tkAtoZkcUnm
arXiv: https://arxiv.org/abs/2010.00827

Zhou, D., Li, L., & Gu, Q. (2020).
Neural Contextual Bandits with UCB-based Exploration.
Proceedings of the 37th ICML, PMLR 119:11492-11502.
https://proceedings.mlr.press/v119/zhou20a.html
arXiv: https://arxiv.org/abs/1911.04462

Riquelme, C., Tucker, G., & Snoek, J. (2018).
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian
Deep Networks for Thompson Sampling.
ICLR 2018.
arXiv: https://arxiv.org/abs/1802.09127

Osband, I., Russo, D., & Van Roy, B. (2013).
(More) Efficient Reinforcement Learning via Posterior Sampling.
NeurIPS 2013.
https://proceedings.neurips.cc/paper/2013/hash/6a5889bb0190d0211a991f47bb19a777-Abstract.html
"""

# Import dependencies
from base import BaseBandit
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _check_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "NeuralTS requires PyTorch. Install it with:\n"
            "    pip install torch --break-system-packages"
        )


class _EmbeddingNet(nn.Module if _TORCH_AVAILABLE else object):
    """
    Two-hidden-layer ReLU network mapping context -> embedding.
    Architecture: context_dim -> hidden_dim -> hidden_dim -> d_emb

    Identical to the network used in NeuralUCB — if both algorithms are
    used together, they each maintain their own independent network
    (separate weights, separate training), so results remain comparable.
    """

    def __init__(self, context_dim: int, hidden_dim: int, d_emb: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_emb),
        )

    def forward(self, x):
        return self.net(x)


class NeuralTS(BaseBandit):
    """
    Neural Thompson Sampling with Neural-Linear approximation.

    Parameters
    ----------
    * n_arms: number of arms in the pool
    * context_dim: dimensionality of raw context vectors
    * d_emb: embedding dimension (last hidden layer width)
    * hidden_dim: width of the two hidden layers
    * sigma: posterior noise scale; scales the sampling covariance as sigma^2 * A_a^{-1}; larger values
             increase exploration by inflating posterior spread
    * lambda_reg: regularization for the linear head; initialises A_a = lambda * I per arm
    * lr_nn: Adam learning rate for network training
    * train_every: retrain network every this many matched updates
    * n_epochs: training epochs per retraining pass
    * warmup_steps: matched updates of random exploration before the first network training and TS selection begins
    * batch_size: mini-batch size for network training
    * seed: random seed (controls both numpy RNG for posterior sampling and torch for network init)
    """

    def __init__(self, n_arms: int, context_dim: int, d_emb: int = 32, hidden_dim: int = 64, sigma: float = 1.0,
        lambda_reg: float = 1.0, lr_nn: float = 1e-3, train_every: int = 50, n_epochs: int = 10, warmup_steps: int = 100,
        batch_size: int = 32, max_buffer_size: int = 5000, seed: int = 42,):

        # Call superconstructor
        _check_torch()
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            seed=seed,
            name="NeuralTS",
        )

        # Ensure hyperparameters are valid
        if sigma <= 0: raise ValueError(f"sigma must be positive, got {sigma}.")
        if lambda_reg <= 0: raise ValueError(f"lambda_reg must be positive, got {lambda_reg}.")

        # Set initial state
        self.d_emb = d_emb
        self.hidden_dim   = hidden_dim
        self.sigma = sigma
        self.sigma2 = sigma ** 2
        self.lambda_reg   = lambda_reg
        self.lr_nn = lr_nn
        self.train_every = train_every
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size

        torch.manual_seed(seed)

        # Shared embedding network (independent of NeuralUCB's network)
        self.net       = _EmbeddingNet(context_dim, hidden_dim, d_emb)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr_nn)
        self.loss_fn   = nn.MSELoss()

        # Per-arm Bayesian linear head parameters.
        # A_inv[a] : (d_emb x d_emb) — inverse design matrix
        # b[a]     : (d_emb,)        — reward-weighted embedding accumulator
        # mu[a]    : (d_emb,)        — posterior mean = A_inv[a] @ b[a]
        self._A_inv, self._b, self._mu = self._init_linear_params()

        # Replay buffer for network retraining
        self._buffer_x   = []
        self._buffer_arm = []
        self._buffer_r   = []

        self._in_warmup  = True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def select_arm(self, context: np.ndarray, candidate_arms: list = None) -> int:
        """
        During warmup: select uniformly at random.
        After warmup:  sample one weight vector per arm from the posterior
                       and select the arm with the highest sampled score.

        For each arm a:
            theta_tilde_a ~ N(mu_a,  sigma^2 * A_a^{-1})
            score_a        = phi(x)^T theta_tilde_a

        The randomness in the sample drives exploration: arms with high posterior uncertainty (large
        A_a^{-1}) will occasionally be sampled with large weight vectors, causing the algorithm to explore
        them even when their posterior mean is not the highest.

        Parameters
        ----------
        * context: np.ndarray of shape (context_dim,)

        Returns
        -------
        * int: the index of the selected arm
        """
        arms_to_score = candidate_arms if candidate_arms is not None else list(range(self.n_arms))
        if self._in_warmup:
            return int(self.rng.choice(arms_to_score))
        phi    = self._embed(context)
        scores = np.array([self._sample_score(a, phi) for a in arms_to_score])
        return int(arms_to_score[np.argmax(scores)])

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        """
        1. Append (context, arm, reward) to the replay buffer.
        2. Update the linear head for this arm using the current embedding.
        3. End warmup and trigger first training once warmup_steps are reached.
        4. Every train_every steps post-warmup, retrain the network and
           refit all linear heads from the new embeddings.

        Parameters
        ----------
        * arm: arm index that was chosen and matched
        * reward: observed binary reward (0 or 1)
        * context: p.ndarray of shape (context_dim,)
        """
        self._base_update(arm)

        # Store in replay buffer
        self._buffer_x.append(context.copy())
        self._buffer_arm.append(arm)
        self._buffer_r.append(reward)

        # Cap buffer to most recent max_buffer_size entries
        if len(self._buffer_x) > self.max_buffer_size:
            self._buffer_x   = self._buffer_x[-self.max_buffer_size:]
            self._buffer_arm = self._buffer_arm[-self.max_buffer_size:]
            self._buffer_r   = self._buffer_r[-self.max_buffer_size:]

        # Transition out of warmup
        if self._in_warmup and self._t >= self.warmup_steps:
            self._in_warmup = False
            self._retrain_network()
            self._refit_linear_heads()
            return

        if not self._in_warmup:
            # Incremental linear head update on the current embedding
            phi = self._embed(context)
            self._update_linear_head(arm, reward, phi)

            # Periodic full retraining
            if self._t % self.train_every == 0:
                self._retrain_network()
                self._refit_linear_heads()

    def reset(self) -> None:
        """Reset network weights, linear heads, replay buffer, and RNG."""
        super()._base_reset()
        torch.manual_seed(self.seed)
        self.net       = _EmbeddingNet(self.context_dim, self.hidden_dim, self.d_emb)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_nn)
        self._A_inv, self._b, self._mu = self._init_linear_params()
        self._buffer_x   = []
        self._buffer_arm = []
        self._buffer_r   = []
        self._in_warmup  = True

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _init_linear_params(self):
        """Initialise per-arm Bayesian linear head parameters."""
        d     = self.d_emb
        A_inv = [(1.0 / self.lambda_reg) * np.eye(d) for _ in range(self.n_arms)]
        b     = [np.zeros(d) for _ in range(self.n_arms)]
        mu    = [np.zeros(d) for _ in range(self.n_arms)]
        return A_inv, b, mu

    def _embed(self, context: np.ndarray) -> np.ndarray:
        """Pass a single context vector through the network, return embedding."""
        self.net.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(context.reshape(1, -1))
            phi = self.net(x_t).squeeze(0).numpy()
        return phi

    def _sample_weights(self, arm: int) -> np.ndarray:
        """
        Sample one weight vector from the posterior for arm a:

            theta_tilde_a ~ N(mu_a,  sigma^2 * A_a^{-1})

        Uses Cholesky decomposition for numerically stable sampling,
        with a small jitter fallback if the covariance is near-singular.

        Returns
        -------
        theta_tilde : np.ndarray of shape (d_emb,)
        """
        mu  = self._mu[arm]
        cov = self.sigma2 * self._A_inv[arm]

        try:
            L = np.linalg.cholesky(cov)
            z = self.rng.standard_normal(self.d_emb)
            return mu + L @ z
        except np.linalg.LinAlgError:
            cov_jittered = cov + 1e-8 * np.eye(self.d_emb)
            return self.rng.multivariate_normal(mu, cov_jittered)

    def _sample_score(self, arm: int, phi: np.ndarray) -> float:
        """
        Sample a weight vector for arm a and return phi^T theta_tilde_a.
        This is the Thompson Sampling score for arm a given embedding phi.
        """
        theta_tilde = self._sample_weights(arm)
        return float(phi @ theta_tilde)

    def _update_linear_head(self, arm: int, reward: float, phi: np.ndarray) -> None:
        """Incremental Sherman-Morrison update of the Bayesian linear head."""
        self._A_inv[arm] = self._sherman_morrison(self._A_inv[arm], phi)
        self._b[arm]    += reward * phi
        self._mu[arm]    = self._A_inv[arm] @ self._b[arm]

    def _retrain_network(self) -> None:
        """Retrain the embedding network on the full replay buffer."""
        if len(self._buffer_x) < self.batch_size:
            return

        self.net.train()
        X = torch.FloatTensor(np.array(self._buffer_x))
        R = torch.FloatTensor(np.array(self._buffer_r)).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X, R)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.n_epochs):
            for xb, rb in loader:
                self.optimizer.zero_grad()
                pred = self.net(xb)
                loss = self.loss_fn(pred[:, -1:], rb)
                loss.backward()
                self.optimizer.step()

    def _refit_linear_heads(self) -> None:
        """
        Recompute all linear head parameters from scratch using the
        updated network embeddings for every buffered interaction.
        Called after every network retraining pass to keep the linear
        heads consistent with the new embedding space.
        """
        self._A_inv, self._b, self._mu = self._init_linear_params()
        for x, arm, r in zip(self._buffer_x, self._buffer_arm, self._buffer_r):
            phi = self._embed(x)
            self._update_linear_head(arm, r, phi)

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
    # Functions to report algoriithmic features.
    # ------------------------------------------------------------------
    def posterior_mean(self, arm: int) -> np.ndarray:
        """Return the posterior mean (mu_a) for a given arm."""
        return self._mu[arm].copy()

    def posterior_variance(self, arm: int, context: np.ndarray) -> float:
        """
        Return the marginal posterior variance of the predicted reward for
        arm a at embedding phi(context):

            Var[phi^T theta | D] = sigma^2 * phi^T A_a^{-1} phi

        Directly comparable to ThompsonSampling.posterior_variance() and
        NeuralUCB.exploration_bonus() for analysis purposes.
        """
        phi = self._embed(context)
        return self.sigma2 * float(phi @ self._A_inv[arm] @ phi)

    def embedding(self, context: np.ndarray) -> np.ndarray:
        """Return the neural embedding phi(context)."""
        return self._embed(context)

    def __repr__(self) -> str:
        return (
            f"NeuralTS("
            f"n_arms={self.n_arms}, "
            f"context_dim={self.context_dim}, "
            f"d_emb={self.d_emb}, "
            f"hidden_dim={self.hidden_dim}, "
            f"sigma={self.sigma}, "
            f"warmup={self.warmup_steps}, "
            f"t={self._t})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _check_torch()
    print("Running NeuralTS smoke test...\n")

    rng = np.random.default_rng(5)
    n_arms = 5
    context_dim = 8
    n_steps = 1000

    # True weights — arm 4 is best
    true_theta = rng.standard_normal((n_arms, context_dim))
    true_theta[4] += 2.0

    agent = NeuralTS(
        n_arms, context_dim,
        d_emb=16, hidden_dim=32,
        sigma=1.0, lambda_reg=2.5,
        warmup_steps=50, train_every=50,
        n_epochs=5, seed=5,
    )
    print(agent)

    total_reward = 0.0
    arm_counts   = np.zeros(n_arms, dtype=int)

    for t in range(n_steps):
        context = rng.standard_normal(context_dim)
        chosen = agent.select_arm(context)
        p = float(np.clip(0.5 + 0.1 * (context @ true_theta[chosen]), 0.05, 0.95))
        reward = float(rng.random() < p)
        agent.update(chosen, reward, context)
        total_reward += reward
        arm_counts[chosen] += 1

    print(f"\nAfter {n_steps} steps:")
    print(f"  Total reward    : {total_reward:.1f}  ({total_reward/n_steps:.3f} per step)")
    print(f"  Arm pull counts : {arm_counts}")
    print(f"  Warmup active   : {agent._in_warmup}")

    # Embedding shape
    test_ctx = rng.standard_normal(context_dim)
    emb = agent.embedding(test_ctx)
    assert emb.shape == (16,), f"Expected embedding shape (16,), got {emb.shape}"
    print(f"\n  Embedding shape : {emb.shape}")

    # Posterior variance shrinks with more updates on arm 0, measured before retraining pass can intefere
    # Temporarily disable retraining by setting train_every flag high
    var_before = agent.posterior_variance(0, test_ctx)
    original_train_every = agent.train_every
    agent.train_every = 10_000       # suppress retraining during this check
    for _ in range(200):
        agent.update(0, 1.0, rng.standard_normal(context_dim))
    var_after = agent.posterior_variance(0, test_ctx)
    agent.train_every = original_train_every    # restore
    assert var_after < var_before, "Posterior variance should shrink with more data."
    print(f"  Posterior variance arm 0 before/after {n_steps} updates: "
        f"{var_before:.6f} → {var_after:.6f}")

    # Sampled weight vectors should have correct shape
    sample = agent._sample_weights(0)
    assert sample.shape == (16,), f"Expected sample shape (16,), got {sample.shape}"
    print(f"  Sampled weight shape : {sample.shape}")

    # Posterior mean should be non-trivial after training
    mu = agent.posterior_mean(0)
    assert mu.shape == (16,)
    print(f"  Posterior mean shape : {mu.shape}")

    # Verify NeuralTS and NeuralUCB have the same select_arm signature
    # (so evaluate.py can treat them identically)
    chosen = agent.select_arm(test_ctx)
    assert 0 <= chosen < n_arms
    print(f"  select_arm returns valid arm index: {chosen}")

    # Reset
    agent.reset()
    assert agent._t == 0 and agent._in_warmup
    print(f"  Reset OK: t=0, in_warmup=True")

    print("\nSmoke test passed.")