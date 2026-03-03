"""
algorithms/neural_ucb.py

========================

NeuralUCB — neural network contextual bandit with UCB-based exploration.


Full NeuralUCB from Zhou et al. (2020) builds a UCB over all neural network parameters, requiring the
inversion of a matrix of dimension equal to the total parameter count; this is intractable for any
non-basic network. This implemenation of the Neural-Linear approximation for UCB, as proposed by Riquelme
et al. (2018) and Xu et al. (2020), which has become the standard practical implementation. This
implementation does the following:
    1. A shared neural network phi(x; w) maps raw context x to a
       d_emb-dimensional embedding (the last hidden layer output).
    2. A *linear* UCB head is placed on top of the embedding:
           UCB(a, x) = phi(x)^T theta_a  +  alpha * sqrt(phi(x)^T A_a^{-1} phi(x))
       where A_a and theta_a are the same ridge-regression quantities as
       LinUCB, just applied to the learned embedding rather than raw features.
    3. The network is periodically retrained on the accumulated matched
       interaction history to keep the embedding up-to-date.

This yields teh representational power of deep learning fof feature extraction, while the bonus for good
exploration remains tractable with the dimension, namely O(d_emb^2). 

The neural network takes the following architecture:
    Input (context_dim)  →  Linear  →  ReLU
                         →  Linear  →  ReLU
                         →  Linear (embedding, d_emb)
    UCB head: per-arm LinUCB applied to the d_emb-dimensional embedding.

The neural network is retrained eevery `train_every` matched updates, using all accumulated pairs of
(context, reward) for each arm. A warm-up period, as conditioned by `warmup_steps`, operates in purely in
the exploration phase (in its random arm selection) to collect enough data before the first training pass.


References
----------
Primary (NeuralUCB theory):
    Zhou, D., Li, L., & Gu, Q. (2020).
    Neural Contextual Bandits with UCB-based Exploration.
    Proceedings of the 37th ICML, PMLR 119:11492-11502.
    https://proceedings.mlr.press/v119/zhou20a.html
    arXiv: https://arxiv.org/abs/1911.04462

Neural-Linear approximation:
    Riquelme, C., Tucker, G., & Snoek, J. (2018).
    Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian
    Deep Networks for Thompson Sampling.
    ICLR 2018.
    arXiv: https://arxiv.org/abs/1802.09127

Neural-LinUCB variant (deep representation, shallow exploration):
    Xu, P., Wen, Z., Zhao, H., & Gu, Q. (2020).
    Neural Contextual Bandits with Deep Representation and Shallow Exploration.
    arXiv: https://arxiv.org/abs/2012.01780
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
            "NeuralUCB requires PyTorch. Install it with:\n"
            "    pip install torch --break-system-packages"
        )


class _EmbeddingNet(nn.Module if _TORCH_AVAILABLE else object):
    """
    Two-hidden-layer ReLU network mapping context -> embedding.
    Architecture: context_dim -> hidden_dim -> hidden_dim -> d_emb
    """

    def __init__(self, context_dim: int, hidden_dim: int, d_emb: int):
        # Ensure torch loaded
        if not _TORCH_AVAILABLE: raise ImportError("PyTorch not available.")

        # Call superconstructor
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


class NeuralUCB(BaseBandit):
    """
    NeuralUCB with Neural-Linear approximation.

    Parameters
    ----------
    * n_arms:  number of arms in the pool
    * context_dim: dimensionality of raw context vectors
    * d_emb: embedding dimension (last hidden layer width)
    * hidden_dim: width of the two hidden layers
    * alpha:  exploration scalar for UCB (same role as LinUCB alpha)
    * lambda_reg: regularization for the linear UCB head
    * lr_nn: learning rate for neural network training (Adam)
    * train_every: retrain the network every this many matched updates
    * n_epochs: number of training epochs per retraining pass
    * warmup_steps: steps of random exploration before first training
    * batch_size: mini-batch size for network training
    * seed: random seed
    """

    def __init__(self, n_arms: int, context_dim: int, d_emb: int = 32, hidden_dim: int = 64,
        alpha: float = 1.0, lambda_reg: float = 1.0, lr_nn: float = 1e-3, train_every: int = 50,
        n_epochs: int = 10, warmup_steps: int = 100, batch_size: int = 32, seed: int = 42,):

        # Call superconstructor
        _check_torch()
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            seed=seed,
            name="NeuralUCB",
        )

        # Set initial state
        self.d_emb = d_emb
        self.hidden_dim   = hidden_dim
        self.alpha = alpha
        self.lambda_reg   = lambda_reg
        self.lr_nn = lr_nn
        self.train_every  = train_every
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.batch_size   = batch_size

        torch.manual_seed(seed)

        # Shared embedding network
        self.net       = _EmbeddingNet(context_dim, hidden_dim, d_emb)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr_nn)
        self.loss_fn   = nn.MSELoss()

        # Per-arm LinUCB head applied to embeddings
        self._A_inv, self._b, self._theta_lin = self._init_linear_params()

        # Replay buffer: store all matched (context, arm, reward) for retraining
        self._buffer_x   = []   # raw context vectors
        self._buffer_arm = []   # arm indices
        self._buffer_r   = []   # rewards

        # Track whether we are still in warmup
        self._in_warmup = True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def select_arm(self, context: np.ndarray) -> int:
        """
        During warmup: select uniformly at random.
        After warmup:  select arm with highest Neural-Linear UCB score.

        UCB(a, x) = phi(x)^T theta_a  +  alpha * sqrt(phi(x)^T A_a^{-1} phi(x))
        where phi(x) is the neural embedding of x.
        """
        if self._in_warmup:
            return int(self.rng.integers(0, self.n_arms))

        phi = self._embed(context)   # shape (d_emb,)
        scores = np.array([self._compute_ucb(a, phi) for a in range(self.n_arms)])
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        """
        1. Append (context, arm, reward) to the replay buffer.
        2. Update the linear UCB head for this arm using the current embedding.
        3. Every train_every steps, retrain the network on the full buffer and
           recompute all linear head parameters from the new embeddings.
        """
        self._base_update(arm)

        # Store in replay buffer
        self._buffer_x.append(context.copy())
        self._buffer_arm.append(arm)
        self._buffer_r.append(reward)

        # End warmup after warmup_steps matched updates
        if self._in_warmup and self._t >= self.warmup_steps:
            self._in_warmup = False
            self._retrain_network()
            self._refit_linear_heads()
            return

        if not self._in_warmup:
            # Incremental linear head update using current embedding
            phi = self._embed(context)
            self._update_linear_head(arm, reward, phi)

            # Periodic full retraining
            if self._t % self.train_every == 0:
                self._retrain_network()
                self._refit_linear_heads()

    def reset(self) -> None:
        """Reset network, linear heads, buffer, and RNG."""
        super()._base_reset()
        torch.manual_seed(self.seed)
        self.net       = _EmbeddingNet(self.context_dim, self.hidden_dim, self.d_emb)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_nn)
        self._A_inv, self._b, self._theta_lin = self._init_linear_params()
        self._buffer_x   = []
        self._buffer_arm = []
        self._buffer_r   = []
        self._in_warmup  = True

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _init_linear_params(self):
        d = self.d_emb
        A_inv     = [(1.0 / self.lambda_reg) * np.eye(d) for _ in range(self.n_arms)]
        b         = [np.zeros(d) for _ in range(self.n_arms)]
        theta_lin = [np.zeros(d) for _ in range(self.n_arms)]
        return A_inv, b, theta_lin

    def _embed(self, context: np.ndarray) -> np.ndarray:
        """Pass a single context through the network and return embedding."""
        self.net.eval()
        with torch.no_grad():
            x_t  = torch.FloatTensor(context.reshape(1, -1))
            phi  = self.net(x_t).squeeze(0).numpy()
        return phi

    def _compute_ucb(self, arm: int, phi: np.ndarray) -> float:
        """LinUCB UCB score applied to embedding phi."""
        exploit = float(phi @ self._theta_lin[arm])
        bonus   = self.alpha * float(np.sqrt(np.clip(phi @ self._A_inv[arm] @ phi, 0, None)))
        return exploit + bonus

    def _update_linear_head(self, arm: int, reward: float, phi: np.ndarray) -> None:
        """Incremental Sherman-Morrison update of the linear head for one arm."""
        self._A_inv[arm] = self._sherman_morrison(self._A_inv[arm], phi)
        self._b[arm]    += reward * phi
        self._theta_lin[arm] = self._A_inv[arm] @ self._b[arm]

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
                # Simple MSE loss on the norm of the embedding vs reward; use supervised reward loss is used.
                # Here we regress the last embedding dimension toward the reward
                # as a lightweight supervised signal.
                loss = self.loss_fn(pred[:, -1:], rb)
                loss.backward()
                self.optimizer.step()

    def _refit_linear_heads(self) -> None:
        """
        After retraining the network, recompute all linear head parameters
        from scratch using the new embeddings for all buffered interactions.
        This ensures the linear heads are consistent with the current embedding.
        """
        self._A_inv, self._b, self._theta_lin = self._init_linear_params()

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
    # Functions to report algorithmic features.
    # ------------------------------------------------------------------
    def embedding(self, context: np.ndarray) -> np.ndarray:
        """Return the neural embedding for a given context."""
        return self._embed(context)

    def ucb_scores(self, context: np.ndarray) -> np.ndarray:
        """Return UCB scores for all arms given a context (post-warmup only)."""
        phi = self._embed(context)
        return np.array([self._compute_ucb(a, phi) for a in range(self.n_arms)])

    def __repr__(self) -> str:
        return (
            f"NeuralUCB("
            f"n_arms={self.n_arms}, "
            f"context_dim={self.context_dim}, "
            f"d_emb={self.d_emb}, "
            f"hidden_dim={self.hidden_dim}, "
            f"alpha={self.alpha}, "
            f"warmup={self.warmup_steps}, "
            f"t={self._t})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _check_torch()
    print("Running NeuralUCB smoke test...\n")

    rng = np.random.default_rng(3)
    n_arms = 5
    context_dim = 8
    n_steps = 1000

    true_theta  = rng.standard_normal((n_arms, context_dim))
    true_theta[1] += 2.0   # arm 1 is best

    agent = NeuralUCB(
        n_arms, context_dim,
        d_emb=16, hidden_dim=32,
        alpha=1.0, lambda_reg=2.5,
        warmup_steps=50, train_every=50,
        n_epochs=5, seed=3,
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
    print(f"  Total reward    : {total_reward:.1f}  ({total_reward/n_steps:.3f} per step)")
    print(f"  Arm pull counts : {arm_counts}")
    print(f"  Warmup active   : {agent._in_warmup}")

    # Embedding shape check
    test_ctx = rng.standard_normal(context_dim)
    emb      = agent.embedding(test_ctx)
    assert emb.shape == (16,), f"Expected embedding shape (16,), got {emb.shape}"
    print(f"  Embedding shape : {emb.shape}")

    # Reset check
    agent.reset()
    assert agent._t == 0 and agent._in_warmup
    print(f"  Reset OK: t=0, in_warmup=True")

    print("\nSmoke test passed.")