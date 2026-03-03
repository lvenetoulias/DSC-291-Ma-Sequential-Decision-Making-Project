"""
algorithms/__init__.py
===========
Contextual bandit algorithm implementations.

All algorithms inherit from BaseBandit (base.py) and implement the same three-method interface:
select_arm(), update(), reset().

Available algorithms
--------------------
    EpsilonGreedy   : contextual epsilon-greedy with per-arm ridge regression
    LinUCB          : disjoint linear upper confidence bound
    ThompsonSampling: linear Gaussian Thompson sampling
    ....            : (other algorithms to implement go here)
"""

from algorithms.base import BaseBandit
from algorithms.epsilon_greedy import EpsilonGreedy
from algorithms.lin_ucb import LinUCB
from algorithms.thompson_sampling import ThompsonSampling
from algorithms.logistic_ucb import LogisticUCB
from algorithms.bootstrap_thompson import BootstrapThompson

# NeuralUCB and NeuralTS require PyTorch — import conditionally so the
# rest of the package remains importable without torch installed.
try:
    from algorithms.neural_ucb import NeuralUCB
    from algorithms.neural_ts import NeuralTS
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False

__all__ = [
    "BaseBandit",
    "EpsilonGreedy",
    "LinUCB",
    "ThompsonSampling",
    "LogisticUCB",
    "BootstrapThompson",
    "NeuralUCB",
    "NeuralTS",
]