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

__all__ = [
    "BaseBandit",
    "EpsilonGreedy",
]