"""Model parameters.

Defaults follow the numerical example in Avellaneda & Stoikov (2008),
"High-frequency trading in a limit order book": one trading session of
length T = 1 split into 200 steps, mid price 100, volatility 2.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MarketParams:
    s0: float = 100.0       # initial mid price
    sigma: float = 2.0      # volatility of the mid price per unit time
    T: float = 1.0          # session length
    n_steps: int = 200      # time steps (dt = T / n_steps)
    A: float = 140.0        # arrival intensity of noise orders at zero distance
    k: float = 1.5          # how fast arrivals decay with quote distance
    # Adverse selection: price jumps that informed traders see one step early.
    jump_prob: float = 0.0  # probability of a jump in each step
    jump_size: float = 0.0  # absolute jump size (direction is random)
    informed_size: int = 0  # units an informed trader takes at the stale quote

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    def with_(self, **kw) -> "MarketParams":
        return replace(self, **kw)
