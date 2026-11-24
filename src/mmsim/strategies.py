"""Quoting strategies.

Every strategy maps (mid price, inventory, time) to a bid and an ask price.
All inputs are numpy arrays over simulation paths, so one call quotes every path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import MarketParams


def as_spread(gamma: float, p: MarketParams, t: float) -> float:
    """Total bid-ask spread of the Avellaneda-Stoikov optimal quotes at time t.

    spread = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
    The first term pays for inventory risk; the second is the monopoly markup
    implied by how quickly order arrivals fall as quotes move away from the mid.
    """
    return gamma * p.sigma**2 * (p.T - t) + (2.0 / gamma) * np.log1p(gamma / p.k)


def reservation_price(mid, inventory, gamma: float, p: MarketParams, t: float):
    """Price at which the market maker is indifferent to holding one more unit.

    r = s - q * gamma * sigma^2 * (T - t): long inventory lowers it (quote lower
    to sell), short inventory raises it.
    """
    return mid - inventory * gamma * p.sigma**2 * (p.T - t)


@dataclass
class SymmetricQuoter:
    """Quote the same spread as Avellaneda-Stoikov, centred on the mid; ignores inventory."""
    gamma: float
    extra_half_spread: float = 0.0
    name: str = "symmetric"

    def quotes(self, mid, inventory, t, p: MarketParams):
        half = as_spread(self.gamma, p, t) / 2 + self.extra_half_spread
        return mid - half, mid + half


@dataclass
class InventoryQuoter:
    """Avellaneda-Stoikov: centre quotes on the reservation price instead of the mid.

    max_inventory: if set, stop quoting the side that would push |inventory| past it.
    """
    gamma: float
    extra_half_spread: float = 0.0
    max_inventory: int | None = None
    name: str = "inventory"

    def quotes(self, mid, inventory, t, p: MarketParams):
        r = reservation_price(mid, inventory, self.gamma, p, t)
        half = as_spread(self.gamma, p, t) / 2 + self.extra_half_spread
        bid, ask = r - half, r + half
        if self.max_inventory is not None:
            bid = np.where(inventory >= self.max_inventory, -np.inf, bid)
            ask = np.where(inventory <= -self.max_inventory, np.inf, ask)
        return bid, ask
