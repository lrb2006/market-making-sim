"""Monte Carlo simulation of a market maker quoting one asset.

Each step, in order:
  1. The market maker posts a bid and an ask (strategy.quotes).
  2. Informed traders: if a price jump is about to happen and the stale quote
     is on the wrong side of the new price, they take `informed_size` units.
  3. Noise traders: a buy order lifts the ask with probability A*exp(-k*delta_a)*dt,
     a sell order hits the bid with probability A*exp(-k*delta_b)*dt
     (delta = distance of the quote from the mid). Each fill is one unit.
  4. The mid moves: +/- sigma*sqrt(dt) with equal probability, plus any jump.

P&L is split exactly into
  spread P&L    = sum of (fill price - mid) edge earned on each fill at fill time
  inventory P&L = sum of inventory * subsequent mid moves
so final wealth = cash + inventory * final mid = spread P&L + inventory P&L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import MarketParams


@dataclass
class SimResult:
    pnl: np.ndarray              # final wealth per path (cash + inventory marked at final mid)
    spread_pnl: np.ndarray       # edge captured from all fills
    inventory_pnl: np.ndarray    # gains/losses from holding inventory as the mid moves
    informed_pnl: np.ndarray     # execution edge vs. post-jump mid; diagnostic, not additive P&L
    final_inventory: np.ndarray
    cash: np.ndarray             # cash account (starts at 0)
    final_mid: np.ndarray
    max_abs_inventory: np.ndarray
    fills: np.ndarray            # noise + informed units traded per path
    avg_spread: float            # average quoted spread (finite quotes only)
    path: dict | None = None     # first path in detail, for plots


def simulate(strategy, p: MarketParams, n_paths: int = 1000, seed: int = 0,
             record_path: bool = False) -> SimResult:
    rng = np.random.default_rng(seed)
    n, dt = p.n_steps, p.dt
    # Draw all randomness up front so every strategy sees the same market (common random numbers).
    diffusion = np.where(rng.random((n, n_paths)) < 0.5, 1.0, -1.0) * p.sigma * np.sqrt(dt)
    jump_dir = np.where(rng.random((n, n_paths)) < 0.5, 1.0, -1.0)
    jumps = np.where(rng.random((n, n_paths)) < p.jump_prob, jump_dir * p.jump_size, 0.0)
    u_buy = rng.random((n, n_paths))
    u_sell = rng.random((n, n_paths))

    mid = np.full(n_paths, p.s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    spread_pnl = np.zeros(n_paths)
    informed_pnl = np.zeros(n_paths)
    inv_pnl = np.zeros(n_paths)
    fills = np.zeros(n_paths)
    max_q = np.zeros(n_paths)
    spreads = []
    # Inventory limit: no single fill may push |inventory| past it (inf = no limit).
    limit = getattr(strategy, "max_inventory", None)
    cap = np.inf if limit is None else float(limit)
    rec = {"mid": [], "bid": [], "ask": [], "q": []} if record_path else None

    for i in range(n):
        t = i * dt
        bid, ask = strategy.quotes(mid, q, t, p)
        finite = np.isfinite(bid) & np.isfinite(ask)
        if finite.any():
            spreads.append(float(np.mean((ask - bid)[finite])))
        if rec is not None:
            for key, val in (("mid", mid), ("bid", bid), ("ask", ask), ("q", q)):
                rec[key].append(float(val[0]))

        # 2. informed traders pick off stale quotes just before a jump
        new_mid = mid + jumps[i]
        if p.informed_size > 0:
            hit_ask = (jumps[i] > 0) & (ask < new_mid)      # they buy cheap before the rise
            hit_bid = (jumps[i] < 0) & (bid > new_mid)      # they sell high before the fall
            # A side switched off by the inventory limit is quoted at +/-inf; np.where below
            # discards those inf * 0 products, so silence the warning they raise.
            with np.errstate(invalid="ignore"):
                # fill size is capped by the room left under the inventory limit
                size_ask = np.where(hit_ask, np.clip(np.minimum(p.informed_size, q + cap), 0, None), 0.0)
                size_bid = np.where(hit_bid, np.clip(np.minimum(p.informed_size, cap - q), 0, None), 0.0)
                # np.where avoids inf * 0 = nan when a side is switched off (quote at +/- inf)
                spread_pnl += np.where(size_ask > 0, (ask - mid) * size_ask, 0.0)
                spread_pnl += np.where(size_bid > 0, (mid - bid) * size_bid, 0.0)
                # the loss relative to the post-jump price, for reporting
                informed_pnl += np.where(size_ask > 0, (ask - new_mid) * size_ask, 0.0)
                informed_pnl += np.where(size_bid > 0, (new_mid - bid) * size_bid, 0.0)
                q += size_bid - size_ask
                cash += np.where(size_ask > 0, ask * size_ask, 0.0) - np.where(size_bid > 0, bid * size_bid, 0.0)
                fills += size_ask + size_bid
                # Record the exposure before noise fills can offset it.
                max_q = np.maximum(max_q, np.abs(q))

        # 3. noise traders
        with np.errstate(over="ignore"):
            p_buy = np.clip(p.A * np.exp(-p.k * (ask - mid)) * dt, 0.0, 1.0)
            p_sell = np.clip(p.A * np.exp(-p.k * (mid - bid)) * dt, 0.0, 1.0)
        lifted = (u_buy[i] < p_buy) & (q - 1 >= -cap)    # someone buys from us at the ask
        hit = (u_sell[i] < p_sell) & (q + 1 <= cap)       # someone sells to us at the bid
        spread_pnl += np.where(lifted, ask - mid, 0.0) + np.where(hit, mid - bid, 0.0)
        q += hit.astype(float) - lifted.astype(float)
        cash += np.where(lifted, ask, 0.0) - np.where(hit, bid, 0.0)
        fills += lifted.astype(float) + hit.astype(float)   # bool + bool would be a logical OR
        max_q = np.maximum(max_q, np.abs(q))

        # 4. the mid moves; inventory is marked to market
        move = jumps[i] + diffusion[i]
        inv_pnl += q * move
        mid = mid + move

    pnl = spread_pnl + inv_pnl
    path = None
    if rec is not None:
        path = {k: np.array(v) for k, v in rec.items()}
        path["t"] = np.arange(n) * dt
    return SimResult(pnl=pnl, spread_pnl=spread_pnl, inventory_pnl=inv_pnl,
                     informed_pnl=informed_pnl, final_inventory=q, cash=cash, final_mid=mid,
                     max_abs_inventory=max_q,
                     fills=fills, avg_spread=float(np.mean(spreads)) if spreads else float("nan"),
                     path=path)


def summarize(res: SimResult) -> dict:
    return {
        "mean_pnl": float(res.pnl.mean()),
        "std_pnl": float(res.pnl.std(ddof=1)),
        "pnl_per_risk": float(res.pnl.mean() / res.pnl.std(ddof=1)),
        "p5_pnl": float(np.percentile(res.pnl, 5)),
        "mean_spread_pnl": float(res.spread_pnl.mean()),
        "mean_inventory_pnl": float(res.inventory_pnl.mean()),
        "mean_informed_loss": float(res.informed_pnl.mean()),
        "std_final_inventory": float(res.final_inventory.std(ddof=1)),
        "mean_max_abs_inventory": float(res.max_abs_inventory.mean()),
        "mean_fills": float(res.fills.mean()),
        "avg_quoted_spread": res.avg_spread,
    }
