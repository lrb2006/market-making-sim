from dataclasses import dataclass

import numpy as np
import pytest

from mmsim.params import MarketParams
from mmsim.simulator import simulate, summarize
from mmsim.strategies import InventoryQuoter, SymmetricQuoter, as_spread, reservation_price

TOXIC = MarketParams(jump_prob=0.05, jump_size=1.5, informed_size=3)


@dataclass
class FixedQuoter:
    """Quotes a constant distance from the mid (for testing fill rates)."""
    delta: float

    def quotes(self, mid, inventory, t, p):
        return mid - self.delta, mid + self.delta


# ---- formulas -------------------------------------------------------------
def test_as_spread_formula():
    p = MarketParams()
    expected = 0.1 * 4.0 * 1.0 + (2 / 0.1) * np.log(1 + 0.1 / 1.5)
    assert as_spread(0.1, p, 0.0) == pytest.approx(expected)
    assert as_spread(0.1, p, 1.0) == pytest.approx((2 / 0.1) * np.log(1 + 0.1 / 1.5))


def test_reservation_price_skews_against_inventory():
    p = MarketParams()
    assert reservation_price(100.0, 0, 0.1, p, 0.0) == 100.0
    assert reservation_price(100.0, 5, 0.1, p, 0.0) == pytest.approx(100.0 - 5 * 0.1 * 4.0)
    assert reservation_price(100.0, -5, 0.1, p, 0.0) > 100.0


def test_symmetric_ignores_inventory_but_inventory_quoter_does_not():
    p = MarketParams()
    mid, q = np.array([100.0, 100.0]), np.array([0.0, 10.0])
    b, a = SymmetricQuoter(0.1).quotes(mid, q, 0.0, p)
    assert np.allclose((a + b) / 2, 100.0)
    b, a = InventoryQuoter(0.1).quotes(mid, q, 0.0, p)
    assert (a + b)[1] / 2 < (a + b)[0] / 2      # long inventory -> quotes move down


# ---- accounting -----------------------------------------------------------
@pytest.mark.parametrize("params", [MarketParams(), TOXIC])
def test_pnl_decomposition_is_exact(params):
    r = simulate(InventoryQuoter(0.1), params, n_paths=300, seed=3)
    # final wealth = cash + inventory marked at the final mid = spread P&L + inventory P&L
    assert np.allclose(r.pnl, r.cash + r.final_inventory * r.final_mid - 0.0 * params.s0)
    assert np.allclose(r.pnl, r.spread_pnl + r.inventory_pnl)


def test_wealth_identity_includes_initial_mid_correctly():
    # Starting flat with zero cash, holding nothing at the end means wealth = cash.
    r = simulate(InventoryQuoter(0.1), MarketParams(), n_paths=200, seed=4)
    flat = r.final_inventory == 0
    assert np.allclose(r.pnl[flat], r.cash[flat])


# ---- market mechanics -----------------------------------------------------
def test_fill_rate_matches_arrival_model():
    p = MarketParams(n_steps=200)
    delta = 0.8
    r = simulate(FixedQuoter(delta), p, n_paths=2000, seed=5)
    expected_per_side = p.A * np.exp(-p.k * delta) * p.dt * p.n_steps
    assert r.fills.mean() == pytest.approx(2 * expected_per_side, rel=0.02)


def test_jumps_without_informed_traders_cost_nothing_on_average():
    # Control experiment: random jumps alone move inventory P&L both ways, with zero mean.
    base = simulate(InventoryQuoter(0.1), MarketParams(), n_paths=4000, seed=11)
    jumps = simulate(InventoryQuoter(0.1), TOXIC.with_(informed_size=0), n_paths=4000, seed=11)
    assert np.all(jumps.informed_pnl == 0)
    diff = jumps.pnl.mean() - base.pnl.mean()
    se = jumps.pnl.std(ddof=1) / np.sqrt(len(jumps.pnl))
    assert abs(diff) < 3 * se
    assert jumps.pnl.std() > base.pnl.std()


def test_no_jumps_means_no_informed_trading():
    r = simulate(SymmetricQuoter(0.1), MarketParams(informed_size=3), n_paths=200, seed=6)
    assert np.all(r.informed_pnl == 0)


def test_informed_traders_only_trade_when_profitable():
    # quotes wider than the jump can never be picked off
    p = TOXIC.with_(jump_size=1.0)
    r = simulate(FixedQuoter(1.2), p, n_paths=200, seed=7)
    assert np.all(r.informed_pnl == 0)
    r = simulate(FixedQuoter(0.5), p, n_paths=200, seed=7)
    assert r.informed_pnl.mean() < 0


@pytest.mark.parametrize("params", [MarketParams(), TOXIC, TOXIC.with_(informed_size=5)])
def test_inventory_limit_is_respected(params):
    # Includes multi-unit informed fills, which must be cut to the room left under the limit.
    r = simulate(InventoryQuoter(0.01, max_inventory=3), params, n_paths=500, seed=8)
    assert r.max_abs_inventory.max() <= 3
    assert np.allclose(r.pnl, r.cash + r.final_inventory * r.final_mid)


def test_common_random_numbers_give_identical_markets():
    a = simulate(SymmetricQuoter(0.1), TOXIC, n_paths=50, seed=9, record_path=True)
    b = simulate(InventoryQuoter(0.5), TOXIC, n_paths=50, seed=9, record_path=True)
    assert np.allclose(a.path["mid"], b.path["mid"])
    assert np.allclose(a.final_mid, b.final_mid)


def test_peak_inventory_includes_informed_fill_before_noise_offset():
    @dataclass
    class CappedFixedQuoter(FixedQuoter):
        max_inventory: int = 3
    # A jump causes a 3-unit informed fill. Guaranteed noise fills then reduce
    # exposure to 2, but the path's maximum exposure must still record 3.
    params = MarketParams(n_steps=1, sigma=0, A=1e6, jump_prob=1,
                          jump_size=2, informed_size=3)
    result = simulate(CappedFixedQuoter(.1), params, n_paths=30, seed=2)
    assert np.all(result.max_abs_inventory == 3)
    assert np.all(np.abs(result.final_inventory) == 2)


# ---- the core economic results -------------------------------------------
def test_inventory_skew_cuts_risk_without_destroying_profit():
    p = MarketParams()
    inv = summarize(simulate(InventoryQuoter(0.1), p, 2000, seed=10))
    sym = summarize(simulate(SymmetricQuoter(0.1), p, 2000, seed=10))
    assert inv["std_pnl"] < 0.6 * sym["std_pnl"]
    assert inv["mean_pnl"] > 0.9 * sym["mean_pnl"]
    assert inv["std_final_inventory"] < 0.5 * sym["std_final_inventory"]
