# Local Setup and Verification

Open this project folder in VS Code, then run the following commands in the terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Expected result: **16 tests pass**.

Run the experiments:

```bash
python scripts/run_experiments.py
```

Summary statistics and three figures are written to `reports/`. All market prices and order flow are simulated. Fixed random seeds make the experiments reproducible; no exchange market data is used.

## Hands-on Experiments

1. **Calculate the reservation price.** With a mid-price of 100, inventory of 5, risk aversion of 0.1, volatility of 2, and remaining time of 1, the reservation price is `100 - 5 * 0.1 * 2**2 * 1 = 98`. Explain why holding a long position shifts quotes downward.
2. **Verify the P&L accounting identity.** Run `python -m pytest -q tests/test_mmsim.py -k pnl_decomposition`. Explain why cash plus the market value of final inventory equals spread P&L plus inventory P&L.
3. **Evaluate adverse selection and spread buffers.** Compare `jumps_only` with `informed_flow`. Then examine how a buffer selected on training paths performs with a fresh test seed. These are tests within a simulated environment, not out-of-sample returns on real market data.

## Model Assumptions and Limitations

Noise trading uses a Bernoulli approximation with at most one fill per side per time step:

```text
fill_probability = clip(A * exp(-k * quote_distance) * dt, 0, 1)
```

This does not simulate exact Poisson arrival counts. Informed traders are assumed to know the next price jump one step in advance; the quoting strategy has no access to that future information.

`informed_pnl` measures the execution edge relative to the post-jump midpoint. It is a diagnostic and must not be added to total P&L, which would double-count its effect.

Explain symmetric quoting and inventory-based quote adjustments before interpreting adverse selection. Results such as lower P&L volatility and improved test-seed P&L should be discussed alongside their assumptions and experimental design.

