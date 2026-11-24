"""Run all experiments; writes reports/results.json and figures to reports/.

Usage: python scripts/run_experiments.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmsim.params import MarketParams                       # noqa: E402
from mmsim.simulator import simulate, summarize             # noqa: E402
from mmsim.strategies import InventoryQuoter, SymmetricQuoter  # noqa: E402

OUT = ROOT / "reports"
N_PATHS = 2000
TRAIN_SEED, TEST_SEED = 1, 2
BASE = MarketParams()
TOXIC = MarketParams(jump_prob=0.05, jump_size=1.5, informed_size=3)
BUFFERS = np.round(np.arange(0.0, 1.51, 0.125), 3)
JUMP_SIZES = [1.0, 1.5, 2.0]


def r2(d):
    return {k: round(v, 3) for k, v in d.items()}


def main():
    OUT.mkdir(exist_ok=True)
    results = {"params": {"base": BASE.__dict__, "toxic": TOXIC.__dict__, "n_paths": N_PATHS}}

    # 1. Replicate Avellaneda-Stoikov: inventory-skewed vs symmetric quotes.
    rep = {}
    for g in [0.01, 0.1, 0.5]:
        rep[str(g)] = {s.name: r2(summarize(simulate(s, BASE, N_PATHS, TRAIN_SEED)))
                       for s in [InventoryQuoter(g), SymmetricQuoter(g)]}
    results["replication"] = rep

    # 2. Add informed traders who pick off stale quotes before price jumps.
    results["adverse_selection"] = {
        "no_informed_flow": r2(summarize(simulate(InventoryQuoter(0.1), BASE, N_PATHS, TRAIN_SEED))),
        # Control: the same jumps, but nobody knows about them in advance.
        "jumps_only": r2(summarize(simulate(InventoryQuoter(0.1), TOXIC.with_(informed_size=0), N_PATHS, TRAIN_SEED))),
        "jumps_only_symmetric": r2(summarize(simulate(SymmetricQuoter(0.1), TOXIC.with_(informed_size=0), N_PATHS, TRAIN_SEED))),
        "informed_flow": r2(summarize(simulate(InventoryQuoter(0.1), TOXIC, N_PATHS, TRAIN_SEED))),
        "informed_flow_symmetric": r2(summarize(simulate(SymmetricQuoter(0.1), TOXIC, N_PATHS, TRAIN_SEED))),
    }

    # 3. Choose a spread buffer on training paths, evaluate on fresh test paths.
    sweep, oos = {}, {}
    for J in JUMP_SIZES:
        p = TOXIC.with_(jump_size=J)
        train = [summarize(simulate(InventoryQuoter(0.1, extra_half_spread=b), p, N_PATHS, TRAIN_SEED))
                 for b in BUFFERS]
        sweep[str(J)] = {"buffers": BUFFERS.tolist(),
                         "mean_pnl": [round(t["mean_pnl"], 3) for t in train],
                         "pnl_per_risk": [round(t["pnl_per_risk"], 3) for t in train]}
        best = float(BUFFERS[int(np.argmax([t["mean_pnl"] for t in train]))])
        test0 = summarize(simulate(InventoryQuoter(0.1), p, N_PATHS, TEST_SEED))
        testb = summarize(simulate(InventoryQuoter(0.1, extra_half_spread=best), p, N_PATHS, TEST_SEED))
        oos[str(J)] = {"chosen_buffer": best, "test_no_buffer": r2(test0), "test_with_buffer": r2(testb),
                       "pnl_improvement": round(testb["mean_pnl"] / test0["mean_pnl"] - 1, 3)}
    results["buffer_sweep_train"] = sweep
    results["buffer_out_of_sample"] = oos
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # ---- figures ------------------------------------------------------------
    sym = simulate(SymmetricQuoter(0.1), BASE, N_PATHS, TRAIN_SEED, record_path=True)
    inv = simulate(InventoryQuoter(0.1), BASE, N_PATHS, TRAIN_SEED, record_path=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col",
                             gridspec_kw={"height_ratios": [2, 1]})
    for col, (name, res) in enumerate([("Symmetric quotes", sym), ("Inventory-skewed quotes", inv)]):
        pth = res.path
        ax = axes[0, col]
        ax.plot(pth["t"], pth["mid"], color="black", lw=1, label="mid")
        ax.plot(pth["t"], pth["ask"], color="tab:red", lw=0.8, label="ask")
        ax.plot(pth["t"], pth["bid"], color="tab:blue", lw=0.8, label="bid")
        ax.set_title(name); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
        axes[1, col].step(pth["t"], pth["q"], where="post", color="tab:green")
        axes[1, col].axhline(0, color="grey", lw=0.6)
        axes[1, col].set_ylabel("inventory"); axes[1, col].set_xlabel("time"); axes[1, col].grid(alpha=0.3)
    fig.suptitle("Same market, two strategies (gamma = 0.1)")
    fig.tight_layout(); fig.savefig(OUT / "sample_path.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(sym.pnl.min(), inv.pnl.min()), max(sym.pnl.max(), inv.pnl.max()), 60)
    ax.hist(sym.pnl, bins=bins, alpha=0.6, label=f"Symmetric (std {sym.pnl.std():.1f})")
    ax.hist(inv.pnl, bins=bins, alpha=0.6, label=f"Inventory-skewed (std {inv.pnl.std():.1f})")
    ax.set_xlabel("P&L per session"); ax.set_ylabel("paths"); ax.legend(frameon=False)
    ax.set_title(f"P&L distribution over {N_PATHS} simulated sessions"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "pnl_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for J in JUMP_SIZES:
        s = sweep[str(J)]
        ax.plot(s["buffers"], s["mean_pnl"], marker="o", ms=3, label=f"jump size {J}")
        ax.scatter([oos[str(J)]["chosen_buffer"]], [max(s["mean_pnl"])], s=60, zorder=3,
                   facecolors="none", edgecolors="black")
    ax.axhline(results["adverse_selection"]["no_informed_flow"]["mean_pnl"], color="grey", ls="--",
               lw=1, label="no informed traders")
    ax.set_xlabel("extra half-spread added to each side"); ax.set_ylabel("mean P&L per session")
    ax.set_title("Widening quotes against informed traders (training paths)")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "spread_buffer.png", dpi=160); plt.close(fig)

    # ---- console summary ----------------------------------------------------
    for g, d in rep.items():
        i, s = d["inventory"], d["symmetric"]
        print(f"gamma {g}: spread {i['avg_quoted_spread']:.2f} | inventory pnl {i['mean_pnl']:.1f} "
              f"(std {i['std_pnl']:.1f}, q std {i['std_final_inventory']:.1f}) | symmetric pnl "
              f"{s['mean_pnl']:.1f} (std {s['std_pnl']:.1f}, q std {s['std_final_inventory']:.1f})")
    a = results["adverse_selection"]
    for k, v in a.items():
        print(k, {x: v[x] for x in ["mean_pnl", "std_pnl", "mean_spread_pnl", "mean_inventory_pnl",
                                    "mean_informed_loss", "mean_fills"]})
    for J, v in oos.items():
        print(f"J={J}: buffer {v['chosen_buffer']} | test pnl {v['test_no_buffer']['mean_pnl']:.1f} -> "
              f"{v['test_with_buffer']['mean_pnl']:.1f} ({v['pnl_improvement']:+.0%}), std "
              f"{v['test_no_buffer']['std_pnl']:.1f} -> {v['test_with_buffer']['std_pnl']:.1f}, informed loss "
              f"{v['test_no_buffer']['mean_informed_loss']:.1f} -> {v['test_with_buffer']['mean_informed_loss']:.1f}")


if __name__ == "__main__":
    main()
