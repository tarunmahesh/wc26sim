"""
simulate_tournament.py — run 10,000 MC simulations and print championship odds

Usage:
    python simulate_tournament.py
    python simulate_tournament.py --sims 50000
    python simulate_tournament.py --bracket          # print deterministic bracket
    python simulate_tournament.py --bracket --png    # also save bracket_2026.png
    python simulate_tournament.py --bootstrap        # add 90% CI (slow)
"""

import argparse
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from data import CONF, GROUPS
from model import load_params, CONF_SHRINKAGE
from simulate import run_mc, make_odds_table, predicted_bracket, MC_SIMS


# -----------------------------------------------------------------------
# Bracket PNG
# -----------------------------------------------------------------------

CONF_COLORS = {
    "UEFA":     "#003087",
    "CONMEBOL": "#006847",
    "CONCACAF": "#BF0A30",
    "CAF":      "#C8960C",
    "AFC":      "#CC0001",
    "OFC":      "#6A1B9A",
}

def _conf_color(team):
    return CONF_COLORS.get(CONF.get(team, "UEFA"), "#555555")


def draw_bracket(bracket, out_path="bracket_2026.png"):
    fig_w, fig_h = 28, 22
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")
    ax.set_facecolor("#0A1628"); fig.patch.set_facecolor("#0A1628")

    round_x    = {"R32": 1.2, "R16": 6.2, "QF": 10.8, "SF": 14.8, "Final": 18.5}
    prev_round = {"R16": "R32", "QF": "R16", "SF": "QF"}
    box_w, box_h, gap = 3.8, 0.52, 0.12
    r32_slot_h = (fig_h - 1.2) / 16

    def r32_y(i):
        return 0.6 + (i + 0.5) * r32_slot_h

    def later_y(rnd, i):
        f = {"R16": 2, "QF": 4, "SF": 8, "Final": 16}[rnd]
        return (r32_y(i*f) + r32_y(i*f + f - 1)) / 2

    def draw_box(x, yc, ta, tb, winner, p_win):
        yt, yb = yc + gap/2, yc - gap/2 - box_h
        for team, yr in [(ta, yt), (tb, yb)]:
            win = (team == winner)
            ax.add_patch(FancyBboxPatch((x, yr), box_w, box_h,
                boxstyle="round,pad=0.02",
                facecolor="#1E3A5F" if win else "#0D1F35",
                edgecolor=_conf_color(team) if win else "#2A4060",
                linewidth=1.8 if win else 0.8, zorder=3))
            ax.add_patch(FancyBboxPatch((x, yr), 0.18, box_h,
                boxstyle="round,pad=0.0", facecolor=_conf_color(team),
                edgecolor="none", zorder=4))
            short = team if len(team) <= 16 else team[:15] + "."
            ax.text(x+0.28, yr+box_h*0.5, short, ha="left", va="center",
                    fontsize=7.5, zorder=5,
                    color="#FFFFFF" if win else "#AABBCC",
                    fontweight="bold" if win else "normal")
        if winner:
            ax.text(x+box_w-0.08, yt+box_h*0.5, f"{p_win:.0%}",
                    ha="right", va="center", fontsize=6.2, color="#88BBFF", zorder=5)
        return yt+box_h/2, yb+box_h/2

    def line(x0, y0, x1, y1):
        xm = (x0 + x1) / 2
        ax.plot([x0,xm,xm,x1],[y0,y0,y1,y1], color="#2A5080", lw=0.8, zorder=2)

    # headers + title
    for label, hx in round_x.items():
        ax.text(hx+box_w/2, fig_h-0.3, label, ha="center", va="top",
                fontsize=10, color="#88CCFF", fontweight="bold", zorder=6)
    ax.text(fig_w/2, fig_h-0.05,
            "FIFA World Cup 2026 — Predicted Bracket  (Dixon-Coles + Elo)",
            ha="center", va="top", fontsize=11, color="#FFFFFF",
            fontweight="bold", zorder=6)

    # draw rounds
    r32_wy = []
    for idx, res in enumerate(bracket["r32"]):
        yt, yb = draw_box(round_x["R32"], r32_y(idx), res["a"], res["b"], res["winner"], res["p_win"])
        r32_wy.append(yt if res["winner"] == res["a"] else yb)

    prev_wy = r32_wy
    for rnd, results in [("R16",bracket["r16"]),("QF",bracket["qf"]),("SF",bracket["sf"])]:
        cur_wy = []
        for idx, res in enumerate(results):
            yc = later_y(rnd, idx)
            yt, yb = draw_box(round_x[rnd], yc, res["a"], res["b"], res["winner"], res["p_win"])
            cur_wy.append(yt if res["winner"] == res["a"] else yb)
            line(round_x[prev_round[rnd]]+box_w, prev_wy[idx*2],   round_x[rnd], yc+gap/2+box_h/2)
            line(round_x[prev_round[rnd]]+box_w, prev_wy[idx*2+1], round_x[rnd], yc-gap/2-box_h/2)
        prev_wy = cur_wy

    fin = bracket["final"]
    yf  = fig_h / 2
    draw_box(round_x["Final"], yf, fin["a"], fin["b"], fin["winner"], fin["p_win"])
    line(round_x["SF"]+box_w, prev_wy[0], round_x["Final"], yf+gap/2+box_h/2)
    line(round_x["SF"]+box_w, prev_wy[1], round_x["Final"], yf-gap/2-box_h/2)

    ax.text(round_x["Final"]+box_w/2, yf-box_h-0.55, f"🏆  {fin['winner']}",
            ha="center", va="top", fontsize=13, color="#FFD700", fontweight="bold", zorder=6)

    # legend
    lx, ly = 22.5, 1.2
    ax.text(lx, ly+0.45*len(CONF_COLORS)+0.3, "Confederation",
            ha="left", va="bottom", fontsize=8, color="#AABBCC", fontweight="bold")
    for i, (conf, col) in enumerate(CONF_COLORS.items()):
        ax.add_patch(FancyBboxPatch((lx, ly+i*0.45), 0.28, 0.32,
                     boxstyle="round,pad=0.0", facecolor=col, edgecolor="none", zorder=4))
        ax.text(lx+0.35, ly+i*0.45+0.18, conf, ha="left", va="center",
                fontsize=7.5, color="#DDDDDD")

    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Bracket PNG saved → {out_path}")


# -----------------------------------------------------------------------
# Bootstrap CI
# -----------------------------------------------------------------------

def _one_boot(b_idx, params, mc_n):
    import numpy as np
    from model import params_from_vector, apply_conf_shrinkage
    from simulate import run_mc
    from data import TEAMS, SEED

    rng = np.random.default_rng([SEED + 77777, b_idx])
    x   = rng.multivariate_normal(params["_x_opt"], params["_hess_inv"])
    try:
        p = params_from_vector(x, params["teams"])
        p = apply_conf_shrinkage(p)
        w, _ = run_mc(p, n=mc_n, seed_offset=b_idx + 5000)
        return {t: w.get(t,0) / mc_n for t in TEAMS}
    except Exception:
        return {}


def bootstrap_ci(params, n_boot=200, mc_per_boot=300):
    from joblib import Parallel, delayed
    from data import TEAMS
    from collections import defaultdict

    workers = os.cpu_count() or 1
    print(f"\nBootstrap: {n_boot} draws × {mc_per_boot} MC on {workers} cores...")

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_one_boot)(b, params, mc_per_boot) for b in range(n_boot)
    )

    boot = defaultdict(list)
    for res in results:
        for t, v in res.items():
            boot[t].append(v)

    import pandas as pd
    rows = [{"team": t, "mean": np.mean(boot[t]),
             "lo90": np.percentile(boot[t], 5),
             "hi90": np.percentile(boot[t], 95)}
            for t in TEAMS if boot[t]]
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims",      type=int, default=MC_SIMS)
    parser.add_argument("--bracket",   action="store_true")
    parser.add_argument("--png",       action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--ratings",   default="data/ratings.json")
    args = parser.parse_args()

    params = load_params(args.ratings)

    # Monte Carlo
    print(f"Running {args.sims:,} simulations...\n")
    wins, rounds = run_mc(params, args.sims)
    table = make_odds_table(wins, rounds, args.sims, params)

    print("=" * 88)
    print(f"{'Rk':<4}{'Team':<26}{'Conf':<10}"
          f"{'Win%':>7}{'Final':>7}{'SF':>7}{'QF':>7}{'R16':>7}{'R32':>7}  Atk")
    print("=" * 88)
    for i, row in table.iterrows():
        atk = f"{row['attack']:+.2f}" if not math.isnan(row["attack"]) else " n/a"
        print(f"{i+1:<4}{row['team']:<26}{row['conf']:<10}"
              f"{row['p_win']:>6.1%}{row['p_final']:>7.1%}{row['p_sf']:>7.1%}"
              f"{row['p_qf']:>7.1%}{row['p_r16']:>7.1%}{row['p_r32']:>7.1%}  {atk}")
    print(f"\nPredicted champion: {table.iloc[0]['team']} ({table.iloc[0]['p_win']:.1%})\n")

    if args.bracket or args.png:
        bracket = predicted_bracket(params)
        print(f"\nPredicted bracket:")
        print(f"  Champion:  {bracket['final']['winner']}")
        print(f"  Runner-up: {bracket['final']['loser']}")
        print(f"  SF losers: {bracket['sf'][0]['loser']}, {bracket['sf'][1]['loser']}")
        if args.png:
            draw_bracket(bracket)

    if args.bootstrap:
        # Need _x_opt and _hess_inv — only available if you pass the raw fit params
        print("\nNote: bootstrap requires raw fit params (run calibrate.py first with --save-raw)")
        ci = bootstrap_ci(params)
        print("\n90% CI — top 20:")
        print("=" * 62)
        for _, row in ci.head(20).iterrows():
            print(f"{row['team']:<26}{row['mean']:>6.1%}  "
                  f"[{row['lo90']:.1%} – {row['hi90']:.1%}]  "
                  f"{'█' * int(row['mean'] * 200)}")


if __name__ == "__main__":
    main()
