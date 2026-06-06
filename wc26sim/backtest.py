"""
backtest.py — walk-forward out-of-sample evaluation on historical internationals

Reproduces the accuracy/Brier/log-loss numbers in the README.
Every match is predicted using only data available before kickoff.

Usage:
    python backtest.py
    python backtest.py --min-year 2015 --burn-in 100
"""

import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from data import load_results, get_elo, SEED
from model import fit, apply_conf_shrinkage, win_draw_loss, MIN_MATCHES


def rps(probs, outcome_idx):
    obs = np.zeros(3); obs[outcome_idx] = 1.0
    return float(np.mean((np.cumsum(probs)[:2] - np.cumsum(obs)[:2]) ** 2))


def run_backtest(min_year=2010, burn_in=150):
    train, _, _ = load_results(min_year=min_year)
    train = train.sort_values("date").reset_index(drop=True)

    print(f"\n{len(train):,} matches loaded  |  burn-in: {burn_in}  |  evaluating: {len(train)-burn_in}\n")

    rps_list, ll_list, brier_list = [], [], []
    correct, total = 0, 0
    correct_fav, total_fav = 0, 0
    bins = defaultdict(list)

    for i in range(burn_in, len(train)):
        chunk = train.iloc[:i]
        row   = train.iloc[i]
        a, b  = row["team_a"], row["team_b"]

        counts    = chunk["team_a"].value_counts().add(chunk["team_b"].value_counts(), fill_value=0)
        all_teams = [t for t in set(chunk["team_a"]) | set(chunk["team_b"])
                     if counts.get(t, 0) >= MIN_MATCHES]
        if a not in all_teams or b not in all_teams:
            continue

        params = fit(chunk, all_teams, verbose=False)
        params = apply_conf_shrinkage(params)
        pw, pd_, pl = win_draw_loss(params, a, b)
        probs = np.array([pw, pd_, pl])

        ga, gb = int(row["goals_a"]), int(row["goals_b"])
        if ga > gb:    oidx, bval, result = 0, 1, "W"
        elif ga == gb: oidx, bval, result = 1, None, "D"
        else:          oidx, bval, result = 2, 0, "L"

        rps_list.append(rps(probs, oidx))
        ll_list.append(-math.log(max(probs[oidx], 1e-9)))
        if bval is not None:
            brier_list.append((pw - bval) ** 2)
            bins[int(pw * 10)].append(bval)

        predicted = ["W","D","L"][np.argmax(probs)]
        total    += 1
        correct  += int(predicted == result)

        if max(probs) >= 0.50:
            total_fav    += 1
            correct_fav  += int(predicted == result)

        if i % 50 == 0:
            print(f"  {i}/{len(train)}  acc={correct/total:.1%}", flush=True)

    n = total
    print(f"\n{'='*60}")
    print(f"Backtest results  ({n} evaluated, {burn_in} burn-in)")
    print(f"{'='*60}")
    print(f"  Correct result (W/D/L)  : {correct/n:.1%}   (always-home ~49%, coin-flip 33%)")
    if total_fav:
        print(f"  When clear favourite    : {correct_fav/total_fav:.1%}   ({total_fav} matches with p≥50%)")
    print(f"  Mean RPS                : {np.mean(rps_list):.4f}  (random=0.333)")
    print(f"  Log-loss (3-outcome)    : {np.mean(ll_list):.4f}  (random=1.099)")
    if brier_list:
        print(f"  Binary Brier            : {np.mean(brier_list):.4f}  (coin-flip=0.250)")

    print("\n  Reliability diagram:")
    print(f"  {'Range':<12}{'Pred':>6}{'Actual':>8}{'N':>6}  Bar")
    print("  " + "-" * 46)
    for b_idx in range(10):
        games = bins.get(b_idx, [])
        if not games: continue
        actual = np.mean(games)
        print(f"  {b_idx/10:.0%}–{(b_idx+1)/10:.0%}   "
              f"  {(b_idx+0.5)/10:.0%}  {actual:.1%}  {len(games):<6}  {'█'*int(actual*24)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--burn-in",  type=int, default=150)
    args = parser.parse_args()
    run_backtest(args.min_year, args.burn_in)


if __name__ == "__main__":
    main()
