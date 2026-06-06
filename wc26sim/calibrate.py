"""
calibrate.py — fit Dixon-Coles ratings from historical results and save to data/ratings.json

Usage:
    python calibrate.py
    python calibrate.py --min-year 2015
    python calibrate.py --no-shrinkage
"""

import argparse
import math

import numpy as np

from data import load_results, CONF, SEED
from model import fit, apply_conf_shrinkage, save_params, CONF_SHRINKAGE, MIN_MATCHES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-year",     type=int, default=2010)
    parser.add_argument("--no-shrinkage", action="store_true")
    parser.add_argument("--out",          default="data/ratings.json")
    args = parser.parse_args()

    train, _, _ = load_results(min_year=args.min_year)
    print(f"Train: {len(train):,} matches ({args.min_year}–present)\n")

    counts    = train["team_a"].value_counts().add(train["team_b"].value_counts(), fill_value=0)
    all_teams = sorted([t for t in set(train["team_a"]) | set(train["team_b"])
                        if counts.get(t, 0) >= MIN_MATCHES])

    print(f"Fitting on {len(all_teams)} teams:")
    params = fit(train, all_teams)
    print(f"  γ = {params['gamma']:.3f}  δ = {params['delta']:.3f}"
          f"  ρ = {params['rho']:.3f}  μ = {params['mu']:.3f}\n")

    if not args.no_shrinkage:
        print("Applying confederation shrinkage:")
        for conf, s in CONF_SHRINKAGE.items():
            print(f"  {conf:<10}: {s:.0%}")

        check = ["Argentina","Brazil","France","Spain","Morocco","Japan","Canada","Iran","Senegal"]
        print("\n  Alpha before → after:")
        params_c = apply_conf_shrinkage(params)
        for t in check:
            raw = params["alpha"].get(t, 0)
            cor = params_c["alpha"].get(t, 0)
            print(f"    {t:<28} {CONF.get(t,'?'):<10}  {raw:+.3f} → {cor:+.3f}")
        params = params_c
        print()

    save_params(params, args.out)


if __name__ == "__main__":
    main()
