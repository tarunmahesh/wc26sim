"""
predict.py — head-to-head match probabilities

Usage:
    python predict.py brazil argentina
    python predict.py usa mexico usa          # 3rd arg = home team
    python predict.py spain germany --scores  # show top correct scores
    python predict.py france england --ko     # show knockout advance prob
"""

import argparse
import sys

from data import ELO, get_elo
from model import (
    load_params, win_draw_loss, xg, over_under,
    btts, correct_score_top, advance_prob,
)


def find_team(name, teams):
    name = name.lower()
    matches = [t for t in teams if t.lower() == name]
    if matches:
        return matches[0]
    # fuzzy fallback
    matches = [t for t in teams if name in t.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous: {matches}")
        sys.exit(1)
    print(f"Team not found: '{name}'")
    print(f"Available: {sorted(teams)}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("home", nargs="?", default=None,
                        help="Home team (optional). Omit for neutral venue.")
    parser.add_argument("--scores",  action="store_true", help="Show top correct scores")
    parser.add_argument("--ko",      action="store_true", help="Show knockout advance probability")
    parser.add_argument("--ratings", default="data/ratings.json")
    args = parser.parse_args()

    params = load_params(args.ratings)
    teams  = params["teams"]

    a = find_team(args.team_a, teams)
    b = find_team(args.team_b, teams)

    neutral = True
    if args.home:
        home_team = find_team(args.home, teams)
        neutral   = False
        if home_team == b:
            a, b = b, a  # put home team first

    elo_a = get_elo(a)
    elo_b = get_elo(b)
    venue = "[neutral]" if neutral else f"[{a} home]"

    print(f"\n  {a} (Elo {elo_a:.0f})  vs  {b} (Elo {elo_b:.0f})   {venue}\n")

    la, lb       = xg(params, a, b)
    pw, pd_, pl  = win_draw_loss(params, a, b)
    ou25         = over_under(params, a, b, 2.5)
    both         = btts(params, a, b)

    w = 30
    print(f"  {'xG':<20} {la:.2f} – {lb:.2f}")
    print(f"  {a[:w]:<20} win   {pw:.1%}  {'█' * int(pw * 40)}")
    print(f"  {'draw':<20}       {pd_:.1%}  {'█' * int(pd_ * 40)}")
    print(f"  {b[:w]:<20} win   {pl:.1%}  {'█' * int(pl * 40)}")
    print(f"\n  Over 2.5 goals    {ou25:.1%}")
    print(f"  Both teams score  {both:.1%}")

    if args.ko:
        pa = advance_prob(params, a, b)
        print(f"\n  P(advance {a[:16]}) = {pa:.1%}")
        print(f"  P(advance {b[:16]}) = {1-pa:.1%}")

    if args.scores:
        print("\n  Top correct scores:")
        for score, prob in correct_score_top(params, a, b, n=8):
            print(f"    {score:<8} {prob:.1%}  {'█' * int(prob * 60)}")

    print()


if __name__ == "__main__":
    main()
