from collections import defaultdict

import numpy as np
import pandas as pd

from data import TEAMS, CONF, GROUPS, R32_PAIRINGS, SEED
from model import sample_match, win_draw_loss, advance_prob

MC_SIMS = 10_000

ROUND_LABEL = {32: "r32", 16: "r16", 8: "qf", 4: "sf", 2: "final", 1: "winner"}


def simulate_group(teams, params, rng):
    pts, gd, gf = defaultdict(int), defaultdict(int), defaultdict(int)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            winner, ga, gb = sample_match(params, a, b, rng, knockout=False)
            gf[a] += ga; gf[b] += gb
            gd[a] += ga - gb; gd[b] += gb - ga
            if winner is None:
                pts[a] += 1; pts[b] += 1
            elif winner == a:
                pts[a] += 3
            else:
                pts[b] += 3
    standings = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    return standings, dict(pts), dict(gd), dict(gf)


def best_thirds(group_results):
    thirds = [(s[2], pts.get(s[2],0), gd.get(s[2],0), gf.get(s[2],0))
              for s, pts, gd, gf in group_results]
    thirds.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return [t[0] for t in thirds[:8]]


def simulate_knockout(group_results, group_names, params, rng):
    winners = {g: r[0][0] for g, r in zip(group_names, group_results)}
    runners = {g: r[0][1] for g, r in zip(group_names, group_results)}
    thirds  = best_thirds(group_results)

    slots = {f"{g}1": winners[g] for g in group_names}
    slots.update({f"{g}2": runners[g] for g in group_names})

    matchups = [(slots[a], slots[b]) for a, b in R32_PAIRINGS]
    for i in range(0, 8, 2):
        matchups.append((thirds[i], thirds[i+1]))
    assert len(matchups) == 16

    exits = {}
    rnd   = 32
    while len(matchups) > 1:
        next_round = []
        for a, b in matchups:
            winner, _, _ = sample_match(params, a, b, rng, knockout=True)
            exits[b if winner == a else a] = rnd
            next_round.append(winner)
        matchups = [(next_round[i], next_round[i+1]) for i in range(0, len(next_round), 2)]
        rnd //= 2

    a, b = matchups[0]
    champion, _, _ = sample_match(params, a, b, rng, knockout=True)
    loser = b if champion == a else a
    exits[loser]    = 2
    exits[champion] = 1
    return champion, exits


def run_mc(params, n=MC_SIMS, seed_offset=0):
    wins   = defaultdict(int)
    rounds = {t: defaultdict(int) for t in TEAMS}
    group_names  = list(GROUPS.keys())
    groups_fixed = [GROUPS[g] for g in group_names]

    for i in range(n):
        rng = np.random.default_rng([SEED + seed_offset, i])
        group_results = [simulate_group(g, params, rng) for g in groups_fixed]
        champion, exits = simulate_knockout(group_results, group_names, params, rng)

        qualifiers = set()
        for res in group_results:
            qualifiers.add(res[0][0]); qualifiers.add(res[0][1])
        for t in best_thirds(group_results):
            qualifiers.add(t)

        for res in group_results:
            for t in res[0][2:]:
                if t not in qualifiers:
                    rounds[t]["group"] += 1

        for team, rnd in exits.items():
            rounds[team][ROUND_LABEL[rnd]] += 1
        wins[champion] += 1

    return wins, rounds


def make_odds_table(wins, rounds, n, params):
    rows = []
    for t in TEAMS:
        rd = rounds[t]
        rows.append({
            "team":    t,
            "conf":    CONF[t],
            "attack":  params["alpha"].get(t, float("nan")),
            "p_win":   wins.get(t, 0) / n,
            "p_final": (rd.get("final",0) + rd.get("winner",0)) / n,
            "p_sf":    sum(rd.get(k,0) for k in ["sf","final","winner"]) / n,
            "p_qf":    sum(rd.get(k,0) for k in ["qf","sf","final","winner"]) / n,
            "p_r16":   sum(rd.get(k,0) for k in ["r16","qf","sf","final","winner"]) / n,
            "p_r32":   sum(rd.get(k,0) for k in ["r32","r16","qf","sf","final","winner"]) / n,
        })
    return pd.DataFrame(rows).sort_values("p_win", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------
# Deterministic bracket — expected-points group rankings, then advance_prob
# -----------------------------------------------------------------------

def _expected_pts(team, others, params):
    return sum(3 * win_draw_loss(params, team, opp)[0] + win_draw_loss(params, team, opp)[1]
               for opp in others if opp != team)


def predicted_bracket(params):
    group_names = list(GROUPS.keys())
    standings, winners, runners, thirds_pool = {}, {}, {}, []

    for grp in group_names:
        members = GROUPS[grp]
        ranked  = sorted(members, key=lambda t: _expected_pts(t, members, params), reverse=True)
        standings[grp] = ranked
        winners[grp]   = ranked[0]
        runners[grp]   = ranked[1]
        thirds_pool.append((ranked[2], _expected_pts(ranked[2], members, params)))

    thirds_pool.sort(key=lambda x: x[1], reverse=True)
    top_thirds = [t for t, _ in thirds_pool[:8]]

    slots = {f"{g}1": winners[g] for g in group_names}
    slots.update({f"{g}2": runners[g] for g in group_names})

    def play_round(matchups):
        out = []
        for a, b in matchups:
            pa = advance_prob(params, a, b)
            w  = a if pa >= 0.5 else b
            out.append({"a": a, "b": b, "winner": w, "loser": b if w==a else a,
                        "p_win": pa if w==a else 1-pa})
        return out

    r32 = play_round([(slots[a], slots[b]) for a, b in R32_PAIRINGS]
                     + [(top_thirds[i], top_thirds[i+1]) for i in range(0, 8, 2)])
    adv = [r["winner"] for r in r32]

    r16 = play_round([(adv[i], adv[i+1]) for i in range(0, 16, 2)])
    adv = [r["winner"] for r in r16]

    qf  = play_round([(adv[i], adv[i+1]) for i in range(0, 8, 2)])
    adv = [r["winner"] for r in qf]

    sf    = play_round([(adv[0], adv[1]), (adv[2], adv[3])])
    adv   = [r["winner"] for r in sf]
    final = play_round([(adv[0], adv[1])])[0]

    return {"group_standings": standings, "top_thirds": top_thirds,
            "r32": r32, "r16": r16, "qf": qf, "sf": sf, "final": final}
