import math
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson as sp_poisson

from data import TEAMS, CONF, ELO, SEED, get_elo

MAX_GOALS   = 12
MIN_MATCHES = 3

# Per-confederation shrinkage for attack/defence ratings.
# Dixon-Coles inflates ratings for teams that mostly play within weak confs —
# Japan or Morocco rack up goals against AFC/CAF sides, then look inflated
# vs UEFA/CONMEBOL. We shrink each team's alpha toward the Elo-implied
# residual and beta toward the conf mean. UEFA shrinks least; OFC most.
CONF_SHRINKAGE = {
    "UEFA":     0.10,
    "CONMEBOL": 0.18,
    "AFC":      0.40,
    "CAF":      0.40,
    "CONCACAF": 0.38,
    "OFC":      0.55,
}


def preprocess(teams, data):
    idx  = {t: i for i, t in enumerate(teams)}
    mask = data["team_a"].isin(idx) & data["team_b"].isin(idx)
    d    = data[mask].copy()

    ia  = np.array([idx[t] for t in d["team_a"]], dtype=np.int32)
    ib  = np.array([idx[t] for t in d["team_b"]], dtype=np.int32)
    ga  = d["goals_a"].values.astype(np.int32)
    gb  = d["goals_b"].values.astype(np.int32)
    wt  = d["weight"].values.astype(np.float64)
    home = (~d["neutral"].values.astype(bool)).astype(np.float64)
    elo_diff = (
        np.array([get_elo(t) for t in d["team_a"]], dtype=np.float64)
      - np.array([get_elo(t) for t in d["team_b"]], dtype=np.float64)
    ) / 400.0
    lgf_a = np.array([math.lgamma(g + 1) for g in ga])
    lgf_b = np.array([math.lgamma(g + 1) for g in gb])

    return ia, ib, ga, gb, wt, home, elo_diff, lgf_a, lgf_b


def neg_ll(x, n, ia, ib, ga, gb, wt, home, elo_diff, lgf_a, lgf_b):
    alpha = np.concatenate([[0.0], x[:n-1]])
    beta  = x[n-1: 2*n-1]
    mu, gamma, delta, rho = x[2*n-1], x[2*n], x[2*n+1], x[2*n+2]

    log_la = mu + alpha[ia] - beta[ib] + gamma * home + delta * elo_diff
    log_lb = mu + alpha[ib] - beta[ia]                - delta * elo_diff
    la, lb = np.exp(log_la), np.exp(log_lb)

    t = np.ones(len(ga))
    t[(ga==0)&(gb==0)] = 1.0 - la[(ga==0)&(gb==0)] * lb[(ga==0)&(gb==0)] * rho
    t[(ga==1)&(gb==0)] = 1.0 + lb[(ga==1)&(gb==0)] * rho
    t[(ga==0)&(gb==1)] = 1.0 + la[(ga==0)&(gb==1)] * rho
    t[(ga==1)&(gb==1)] = 1.0 - rho
    if np.any(t <= 0):
        return 1e9

    ll = np.log(t) + ga*log_la - la - lgf_a + gb*log_lb - lb - lgf_b
    return -float(np.dot(wt, ll))


def fit(train, teams, verbose=True):
    n    = len(teams)
    args = preprocess(teams, train)
    rng  = np.random.default_rng(SEED)
    x0   = np.concatenate([rng.normal(0, 0.05, n-1), rng.normal(0, 0.05, n), [0.0, 0.3, 0.1, 0.0]])
    bounds = [(-4,4)]*(n-1) + [(-4,4)]*n + [(-2,2),(0.0,1.0),(-1.0,1.0),(-0.99,0.5)]

    if verbose:
        print(f"  Fitting on {len(args[0])} matches, {n} teams...", end=" ", flush=True)

    res = minimize(neg_ll, x0, args=(n, *args), method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-7})

    if verbose:
        print(f"{'✓' if res.success else '~'} LL={-res.fun:.1f}")

    a_all = np.concatenate([[0.0], res.x[:n-1]]); a_all -= a_all.mean()
    b_all = res.x[n-1: 2*n-1];                    b_all -= b_all.mean()

    try:
        hess_inv = np.array(res.hess_inv.todense())
    except AttributeError:
        hess_inv = np.eye(len(res.x)) * 1e-4

    return {
        "teams":     teams,
        "alpha":     dict(zip(teams, a_all)),
        "beta":      dict(zip(teams, b_all)),
        "mu":        float(res.x[2*n-1]),
        "gamma":     float(res.x[2*n]),
        "delta":     float(res.x[2*n+1]),
        "rho":       float(res.x[2*n+2]),
        "_x_opt":    res.x.tolist(),
        "_hess_inv": hess_inv.tolist(),
    }


def params_from_vector(x, teams):
    n     = len(teams)
    a_all = np.concatenate([[0.0], x[:n-1]]); a_all -= a_all.mean()
    b_all = x[n-1: 2*n-1];                    b_all -= b_all.mean()
    return {
        "teams": teams,
        "alpha": dict(zip(teams, a_all)),
        "beta":  dict(zip(teams, b_all)),
        "mu":    float(x[2*n-1]),
        "gamma": max(0.0, float(x[2*n])),
        "delta": float(x[2*n+1]),
        "rho":   float(np.clip(x[2*n+2], -0.99, 0.5)),
    }


def apply_conf_shrinkage(params):
    conf_alphas = defaultdict(list)
    conf_betas  = defaultdict(list)
    for t in params["teams"]:
        c = CONF.get(t, "UEFA")
        conf_alphas[c].append(params["alpha"][t])
        conf_betas[c].append(params["beta"][t])

    mean_beta = {c: np.mean(v) for c, v in conf_betas.items()}

    new_alpha, new_beta = {}, {}
    for t in params["teams"]:
        c  = CONF.get(t, "UEFA")
        s  = CONF_SHRINKAGE.get(c, 0.2)
        elo_r = (ELO.get(t, 1700) - 1800) / 600.0
        new_alpha[t] = (1 - s) * params["alpha"][t] + s * elo_r
        new_beta[t]  = (1 - s*0.6) * params["beta"][t] + (s*0.6) * mean_beta[c]

    all_a = np.array(list(new_alpha.values()))
    all_b = np.array(list(new_beta.values()))
    new_alpha = {t: v - all_a.mean() for t, v in new_alpha.items()}
    new_beta  = {t: v - all_b.mean() for t, v in new_beta.items()}

    return {**params, "alpha": new_alpha, "beta": new_beta}


# -----------------------------------------------------------------------
# Match probability functions — all take a params dict, no class needed
# -----------------------------------------------------------------------

def _dc_matrix(la, lb, rho, mg=MAX_GOALS):
    gs  = np.arange(mg + 1)
    mat = np.outer(sp_poisson.pmf(gs, la), sp_poisson.pmf(gs, lb))
    mat[0,0] = max(mat[0,0] * (1.0 - la*lb*rho), 0.0)
    mat[1,0] = max(mat[1,0] * (1.0 + lb*rho),    0.0)
    mat[0,1] = max(mat[0,1] * (1.0 + la*rho),    0.0)
    mat[1,1] = max(mat[1,1] * (1.0 - rho),        0.0)
    s = mat.sum()
    return mat / s if s > 0 else mat


def lambdas(params, a, b, neutral=True):
    ga  = params["alpha"].get(a, (get_elo(a) - 1800) / 600)
    gb  = params["alpha"].get(b, (get_elo(b) - 1800) / 600)
    da  = params["beta"].get(a,  -(get_elo(a) - 1800) / 600)
    db  = params["beta"].get(b,  -(get_elo(b) - 1800) / 600)
    ed  = (get_elo(a) - get_elo(b)) / 400.0
    h   = 0.0 if neutral else params["gamma"]
    la  = math.exp(params["mu"] + ga - db + h + params["delta"] * ed)
    lb  = math.exp(params["mu"] + gb - da     - params["delta"] * ed)
    return la, lb


def match_matrix(params, a, b):
    la, lb = lambdas(params, a, b)
    return _dc_matrix(la, lb, params["rho"])


def win_draw_loss(params, a, b):
    mat = match_matrix(params, a, b)
    w   = float(np.tril(mat, -1).sum())
    d   = float(np.diag(mat).sum())
    return w, d, 1.0 - w - d


def xg(params, a, b):
    return lambdas(params, a, b)


def over_under(params, a, b, line=2.5):
    mat  = match_matrix(params, a, b)
    mg   = MAX_GOALS
    R, C = np.meshgrid(np.arange(mg+1), np.arange(mg+1), indexing="ij")
    return float(mat[R + C > line].sum())


def btts(params, a, b):
    mat = match_matrix(params, a, b)
    return float(1.0 - mat[0,:].sum() - mat[:,0].sum() + mat[0,0])


def correct_score_top(params, a, b, n=10):
    mat  = match_matrix(params, a, b)
    mg   = MAX_GOALS
    rows = [(f"{i}-{j}", mat[i,j]) for i in range(mg+1) for j in range(mg+1) if mat[i,j] > 1e-6]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:n]


def advance_prob(params, a, b):
    pw, pd90, _ = win_draw_loss(params, a, b)
    la, lb      = lambdas(params, a, b)
    et_mat      = _dc_matrix(la/3, lb/3, 0.0, 5)
    et_win      = float(np.tril(et_mat, -1).sum())
    et_draw     = float(np.diag(et_mat).sum())
    mat         = match_matrix(params, a, b)
    p_pen       = 0.5 + 0.04 * (float(np.tril(mat, -1).sum()) - 0.5)
    return float(pw + pd90 * (et_win + et_draw * p_pen))


def sample_match(params, a, b, rng, knockout=False):
    mat  = match_matrix(params, a, b)
    flat = mat.ravel()
    mg   = MAX_GOALS
    idx  = rng.choice(len(flat), p=flat)
    ga, gb = divmod(idx, mg + 1)

    if ga != gb:
        return (a if ga > gb else b), ga, gb
    if not knockout:
        return None, ga, gb

    la, lb = lambdas(params, a, b)
    ga += int(rng.poisson(la/3)); gb += int(rng.poisson(lb/3))
    if ga != gb:
        return (a if ga > gb else b), ga, gb

    p_pen = 0.5 + 0.04 * (float(np.tril(mat, -1).sum()) - 0.5)
    return (a if rng.random() < p_pen else b), ga, gb


# -----------------------------------------------------------------------
# Serialisation helpers — save/load params to JSON for calibrate.py
# -----------------------------------------------------------------------

def save_params(params, path="data/ratings.json"):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {k: v for k, v in params.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Ratings saved → {path}")


def load_params(path="data/ratings.json"):
    with open(path) as f:
        return json.load(f)
