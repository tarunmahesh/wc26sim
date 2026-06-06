# wc26sim

A World Cup 2026 Monte Carlo simulator powered by Dixon-Coles match modeling. Run 10,000+ simulations of the full 48-team tournament to get championship odds, bracket predictions, and head-to-head match probabilities for every qualified team.

---

## How It Works

The model fits **Dixon-Coles** attack and defense ratings for every international team using historical match results (downloaded automatically from [martj42/international_results](https://github.com/martj42/international_results)). Ratings are time-weighted so recent results matter more, and confederation strength adjustments prevent teams from being inflated by weaker regional opposition.

Tournament simulations use these ratings to sample realistic scorelines across all group-stage and knockout rounds, producing probabilistic championship odds from thousands of simulated tournaments.

**Model parameters:**
- `α` (alpha) — team attack strength
- `β` (beta) — team defensive strength
- `γ` (gamma) — home advantage
- `δ` (delta) — Elo-based prior weight
- `ρ` (rho) — Dixon-Coles low-score correction

---

## Installation

```bash
pip install numpy pandas scipy matplotlib
```

---

## Usage

### 1. Calibrate — fit ratings from historical data *(run once)*

```bash
cd wc26sim
python calibrate.py
```

Downloads international results, fits Dixon-Coles ratings, and saves them to `data/ratings.json`. This is a prerequisite for all other scripts.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--min-year YEAR` | `2010` | Only use matches from this year onward |
| `--no-shrinkage` | off | Skip confederation strength adjustment |
| `--out PATH` | `data/ratings.json` | Output path for ratings file |

---

### 2. Simulate the tournament

```bash
python simulate_tournament.py
```

Runs 10,000 Monte Carlo simulations and prints championship odds, final probabilities, and round-by-round survival rates for all 48 teams.

**Options:**
| Flag | Description |
|------|-------------|
| `--sims N` | Number of simulations (default: 10,000) |
| `--bracket` | Print a single deterministic predicted bracket |
| `--bracket --png` | Also save `bracket_2026.png` |
| `--bootstrap` | Add 90% confidence intervals (slow) |

---

### 3. Head-to-head match predictions

```bash
python predict.py <team_a> <team_b> [home_team] [--scores] [--ko]
```

**Examples:**
```bash
python predict.py france england               # neutral venue
python predict.py usa mexico usa               # USA at home
python predict.py spain germany --scores       # show top scorelines
python predict.py brazil argentina --ko        # knockout advance probability
```

Team names are case-insensitive and support partial matching (e.g. `usa` → `United States`).

**Output includes:** xG, win/draw/loss probabilities, over 2.5 goals, both-teams-to-score, and optionally top correct scorelines or knockout advance probability.

---

### 4. Backtest model accuracy

```bash
python backtest.py
```

Walk-forward out-of-sample evaluation on historical matches. Prints accuracy, Brier score, log-loss, RPS, and a reliability diagram.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--min-year YEAR` | `2010` | Start of evaluation window |
| `--burn-in N` | `150` | Matches used to warm up before evaluating |

---

## Project Structure

```
wc26sim/
├── data.py                  # Team data, Elo ratings, groups, WC pairings
├── model.py                 # Dixon-Coles fitting, scoring, match sampling
├── calibrate.py             # Fit ratings and save to data/ratings.json
├── simulate.py              # Group stage + knockout MC simulation logic
├── simulate_tournament.py   # Main entry point: run MC, print odds table
├── predict.py               # Head-to-head match probability CLI
└── backtest.py              # Walk-forward model evaluation
```

---

## Quickstart

```bash
cd wc26sim
pip install numpy pandas scipy matplotlib
python calibrate.py                         # ~1 min, run once
python simulate_tournament.py               # championship odds
python predict.py france england --ko       # head-to-head
```
